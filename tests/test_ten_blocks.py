import json
from pathlib import Path
import numpy as np
import pytest
import torch
from streamlit.testing.v1 import AppTest
from cnn_lab.model import CNN,Config,Block,Conv,MAX_CONV_BLOCKS
from cnn_lab.explain import gradcam_all
from cnn_lab.interpretability import layer_title
from cnn_lab.visualization import architecture_nodes


def ten_config():
    return Config(blocks=[Block([Conv(4)],pooling='none') for _ in range(MAX_CONV_BLOCKS)],head='gap',dense=[])


def test_ten_blocks_forward_roundtrip_shapes_rf_and_figures():
    config=ten_config(); restored=Config.from_dict(json.loads(json.dumps(config.dict())))
    assert restored.dict()==config.dict() and MAX_CONV_BLOCKS==10
    model=CNN(restored)
    output=model(torch.rand(2,1,8,8))
    assert output.shape==(2,10) and torch.isfinite(output).all()
    assert len(model.conv_names())==10 and model.conv_names()[-1]=='b10_conv1'
    last=next(row for row in model.rows if row['capa']=='b10_conv1')
    assert last['salida']=='4 × 8 × 8' and last['RF teórico']==21
    nodes=architecture_nodes(model)
    assert sum(node['parameters'] for node in nodes)==sum(parameter.numel() for parameter in model.parameters())
    assert len([node for node in nodes if node['label'].startswith('Bloque convolucional')])==10
    assert next(node for node in nodes if node['label']=='Bloque convolucional 10')['shape']=='4 × 8 × 8'
    assert layer_title('b10_conv1')=='Bloque 10 · convolución 1'
    result=gradcam_all(model,torch.rand(1,1,8,8),0)
    assert len(result['layers'])==10 and all(np.isfinite(layer['heat']).all() for layer in result['layers'].values())


def test_eleven_blocks_rejected_and_pool_collapse_still_actionable():
    config=ten_config(); config.blocks.append(Block(pooling='none'))
    with pytest.raises(ValueError,match='1–10 bloques'): CNN(config)
    config=ten_config()
    for block in config.blocks: block.pooling='max'
    with pytest.raises(ValueError,match='Pooling.*colapsa'): CNN(config)
    assert len(Config().blocks)==2


def test_app_ten_blocks_controls_invalid_then_configurable():
    at=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    at.number_input(key='cfg_blocks').set_value(10).run(timeout=30)
    assert not at.exception
    assert len([expander for expander in at.expander if expander.label.startswith('Bloque convolucional')])==10
    assert next(button for button in at.button if button.label=='Entrenar CNN').disabled
    assert any('colapsa' in error.value for error in at.error)
    for index in range(10): at.selectbox(key=f'cfg_{index}_pool').set_value('none')
    at.run(timeout=30)
    assert not at.exception and not next(button for button in at.button if button.label=='Entrenar CNN').disabled
    assert at.number_input(key='cfg_blocks').value==10
    assert any('b10_conv1' in str(table.value) for table in at.dataframe)
