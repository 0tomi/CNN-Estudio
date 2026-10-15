import ast
import json
from pathlib import Path
import pytest
import torch
from cnn_lab.model import CNN,Config,Block,Conv,MAX_CONV_FILTERS,validate
from cnn_lab.config_export import config_json

@pytest.mark.parametrize('channels,accepted',[(300,True),(301,False),(0,False)])
def test_filter_boundaries(channels,accepted):
    config=Config(blocks=[Block([Conv(channels)],pooling='none')],head='gap',dense=[])
    if accepted:validate(config)
    else:
        with pytest.raises(ValueError,match='filtros 1–300'):validate(config)


def test_filters_json_forward_and_ui_limit_without_training():
    config=Config(blocks=[Block([Conv(300)],pooling='none')],head='gap',dense=[])
    restored=Config.from_dict(json.loads(config_json(config)))
    assert MAX_CONV_FILTERS==300 and restored.dict()==config.dict()
    model=CNN(restored)
    with torch.no_grad():output=model(torch.zeros(1,1,8,8))
    assert output.shape==(1,10) and torch.isfinite(output).all()
    assert model.rows[0]['salida']=='300 × 8 × 8'
    assert Config().blocks[0].layers[0].channels==16 and Config().blocks[1].layers[0].channels==32
    tree=ast.parse((Path(__file__).resolve().parents[1]/'app.py').read_text())
    field=next(call for call in ast.walk(tree) if isinstance(call,ast.Call) and call.args and isinstance(call.args[0],ast.Constant) and call.args[0].value=='Cantidad de filtros / mapas de salida')
    assert isinstance(field.args[2],ast.Name) and field.args[2].id=='MAX_CONV_FILTERS'
