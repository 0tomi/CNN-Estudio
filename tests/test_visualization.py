import pytest
from torch import nn
from cnn_lab.model import CNN,Config,Block,Conv,DenseLayer
from cnn_lab.visualization import architecture_figure,architecture_nodes,neuron_figure

@pytest.mark.parametrize('dataset',['Digits','CIFAR10Animals'])
def test_block_and_detailed_nodes_match_real_model(dataset):
    model=CNN(Config(dataset=dataset,blocks=[Block([Conv(16),Conv(16)],batchnorm=True),Block([Conv(32)])]))
    compact=architecture_nodes(model)
    assert len(compact)==1+len(model.config.blocks)+1+len(model.config.dense)+1
    assert sum(node['parameters'] for node in compact)==sum(p.numel() for p in model.parameters())
    assert compact[0]['shape']==f'{model.input_channels} × {model.input_size} × {model.input_size}'
    detailed=architecture_nodes(model,True)
    assert len(detailed)==len(model.rows)+1
    assert [node['shape'] for node in detailed[1:]]==[row['salida'] for row in model.rows]
    assert detailed[1]['parameters']==model.rows[0]['parámetros']
    assert 'Kernel (3, 3)' in detailed[1]['hover']
    for mode in [False,True]:
        fig=architecture_figure(model,mode)
        assert fig.layout.meta['edge_count']==len(fig.layout.meta['nodes'])-1
        assert len(fig.data[1].x)==len(fig.layout.meta['nodes'])
        assert fig.layout.height>=480
        assert fig.layout.dragmode=='pan'
        assert all(node['rf'] is None for node in fig.layout.meta['nodes'] if node['kind'] in ('densa','salida'))

@pytest.mark.parametrize('depth',[0,1,5])
@pytest.mark.parametrize('dataset,outputs',[('Digits',10),('CIFAR10Animals',6)])
def test_dense_neuron_caps_connectivity_and_outputs(depth,dataset,outputs):
    model=CNN(Config(dataset=dataset,dense=[DenseLayer(128)]*depth))
    fig=neuron_figure(model)
    meta=fig.layout.meta
    linear=[module for module in model.classifier if isinstance(module,nn.Linear)]
    assert meta['actual_sizes']==[linear[0].in_features]+[layer.out_features for layer in linear]
    assert len(meta['shown_indices'])==depth+2
    assert all(len(indices)<=12 for indices in meta['shown_indices'][:-1])
    assert meta['shown_indices'][-1]==list(range(outputs))
    assert len(meta['output_classes'])==outputs
    edges=sum(len(a)*len(b) for a,b in zip(meta['shown_indices'],meta['shown_indices'][1:]))
    assert meta['edge_count']==edges
    assert len(fig.data[0].x)==edges*3
    assert sum(len(trace.x) for trace in fig.data[1:])==sum(len(indices) for indices in meta['shown_indices'])
    assert all(0<=index<size for indices,size in zip(meta['shown_indices'],meta['actual_sizes']) for index in indices)
    assert fig.layout.height>=520
    assert any('mostradas de' in annotation.text for annotation in fig.layout.annotations)

def test_gap_features_and_invalid_cap():
    model=CNN(Config(head='gap',dense=[]))
    fig=neuron_figure(model)
    assert fig.layout.meta['actual_sizes']==[32,10]
    assert architecture_nodes(model)[3]['label']=='GAP + Flatten'
    with pytest.raises(ValueError):neuron_figure(model,cap=100)

def test_app_switches_readable_architecture_modes():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    assert not at.exception
    for mode in ['Operaciones detalladas','Neuronas de la MLP','Bloques CNN + MLP']:
        at.radio(key='architecture_view').set_value(mode).run(timeout=30)
        assert not at.exception
    at.number_input(key='cfg_hidden_count').set_value(5).run(timeout=30)
    at.radio(key='architecture_view').set_value('Neuronas de la MLP').run(timeout=30)
    assert not at.exception
    assert any('hasta 12' in caption.value for caption in at.caption)
