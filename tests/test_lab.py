import copy
import numpy as np
import pytest
import torch
from cnn_lab.model import CNN,Config,Conv,Block
from cnn_lab.data import prepare
from cnn_lab.metrics import evaluate
from cnn_lab.explain import explain

def test_default_shapes_rf_and_roundtrip():
    c=Config(); m=CNN(c)
    assert m(torch.zeros(2,1,8,8)).shape==(2,10)
    assert m.rows[0]['RF teórico']==3
    assert m.rows[3]['RF teórico']==8
    assert Config.from_dict(c.dict()).dict()==c.dict()
    assert not isinstance(m.classifier[-1],torch.nn.Softmax)

def test_hidden_ten_is_not_output():
    m=CNN(Config(hidden=[10,10]))
    assert len([x for x in m.classifier if isinstance(x,torch.nn.Linear)])==3
    assert isinstance(m.classifier[-1],torch.nn.Linear)

@pytest.mark.parametrize('config',[Config(blocks=[Block([Conv(kernel=5,padding=0)],pooling='max'),Block([Conv(kernel=5,padding=0)])]),Config(lr=-1),Config(hidden=[1000]),Config(blocks=[]),Config(dropout=.9)])
def test_invalid(config):
    with pytest.raises(ValueError): CNN(config)

def test_split_and_train_normalization():
    d=prepare(Config())
    assert not set(d.train)&set(d.val)
    assert not set(d.train)&set(d.test)
    assert not set(d.val)&set(d.test)
    assert len(d.train)+len(d.val)+len(d.test)==1797
    assert abs(d.mean-float(d.images[d.train].mean()))<1e-7
    assert abs(float(d.normalized(d.train).mean()))<1e-5
    again=prepare(Config())
    assert np.array_equal(d.train,again.train)

def test_metrics_and_undefined_auc():
    y=np.arange(10); p=np.eye(10)
    result=evaluate(y,p)
    assert result['accuracy']==result['f1']==result['auc']==1
    assert evaluate(np.zeros(3,dtype=int),np.eye(10)[[0,0,0]])['auc'] is None
    with pytest.raises(ValueError): evaluate(y,p*2)

def test_explanations_cleanup_and_eval_restore():
    m=CNN(Config(blocks=[Block(batchnorm=True),Block([Conv(32)],batchnorm=True)])); m.train(); x=torch.rand(1,1,8,8)
    layer=m.conv_names()[0]; module=m.features.get_submodule(layer)
    state=copy.deepcopy(m.state_dict())
    for method in ['gradcam','neuron']:
        r=explain(m,x,layer,method=method)
        assert r['heat'].shape==(8,8)
        assert np.isfinite(r['heat']).all()
        assert m.training
        assert not module._forward_hooks
    for k,v in state.items(): assert torch.equal(v,m.state_dict()[k])
    with pytest.raises(IndexError): explain(m,x,layer,channel=999,method='neuron')
    assert not module._forward_hooks
    assert m.training

def test_app_empty_state():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(__import__('pathlib').Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    assert not at.exception
    assert any('Entrenar CNN'==b.label for b in at.button)


def test_app_real_training_flow():
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    at=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    at.number_input(key="cfg_epochs").set_value(2)
    next(b for b in at.button if b.label=="Entrenar CNN").click()
    at.run(timeout=60)
    assert not at.exception
    assert len(at.session_state["runs"])==1
    assert len(at.session_state["runs"][0]["history"])==2
    at.number_input(key="cfg_lr").set_value(.01).run(timeout=30)
    assert not at.exception
    assert len(at.session_state["runs"])==1
    assert any("no coinciden" in x.value for x in at.warning)


def test_batchnorm_singleton_actionable():
    from cnn_lab.training import train
    c=Config(samples=230,batch_size=16,epochs=1,blocks=[Block(),Block([Conv(32)]),Block([Conv(32,stride=2)],batchnorm=True,pooling="none")])
    with pytest.raises(ValueError,match="último batch"):
        train(c)
