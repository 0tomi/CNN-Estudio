import copy
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from cnn_lab.model import Config,CNN,Conv,Block
from cnn_lab.data import prepare
from cnn_lab.training import train,EarlyStopping
from cnn_lab.cross_validation import cross_validate

@pytest.mark.parametrize('settings',[{'early_stopping':1},{'patience':True},{'patience':0},{'patience':31},{'patience':1.5},{'min_delta':float('nan')},{'min_delta':float('inf')},{'min_delta':-1},{'min_delta':True},{'min_delta':'bad'}])
def test_early_config_validation(settings):
    with pytest.raises(ValueError,match='Parada temprana'):CNN(Config(**settings))

def test_strict_delta_and_reference_baseline():
    stop=EarlyStopping(2,.125)
    assert not stop.update(1.)
    assert not stop.update(.875) # Exactly delta is NOT a significant improvement.
    assert stop.reference_loss==1. and stop.wait==1
    assert not stop.update(.75)
    assert stop.wait==0 and stop.reference_loss==.75
    assert not stop.update(.75)
    assert stop.update(.75)
    zero=EarlyStopping(1,0.)
    assert not zero.update(1.) and zero.update(1.)

def controlled_train(monkeypatch,losses,**settings):
    import cnn_lab.training as training
    config=Config(samples=200,batch_size=64,epochs=len(losses),**settings)
    data=prepare(config);original=training.predict;states=[];seen=[]
    def controlled(model,fold_data,indices):
        seen.append(indices.copy())
        if np.array_equal(indices,data.val) and len(states)<len(losses):
            states.append(copy.deepcopy(model.state_dict()))
            loss=losses[len(states)-1]
            labels=data.labels[data.val]
            logits=torch.zeros(len(labels),10)
            logits[torch.arange(len(labels)),labels]=math.log(9)-math.log(math.expm1(loss))
            return logits.softmax(1).numpy(),logits
        return original(model,fold_data,indices)
    monkeypatch.setattr(training,'predict',controlled)
    result=train(config,data=data)
    assert np.array_equal(seen[-1],data.test)
    assert sum(np.array_equal(x,data.test) for x in seen)==1
    expected=states[result['best_epoch']-1]
    for key,value in result['model'].state_dict().items():assert torch.equal(value,expected[key])
    assert result['epochs_completed']==len(result['history'])
    assert len(result['batches'])==result['epochs_completed']*math.ceil(len(data.train)/config.batch_size)
    return result

def test_disabled_runs_all_epochs(monkeypatch):
    result=controlled_train(monkeypatch,[1.,1.,1.,1.,1.],early_stopping=False,patience=1,min_delta=10.)
    assert result['epochs_completed']==5 and not result['stopped_early']
    assert result['stop_reason']=='max_epochs'

def test_patience_and_subdelta_restore_absolute_best(monkeypatch):
    result=controlled_train(monkeypatch,[1.,.9,.8,.7,.6],early_stopping=True,patience=2,min_delta=10.)
    assert result['epochs_completed']==3 and result['best_epoch']==3
    assert result['stopped_early'] and result['stop_reason']=='patience'
    assert [r['patience_count'] for r in result['history']]==[0,1,2]

def test_significant_improvement_resets_counter(monkeypatch):
    result=controlled_train(monkeypatch,[1.,.9,.7,.65,.6,.6,.6],early_stopping=True,patience=2,min_delta=.2)
    assert result['epochs_completed']==5
    assert [r['patience_count'] for r in result['history']]==[0,1,0,1,2]
    assert result['best_epoch']==5

def test_last_epoch_patience_does_not_claim_skipped_epochs(monkeypatch):
    result=controlled_train(monkeypatch,[1.,1.,1.],early_stopping=True,patience=2,min_delta=0.)
    assert result['epochs_completed']==result['max_epochs']==3
    assert not result['stopped_early'] and result['stop_reason']=='patience'

def test_legacy_default_and_even_kernel_pool():
    payload=Config().dict()
    for name in ['early_stopping','patience','min_delta']:payload.pop(name)
    c=Config.from_dict(payload)
    assert c.early_stopping is False and c.patience==3 and c.min_delta==.0001
    model=CNN(Config(blocks=[Block([Conv(kernel=4)],pooling='max',pool_size=3)],dense=[]))
    assert model.rows[0]['salida']=='16 × 7 × 7'
    assert model.rows[2]['salida']=='16 × 2 × 2'
    assert model(torch.rand(2,1,8,8)).shape==(2,10)
    for conv in [Conv(kernel=0),Conv(kernel=8),Conv(kernel=2.5),Conv(kernel=True)]:
        with pytest.raises(ValueError):CNN(Config(blocks=[Block([conv])]))

def test_kfold_independent_early_stops_no_test(monkeypatch):
    import cnn_lab.training as training
    c=Config(samples=300,epochs=8,early_stopping=True,patience=2,min_delta=10.)
    d=prepare(c);original=training.predict;calls=[]
    def inspect(model,data,indices):
        assert not set(indices)&set(d.test)
        calls.append(list(indices))
        return original(model,data,indices)
    monkeypatch.setattr(training,'predict',inspect)
    result=cross_validate(c,3,data=d)
    assert [f['epochs_completed'] for f in result['folds']]==[3,3,3]
    assert all(f['stopped_early'] and f['stop_reason']=='patience' for f in result['folds'])
    assert all([h['patience_count'] for h in f['history']]==[0,1,2] for f in result['folds'])
    assert len(calls)==12

def test_app_early_toggle_and_real_completion():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    assert not at.exception
    assert at.checkbox(key='cfg_early_stopping').value is False
    assert not any(n.key=='cfg_patience' for n in at.number_input)
    at.checkbox(key='cfg_early_stopping').check().run(timeout=30)
    at.number_input(key='cfg_patience').set_value(2)
    at.number_input(key='cfg_min_delta').set_value(10.)
    at.number_input(key='cfg_samples').set_value(300)
    next(b for b in at.button if b.label=='Entrenar CNN').click()
    at.run(timeout=60)
    assert not at.exception
    run=at.session_state['runs'][-1]
    assert run['epochs_completed']==3 and run['max_epochs']==8
    assert any('3 de 8' in s.value and 'paciencia' in s.value for s in at.success)
    assert any('lr 0.00300' in m.value for m in at.markdown)
    at.selectbox(key='cfg_optimizer').set_value('SGD').run(timeout=30)
    assert any(n.key=='cfg_momentum' for n in at.number_input)
    at.number_input(key='cfg_0_0_k').set_value(4).run(timeout=30)
    assert not at.exception
    assert any('4×4' in c.value for c in at.caption)

def test_zero_delta_improvement_resets_then_equality_stops(monkeypatch):
    result=controlled_train(monkeypatch,[1.,1.,.8,.8,.8,.7],early_stopping=True,patience=2,min_delta=0.)
    assert result['epochs_completed']==5 and result['best_epoch']==3
    assert [row['patience_count'] for row in result['history']]==[0,1,0,1,2]
