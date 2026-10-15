import copy
import numpy as np
import torch
from streamlit.testing.v1 import AppTest
from cnn_lab.model import Config,CNN
from cnn_lab.data import Data
from cnn_lab.comparison import test_fingerprint,comparison_tables,comparison_figure


def example(identity,epochs=2):
    config=Config();images=torch.arange(30*64,dtype=torch.float32).reshape(30,1,8,8)/(30*64)
    data=Data(images,torch.arange(30)%10,np.arange(10,20),np.arange(20,30),np.arange(10),0.,1.,list(map(str,range(10))))
    history=[dict(epoch=i,train_loss=2/i,val_loss=3/i,train_accuracy=i/10,val_accuracy=i/20) for i in range(1,epochs+1)]
    return dict(_experiment_id=identity,config=config.dict(),model=CNN(config),data=data,history=history,probabilities=np.eye(10),best_epoch=epochs,epochs_completed=epochs,elapsed=.5)


def test_comparison_exact_curves_metrics_and_order_independent_test_identity():
    first=example('a',2);second=example('b',3)
    assert test_fingerprint(first)==test_fingerprint(second)
    second['data'].test=second['data'].test[::-1].copy();second['probabilities']=second['probabilities'][::-1].copy()
    assert test_fingerprint(first)==test_fingerprint(second)
    metrics,settings=comparison_tables([first,second])
    assert list(metrics['Accuracy · TEST'])==[1.,1.] and list(metrics['Épocas completadas'])==[2,3]
    assert list(settings['Máximo de épocas'])==[8,8]
    figure=comparison_figure([first,second])
    assert len(figure.data)==4 and list(figure.data[0].x)==[1,2] and list(figure.data[2].x)==[1,2,3]
    assert list(figure.data[0].y)==[2,1] and figure.data[0].line.dash=='solid' and figure.data[1].line.dash=='dash'
    assert figure.data[0].line.color==figure.data[1].line.color!=figure.data[2].line.color


def test_comparison_mismatched_content_guard_and_ui_without_inference(monkeypatch):
    first=example('first');second=example('second')
    second['data'].images=second['data'].images.clone();second['data'].images[0,0,0,0]=.99
    assert test_fingerprint(first)!=test_fingerprint(second)
    def forbidden(*args,**kwargs):raise AssertionError('No se debe inferir ni entrenar para comparar')
    for candidate in [first,second]:monkeypatch.setattr(candidate['model'],'forward',forbidden)
    app=AppTest.from_string('import streamlit as st\nfrom cnn_lab.comparison import render_comparison\nrender_comparison(st.session_state.runs)')
    app.session_state.runs=[first,second];app.run(timeout=15)
    assert not app.exception and any('test distintos' in warning.value for warning in app.warning)
    assert len(app.get('plotly_chart'))==1 and len(app.dataframe)==2
    assert list(app.dataframe[0].value['Accuracy · TEST'])==[1.,1.]
    app.radio(key='comparison_metric').set_value('Accuracy').run(timeout=15)
    assert not app.exception
    app.session_state.runs=[first];app.run(timeout=15)
    assert any('al menos2' in info.value for info in app.info)
