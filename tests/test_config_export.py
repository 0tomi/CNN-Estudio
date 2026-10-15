import json
from pathlib import Path
from cnn_lab.model import Config,Block,Conv,DenseLayer
from cnn_lab.config_export import config_json
from streamlit.testing.v1 import AppTest


def test_config_json_complete_roundtrip_and_legacy_migration():
    config=Config(dataset='CIFAR10Animals',blocks=[Block([Conv(8),Conv(12)],batchnorm=True,pooling='none')],dense=[DenseLayer(128,'gelu',.2),DenseLayer(64,'tanh',.3)],epochs=12,early_stopping=True,patience=4,min_delta=.002,augment=True)
    exported=json.loads(config_json(config))
    assert Config.from_dict(exported).dict()==config.dict()
    assert not {'weights','state_dict','probabilities','history','model'}&exported.keys()
    legacy=Config().dict();legacy.pop('dense');legacy.update(hidden=[128,64],activation='tanh',dropout=.25)
    migrated=json.loads(config_json(legacy))
    assert [layer['units'] for layer in migrated['dense']]==[128,64]
    assert Config.from_dict(migrated).dict()==Config.from_dict(legacy).dict()


def test_current_and_frozen_exports_visible_and_differ_after_edit(monkeypatch):
    import streamlit as st
    original=st.download_button;captured={}
    def download(label,data,*args,**kwargs):
        if label in ('Exportar arquitectura e hiperparámetros (JSON)','Exportar configuración del experimento (JSON)'):
            captured[label]=json.loads(data)
        return original(label,data,*args,**kwargs)
    monkeypatch.setattr(st,'download_button',download)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    assert 'Exportar arquitectura e hiperparámetros (JSON)' in captured
    app.number_input(key='cfg_epochs').set_value(1)
    next(button for button in app.button if button.label=='Entrenar CNN').click().run(timeout=60)
    next(button for button in app.button if button.label=='Volver a configuración').click().run(timeout=30)
    app.number_input(key='cfg_lr').set_value(.02).run(timeout=30)
    current=captured['Exportar arquitectura e hiperparámetros (JSON)'];frozen=captured['Exportar configuración del experimento (JSON)']
    assert current['lr']==.02 and frozen['lr']==.003
    assert Config.from_dict(frozen).dict()==app.session_state['runs'][0]['config']
    assert not app.exception
    # A new session restores the experiment; its export remains the frozen config.
    recovered=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    assert captured['Exportar configuración del experimento (JSON)']==frozen
    assert recovered.session_state['store_restored']==1 and not recovered.exception
