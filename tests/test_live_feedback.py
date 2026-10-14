from pathlib import Path
import pytest
from cnn_lab.model import Config
from cnn_lab.training import train

def test_preparation_phase_precedes_slow_dataset(monkeypatch):
    import cnn_lab.training as training
    phases=[]
    def blocked(_config,**kwargs):
        assert phases and phases[-1]=='preparing', 'No hay estado visible antes de preparar/descargar'
        raise TimeoutError('lectura de red bloqueada')
    monkeypatch.setattr(training,'prepare',blocked)
    with pytest.raises(TimeoutError):
        train(Config(),on_event=lambda event:phases.append(event['phase']))

def test_button_acknowledged_before_blocking_prepare(monkeypatch):
    import cnn_lab.training as training
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    def blocked(_config,**kwargs):
        live=st.session_state.get('live_training',{})
        assert live.get('phase')=='preparing', 'Click sin reconocimiento previo a la preparación'
        raise TimeoutError('preparación detenida para reproducir')
    monkeypatch.setattr(training,'prepare',blocked)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    next(b for b in app.button if b.label=='Entrenar CNN').click()
    app.run(timeout=30)
    assert not app.exception
    assert any('preparación detenida' in error.value for error in app.error)
    assert app.session_state['live_training']['phase']=='error'

def test_actual_events_batch_stats_and_incremental_validation():
    from cnn_lab.data import prepare
    config=Config(samples=300,epochs=3);data=prepare(config);events=[];snapshots=[]
    def batch(trace,history,batches,total):
        assert trace['samples_seen']>0 and trace['train_loss_running']>0
        assert trace['step']==len(batches)
        if trace.get('epoch_complete'):snapshots.append([row.copy() for row in history])
    run=train(config,batch,data=data,on_event=events.append)
    assert [len(rows) for rows in snapshots]==[1,2,3]
    assert all('val_loss' in row and 'train_loss' in row for rows in snapshots for row in rows)
    phases=[event['phase'] for event in events]
    assert phases.index('preparing')<phases.index('initializing')<phases.index('training')<phases.index('validation')<phases.index('test')
    assert phases[-1]=='completed'
    assert len(run['history'])==3

def test_live_job_consumed_once_retry_and_preview(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import cnn_lab.training as training
    original=training.prepare;calls=[]
    def prepare_once(config,**kwargs):calls.append(config.dict());return original(config,**kwargs)
    monkeypatch.setattr(training,'prepare',prepare_once)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    app.number_input(key='cfg_epochs').set_value(2)
    next(b for b in app.button if b.label=='Entrenar CNN').click().run(timeout=60)
    assert len(calls)==1 and not app.exception
    assert app.session_state['view']=='live'
    assert app.session_state['live_training']['phase']=='completed'
    assert len(app.session_state['live_training']['history'])==2
    assert not any(b.label=='Entrenar CNN' for b in app.button)
    app.run(timeout=30)
    assert len(calls)==1 and len(app.session_state['runs'])==1
    next(b for b in app.button if b.label=='Volver a configuración').click().run(timeout=30)
    next(b for b in app.button if b.label=='Preparar y visualizar datos').click().run(timeout=30)
    assert app.session_state['data_preparation']['phase']=='completed'
    assert 'preview_data' in app.session_state

def test_error_keeps_last_model_and_explicit_retry(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import cnn_lab.training as training
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    app.number_input(key='cfg_epochs').set_value(1)
    next(b for b in app.button if b.label=='Entrenar CNN').click().run(timeout=60)
    previous=app.session_state['runs'][-1]['model']
    next(b for b in app.button if b.label=='Volver a configuración').click().run(timeout=30)
    original=training.prepare
    def failed(*args,**kwargs):raise TimeoutError('red detenida')
    monkeypatch.setattr(training,'prepare',failed)
    next(b for b in app.button if b.label=='Entrenar CNN').click().run(timeout=30)
    assert app.session_state['live_training']['phase']=='error'
    assert len(app.session_state['runs'])==1 and app.session_state['runs'][-1]['model'] is previous
    monkeypatch.setattr(training,'prepare',original)
    next(b for b in app.button if b.label=='Volver a configuración').click().run(timeout=30)
    next(b for b in app.button if b.label=='Entrenar CNN').click().run(timeout=60)
    assert len(app.session_state['runs'])==2 and app.session_state['live_training']['phase']=='completed'

def test_interrupted_view_has_recovery_without_auto_execution(monkeypatch):
    import time
    import cnn_lab.training as training
    from streamlit.testing.v1 import AppTest
    def forbidden(*args,**kwargs):raise AssertionError('No debe reejecutar un trabajo consumido')
    monkeypatch.setattr(training,'train',forbidden)
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py')
    app.session_state['view']='live'
    app.session_state['live_training']={'phase':'training','config':Config().dict(),'started':time.monotonic(),'history':[],'batches':[]}
    app.run(timeout=30)
    assert not app.exception
    assert app.session_state['live_training']['phase']=='error'
    assert any('interrumpida' in error.value for error in app.error)
    assert any(b.label=='Volver a configuración' for b in app.button)
