import copy
import io
import json
from pathlib import Path
import uuid
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest
from cnn_lab.experiment_store import ExperimentStore
from cnn_lab.training import train
from cnn_lab.model import Config,Block,Conv
from cnn_lab.data import Data,training_statistics,display_image
from cnn_lab.explain import gradcam_all
from cnn_lab.inference import classify_latest

@pytest.fixture
def digits():
    return train(Config(samples=300,epochs=1))


def test_digits_roundtrip_offline_exact_weights_probabilities_maps(tmp_path,digits,monkeypatch):
    store=ExperimentStore(tmp_path/'store');count,warnings=store.save(digits)
    assert count==1 and not warnings
    import cnn_lab.training as training
    monkeypatch.setattr(training,'prepare',lambda *a,**k: (_ for _ in ()).throw(AssertionError('No preparar al recuperar')))
    original=digits['data']; sample=original.test[0]
    before=gradcam_all(digits['model'],original.normalized([sample]),0)
    loaded,warnings,count=store.load()
    assert len(loaded)==1 and not warnings and count==1
    recovered=loaded[0];data=recovered['data']
    assert recovered['_experiment_id']==digits['_experiment_id'] and recovered['history']==digits['history']
    assert recovered['batches']==digits['batches'] and np.array_equal(recovered['probabilities'],digits['probabilities'])
    assert torch.equal(original.images,data.images) and np.array_equal(original.test,data.test)
    assert original.mean==data.mean and original.std==data.std
    assert not recovered['model'].training
    for name,value in digits['model'].state_dict().items(): assert torch.equal(value,recovered['model'].state_dict()[name])
    after=gradcam_all(recovered['model'],data.normalized([sample]),0)
    for name,result in before['layers'].items(): assert np.array_equal(result['heat'],after['layers'][name]['heat'])
    metadata=json.loads((store.root/digits['_experiment_id']/'run.json').read_text())
    assert metadata['checkpoint_format_version']==4 and 'accuracy' in metadata['metrics']


def test_rgb_stats_upload_and_source_identity_roundtrip(tmp_path):
    config=Config(dataset='CIFAR10Animals',samples=300,epochs=1,blocks=[Block([Conv(4)],pooling='none')],head='gap',dense=[])
    images=torch.rand(30,3,32,32);labels=torch.arange(30)%6
    mean,std=training_statistics(images,np.arange(18),True)
    data=Data(images,labels,np.arange(18),np.arange(18,24),np.arange(24,30),mean,std,['Pájaro','Gato','Ciervo','Perro','Rana','Caballo'],[('train',i) if i<24 else ('test',i-24) for i in range(30)])
    run=train(config,data=data);store=ExperimentStore(tmp_path/'store');store.save(run)
    loaded,warnings,_=store.load(); assert not warnings
    recovered=loaded[0]; assert torch.equal(mean,recovered['data'].mean) and recovered['data'].mean.shape==(1,3,1,1)
    assert [tuple(item) for item in recovered['data'].source_indices]==data.source_indices
    buffer=io.BytesIO();Image.fromarray(np.rint(display_image(images[24])*255).astype('uint8')).save(buffer,format='PNG')
    first=classify_latest([run],buffer.getvalue()); second=classify_latest(loaded,buffer.getvalue())
    assert np.array_equal(first['probabilities'],second['probabilities']) and len(second['probabilities'])==6


def test_last_six_idempotent_order_and_only_owned_pruning(tmp_path,digits):
    store=ExperimentStore(tmp_path/'store');ids=[]
    sentinel=store.root/uuid.uuid4().hex;sentinel.mkdir(parents=True);(sentinel/'my-file.txt').write_text('preserve')
    for index in range(7):
        run=copy.copy(digits);run.pop('_experiment_id',None);store.save(run);ids.append(run['_experiment_id'])
    runs,warnings,count=store.load()
    assert not warnings and count==7 and [run['_experiment_id'] for run in runs]==ids[1:]
    assert not (store.root/ids[0]).exists() and (sentinel/'my-file.txt').exists()
    count,_=store.save(run);assert count==7
    assert store.load()[2]==7 and len(store.load()[0])==6


@pytest.mark.parametrize('damage',['json','tensor','version','missing','dtype','split','nonfinite'])
def test_corruption_isolated(tmp_path,digits,damage):
    store=ExperimentStore(tmp_path/'store');store.save(digits)
    second=copy.copy(digits);second.pop('_experiment_id');store.save(second)
    directory=store.root/digits['_experiment_id']
    if damage=='json': (directory/'run.json').write_text('{broken')
    elif damage=='missing': (directory/'weights.pt').unlink()
    elif damage=='version':
        payload=json.loads((directory/'run.json').read_text());payload['format_version']=999;(directory/'run.json').write_text(json.dumps(payload))
    elif damage=='tensor': (directory/'weights.pt').write_bytes(b'invalid checkpoint')
    else:
        filename='weights.pt' if damage=='dtype' else 'data.pt'
        payload=torch.load(directory/filename,weights_only=True)
        if damage=='dtype':
            first=next(iter(payload['state_dict']));payload['state_dict'][first]=payload['state_dict'][first].double()
        if damage=='split':payload['test'][0]=payload['train'][0]
        if damage=='nonfinite':payload['probabilities'][0,0]=float('nan')
        torch.save(payload,directory/filename)
    loaded,warnings,count=store.load()
    assert len(loaded)==1 and loaded[0]['_experiment_id']==second['_experiment_id'] and warnings and count==2


def test_atomic_failure_preserves_previous_index_and_all_old_files(tmp_path,digits,monkeypatch):
    store=ExperimentStore(tmp_path/'store');store.save(digits)
    previous=(store.root/'index.json').read_bytes()
    next_run=copy.copy(digits);next_run.pop('_experiment_id')
    monkeypatch.setattr(store,'_write_index',lambda payload: (_ for _ in ()).throw(OSError('disco lleno')))
    with pytest.raises(OSError,match='disco lleno'):store.save(next_run)
    assert (store.root/'index.json').read_bytes()==previous
    assert (store.root/digits['_experiment_id']/'weights.pt').exists()
    assert len(store.load()[0])==1 and not list(store.root.glob('.pending-*'))


def test_corrupt_index_recovers_committed_runs_without_deleting(tmp_path,digits):
    store=ExperimentStore(tmp_path/'store');store.save(digits)
    (store.root/'index.json').write_text('{broken')
    runs,warnings,count=store.load()
    assert len(runs)==1 and warnings and count==1
    second=copy.copy(digits);second.pop('_experiment_id');store.save(second)
    assert len(store.load()[0])==2 and (store.root/digits['_experiment_id']).exists()


def test_concurrent_sessions_append_under_lock(tmp_path,digits):
    root=tmp_path/'store'
    runs=[copy.copy(digits) for _ in range(2)]
    with ThreadPoolExecutor(2) as executor:list(executor.map(lambda run:ExperimentStore(root).save(run),runs))
    loaded,warnings,count=ExperimentStore(root).load()
    assert len(loaded)==2 and count==2 and not warnings
    assert {run['_experiment_id'] for run in loaded}=={run['_experiment_id'] for run in runs}


def test_app_new_session_restores_without_train_and_migration_once(monkeypatch):
    path=Path(__file__).resolve().parents[1]/'app.py'
    first=AppTest.from_file(path).run(timeout=30)
    first.number_input(key='cfg_epochs').set_value(1)
    next(button for button in first.button if button.label=='Entrenar CNN').click().run(timeout=60)
    assert not first.exception
    original=first.session_state['runs'][0];assert original['_saved_local']
    import cnn_lab.training as training
    monkeypatch.setattr(training,'train',lambda *a,**k: (_ for _ in ()).throw(AssertionError('No entrenar durante restore')))
    second=AppTest.from_file(path).run(timeout=30)
    assert not second.exception and second.session_state['store_restored']==1
    recovered=second.session_state['runs'][0]
    assert recovered['_experiment_id']==original['_experiment_id']
    assert np.array_equal(recovered['probabilities'],original['probabilities'])
    assert any('Grad-CAM · todas' in header.value for header in second.subheader)
    second.run(timeout=30);assert second.session_state['completed_runs']==1
    # A pre-feature in-memory model is migrated exactly once on its first rerun.
    migrated=AppTest.from_file(path);extra=copy.copy(original);extra.pop('_experiment_id');extra.pop('_saved_local');extra.pop('_interpret_id',None)
    migrated.session_state.runs=[extra];migrated.run(timeout=30)
    assert not migrated.exception and extra['_saved_local']
    assert ExperimentStore().load()[2]==2
    migrated.run(timeout=30);assert ExperimentStore().load()[2]==2


def test_app_save_failure_keeps_result_and_retry_does_not_train(monkeypatch):
    original_save=ExperimentStore.save
    monkeypatch.setattr(ExperimentStore,'save',lambda *a: (_ for _ in ()).throw(OSError('sin espacio')))
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    app.number_input(key='cfg_epochs').set_value(1)
    next(button for button in app.button if button.label=='Entrenar CNN').click().run(timeout=60)
    assert app.session_state['live_training']['phase']=='completed' and len(app.session_state['runs'])==1
    assert any('sin espacio' in warning.value for warning in app.warning)
    monkeypatch.setattr(ExperimentStore,'save',original_save)
    next(button for button in app.button if button.label=='Reintentar guardado local').click().run(timeout=30)
    assert app.session_state['runs'][0]['_saved_local'] and len(ExperimentStore().load()[0])==1


def test_valid_json_but_malformed_index_recovers(tmp_path,digits):
    store=ExperimentStore(tmp_path/'store');store.save(digits)
    (store.root/'index.json').write_text('[]')
    loaded,warnings,count=store.load()
    assert len(loaded)==1 and warnings and count==1


def test_failed_directory_sync_does_not_prune_previous_snapshot(tmp_path,digits,monkeypatch):
    import cnn_lab.experiment_store as module
    store=ExperimentStore(tmp_path/'store');ids=[]
    for _ in range(6):
        run=copy.copy(digits);store.save(run);ids.append(run['_experiment_id'])
    original=module._sync_dir
    def failed_root(path):
        if path==store.root:raise OSError('sync no disponible')
        return original(path)
    monkeypatch.setattr(module,'_sync_dir',failed_root)
    newest=copy.copy(digits);_,warnings=store.save(newest)
    assert warnings and (store.root/ids[0]/'weights.pt').exists()
    assert len(store.load()[0])==6 and store.load()[2]==7
