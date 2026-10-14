import copy
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from cnn_lab.model import Config,DenseLayer,CNN
from cnn_lab.data import prepare,training_statistics,Data
from cnn_lab.cross_validation import make_folds,cross_validate


def test_legacy_dense_migration_and_state_keys():
    legacy=Config().dict()
    legacy.pop('dense')
    legacy.update(hidden=[128,64],activation='tanh',dropout=.25)
    c=Config.from_dict(legacy)
    assert c.dense==[DenseLayer(128,'tanh',.25),DenseLayer(64,'tanh',.25)]
    assert not {'hidden','activation','dropout'}&c.dict().keys()
    assert Config.from_dict(c.dict()).dict()==c.dict()
    old_style=CNN(Config(hidden=[64],activation='relu',dropout=.1))
    canonical=CNN(Config(dense=[DenseLayer()]))
    canonical.load_state_dict(old_style.state_dict())
    assert list(canonical.state_dict())==list(old_style.state_dict())
    canonical.eval();old_style.eval()
    x=torch.rand(2,1,8,8)
    assert torch.equal(canonical(x),old_style(x))


def test_dense_layers_each_activation_dropout_bias_and_zero():
    model=CNN(Config(dataset='CIFAR10Animals',dense=[DenseLayer(128,'relu',.2),DenseLayer(64,'gelu',.35,False)]))
    linears=[m for m in model.classifier if isinstance(m,torch.nn.Linear)]
    assert [m.out_features for m in linears]==[128,64,6]
    assert linears[1].bias is None
    assert any(isinstance(m,torch.nn.GELU) for m in model.classifier)
    assert [m.p for m in model.classifier if isinstance(m,torch.nn.Dropout)]==[.2,.35]
    assert model(torch.rand(2,3,32,32)).shape==(2,6)
    assert model.rows[-1]['capa']=='salida · logits'
    zero=CNN(Config(dense=[]))
    assert len([m for m in zero.classifier if isinstance(m,torch.nn.Linear)])==1
    assert not any(isinstance(m,torch.nn.Dropout) for m in zero.classifier)
    assert zero(torch.rand(2,1,8,8)).shape==(2,10)


@pytest.mark.parametrize('dense',[[DenseLayer(units=1000)],[DenseLayer(activation='bad')],[DenseLayer(dropout=.8)],[DenseLayer(dropout='bad')],[DenseLayer(bias='yes')],[DenseLayer()]*6])
def test_invalid_dense(dense):
    with pytest.raises(ValueError):CNN(Config(dense=dense))


def test_fold_partitions_stats_and_reproducibility():
    c=Config(samples=600);d=prepare(c)
    folds=list(make_folds(d,3,c.seed))
    validation=np.concatenate([f.val for f in folds])
    development=np.sort(np.concatenate([d.train,d.val]))
    assert np.array_equal(np.sort(validation),development)
    assert len(np.unique(validation))==len(validation)
    for fold in folds:
        assert not set(fold.train)&set(fold.val)
        assert not (set(fold.train)|set(fold.val))&set(d.test)
        assert fold.mean==float(d.images[fold.train].mean())
        assert fold.std==max(float(d.images[fold.train].std()),1e-6)
    repeated=list(make_folds(d,3,c.seed))
    assert all(np.array_equal(a.train,b.train) for a,b in zip(folds,repeated))
    unchanged=list(make_folds(d,2,c.seed,normalize=False))
    assert all(f.mean==0 and f.std==1 for f in unchanged)
    missing=replace(d,train=d.train[:1],val=d.val[:1])
    with pytest.raises(ValueError,match='muestras por clase'):list(make_folds(missing,3,c.seed))


def test_cross_validation_no_test_predictions_fresh_models_and_optimizers(monkeypatch):
    import cnn_lab.training as training
    c=Config(samples=300,epochs=1);d=prepare(c)
    original_predict=training.predict;original_cnn=training.CNN;original_adam=torch.optim.Adam
    models=[];initial=[];optimizers=[];evaluated=[]
    def predict(model,data,indices):
        assert not set(indices)&set(d.test)
        evaluated.append(np.array(indices))
        return original_predict(model,data,indices)
    def cnn(*args,**kwargs):
        model=original_cnn(*args,**kwargs)
        models.append(model);initial.append(next(model.parameters()).detach().clone())
        return model
    def adam(*args,**kwargs):
        optimizer=original_adam(*args,**kwargs);optimizers.append(optimizer)
        assert not optimizer.state
        return optimizer
    monkeypatch.setattr(training,'predict',predict)
    monkeypatch.setattr(training,'CNN',cnn)
    monkeypatch.setattr(torch.optim,'Adam',adam)
    result=cross_validate(c,2,data=d)
    assert len(models)==len(optimizers)==2
    assert models[0] is not models[1] and optimizers[0] is not optimizers[1]
    assert not torch.equal(initial[0],initial[1])
    assert len(evaluated)==4
    for fold in result['folds']:
        assert 'model' not in fold
        assert fold['seed']==c.seed+fold['fold']-1
    for metric,summary in result['aggregate'].items():
        values=[f[metric] for f in result['folds']]
        assert summary['mean']==pytest.approx(np.mean(values))
        assert summary['std']==pytest.approx(np.std(values,ddof=1))
    assert result['std_ddof']==1
    second=cross_validate(c,2,data=d)
    assert [f['accuracy'] for f in result['folds']]==[f['accuracy'] for f in second['folds']]
    assert [f['history'] for f in result['folds']]==[f['history'] for f in second['folds']]


def test_rgb_fold_channel_stats_official_test_boundary():
    generator=torch.Generator().manual_seed(8)
    images=torch.rand(120,3,32,32,generator=generator)
    labels=torch.arange(120)%6
    data=Data(images,labels,np.arange(60),np.arange(60,90),np.arange(90,120),0.,1.,list('abcdef'),[('train' if i<90 else 'test',i) for i in range(120)])
    folds=list(make_folds(data,3,42))
    for fold in folds:
        mean,std=training_statistics(images,fold.train)
        assert torch.equal(fold.mean,mean) and torch.equal(fold.std,std)
        assert mean.shape==(1,3,1,1)
        assert all(data.source_indices[i][0]=='train' for i in np.concatenate([fold.train,fold.val]))
    result=cross_validate(Config(dataset='CIFAR10Animals',epochs=1),3,data=data)
    assert all(0<=f['accuracy']<=1 and np.isfinite(f['f1']) for f in result['folds'])


def test_app_dense_controls_sidebar_and_kfold_preserves_normal_run():
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    assert not at.exception
    assert any('NO redes preentrenadas' in value.value for value in at.caption)
    assert not any('GEA' in value.value for value in at.info)
    at.number_input(key='cfg_hidden_count').set_value(2).run(timeout=30)
    at.number_input(key='cfg_dense_units_0').set_value(128)
    at.number_input(key='cfg_dense_units_1').set_value(64)
    at.selectbox(key='cfg_dense_activation_1').set_value('gelu')
    at.number_input(key='cfg_dense_dropout_1').set_value(.3)
    at.number_input(key='cfg_samples').set_value(300)
    at.number_input(key='cfg_epochs').set_value(1)
    next(b for b in at.button if b.label=='Entrenar CNN').click()
    at.run(timeout=60)
    assert not at.exception
    normal=at.session_state['runs'][-1]
    assert normal['config']['dense'][1]['activation']=='gelu'
    assert normal['config']['dense'][1]['dropout']==.3
    next(b for b in at.button if b.label=='Volver a configuración').click().run(timeout=30)
    at.selectbox(key='cv_k').set_value(2)
    at.number_input(key='cv_epochs').set_value(1)
    next(b for b in at.button if b.label=='Ejecutar validación K-fold').click()
    at.run(timeout=60)
    assert not at.exception
    assert len(at.session_state['runs'])==1
    assert at.session_state['runs'][-1]['model'] is normal['model']
    assert len(at.session_state['cv_result']['folds'])==2
    at.number_input(key='cfg_hidden_count').set_value(0).run(timeout=30)
    assert not at.exception
    assert any('Sin ocultas' in value.value for value in at.caption)
    assert any('validación K-fold' in value.value for value in at.warning)


def test_kfold_batchnorm_singleton_is_actionable():
    from cnn_lab.model import Block,Conv
    c=Config(samples=230,batch_size=16,epochs=1,blocks=[Block(),Block([Conv(32)]),Block([Conv(32,stride=2)],batchnorm=True,pooling='none')])
    with pytest.raises(ValueError,match='Fold 1/2:.*último batch'):
        cross_validate(c,2)


def test_app_legacy_import_and_zero_hidden():
    from streamlit.testing.v1 import AppTest
    legacy=Config().dict();legacy.pop('dense')
    legacy.update(hidden=[128,64],activation='tanh',dropout=.2)
    at=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py')
    at.session_state['base']=legacy
    at.run(timeout=30)
    assert not at.exception
    assert at.number_input(key='cfg_hidden_count').value==2
    assert at.number_input(key='cfg_dense_units_0').value==128
    assert at.selectbox(key='cfg_dense_activation_1').value=='tanh'
    assert at.number_input(key='cfg_dense_dropout_0').value==.2
    at.number_input(key='cfg_hidden_count').set_value(0).run(timeout=30)
    assert not at.exception
    rows=at.dataframe[0].value
    assert not any('densa oculta' in name for name in rows['capa'])
    assert rows.iloc[-1]['capa']=='salida · logits'
