import copy
import io
import numpy as np
import pytest
import torch
from PIL import Image
from cnn_lab.model import Config, CNN
from cnn_lab.data import prepare, ANIMAL_IDS, ANIMAL_NAMES, display_image, augment
from cnn_lab.inference import decode_image, preprocess_image, classify_latest, MAX_UPLOAD_BYTES
from cnn_lab.metrics import evaluate
from cnn_lab.explain import explain

@pytest.fixture
def animals(monkeypatch):
    from torchvision import datasets
    class MockCIFAR:
        def __init__(self,root,train,download):
            self.targets = np.repeat(np.arange(10),100).tolist()
            self.classes = list(map(str,range(10)))
            self.data = np.zeros((1000,32,32,3),dtype=np.uint8)
            self.data[:,:,:,0] = np.array(self.targets)[:,None,None]*8+10+(0 if train else 50)
            self.data[:,:,:,1] = 70+(0 if train else 40)
            self.data[:,:,:,2] = 140+(0 if train else 30)
            self.data[:,0,0,0] = np.arange(1000)%255
    monkeypatch.setattr(datasets,'CIFAR10',MockCIFAR)
    return prepare(Config(dataset='CIFAR10Animals',samples=300))

def encode(image,format='PNG',**kwargs):
    output=io.BytesIO(); image.save(output,format=format,**kwargs)
    return output.getvalue()

def test_animals_filter_remap_splits_rgb_normalization(animals):
    d=animals
    assert d.images.shape==(500,3,32,32)
    assert d.names==ANIMAL_NAMES
    assert set(d.labels.tolist())==set(range(6))
    assert len(d.train)==240 and len(d.val)==60 and len(d.test)==200
    assert not set(d.train)&set(d.val) and not set(d.train)&set(d.test)
    for local,(source,original) in enumerate(d.source_indices):
        original_label=original//100
        assert original_label in ANIMAL_IDS
        assert int(d.labels[local])==ANIMAL_IDS.index(original_label)
        assert source==('test' if local in d.test else 'train')
    assert d.mean.shape==d.std.shape==(1,3,1,1)
    assert torch.allclose(d.mean,d.images[d.train].mean((0,2,3),keepdim=True))
    assert torch.allclose(d.std,d.images[d.train].std((0,2,3),keepdim=True).clamp_min(1e-6))
    assert not torch.allclose(d.mean,d.images.mean((0,2,3),keepdim=True))
    assert torch.allclose(d.normalized(d.train).mean((0,2,3)),torch.zeros(3),atol=1e-4)
    assert display_image(d.images[0]).shape==(32,32,3)
    assert augment(d.images[:2]).shape==(2,3,32,32)

def test_animals_reproducible(animals):
    second=prepare(Config(dataset='CIFAR10Animals',samples=300))
    assert np.array_equal(animals.train,second.train)
    assert animals.source_indices==second.source_indices

def test_rgb_model_metrics_and_explanations(animals):
    model=CNN(Config(dataset='CIFAR10Animals'))
    assert model(torch.zeros(2,3,32,32)).shape==(2,6)
    assert model.rows[-1]['salida']=='6'
    result=evaluate(np.arange(6),np.eye(6))
    assert result['accuracy']==result['f1']==result['auc']==1
    assert result['confusion'].shape==(6,6)
    for method in ['gradcam','neuron']:
        result=explain(model,animals.normalized([0]),model.conv_names()[0],target=5,method=method)
        assert result['heat'].shape==(32,32)
        assert np.isfinite(result['heat']).all()
        assert not model.features.get_submodule(model.conv_names()[0])._forward_hooks

@pytest.mark.parametrize('mode',['RGB','L','RGBA'])
def test_upload_converts_color_and_normalizes_identically(animals,mode):
    source=Image.new(mode,(45,20), (10,70,140,128) if mode=='RGBA' else (10,70,140) if mode=='RGB' else 80)
    original,raw,normalized=preprocess_image(encode(source),animals)
    assert original.mode=='RGB'
    assert raw.shape==(3,32,32)
    assert torch.allclose(normalized,(raw.unsqueeze(0)-animals.mean)/animals.std)
    if mode=='RGBA': assert float(raw[0,0,0])>10/255

def test_upload_exif_and_bad_files(monkeypatch):
    exif=Image.Exif(); exif[274]=6
    original=decode_image(encode(Image.new('RGB',(15,7)),format='JPEG',exif=exif))
    assert original.size==(7,15)
    for content in [b'',b'not an image',b'x'*(MAX_UPLOAD_BYTES+1),encode(Image.new('RGB',(4,4)),format='BMP')]:
        with pytest.raises(ValueError): decode_image(content)
    import cnn_lab.inference as inference
    monkeypatch.setattr(inference,'MAX_IMAGE_PIXELS',8)
    with pytest.raises(ValueError,match='píxeles'): decode_image(encode(Image.new('RGB',(4,4))))

def test_upload_grayscale_old_dataset_and_no_training():
    d=prepare(Config())
    _,raw,normalized=preprocess_image(encode(Image.new('RGB',(16,12),(80,90,100))),d)
    assert raw.shape==(1,8,8)
    assert torch.allclose(normalized,(raw.unsqueeze(0)-d.mean)/d.std)
    with pytest.raises(ValueError,match='Primero entrená'): classify_latest([],b'')

def test_latest_frozen_run_and_state_unchanged(animals):
    old_data=prepare(Config())
    old={'model':CNN(Config()),'data':old_data,'config':Config().dict(),'best_epoch':1}
    config=Config(dataset='CIFAR10Animals')
    model=CNN(config); model.train()
    model.classifier.eval()
    latest={'model':model,'data':animals,'config':copy.deepcopy(config.dict()),'best_epoch':3}
    state=copy.deepcopy(model.state_dict())
    content=encode(Image.fromarray(np.rint(display_image(animals.images[0])*255).astype('uint8')))
    result=classify_latest([old,latest],content)
    assert result['config']['dataset']=='CIFAR10Animals'
    assert result['best_epoch']==3 and result['run_number']==2
    assert result['probabilities'].shape==(6,)
    assert np.isclose(result['probabilities'].sum(),1)
    assert model.training and not model.classifier.training
    for key,value in state.items(): assert torch.equal(value,model.state_dict()[key])
    assert all(parameter.grad is None for parameter in model.parameters())

def test_app_switch_animals_preview_empty_upload():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    at.selectbox(key='cfg_dataset').set_value('CIFAR10Animals').run(timeout=30)
    assert not at.exception
    assert at.number_input(key='cfg_samples').value==3000
    assert any('Probar imagen' in tab.label for tab in at.tabs)
    assert any('Primero entrená' in value.value for value in at.info)
    architecture=at.dataframe[0].value
    assert architecture.iloc[-1]['salida']=='6'


def test_app_animal_training_rgb_rendering(animals):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    at=AppTest.from_file(Path(__file__).resolve().parents[1]/'app.py').run(timeout=30)
    at.selectbox(key='cfg_dataset').set_value('CIFAR10Animals').run(timeout=30)
    at.number_input(key='cfg_samples').set_value(300)
    at.number_input(key='cfg_epochs').set_value(1)
    next(button for button in at.button if button.label=='Entrenar CNN').click()
    at.run(timeout=60)
    assert not at.exception
    run=at.session_state['runs'][-1]
    assert run['probabilities'].shape==(200,6)
    assert run['data'].images.shape[1:]==(3,32,32)
    assert any('6 clases' in value.value for value in at.info)
    assert any('clasificador cerrado' in value.value.lower() for value in at.warning)
