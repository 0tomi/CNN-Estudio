import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from streamlit.testing.v1 import AppTest
from cnn_lab.model import Config, CNN, Block, Conv
from cnn_lab.data import prepare, Data
from cnn_lab.explain import gradcam_all, explain, feature_maps
from cnn_lab.interpretability import page_bounds, selection_state, overlay_image

@pytest.mark.parametrize('total,page,expected',[(0,0,(0,1,0,0)),(1,9,(0,1,0,1)),(10,0,(0,1,0,10)),(11,1,(1,2,10,11)),(31,3,(3,4,30,31)),(11,-2,(0,2,0,10))])
def test_pagination(total,page,expected):
    assert page_bounds(total,page)==expected

def test_selection_resets_by_identity_and_split_size():
    old={'run_id':'a','total':31,'page':2,'sample':25}
    assert selection_state(old,'a',31)==old
    assert selection_state(old,'b',31)['sample']==0
    assert selection_state(old,'a',11)['page']==0

def three_config(dataset='Digits'):
    return Config(dataset=dataset,blocks=[Block([Conv(8),Conv(8)],batchnorm=True),Block([Conv(12)])])

@pytest.mark.parametrize('dataset,shape',[('Digits',(1,1,8,8)),('CIFAR10Animals',(1,3,32,32))])
def test_all_convs_one_forward_preserve_hooks_modes_weights_grads(dataset,shape):
    model=CNN(three_config(dataset)); model.train(); model.classifier.eval()
    modes=[module.training for module in model.modules()]
    state=copy.deepcopy(model.state_dict())
    # Existing parameter gradients are not accumulated or cleared by explanation.
    for parameter in model.parameters(): parameter.grad=torch.ones_like(parameter)
    forwards=[]; hook=model.register_forward_hook(lambda *args: forwards.append(1))
    result=gradcam_all(model,torch.rand(shape),2)
    hook.remove()
    assert len(forwards)==1 and len(result['layers'])==3
    assert list(result['layers'])==model.conv_names()
    for layer in result['layers'].values():
        assert layer['heat'].shape==shape[-2:] and np.isfinite(layer['heat']).all()
        assert layer['heat'].min()>=0 and layer['heat'].max()<=1
    assert [module.training for module in model.modules()]==modes
    assert all(not model.features.get_submodule(name)._forward_hooks for name in model.conv_names())
    for key,value in state.items(): assert torch.equal(value,model.state_dict()[key])
    assert all(torch.equal(p.grad,torch.ones_like(p)) for p in model.parameters())
    with pytest.raises(ValueError,match='objetivo'): gradcam_all(model,torch.rand(shape),999)
    assert [module.training for module in model.modules()]==modes
    assert all(not model.features.get_submodule(name)._forward_hooks for name in model.conv_names())
    feature_maps(model,torch.rand(shape),model.conv_names()[0])
    explain(model,torch.rand(shape),model.conv_names()[0])
    assert [module.training for module in model.modules()]==modes

def test_zero_cam_overlay_no_false_heat_and_target_changes():
    model=CNN(three_config()); x=torch.rand(1,1,8,8)
    a=gradcam_all(model,x,0); b=gradcam_all(model,x,1)
    assert any(not np.array_equal(a['layers'][n]['heat'],b['layers'][n]['heat']) for n in model.conv_names())
    with torch.no_grad():
        for parameter in model.parameters(): parameter.zero_()
    result=gradcam_all(model,x,1)
    assert all(layer['constant'] and not layer['heat'].any() for layer in result['layers'].values())
    image=np.ones((32,32,3),dtype='float32')*.5
    overlay=overlay_image(image,np.zeros((32,32)))
    assert overlay.shape==(256,256,3) and np.allclose(overlay,128/255)

def rgb_run(total=23):
    config=three_config('CIFAR10Animals'); model=CNN(config)
    images=torch.rand(total,3,32,32); labels=torch.arange(total)%6
    data=Data(images,labels,np.array([],int),np.array([],int),np.arange(total),0.,1.,['Pájaro','Gato','Ciervo','Perro','Rana','Caballo'])
    with torch.no_grad(): probabilities=model.eval()(images).softmax(1).numpy()
    return {'model':model,'data':data,'config':config.dict(),'probabilities':probabilities,'best_epoch':1}

def test_gallery_rgb_pagination_selection_target_cache_and_run_reset():
    script='''import streamlit as st\nfrom cnn_lab.interpretability import render_interpretability\nrender_interpretability(st.session_state.run)'''
    at=AppTest.from_string(script); run=rgb_run(); at.session_state.run=run
    at.run(timeout=30); assert not at.exception
    def tiles():return [button for button in at.button if 'Imagen ' in button.label]
    assert len(tiles())==10 and tiles()[0].label=='✓ Imagen 1'
    cached=at.session_state['interpret_maps']; assert len(cached['result']['layers'])==3
    tiles()[3].click().run(timeout=30)
    assert at.session_state['interpret_selection']['sample']==3
    assert at.session_state['interpret_maps']['key'][1]==3
    assert any('Imagen 4 seleccionada'==header.value.removeprefix('2 · ') for header in at.subheader)
    next(button for button in at.button if button.label=='Siguiente').click().run(timeout=30)
    assert at.session_state['interpret_selection']['sample']==10 and len(tiles())==10
    next(button for button in at.button if button.label=='Siguiente').click().run(timeout=30)
    assert at.session_state['interpret_selection']['sample']==20 and len(tiles())==3
    assert next(button for button in at.button if button.label=='Siguiente').disabled
    next(button for button in at.button if button.label=='Anterior').click().run(timeout=30)
    assert at.session_state['interpret_selection']['sample']==10
    at.radio[0].set_value('Elegir clase').run(timeout=30)
    target=next(widget for widget in at.selectbox if widget.label=='Clase objetivo')
    target.set_value(4).run(timeout=30)
    assert at.session_state['interpret_maps']['key'][2]==4
    assert at.session_state['interpret_selection']['sample']==10
    cached=at.session_state['interpret_maps']
    at.run(timeout=30)
    assert at.session_state['interpret_maps']['key']==cached['key']
    at.session_state.run=rgb_run(11); at.run(timeout=30)
    assert at.session_state['interpret_selection']['sample']==0
    assert not at.exception
