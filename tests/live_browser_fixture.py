"""Browser proof fixture ONLY: slow preparation/validation to inspect intermediate UI.
Run on isolated port8502, never use this fixture as the normal study app.
"""
import runpy
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
import cnn_lab.training as training
from cnn_lab.model import Config
if getattr(training.predict,'_fixture_version',None)!=2:
    previous_predict=training.predict
    original_predict=getattr(previous_predict,'_original',previous_predict.__globals__.get('original_predict',previous_predict))
    def controlled_predict(*args,**kwargs):
        time.sleep(.55)
        return original_predict(*args,**kwargs)
    controlled_predict._fixture=True
    controlled_predict._fixture_version=2
    controlled_predict._original=original_predict
    training.predict=controlled_predict
    previous_prepare=training.prepare
    original_prepare=getattr(previous_prepare,'_original',previous_prepare.__globals__.get('original_prepare',previous_prepare))
    def controlled_prepare(*args,**kwargs):
        time.sleep(1.)
        event=kwargs.get("on_event")
        if event:
            for downloaded in (0,25,50,75,100):
                event({"phase":"downloading","message":"SIMULACIÓN DE TRANSPORTE · sólo fixture de prueba, sin red","downloaded":downloaded*1024*1024,"total":100*1024*1024,"elapsed":downloaded/25,"speed":25*1024*1024})
                time.sleep(.3)
        return original_prepare(*args,**kwargs)
    controlled_prepare._original=original_prepare
    training.prepare=controlled_prepare
if 'base' not in st.session_state:st.session_state.base=Config(samples=600,epochs=8).dict()
runpy.run_path(str(Path(__file__).resolve().parents[1]/'app.py'),run_name='__main__')
