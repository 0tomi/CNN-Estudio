"""Isolated UI proof ONLY; synthetic byte events, no network or dataset files."""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
from cnn_lab.model import Config
from cnn_lab.live import Monitor
st.set_page_config(layout='wide')
st.title('Prueba aislada: progreso de transporte')
st.caption('SIMULACIÓN SOLAMENTE: no descarga CIFAR, no hay conexión de red ni archivos.')
monitor=Monitor('mock_download',Config(dataset='CIFAR10Animals').dict())
if st.button('Simular progreso'):
    for percent in (0,25,50,75,100):
        monitor.event({'phase':'downloading','message':'SIMULACIÓN DE TRANSPORTE · sólo prueba','downloaded':percent*1024*1024,'total':100*1024*1024,'speed':25*1024*1024})
        time.sleep(1.)
    monitor.event({'phase':'completed','message':'Simulación finalizada: no se descargó un dataset real'})
