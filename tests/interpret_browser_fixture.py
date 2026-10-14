"""Isolated browser check: real cached animal data, tiny CPU run, no downloads."""
import runpy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
from torchvision.datasets import CIFAR10
from cnn_lab.model import Config,Block,Conv
from cnn_lab.data import prepare
from cnn_lab.training import train
if 'runs' not in st.session_state:
    # Refuse unavailable caches rather than ever downloading in this fixture.
    CIFAR10('data',train=True,download=False)
    CIFAR10('data',train=False,download=False)
    config=Config(dataset='CIFAR10Animals',samples=300,epochs=1,
                  blocks=[Block([Conv(8),Conv(8)],batchnorm=True),Block([Conv(12)])])
    data=prepare(config)
    data.test=data.test[:23]
    st.session_state.runs=[train(config,data=data)]
runpy.run_path('app.py',run_name='__main__')
st.caption('PRUEBA AISLADA: CIFAR-10 animal real en caché · 300 ejemplos de desarrollo · 1 época · 23 muestras test para probar paginación. No modifica la sesión del usuario ni descarga archivos.')
