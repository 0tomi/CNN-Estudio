"""Offline browser reload proof; MUST point at an isolated temporary store."""
import os
import runpy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
from cnn_lab.experiment_store import ExperimentStore
from cnn_lab.model import Config,Block,Conv
from cnn_lab.data import prepare
from cnn_lab.training import train
assert os.environ.get('CNN_LAB_EXPERIMENT_DIR'), 'Definí una carpeta temporal aislada para esta prueba.'
if not (ExperimentStore().root/'index.json').exists() and 'runs' not in st.session_state:
    from torchvision.datasets import CIFAR10
    CIFAR10('data',train=True,download=False)
    CIFAR10('data',train=False,download=False)
    config=Config(dataset='CIFAR10Animals',samples=300,epochs=1,blocks=[Block([Conv(8),Conv(8)]),Block([Conv(12)])])
    data=prepare(config);data.test=data.test[:23]
    st.session_state.runs=[train(config,data=data)]
runpy.run_path('app.py',run_name='__main__')
st.caption('PRUEBA AISLADA DE PERSISTENCIA: datos animales reales en caché; el primer acceso entrena1época y migra. Tras reload sólo recupera archivos, no vuelve a entrenar.')
