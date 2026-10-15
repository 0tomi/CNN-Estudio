"""Atomic local inference snapshots; only safe tensors and versioned JSON are loaded."""
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid
import numpy as np
import torch
from filelock import FileLock
from .data import Data
from .metrics import evaluate
from .model import CNN, Config, dataset_spec

VERSION = 1
KEEP = 6
ID = re.compile(r'^[0-9a-f]{32}$')


def _json_default(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(f'Tipo JSON no admitido: {type(value).__name__}')


def _write_json(path, payload):
    with open(path, 'w', encoding='utf8') as output:
        json.dump(payload, output, ensure_ascii=False, allow_nan=False, default=_json_default)
        output.flush(); os.fsync(output.fileno())


def _sync_dir(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def _save_tensor(path, payload):
    with open(path, 'wb') as output:
        torch.save(payload, output)
        output.flush(); os.fsync(output.fileno())


def _read_json(path):
    with open(path, encoding='utf8') as source:
        return json.load(source, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f'Número JSON inválido: {value}')))


def _entry(metadata):
    identity = metadata['id']
    if not isinstance(identity, str) or not ID.fullmatch(identity): raise ValueError('Identificador de experimento inválido.')
    if type(metadata['sequence']) is not int or metadata['sequence'] < 1: raise ValueError('Secuencia inválida.')
    timestamp = metadata['completed_at']
    if not isinstance(timestamp, str) or datetime.fromisoformat(timestamp).tzinfo is None: raise ValueError('Fecha inválida.')
    return {key: metadata[key] for key in ('id', 'sequence', 'completed_at')}


def _validate_run(run):
    config = Config.from_dict(run['config'])
    size, channels, classes = dataset_spec(config.dataset)
    data = run['data']; x, y = data.images, data.labels
    if x.dtype != torch.float32 or x.ndim != 4 or tuple(x.shape[1:]) != (channels, size, size) or not 1 <= len(x) <= 12000:
        raise ValueError('Imágenes incompatibles con la configuración guardada.')
    if not torch.isfinite(x).all() or torch.any(x < 0) or torch.any(x > 1): raise ValueError('Imágenes fuera de [0,1] o no finitas.')
    if y.dtype != torch.int64 or y.shape != (len(x),) or torch.any(y < 0) or torch.any(y >= classes): raise ValueError('Etiquetas inválidas.')
    if len(data.names) != classes or any(not isinstance(name, str) or not name for name in data.names): raise ValueError('Clases incompatibles.')
    joined=[]
    for indices in (data.train, data.val, data.test):
        a=np.asarray(indices)
        if a.ndim != 1 or not np.issubdtype(a.dtype,np.integer) or len(a)==0 or np.any(a<0) or np.any(a>=len(x)):
            raise ValueError('Índices de partición inválidos.')
        joined.extend(a.tolist())
    if len(set(joined)) != len(joined): raise ValueError('Las particiones guardadas se superponen.')
    for name in ('mean', 'std'):
        tensor=torch.as_tensor(getattr(data,name))
        if tensor.shape not in (torch.Size([]),torch.Size([1,channels,1,1])) or not torch.isfinite(tensor).all(): raise ValueError('Normalización incompatible.')
        if name=='std' and torch.any(tensor<=0): raise ValueError('Desvío inválido.')
    if data.source_indices is not None:
        if len(data.source_indices)!=len(x): raise ValueError('Identidades fuente incompletas.')
        for item in data.source_indices:
            if not isinstance(item,(tuple,list)) or len(item)!=2 or not isinstance(item[0],str) or type(item[1]) is not int or item[1]<0: raise ValueError('Identidad fuente inválida.')
        if config.dataset != 'Digits':
            for indices,expected in ((data.train,'train'),(data.val,'train'),(data.test,'test')):
                if any(data.source_indices[int(i)][0]!=expected for i in indices): raise ValueError('Se violó el límite train/test oficial.')
    probabilities=np.asarray(run['probabilities'])
    if probabilities.shape!=(len(data.test),classes): raise ValueError('Probabilidades incompatibles con test.')
    evaluate(y[data.test],probabilities)
    history=run['history']
    if not isinstance(history,list) or not history or len(history)>config.epochs: raise ValueError('Historial de épocas inválido.')
    for index,row in enumerate(history,1):
        if row.get('epoch')!=index or any(not math.isfinite(float(row[name])) for name in ('train_loss','val_loss','train_accuracy','val_accuracy')): raise ValueError('Métricas por época inválidas.')
    if type(run['best_epoch']) is not int or not 1<=run['best_epoch']<=len(history): raise ValueError('Checkpoint inválido.')
    if run.get('evaluation_split','test')!='test': raise ValueError('Sólo se guardan entrenamientos normales completados, no folds.')
    if run.get('epochs_completed',len(history))!=len(history) or run.get('max_epochs',config.epochs)!=config.epochs:
        raise ValueError('Cantidad de épocas incompatible.')
    if not math.isfinite(float(run['elapsed'])) or run['elapsed']<0: raise ValueError('Duración inválida.')
    # Strict loading validates all parameter keys, dimensions and dtypes separately.
    expected=CNN(config)
    state=run['model'].state_dict()
    if state.keys()!=expected.state_dict().keys(): raise ValueError('Claves de pesos incompatibles.')
    for name,value in state.items():
        other=expected.state_dict()[name]
        if value.shape!=other.shape or value.dtype!=other.dtype or not torch.isfinite(value).all(): raise ValueError('Pesos incompatibles o no finitos.')
    return config


class ExperimentStore:
    def __init__(self, root=None):
        self.root=Path(root if root is not None else os.environ.get('CNN_LAB_EXPERIMENT_DIR','experiments'))

    def _directory(self, identity):
        if not isinstance(identity,str) or not ID.fullmatch(identity): raise ValueError('Identificador inseguro.')
        path=self.root/identity
        if path.is_symlink(): raise ValueError('No se admiten enlaces simbólicos de experimentos.')
        return path

    def _index(self, warnings):
        index_path=self.root/'index.json'
        if index_path.exists():
            try:
                index=_read_json(index_path)
                if not isinstance(index,dict) or index.get('format_version')!=VERSION: raise ValueError('Versión de índice desconocida.')
                entries=[_entry(item) for item in index['runs']]
                if len(entries)>KEEP or len({item['id'] for item in entries})!=len(entries): raise ValueError('Índice duplicado o demasiado grande.')
                count=index['completed_count']
                if type(count) is not int or count<max([item['sequence'] for item in entries],default=0): raise ValueError('Contador inválido.')
                return dict(format_version=VERSION,completed_count=count,runs=sorted(entries,key=lambda item:item['sequence']))
            except (OSError,ValueError,KeyError,TypeError) as error:
                warnings.append(f'Índice local ilegible: {error}. Se buscaron manifiestos completos; no se borró ningún archivo.')
        entries=[]
        for path in self.root.iterdir():
            if not ID.fullmatch(path.name) or path.is_symlink() or not path.is_dir(): continue
            try:
                metadata=_read_json(path/'run.json')
                if not isinstance(metadata,dict) or metadata.get('format_version')!=VERSION or metadata.get('id')!=path.name or not all((path/name).is_file() for name in ('weights.pt','data.pt')): continue
                entries.append(_entry(metadata))
            except (OSError,ValueError,KeyError,TypeError): continue
        entries.sort(key=lambda item:(item['sequence'],item['completed_at'],item['id']))
        return dict(format_version=VERSION,completed_count=max([item['sequence'] for item in entries],default=0),runs=entries[-KEEP:])

    def _write_index(self,index):
        path=self.root/f'.index-{uuid.uuid4().hex}.tmp'
        try:
            _write_json(path,index)
            os.replace(path,self.root/'index.json')
        finally:
            path.unlink(missing_ok=True)

    def save(self,run):
        if not run.get('_experiment_id'):
            candidate=run.get('_interpret_id')
            run['_experiment_id']=candidate if isinstance(candidate,str) and ID.fullmatch(candidate) else uuid.uuid4().hex
        identity=run['_experiment_id']; directory=self._directory(identity)
        self.root.mkdir(parents=True,exist_ok=True)
        warnings=[]
        with FileLock(str(self.root/'.store.lock'),timeout=10):
            index=self._index(warnings)
            existing=next((item for item in index['runs'] if item['id']==identity),None)
            if existing:
                run.update(_saved_local=True,completed_at=existing['completed_at'],experiment_sequence=existing['sequence'])
                return index['completed_count'],warnings
            _validate_run(run)
            if directory.exists(): raise ValueError('El identificador ya tiene archivos fuera del índice; no se sobrescriben.')
            sequence=index['completed_count']+1
            completed=run.get('completed_at') or datetime.now(timezone.utc).isoformat()
            entry=_entry(dict(id=identity,sequence=sequence,completed_at=completed))
            data=run['data']
            metadata=dict(format_version=VERSION,checkpoint_format_version=4,**entry,config=copy.deepcopy(run['config']),classes=data.names,
                          source_indices=data.source_indices,metrics=evaluate(data.labels[data.test],run['probabilities']))
            for key in ('history','batches','best_epoch','elapsed'):
                metadata[key]=run[key]
            metadata.update(epochs_completed=run.get('epochs_completed',len(run['history'])),max_epochs=run.get('max_epochs',run['config']['epochs']),
                            stopped_early=run.get('stopped_early',False),stop_reason=run.get('stop_reason','max_epochs'),evaluation_split='test')
            temporary=Path(tempfile.mkdtemp(prefix='.pending-',dir=self.root))
            committed=False
            try:
                _write_json(temporary/'run.json',metadata)
                _save_tensor(temporary/'weights.pt',dict(format_version=4,state_dict={key:value.detach().cpu() for key,value in run['model'].state_dict().items()}))
                _save_tensor(temporary/'data.pt',dict(images=data.images.detach().cpu(),labels=data.labels.detach().cpu(),
                     train=torch.as_tensor(data.train,dtype=torch.int64),val=torch.as_tensor(data.val,dtype=torch.int64),test=torch.as_tensor(data.test,dtype=torch.int64),
                     mean=data.mean.detach().cpu() if isinstance(data.mean,torch.Tensor) else data.mean,
                     std=data.std.detach().cpu() if isinstance(data.std,torch.Tensor) else data.std,probabilities=torch.as_tensor(run['probabilities'])))
                _sync_dir(temporary)
                os.replace(temporary,directory)
                previous=index['runs']; entries=(previous+[entry])[-KEEP:]
                self._write_index(dict(format_version=VERSION,completed_count=sequence,runs=entries))
                committed=True
                durable=True
                try: _sync_dir(self.root)
                except OSError as error:
                    durable=False
                    warnings.append(f'Archivos escritos; no se pudo sincronizar el directorio: {error}. No se eliminan experimentos antiguos.')
                # Only directories previously indexed and now evicted may be pruned.
                for old in previous if durable else []:
                    if old['id'] not in {item['id'] for item in entries}:
                        try:
                            old_directory=self._directory(old['id'])
                            owned=_read_json(old_directory/'run.json')
                            if not isinstance(owned,dict) or owned.get('id')!=old['id'] or owned.get('format_version')!=VERSION:
                                raise ValueError('No se reconoce como carpeta propia; no se borra.')
                            shutil.rmtree(old_directory)
                        except (OSError,ValueError) as error: warnings.append(f'No se pudo liberar el experimento antiguo {old["id"]}: {error}')
            finally:
                shutil.rmtree(temporary,ignore_errors=True)
                if not committed and directory.exists():
                    # An error reported after replace must never remove referenced data.
                    try:
                        referenced=any(item.get('id')==identity for item in _read_json(self.root/'index.json')['runs']) if (self.root/'index.json').exists() else False
                        if not referenced: shutil.rmtree(directory,ignore_errors=True)
                    except (OSError,ValueError,KeyError,TypeError):
                        pass
            run.update(_saved_local=True,completed_at=completed,experiment_sequence=sequence)
            return sequence,warnings

    def load(self):
        if not self.root.exists(): return [],[],0
        warnings=[];runs=[]
        with FileLock(str(self.root/'.store.lock'),timeout=10):
            index=self._index(warnings)
            for entry in index['runs']:
                try:
                    directory=self._directory(entry['id'])
                    metadata=_read_json(directory/'run.json')
                    if metadata.get('format_version')!=VERSION or metadata.get('checkpoint_format_version')!=4 or _entry(metadata)!=entry: raise ValueError('Versión o manifiesto incompatible.')
                    for filename in ('data.pt','weights.pt','run.json'):
                        if (directory/filename).is_symlink(): raise ValueError('No se admiten enlaces simbólicos.')
                    payload=torch.load(directory/'data.pt',map_location='cpu',weights_only=True)
                    checkpoint=torch.load(directory/'weights.pt',map_location='cpu',weights_only=True)
                    if checkpoint.get('format_version')!=4: raise ValueError('Checkpoint desconocido.')
                    config=Config.from_dict(metadata['config']);model=CNN(config)
                    # Validate exact dtype/key/shape before load_state_dict (which can cast).
                    expected=model.state_dict();state=checkpoint['state_dict']
                    if state.keys()!=expected.keys() or any(value.dtype!=expected[key].dtype or value.shape!=expected[key].shape for key,value in state.items()): raise ValueError('Pesos incompatibles.')
                    model.load_state_dict(state,strict=True);model.eval()
                    for key in ('train','val','test'):
                        if payload[key].dtype!=torch.int64: raise ValueError('Índices deben ser enteros.')
                    data=Data(payload['images'],payload['labels'],payload['train'].numpy(),payload['val'].numpy(),payload['test'].numpy(),payload['mean'],payload['std'],metadata['classes'],metadata['source_indices'])
                    run={key:metadata[key] for key in ('config','history','batches','best_epoch','elapsed','epochs_completed','max_epochs','stopped_early','stop_reason','evaluation_split')}
                    run.update(model=model,data=data,probabilities=payload['probabilities'].numpy(),_experiment_id=entry['id'],_interpret_id=entry['id'],_saved_local=True,completed_at=entry['completed_at'],experiment_sequence=entry['sequence'])
                    _validate_run(run)
                    runs.append(run)
                except Exception as error:
                    warnings.append(f'Experimento {entry["id"]} omitido: {type(error).__name__}: {error}')
        return runs,warnings,index['completed_count']
