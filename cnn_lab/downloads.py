"""Bounded, integrity-checked local downloads with truthful progress events."""
import gzip
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import time
import requests
from filelock import FileLock, Timeout


def emit(callback,phase,**details):
    if callback: callback({'phase':phase,**details})


def checksum(path):
    digest=hashlib.md5()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def active_legacy_file(path):
    """Avoid overwriting a recently changing archive from older torchvision code."""
    path=Path(path)
    if not path.exists():return False
    # Older torchvision may still own a writer even when the socket has stalled.
    descriptors=Path("/proc/self/fd")
    if descriptors.exists():
        for descriptor in descriptors.iterdir():
            try:
                if descriptor.resolve()==path.resolve():
                    info=Path(f"/proc/self/fdinfo/{descriptor.name}").read_text()
                    flags=int(next(line.split()[1] for line in info.splitlines() if line.startswith("flags:")),8)
                    if flags & os.O_ACCMODE != os.O_RDONLY:return True
            except (OSError,StopIteration,ValueError):pass
    # Existing unverified recent files are conservatively considered active.
    return time.time()-path.stat().st_mtime<60


def download_file(url,target,md5,on_event=None,*,read_timeout=20,max_seconds=300,session=None):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    emit(on_event,'cache_check',message=f'Verificando {target.name}')
    if target.exists() and checksum(target)==md5:
        emit(on_event,'cached',message=f'Archivo verificado: {target.name}')
        return target
    if active_legacy_file(target):
        raise RuntimeError(f'{target.name}: hay un archivo incompleto reciente (posible descarga en curso). No se sobrescribe. Esperá a que termine; si dejó de descargarse, reintentá tras 60 segundos. Podés colocar manualmente el archivo oficial completo: se verificará su MD5.')
    temporary=target.with_name(target.name+'.part')
    started=time.monotonic();received=0;owned=False
    client=session or requests.Session()
    try:
        with FileLock(str(target)+'.lock',timeout=0):
            owned=True
            emit(on_event,'downloading',message=f'Descargando {target.name}',downloaded=0,total=None,elapsed=0)
            with client.get(url,stream=True,timeout=(10,read_timeout)) as response:
                response.raise_for_status()
                raw_length=response.headers.get('Content-Length')
                total=int(raw_length) if raw_length and raw_length.isdigit() else None
                emit(on_event,'downloading',message=f'Descargando {target.name}',downloaded=0,total=total,elapsed=time.monotonic()-started)
                with temporary.open('wb') as output:
                    for chunk in response.iter_content(chunk_size=16*1024):
                        elapsed=time.monotonic()-started
                        if elapsed>max_seconds:raise TimeoutError(f'Descarga superó {max_seconds}s. La fuente es demasiado lenta; reintentá más tarde o descargá manualmente el archivo oficial.')
                        if not chunk:continue
                        output.write(chunk);received+=len(chunk)
                        emit(on_event,'downloading',message=f'Descargando {target.name}',downloaded=received,total=total,elapsed=elapsed,speed=received/max(elapsed,.001))
            emit(on_event,'verifying',message=f'Verificando MD5 de {target.name}',downloaded=received,total=total)
            if checksum(temporary)!=md5:raise ValueError(f'Integridad incorrecta para {target.name}: MD5 no coincide. El parcial no se usará.')
            if target.exists():target.rename(target.with_name(target.name+f'.invalid-{time.time_ns()}'))
            os.replace(temporary,target)
            return target
    except Timeout as error:
        raise RuntimeError(f'{target.name}: otra sesión está descargando este archivo. Esperá; no se inicia una segunda descarga.') from error
    except requests.RequestException as error:
        raise TimeoutError(f'No se pudo descargar {target.name}: conexión/lectura limitada a {read_timeout}s. Reintentá o colocá el archivo oficial en {target.parent}/; no se usará un parcial.') from error
    finally:
        # A lock timeout must not remove another session's partial file.
        if owned and temporary.exists():temporary.unlink()
        if session is None:client.close()


def extract_tar(archive,destination,on_event=None):
    root=Path(destination).resolve()
    emit(on_event,'extracting',message=f'Extrayendo archivo verificado {Path(archive).name}')
    with tarfile.open(archive,'r:gz') as source:
        for member in source.getmembers():
            if member.issym() or member.islnk() or not (root/member.name).resolve().is_relative_to(root):
                raise ValueError('Archivo inseguro: ruta o enlace fuera del directorio de datos.')
        source.extractall(root,filter='data')


def ensure_dataset(factory,root,on_event=None):
    """Use torchvision's own metadata/checks; never delegate an unbounded download."""
    root=Path(root)
    try:
        factory(str(root),train=True,download=False)
        factory(str(root),train=False,download=False)
        emit(on_event,'cached',message='Dataset local disponible; leyendo archivos')
        return
    except RuntimeError:
        pass
    if hasattr(factory,'tgz_md5'):
        archive=download_file(factory.url,root/factory.filename,factory.tgz_md5,on_event)
        extract_tar(archive,root,on_event)
    else:
        raw=root/factory.__name__/'raw';raw.mkdir(parents=True,exist_ok=True)
        for filename,md5 in factory.resources:
            # Metadata is torchvision's official source, with no blind retries/mirror loops.
            archive=download_file(factory.mirrors[0]+filename,raw/filename,md5,on_event)
            emit(on_event,'extracting',message=f'Descomprimiendo {filename} verificado')
            output=raw/filename.removesuffix('.gz')
            with gzip.open(archive,'rb') as source,output.open('wb') as destination:shutil.copyfileobj(source,destination)
