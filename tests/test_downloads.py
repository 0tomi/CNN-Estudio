import hashlib
import io
import os
from pathlib import Path
import tarfile
import time
import pytest
import requests
from cnn_lab.downloads import download_file,extract_tar

class Response:
    def __init__(self,chunks):self.chunks=chunks;self.headers={'Content-Length':'6'}
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def raise_for_status(self):pass
    def iter_content(self,chunk_size):
        for value in self.chunks:
            if isinstance(value,Exception):raise value
            yield value
class Session:
    def __init__(self,chunks):self.chunks=chunks;self.calls=[]
    def get(self,url,**kwargs):self.calls.append(kwargs);return Response(self.chunks)

def test_download_progress_integrity_cache(tmp_path):
    target=tmp_path/'archive.gz';events=[];client=Session([b'abc',b'def']);md5=hashlib.md5(b'abcdef').hexdigest()
    download_file('https://official.test/file',target,md5,events.append,session=client)
    assert target.read_bytes()==b'abcdef'
    assert any(event['phase']=='downloading' and event.get('downloaded')==3 and event['total']==6 for event in events)
    assert client.calls[0]['timeout']==(10,20)
    download_file('https://official.test/file',target,md5,events.append,session=client)
    assert len(client.calls)==1 and events[-1]['phase']=='cached'
    assert not target.with_name(target.name+'.part').exists()

@pytest.mark.parametrize('chunks',[ [b'abc',requests.exceptions.ReadTimeout('stalled')],[b'corrupt'] ])
def test_partial_never_becomes_valid_cache(tmp_path,chunks):
    target=tmp_path/'archive';client=Session(chunks)
    with pytest.raises((TimeoutError,ValueError)):
        download_file('https://official.test/file',target,hashlib.md5(b'abcdef').hexdigest(),session=client)
    assert not target.exists() and not target.with_name('archive.part').exists()

def test_existing_active_partial_not_touched(tmp_path):
    target=tmp_path/'archive';target.write_bytes(b'user download')
    client=Session([b'abc'])
    with pytest.raises(RuntimeError,match='incompleto reciente'):
        download_file('https://official.test/file',target,'wrong',session=client)
    assert target.read_bytes()==b'user download' and not client.calls

def test_deadline_and_safe_extraction(tmp_path,monkeypatch):
    import cnn_lab.downloads as downloads
    values=iter([0.,0.,301.])
    monkeypatch.setattr(downloads.time,'monotonic',lambda:next(values))
    with pytest.raises(TimeoutError,match='superó'):
        download_file('https://official.test/file',tmp_path/'archive','md5',session=Session([b'abc']))
    assert not (tmp_path/'archive.part').exists()
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as output:
        info=tarfile.TarInfo('../escape');info.size=1;output.addfile(info,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='inseguro'):extract_tar(archive,tmp_path/'data')
    assert not (tmp_path/'escape').exists()

def test_lock_conflict_does_not_delete_other_partial(tmp_path):
    from filelock import FileLock
    target=tmp_path/'archive';partial=tmp_path/'archive.part';partial.write_bytes(b'other session')
    client=Session([b'abc'])
    with FileLock(str(target)+'.lock'):
        with pytest.raises(RuntimeError,match='otra sesión'):
            download_file('https://official.test/file',target,'md5',session=client)
    assert partial.read_bytes()==b'other session' and not client.calls

def test_stalled_legacy_writer_even_with_old_mtime_is_preserved(tmp_path):
    if not Path('/proc/self/fd').exists():pytest.skip('Linux descriptor safety check')
    target=tmp_path/'archive';client=Session([b'abc'])
    with target.open('wb') as writer:
        writer.write(b'legacy writer');writer.flush();os.utime(target,(1,1))
        with pytest.raises(RuntimeError,match='incompleto reciente'):
            download_file('https://official.test/file',target,'md5',session=client)
        assert target.read_bytes()==b'legacy writer' and not client.calls

def test_ready_dataset_uses_cache_without_any_download(tmp_path,monkeypatch):
    import cnn_lab.downloads as downloads
    calls=[];events=[]
    class Ready:
        def __init__(self,root,train,download):
            assert download is False
            calls.append(train)
    def forbidden(*args,**kwargs):raise AssertionError('No debe abrir red con caché disponible')
    monkeypatch.setattr(downloads,'download_file',forbidden)
    downloads.ensure_dataset(Ready,tmp_path,events.append)
    assert calls==[True,False] and events[-1]['phase']=='cached'
