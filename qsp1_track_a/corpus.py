#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, tempfile, urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''): h.update(block)
    return h.hexdigest()


def download(url: str, destination: Path, timeout: float=180.0) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as f:
        temp=Path(f.name)
    try:
        req=urllib.request.Request(url, headers={'User-Agent':'QSHRINK-TrackA-Public/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as response, temp.open('wb') as out:
            shutil.copyfileobj(response, out, length=1<<20)
        if temp.stat().st_size <= 0: raise RuntimeError(f'empty download: {url}')
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def canonicalize(source: Path, destination: Path, frames: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp=destination.with_suffix('.part.y4m')
    temp.unlink(missing_ok=True)
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(source),'-map','0:v:0','-an','-frames:v',str(frames),'-vf','format=yuv420p','-pix_fmt','yuv420p','-f','yuv4mpegpipe',str(temp)]
    p=subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode: raise RuntimeError(f'canonicalization failed: {p.stderr[-2000:]}')
    os.replace(temp,destination)


def acquire(manifest_path: Path, registry_path: Path, output: Path) -> dict:
    manifest_bytes=manifest_path.read_bytes(); manifest=json.loads(manifest_bytes)
    registry=json.loads(registry_path.read_text())
    if registry['benchmark_id'] != manifest['benchmark_id']: raise RuntimeError('benchmark ID mismatch')
    if registry['manifest_sha256'] != hashlib.sha256(manifest_bytes).hexdigest(): raise RuntimeError('manifest hash mismatch')
    expected={x['id']:x for x in registry['clips']}
    frames=int(manifest['canonicalization']['frames'])
    raw=output/'raw'; canonical=output/'canonical'; raw.mkdir(parents=True,exist_ok=True); canonical.mkdir(parents=True,exist_ok=True)
    entries=[]
    for clip in manifest['clips']:
        e=expected[clip['id']]
        src=raw/clip['filename']; dst=canonical/f"{clip['id']}.{frames}f.y4m"
        download(clip['url'],src); canonicalize(src,dst,frames)
        actual={'id':clip['id'],'split':clip['split'],'url':clip['url'],'filename':clip['filename'],'source_bytes':src.stat().st_size,'canonical_bytes':dst.stat().st_size,'source_sha256':sha256(src),'canonical_sha256':sha256(dst),'frames':frames}
        for key,value in e.items():
            if key in actual and actual[key] != value: raise RuntimeError(f"{clip['id']} {key} mismatch: {actual[key]!r} != {value!r}")
        actual['canonical_path']=str(dst.resolve()); entries.append(actual)
    lock={'schema':'qshrink.track-a-public-lock/v1','benchmark_id':manifest['benchmark_id'],'manifest_sha256':registry['manifest_sha256'],'source_registry_sha256':registry['source_registry_sha256'],'registry_projection_sha256':sha256(registry_path),'status':'LOCKED','complete':True,'clips':entries}
    output.mkdir(parents=True,exist_ok=True); (output/'corpus_lock.json').write_text(json.dumps(lock,indent=2,sort_keys=True)+'\n')
    return lock


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--registry',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    print(json.dumps(acquire(a.manifest,a.registry,a.output),indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
