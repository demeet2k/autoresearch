#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, tempfile, urllib.request
from pathlib import Path

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def download(url:str,dst:Path)->None:
    dst.parent.mkdir(parents=True,exist_ok=True)
    req=urllib.request.Request(url,headers={'User-Agent':'QSHRINK-TrackA-Confirmatory/1.0'})
    with tempfile.NamedTemporaryFile(dir=dst.parent,delete=False) as t: tmp=Path(t.name)
    try:
        with urllib.request.urlopen(req,timeout=180) as r,tmp.open('wb') as out: shutil.copyfileobj(r,out,1<<20)
        os.replace(tmp,dst)
    finally: tmp.unlink(missing_ok=True)

def canonicalize(src:Path,dst:Path,frames:int,filt:str)->None:
    dst.parent.mkdir(parents=True,exist_ok=True); tmp=dst.with_suffix('.part.y4m')
    cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(src),'-map','0:v:0','-an','-frames:v',str(frames),'-vf',filt,'-pix_fmt','yuv420p','-color_range','tv','-colorspace','smpte170m','-color_primaries','smpte170m','-color_trc','smpte170m','-f','yuv4mpegpipe',str(tmp)]
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode: raise RuntimeError(p.stderr[-4000:])
    os.replace(tmp,dst)

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--registry',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args()
    m=json.loads(a.manifest.read_text()); r=json.loads(a.registry.read_text())
    if r['benchmark_id']!=m['benchmark_id'] or r['manifest_sha256']!=hashlib.sha256(a.manifest.read_bytes()).hexdigest(): raise RuntimeError('manifest/registry mismatch')
    by={x['id']:x for x in r['clips']}; frames=int(m['canonicalization']['frames']); filt=m['canonicalization']['filter']; locked=[]
    for c in m['clips']:
        e=by[c['id']]; raw=a.output_dir/'raw'/c['filename']; can=a.output_dir/'canonical'/f"{c['id']}.{frames}f.limited.y4m"
        download(c['url'],raw)
        if sha256(raw)!=e['source_sha256']: raise RuntimeError(f"source hash mismatch: {c['id']}")
        canonicalize(raw,can,frames,filt)
        if sha256(can)!=e['canonical_sha256']: raise RuntimeError(f"canonical hash mismatch: {c['id']}")
        locked.append({'id':c['id'],'split':c['split'],'source_path':str(raw.resolve()),'canonical_path':str(can.resolve()),'source_sha256':e['source_sha256'],'canonical_sha256':e['canonical_sha256'],'frames':frames})
    out={'schema':'qshrink.track-a-confirmatory-runtime-lock/v1','status':'LOCKED','benchmark_id':m['benchmark_id'],'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'registry_sha256':hashlib.sha256(a.registry.read_bytes()).hexdigest(),'clips':locked}
    (a.output_dir/'corpus_lock.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'status':'LOCKED','clips':len(locked)},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
