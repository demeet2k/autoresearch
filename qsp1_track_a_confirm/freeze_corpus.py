#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, tempfile, time, urllib.request
from pathlib import Path

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def probe(path: Path) -> dict:
    p=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames,color_range,color_space,color_primaries,color_transfer','-of','json',str(path)],capture_output=True,text=True)
    if p.returncode: raise RuntimeError(p.stderr[-4000:])
    streams=json.loads(p.stdout).get('streams',[])
    if not streams: raise RuntimeError(f'no video stream: {path}')
    return streams[0]

def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True,exist_ok=True)
    req=urllib.request.Request(url,headers={'User-Agent':'QSHRINK-TrackA-Confirmatory/1.0'})
    with tempfile.NamedTemporaryFile(dir=dst.parent,delete=False) as t: tmp=Path(t.name)
    try:
        with urllib.request.urlopen(req,timeout=180) as r,tmp.open('wb') as out:
            shutil.copyfileobj(r,out,1<<20)
        if tmp.stat().st_size==0: raise RuntimeError(f'empty download: {url}')
        os.replace(tmp,dst)
    finally: tmp.unlink(missing_ok=True)

def canonicalize(src: Path,dst: Path,frames:int,filter_text:str)->None:
    dst.parent.mkdir(parents=True,exist_ok=True)
    tmp=dst.with_suffix('.part.y4m')
    cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-i',str(src),'-map','0:v:0','-an','-frames:v',str(frames),'-vf',filter_text,'-pix_fmt','yuv420p','-color_range','tv','-colorspace','smpte170m','-color_primaries','smpte170m','-color_trc','smpte170m','-f','yuv4mpegpipe',str(tmp)]
    p=subprocess.run(cmd,capture_output=True,text=True)
    if p.returncode: raise RuntimeError(f'canonicalization failed: {src}\n{p.stderr[-4000:]}')
    os.replace(tmp,dst)

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--registry-out',type=Path,required=True); ap.add_argument('--receipt-out',type=Path,required=True); a=ap.parse_args()
    manifest_bytes=a.manifest.read_bytes(); m=json.loads(manifest_bytes); frames=int(m['canonicalization']['frames']); filt=m['canonicalization']['filter']
    raw=a.output_dir/'raw'; can=a.output_dir/'canonical'; entries=[]
    for c in m['clips']:
        cid=c['id']; src=raw/c['filename']; dst=can/f'{cid}.{frames}f.limited.y4m'
        download(c['url'],src); canonicalize(src,dst,frames)
        sp=probe(src); cp=probe(dst); count=int(cp.get('nb_read_frames','0'))
        if count!=frames: raise RuntimeError(f'{cid}: {count} frames != {frames}')
        entries.append({'id':cid,'split':c['split'],'url':c['url'],'filename':c['filename'],'source_bytes':src.stat().st_size,'canonical_bytes':dst.stat().st_size,'source_sha256':sha256(src),'canonical_sha256':sha256(dst),'source_probe':sp,'canonical_probe':cp,'frames':frames})
    registry={'schema':'qshrink.track-a-confirmatory-registry/v1','benchmark_id':m['benchmark_id'],'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'canonicalization':m['canonicalization'],'complete':True,'status':'LOCKED','clips':entries}
    a.registry_out.parent.mkdir(parents=True,exist_ok=True); a.registry_out.write_text(json.dumps(registry,indent=2,sort_keys=True)+'\n')
    receipt={'schema':'qshrink.track-a-confirmatory-freeze-receipt/v1','status':'COMPLETE','benchmark_id':m['benchmark_id'],'created_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'manifest_sha256':registry['manifest_sha256'],'registry_sha256':hashlib.sha256(a.registry_out.read_bytes()).hexdigest(),'clip_count':len(entries),'new_heldout_count':sum(x['split']=='novel_heldout' for x in entries),'benchmark_executed':False,'total_source_bytes':sum(x['source_bytes'] for x in entries),'total_canonical_bytes':sum(x['canonical_bytes'] for x in entries)}
    a.receipt_out.parent.mkdir(parents=True,exist_ok=True); a.receipt_out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n'); print(json.dumps(receipt,indent=2,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
