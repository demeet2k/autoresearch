#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, itertools, json, subprocess
from functools import lru_cache
from pathlib import Path
import numpy as np
A=((0,1,2,3),(2,3,0,1),(3,2,1,0),(1,0,3,2)); B=((0,1,2,3),(3,2,1,0),(1,0,3,2),(2,3,0,1)); PERMS=tuple(itertools.permutations(range(4)))
def cell(variant,perm_rank,orient,row,col,m=3):
    n=4**m
    if orient==1: row,col=n-1-col,row
    elif orient==2: row,col=n-1-row,n-1-col
    elif orient==3: row,col=col,n-1-row
    mat=A if variant==0 else B; perm=PERMS[perm_rank]; value=0; place=1
    for _ in range(m): value+=perm[mat[row&3][col&3]]*place; place*=4; row>>=2; col>>=2
    return value
@lru_cache(maxsize=1)
def bank():
    out=[]
    for v in (0,1):
      for p in range(24):
       for o in range(4):
        x=np.array([cell(v,p,o,r,c) for r in range(64) for c in range(64)],dtype=np.float64).reshape(64,64); x=(x-x.mean())/(x.std()+1e-12); out.append(x)
    return tuple(out)
def run(cmd):
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if p.returncode: raise RuntimeError(p.stderr.decode(errors='replace')[-3000:])
    return p.stdout
def probe(path):
    d=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height','-of','json',str(path)])); return d['streams'][0]
def frames_gray(path,frames,w,h):
    data=run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(path),'-frames:v',str(frames),'-vf','format=gray','-f','rawvideo','pipe:1']); a=np.frombuffer(data,dtype=np.uint8)
    if a.size!=frames*w*h: raise RuntimeError(f'gray samples {a.size} != {frames*w*h}')
    return a.reshape(frames,h,w)
def first64(path): return np.frombuffer(run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(path),'-frames:v','1','-vf','scale=64:64:flags=bilinear,format=gray','-f','rawvideo','pipe:1']),dtype=np.uint8).reshape(64,64)
def entropy(x):
    h=np.bincount(x.ravel(),minlength=256).astype(float); p=h[h>0]/x.size; return float(-(p*np.log2(p)).sum())
def repeat(x,b=8):
    hs=[hashlib.blake2s(x[y:y+b,x0:x0+b].tobytes(),digest_size=8).digest() for y in range(0,x.shape[0]-b+1,b) for x0 in range(0,x.shape[1]-b+1,b)]; return 1-len(set(hs))/len(hs) if hs else 0.0
def dls(x):
    z=x.astype(float); z=(z-z.mean())/(z.std()+1e-12); return max(abs(float(np.mean(z*c))) for c in bank())
def choose(f):
    if f['dls_coherence']>=0.25 or f['repeat_ratio']>=0.02: return 'screen_tools','high structural coherence'
    if f['temporal_mad']>=12.0: return 'complexity_aq','high temporal change'
    if f['entropy_bits']<=6.85: return 'quality_slow','low entropy'
    return 'baseline_fast','default baseline'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--lock',type=Path,required=True); ap.add_argument('--policy',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); lock=json.loads(a.lock.read_text()); pol=json.loads(a.policy.read_text()); result={}
    for c in lock['clips']:
        path=Path(c['canonical_path']); pr=probe(path); fs=frames_gray(path,c['frames'],int(pr['width']),int(pr['height'])); f={'entropy_bits':float(np.mean([entropy(x) for x in fs])),'temporal_mad':float(np.abs(np.diff(fs.astype(np.int16),axis=0)).mean()),'repeat_ratio':float(np.mean([repeat(x) for x in fs])),'dls_coherence':dls(first64(path)),'spatial_std':float(fs.astype(float).std())}; route,reason=choose(f); result[c['id']]={'split':c['split'],'features':f,'route':route,'reason':reason}
    exp=pol['calibration_route_expectation']
    for cid,route in exp.items():
        if result[cid]['route']!=route: raise RuntimeError(f'calibration route mismatch {cid}: {result[cid]["route"]} != {route}')
    out={'schema':'qshrink.track-a-confirmatory-features/v1','benchmark_id':lock['benchmark_id'],'policy_id':pol['policy_id'],'calibration_routes_verified':True,'clips':result}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({k:v['route'] for k,v in result.items()},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
