#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, struct
from pathlib import Path
def digest(path:Path):
    h=hashlib.sha256(); count=0; total=0
    with path.open('rb') as f:
        header=f.read(32)
        if len(header)!=32 or header[:4]!=b'DKIF': raise RuntimeError('invalid IVF')
        declared=struct.unpack_from('<I',header,24)[0]
        while True:
            fh=f.read(12)
            if not fh: break
            if len(fh)!=12: raise RuntimeError('truncated IVF frame header')
            size=struct.unpack_from('<I',fh,0)[0]; payload=f.read(size)
            if len(payload)!=size: raise RuntimeError('truncated IVF frame')
            h.update(payload); count+=1; total+=size
        if declared not in (0,count): raise RuntimeError(f'IVF frame count {count} != {declared}')
    return {'payload_sha256':h.hexdigest(),'frames':count,'payload_bytes':total}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('path',type=Path); a=ap.parse_args(); print(json.dumps(digest(a.path),sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
