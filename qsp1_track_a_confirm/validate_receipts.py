#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, statistics, time
from collections import defaultdict
from pathlib import Path

def sha(path:Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--points',type=Path,required=True); ap.add_argument('--features',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--registry',type=Path,required=True); ap.add_argument('--policy',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    rows=list(csv.DictReader(a.points.open(),delimiter='\t')); groups=defaultdict(list)
    for r in rows:
        groups[(r['clip_id'],r['config'],int(r['crf']))].append(r)
        if int(r['decoded_frames'])!=72 or int(r['payload_frames'])!=72: raise RuntimeError('decoded/payload frame count mismatch')
    if len(rows)!=576 or len(groups)!=192: raise RuntimeError(f'rows/groups {len(rows)}/{len(groups)}')
    nondet=[]; metric_warn=[]; timings=[]
    for key,rs in groups.items():
        if sorted(int(r['trial']) for r in rs)!=[1,2,3]: raise RuntimeError(f'trial set mismatch: {key}')
        if len({r['payload_sha256'] for r in rs})!=1 or len({r['decoded_sha256'] for r in rs})!=1: nondet.append(key)
        primary=[r for r in rs if r['trial']=='1'][0]
        if not primary['psnr_y_db'] or not primary['ssim_y']: raise RuntimeError(f'missing primary metrics: {key}')
        if int(primary['metric_warning_count'])!=0: metric_warn.append({'key':key,'warnings':int(primary['metric_warning_count'])})
        timings.append({'clip_id':key[0],'config':key[1],'crf':key[2],'median_seconds':statistics.median(float(r['encode_seconds']) for r in rs),'trials':[float(r['encode_seconds']) for r in rs]})
    f=json.loads(a.features.read_text()); routes=defaultdict(int)
    for x in f['clips'].values(): routes[x['route']]+=1
    receipt={'schema':'qshrink.track-a-confirmatory-execution-receipt/v1','status':'COMPLETE','generated_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'benchmark_id':json.loads(a.manifest.read_text())['benchmark_id'],'row_count':len(rows),'group_count':len(groups),'expected_encodes':576,'deterministic_groups':len(nondet)==0,'nondeterministic_groups':[list(x) for x in nondet],'metric_warning_groups':metric_warn,'route_counts':dict(routes),'calibration_routes_verified':f['calibration_routes_verified'],'manifest_sha256':sha(a.manifest),'registry_sha256':sha(a.registry),'policy_sha256':sha(a.policy),'features_sha256':sha(a.features),'points_sha256':sha(a.points),'timing_summary':timings,'claim_status':'UNANALYZED_CONFIRMATORY_RECEIPTS'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    if nondet or metric_warn: raise RuntimeError(f'determinism/metric warnings: {len(nondet)}/{len(metric_warn)}')
    print(json.dumps({'status':'COMPLETE','rows':len(rows),'routes':dict(routes)},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
