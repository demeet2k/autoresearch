#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, math, statistics, time
from collections import defaultdict
from pathlib import Path
from bd_rate import bd_rate, mean_median

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_points(path):
    rows=list(csv.DictReader(Path(path).open(),delimiter='\t')); groups=defaultdict(list)
    for r in rows:
        for k in ('crf','trial','bytes','payload_bytes','decoded_frames','payload_frames','metric_warning_count'): r[k]=int(r[k])
        for k in ('bitrate_kbps','encode_seconds'): r[k]=float(r[k])
        r['psnr_y_db']=float(r['psnr_y_db']) if r['psnr_y_db'] else None; r['ssim_y']=float(r['ssim_y']) if r['ssim_y'] else None
        groups[(r['clip_id'],r['config'],r['crf'])].append(r)
    primary={k:next(x for x in v if x['trial']==1) for k,v in groups.items()}; medtime={k:statistics.median(x['encode_seconds'] for x in v) for k,v in groups.items()}; return rows,primary,medtime
def route(f,ab):
    if ab=='always_baseline': return 'baseline_fast','ablation baseline'
    if (ab!='no_dls' and f['dls_coherence']>=0.25) or f['repeat_ratio']>=0.02:return 'screen_tools','structural rule'
    if ab!='no_temporal' and f['temporal_mad']>=12:return 'complexity_aq','temporal rule'
    if ab!='no_entropy' and f['entropy_bits']<=6.85:return 'quality_slow','entropy rule'
    return 'baseline_fast','baseline rule'
def floor_ok(c,b,p):
    q=p['quality_floor']; return c['psnr_y_db']>=b['psnr_y_db']-q['psnr_y_db_below_baseline_max'] and c['ssim_y']>=b['ssim_y']-q['ssim_y_below_baseline_max']
def curve(clip,f,primary,medtime,p,ab):
    crfs=p['rate_points_crf']; base=p['baseline']; configs=sorted(p['configs'])
    if ab=='offline_oracle_non_deployable':
        candidates=[]
        for config in configs:
            if all(floor_ok(primary[(clip,config,c)],primary[(clip,base,c)],p) for c in crfs): candidates.append((sum(primary[(clip,config,c)]['bytes'] for c in crfs),config))
        requested=min(candidates)[1] if candidates else base; reason='offline byte oracle'
    else: requested,reason=route(f,ab)
    effective=[]
    for c in crfs:
        b=primary[(clip,base,c)]; t=primary[(clip,requested,c)]; fallback=not floor_ok(t,b,p); x=b if fallback else t; cfg=base if fallback else requested
        effective.append({'crf':c,'requested_config':requested,'effective_config':cfg,'quality_floor_fallback':fallback,'bytes':x['bytes'],'bitrate_kbps':x['bitrate_kbps'],'psnr_y_db':x['psnr_y_db'],'ssim_y':x['ssim_y'],'median_encode_seconds':medtime[(clip,cfg,c)],'payload_sha256':x['payload_sha256'],'decoded_sha256':x['decoded_sha256']})
    bp=[primary[(clip,base,c)] for c in crfs]; br=[x['bitrate_kbps'] for x in bp]; bps=[x['psnr_y_db'] for x in bp]; bss=[x['ssim_y'] for x in bp]; er=[x['bitrate_kbps'] for x in effective]; eps=[x['psnr_y_db'] for x in effective]; ess=[x['ssim_y'] for x in effective]
    return {'ablation':ab,'selected_config':requested,'selection_reason':reason,'effective_points':effective,'fallback_count':sum(x['quality_floor_fallback'] for x in effective),'total_baseline_bytes':sum(x['bytes'] for x in bp),'total_effective_bytes':sum(x['bytes'] for x in effective),'pointwise_byte_change_percent':(sum(x['bytes'] for x in effective)/sum(x['bytes'] for x in bp)-1)*100,'median_time_change_percent':(sum(x['median_encode_seconds'] for x in effective)/sum(medtime[(clip,base,c)] for c in crfs)-1)*100,'bd_rate_psnr_y':bd_rate(br,bps,er,eps).to_dict(),'bd_rate_ssim_y':bd_rate(br,bss,er,ess).to_dict()}
def summary(curves,ids,ab):
    xs=[curves[c][ab] for c in ids]
    return {'clips':ids,'bd_rate_psnr_y_percent':mean_median([x['bd_rate_psnr_y']['percent'] for x in xs]),'bd_rate_ssim_y_percent':mean_median([x['bd_rate_ssim_y']['percent'] for x in xs]),'pointwise_byte_change_percent':mean_median([x['pointwise_byte_change_percent'] for x in xs]),'median_time_change_percent':mean_median([x['median_time_change_percent'] for x in xs]),'fallback_count':sum(x['fallback_count'] for x in xs),'nonpositive_both_count':sum(x['bd_rate_psnr_y']['percent']<=0 and x['bd_rate_ssim_y']['percent']<=0 for x in xs)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--points',required=True); ap.add_argument('--features',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--registry',required=True); ap.add_argument('--policy',required=True); ap.add_argument('--execution-receipt',required=True); ap.add_argument('--output',required=True); ap.add_argument('--markdown',required=True); a=ap.parse_args()
    m=json.load(open(a.manifest)); p=json.load(open(a.policy)); f=json.load(open(a.features)); receipt=json.load(open(a.execution_receipt)); rows,primary,medtime=load_points(a.points)
    if not receipt['deterministic_groups'] or receipt['metric_warning_groups']: raise RuntimeError('execution validation failed')
    curves={clip:{ab:curve(clip,x['features'],primary,medtime,p,ab) for ab in p['ablations']} for clip,x in f['clips'].items()}; summaries={split:{ab:summary(curves,ids,ab) for ab in p['ablations']} for split,ids in m['splits'].items()}; s=summaries['novel_heldout']['full']; promoted=s['bd_rate_psnr_y_percent']['mean']<0 and s['bd_rate_ssim_y_percent']['mean']<0 and s['nonpositive_both_count']>=4
    routes={c:x['route'] for c,x in f['clips'].items()}; counts={r:list(routes.values()).count(r) for r in sorted(set(routes.values()))}
    result={'schema':'qshrink.track-a-confirmatory-analysis/v1','status':'COMPLETE','generated_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'benchmark_id':m['benchmark_id'],'policy_id':p['policy_id'],'promotion_classification':'CONFIRMATORY_HELDOUT_GAIN' if promoted else 'NO_CONFIRMATORY_GAIN','integrity':{'manifest_sha256':sha(a.manifest),'registry_sha256':sha(a.registry),'policy_sha256':sha(a.policy),'points_sha256':sha(a.points),'features_sha256':sha(a.features),'execution_receipt_sha256':sha(a.execution_receipt),'rows':len(rows)},'route_counts':counts,'routes':routes,'features':f['clips'],'curves':curves,'summaries':summaries}
    Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    lines=['# QSP1 Track A confirmatory AV1 analysis','',f"- Classification: **{result['promotion_classification']}**",f"- Novel held-out PSNR-Y BD-rate mean: **{s['bd_rate_psnr_y_percent']['mean']:.3f}%**",f"- Novel held-out SSIM-Y BD-rate mean: **{s['bd_rate_ssim_y_percent']['mean']:.3f}%**",f"- Novel clips non-positive on both: **{s['nonpositive_both_count']}/6**",f"- Quality-floor fallbacks: **{s['fallback_count']}**",f"- Route counts: `{json.dumps(counts,sort_keys=True)}`",'', '## Novel held-out clips']
    for c in m['splits']['novel_heldout']:
        x=curves[c]['full']; lines.append(f"- **{c}** → `{x['selected_config']}`: PSNR {x['bd_rate_psnr_y']['percent']:.3f}%, SSIM {x['bd_rate_ssim_y']['percent']:.3f}%, fallbacks {x['fallback_count']}")
    lines+=['','## Attribution ablations']
    for ab in p['ablations']:
        z=summaries['novel_heldout'][ab]; lines.append(f"- `{ab}`: PSNR mean {z['bd_rate_psnr_y_percent']['mean']:.3f}%, SSIM mean {z['bd_rate_ssim_y_percent']['mean']:.3f}%")
    Path(a.markdown).write_text('\n'.join(lines)+'\n'); print('\n'.join(lines)); return 0
if __name__=='__main__': raise SystemExit(main())
