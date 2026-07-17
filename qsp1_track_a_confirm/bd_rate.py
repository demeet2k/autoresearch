#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from typing import Iterable, Sequence
import numpy as np
@dataclass(frozen=True)
class BDRateResult:
    status:str; percent:float|None; overlap_low:float|None; overlap_high:float|None; reference_points:int; test_points:int; polynomial_degree:int|None; detail:str|None=None
    def to_dict(self): return asdict(self)
def _clean(rates:Sequence[float],qualities:Sequence[float]):
    pairs=[]
    for r,q in zip(rates,qualities,strict=True):
        r=float(r); q=float(q)
        if math.isfinite(r) and math.isfinite(q) and r>0:pairs.append((q,r))
    best={}
    for q,r in pairs:best[q]=min(r,best.get(q,float('inf')))
    ordered=sorted(best.items()); return np.asarray([r for q,r in ordered]),np.asarray([q for q,r in ordered])
def bd_rate(rrates,rquals,trates,tquals):
    rr,rq=_clean(rrates,rquals); tr,tq=_clean(trates,tquals)
    if len(rr)<2 or len(tr)<2:return BDRateResult('INSUFFICIENT_POINTS',None,None,None,len(rr),len(tr),None)
    lo=max(float(rq.min()),float(tq.min())); hi=min(float(rq.max()),float(tq.max()))
    if hi<=lo:return BDRateResult('NO_QUALITY_OVERLAP',None,lo,hi,len(rr),len(tr),None)
    degree=min(3,len(rr)-1,len(tr)-1)
    try:
        pref=np.polyfit(rq,np.log(rr),degree); ptest=np.polyfit(tq,np.log(tr),degree); iref=np.polyint(pref); itest=np.polyint(ptest); ar=float(np.polyval(iref,hi)-np.polyval(iref,lo))/(hi-lo); at=float(np.polyval(itest,hi)-np.polyval(itest,lo))/(hi-lo); pct=(math.exp(at-ar)-1)*100
    except Exception as e:return BDRateResult('FIT_FAILED',None,lo,hi,len(rr),len(tr),degree,str(e))
    return BDRateResult('OK',pct,lo,hi,len(rr),len(tr),degree)
def mean_median(values:Iterable[float|None]):
    xs=np.asarray([float(x) for x in values if x is not None and math.isfinite(float(x))])
    if xs.size==0:return {'count':0,'mean':None,'median':None,'minimum':None,'maximum':None}
    return {'count':int(xs.size),'mean':float(xs.mean()),'median':float(np.median(xs)),'minimum':float(xs.min()),'maximum':float(xs.max())}
