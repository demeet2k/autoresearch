#!/usr/bin/env python3
"""Source-certified six-family Voynich latent-geometry tournament.

Families: D, O, QO, s, sh, she. The script reuses the V3 IVTFF source
parser, scores every complement-quotiented binary partition, and searches
injective three-bit codebooks modulo bit permutation/complement gauge.
No semantic labels are model features.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence
import v3_dual_witness as v3

EPS=1e-12
FAMS=("D","O","QO","S","SH","SHE")
MODES=("primitive","s_scope","sh_scope","primitive_broad_q")

@dataclass(frozen=True)
class R:
    witness:str; folio:str; family:str; remainder:str; position:float
    position_bin:int; initial:bool; terminal:bool; language:str; hand:str; section:str

def parse_token(t:str,mode:str):
    if t.startswith("qo"): return "QO",t[2:] or "<EMPTY>"
    if t.startswith("q"): return ("QO",t[1:] or "<EMPTY>") if mode=="primitive_broad_q" else None
    if t.startswith("she"):
        return "SHE",(t[1:] if mode=="s_scope" else t[2:] if mode=="sh_scope" else t[3:]) or "<EMPTY>"
    if t.startswith("sh"): return "SH",(t[1:] if mode=="s_scope" else t[2:]) or "<EMPTY>"
    if t.startswith("s"): return "S",t[1:] or "<EMPTY>"
    if t.startswith("d"): return "D",t[1:] or "<EMPTY>"
    if t.startswith("o"): return "O",t[1:] or "<EMPTY>"
    return None

def make_records(lines:Sequence[v3.Line],mode:str,bins:int=5)->list[R]:
    out=[]
    for line in lines:
        n=len(line.tokens)
        for i,t in enumerate(line.tokens):
            p=parse_token(t,mode)
            if not p: continue
            fam,rem=p; x=.5 if n<=1 else i/(n-1)
            out.append(R(line.witness,line.folio,fam,rem,x,min(bins-1,int(x*bins)),i==0,i==n-1,line.language,line.hand,line.section))
    return out

def canon(sub:Iterable[str])->tuple[str,...]:
    a=tuple(sorted(set(sub))); b=tuple(sorted(set(FAMS)-set(a))); return min(a,b)

def partitions():
    return tuple(sorted({canon(c) for n in range(1,6) for c in itertools.combinations(FAMS,n)},key=lambda x:(len(x),x)))
PARTS=partitions()

def bit(f:str,p:tuple[str,...])->int: return 0 if f in p else 1

def add(d,key,n,o):
    a,b=d.get(key,(0,0)); d[key]=(a+n,b+o)

def stats(rs:Sequence[R],p:tuple[str,...],group:Callable[[R],str]):
    d={k:{} for k in ("g","x","gx","m","gm","r","gr","j","gj")}; total=ones=0
    for z in rs:
        y=bit(z.family,p); g=group(z); m=(z.section,z.language,z.hand); total+=1; ones+=y
        for name,key in (("g",g),("x",z.position_bin),("gx",(g,z.position_bin)),("m",m),("gm",(g,m)),("r",z.remainder),("gr",(g,z.remainder)),("j",(z.remainder,z.position_bin)),("gj",(g,z.remainder,z.position_bin))): add(d[name],key,1,y)
    return total,ones,d

def sub(a,b): return a[0]-b[0],a[1]-b[1]
def sm(c,prior,a=4.): return (c[1]+a*prior)/(c[0]+a)
def clip(x): return min(1-EPS,max(EPS,x))

def prob(st,z:R,held:str,use_rem=True):
    total,ones,d=st; gh=d["g"].get(held,(0,0)); n=total-gh[0]; g=(ones-gh[1]+4)/(n+8)
    pc=sub(d["x"].get(z.position_bin,(0,0)),d["gx"].get((held,z.position_bin),(0,0)))
    mk=(z.section,z.language,z.hand); mc=sub(d["m"].get(mk,(0,0)),d["gm"].get((held,mk),(0,0)))
    px=sm(pc,g) if pc[0] else g; pm=sm(mc,g) if mc[0] else g
    if not use_rem:return clip((px+pm+g)/3)
    rc=sub(d["r"].get(z.remainder,(0,0)),d["gr"].get((held,z.remainder),(0,0)))
    jk=(z.remainder,z.position_bin); jc=sub(d["j"].get(jk,(0,0)),d["gj"].get((held,*jk),(0,0)))
    pr=sm(rc,g) if rc[0] else g; pj=sm(jc,(pr+px)/2) if jc[0] else (pr+px)/2
    return clip((2*pj+pr+px+pm+g)/6)

def part_eval(rs:Sequence[R],group,use_rem=True):
    gs=[group(z) for z in rs]; scores={}; probs={}
    for p in PARTS:
        st=stats(rs,p,group); loss=0.; ps=[]
        for z,h in zip(rs,gs):
            q=prob(st,z,h,use_rem); y=bit(z.family,p); loss-=math.log(q if y else 1-q); ps.append(q)
        k=",".join(p); scores[k]=loss/max(1,len(rs)); probs[k]=ps
    return scores,probs

def codes(ps): return {f:"".join("0" if f in p else "1" for p in ps) for f in FAMS}
def injective(ps): return len(set(codes(ps).values()))==6

def orbit(c:Mapping[str,str]):
    best=None
    for perm in itertools.permutations(range(3)):
        for mask in range(8):
            x=tuple("".join(str(int(c[f][perm[j]])^((mask>>j)&1)) for j in range(3)) for f in FAMS)
            if best is None or x<best:best=x
    return best

def code_loss(rs,ps,pp):
    c=codes(ps); loss=0.
    for i,z in enumerate(rs):
        w={}
        for f in FAMS:
            a=1.
            for j,p in enumerate(ps):
                q=pp[",".join(p)][i]; a*=q if c[f][j]=="1" else 1-q
            w[f]=a
        loss-=math.log(max(EPS,w[z.family]/sum(w.values())))
    return loss/max(1,len(rs))

def flat_loss(rs:Sequence[R],group,use_rem=True):
    gs=[group(z) for z in rs]; T=Counter(z.family for z in rs); GN=Counter(gs); G=Counter((g,z.family) for g,z in zip(gs,rs))
    XN=Counter(z.position_bin for z in rs); X=Counter((z.position_bin,z.family) for z in rs); GXN=Counter((g,z.position_bin) for g,z in zip(gs,rs)); GX=Counter((g,z.position_bin,z.family) for g,z in zip(gs,rs))
    MN=Counter((z.section,z.language,z.hand) for z in rs); M=Counter(((z.section,z.language,z.hand),z.family) for z in rs); GMN=Counter((g,(z.section,z.language,z.hand)) for g,z in zip(gs,rs)); GM=Counter((g,(z.section,z.language,z.hand),z.family) for g,z in zip(gs,rs))
    RN=Counter(z.remainder for z in rs); RC=Counter((z.remainder,z.family) for z in rs); GRN=Counter((g,z.remainder) for g,z in zip(gs,rs)); GRC=Counter((g,z.remainder,z.family) for g,z in zip(gs,rs))
    JN=Counter((z.remainder,z.position_bin) for z in rs); J=Counter((z.remainder,z.position_bin,z.family) for z in rs); GJN=Counter((g,z.remainder,z.position_bin) for g,z in zip(gs,rs)); GJ=Counter((g,z.remainder,z.position_bin,z.family) for g,z in zip(gs,rs))
    loss=0.
    for z,h in zip(rs,gs):
        nt=len(rs)-GN[h]; glob={f:(T[f]-G[(h,f)]+4)/(nt+24) for f in FAMS}; ds=[glob]
        nx=XN[z.position_bin]-GXN[(h,z.position_bin)]
        if nx>0:ds.append({f:(X[(z.position_bin,f)]-GX[(h,z.position_bin,f)]+4*glob[f])/(nx+4) for f in FAMS})
        m=(z.section,z.language,z.hand); nm=MN[m]-GMN[(h,m)]
        if nm>0:ds.append({f:(M[(m,f)]-GM[(h,m,f)]+4*glob[f])/(nm+4) for f in FAMS})
        if use_rem:
            nr=RN[z.remainder]-GRN[(h,z.remainder)]
            if nr>0:ds.append({f:(RC[(z.remainder,f)]-GRC[(h,z.remainder,f)]+4*glob[f])/(nr+4) for f in FAMS})
            j=(z.remainder,z.position_bin); nj=JN[j]-GJN[(h,*j)]
            if nj>0:
                jd={f:(J[(*j,f)]-GJ[(h,*j,f)]+4*glob[f])/(nj+4) for f in FAMS}; ds.extend((jd,jd))
        mix={f:sum(d[f] for d in ds)/len(ds) for f in FAMS}; norm=sum(mix.values()); loss-=math.log(max(EPS,mix[z.family]/norm))
    return loss/max(1,len(rs))

def tournament(rs:Sequence[R],group,use_rem=True,top=10):
    sc,pp=part_eval(rs,group,use_rem); cand=[]
    for ps in itertools.combinations(PARTS,3):
        if injective(ps):cand.append((sum(sc[",".join(p)] for p in ps),ps,codes(ps),orbit(codes(ps))))
    cand.sort(key=lambda x:x[0]); uniq=[]; seen=set()
    for x in cand:
        if x[3] in seen:continue
        seen.add(x[3]); uniq.append(x)
        if len(uniq)>=top:break
    exact=[]
    for approx,ps,c,o in uniq:exact.append({"bit_loss":approx,"six_class_loss":code_loss(rs,ps,pp),"partitions":[list(p) for p in ps],"codes":c,"orbit":list(o)})
    exact.sort(key=lambda x:x["six_class_loss"])
    return {"records":len(rs),"best_partitions":[{"partition":k,"loss":v} for k,v in sorted(sc.items(),key=lambda q:q[1])[:10]],"best_codebooks":exact,"flat_loss":flat_loss(rs,group,use_rem)}

def metrics(rs):
    d=defaultdict(list)
    for z in rs:d[z.family].append(z)
    return {f:{"n":len(x),"mean_position":sum(z.position for z in x)/len(x),"initial_rate":sum(z.initial for z in x)/len(x),"terminal_rate":sum(z.terminal for z in x)/len(x),"unique_remainders":len({z.remainder for z in x})} for f,x in sorted(d.items())}

def overlap(rs):
    s={f:{z.remainder for z in rs if z.family==f} for f in FAMS}; out={}
    for a,b in itertools.combinations(FAMS,2):
        i=len(s[a]&s[b]); u=len(s[a]|s[b]); out[f"{a}|{b}"]={"intersection":i,"jaccard":i/u if u else 0.}
    return out

def run(zlp,itp,zsha=None,isha=None):
    zl=v3.parse_ivtff(zlp,"ZL3b",True); it=v3.parse_ivtff(itp,"IT",True)
    rep={"schema":"VBR.SIX_FAMILY.v4","sources":{"ZL3b":{"sha256":v3.sha256(zlp),"verified":v3.sha256(zlp)==zsha if zsha else None},"IT":{"sha256":v3.sha256(itp),"verified":v3.sha256(itp)==isha if isha else None}},"modes":{}}
    tests=(("folio_pooled",lambda z:z.folio,True),("position_only",lambda z:z.folio,False),("cross_witness",lambda z:z.witness,True),("leave_remainder",lambda z:z.remainder,True),("leave_section",lambda z:z.section,True),("leave_hand",lambda z:f"{z.witness}:{z.hand}",True))
    for mode in MODES:
        rz=make_records(zl,mode);ri=make_records(it,mode);rp=rz+ri
        mr={"metrics":metrics(rp),"remainder_overlap":overlap(rp),"tests":{}}
        for name,g,rem in tests:mr["tests"][name]=tournament(rp,g,rem)
        mr["tests"]["folio_zl"]=tournament(rz,lambda z:z.folio,True);mr["tests"]["folio_it"]=tournament(ri,lambda z:z.folio,True)
        rep["modes"][mode]=mr
    raw=json.dumps(rep,sort_keys=True,separators=(",",":"));rep["receipt_sha256"]=hashlib.sha256(raw.encode()).hexdigest();return rep

def render(rep):
    x=["# Voynich × BR21 V4 six-family latent geometry","",f"Receipt: `{rep['receipt_sha256']}`","","| mode | test | codebook | flat | codes |","|---|---|---:|---:|---|"]
    for m,d in rep["modes"].items():
        for t,r in d["tests"].items():
            b=r["best_codebooks"][0]; c=" ".join(f"{f}:{b['codes'][f]}" for f in FAMS);x.append(f"| {m} | {t} | {b['six_class_loss']:.6f} | {r['flat_loss']:.6f} | `{c}` |")
    x += ["","## Primitive family metrics","","| family | n | mean position | initial | terminal | remainders |","|---|---:|---:|---:|---:|---:|"]
    for f,z in rep["modes"]["primitive"]["metrics"].items():x.append(f"| {f} | {z['n']:,} | {z['mean_position']:.4f} | {z['initial_rate']:.2%} | {z['terminal_rate']:.2%} | {z['unique_remainders']:,} |")
    return "\n".join(x)+"\n"

def self_test():
    p=Path('.v4test');p.write_text('#=IVTFF EvaT 1.7\n<f1r> <! $I=T $L=A $H=1>\n<f1r.1,@P0> sain.shain.sheain.qoain.oain.dain\n<f2r> <! $I=H $L=B $H=2>\n<f2r.1,@P0> sol.shol.sheol.qool.ool.dol\n')
    try:
        ls=v3.parse_ivtff(p,'X',True);rs=make_records(ls,'primitive');assert len(rs)==12 and len(PARTS)==31 and tournament(rs,lambda z:z.folio)["best_codebooks"]
    finally:p.unlink(missing_ok=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--zl');a.add_argument('--it');a.add_argument('--zl-sha');a.add_argument('--it-sha');a.add_argument('--json',default='six_family.json');a.add_argument('--markdown',default='six_family.md');a.add_argument('--self-test',action='store_true');q=a.parse_args()
    if q.self_test:self_test();print('self-test: PASS');return 0
    if not q.zl or not q.it:a.error('--zl and --it required')
    r=run(Path(q.zl),Path(q.it),q.zl_sha,q.it_sha);Path(q.json).write_text(json.dumps(r,indent=2,sort_keys=True));Path(q.markdown).write_text(render(r));print(json.dumps({"receipt":r['receipt_sha256']},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
