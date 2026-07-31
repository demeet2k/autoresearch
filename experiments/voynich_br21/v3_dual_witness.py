#!/usr/bin/env python3
"""Source-certified Voynich × BR21 dual-witness tournament.

Parses IVTFF ZL/IT witnesses, preserves uncertainty, aligns loci, and tests
all three gauge-quotiented two-bit factorization classes. No semantic or
alchemical labels participate in scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

EPS = 1e-12
FAMS = ("D", "O", "S", "QO")
AB_SIGNATURE = (("D", "O"), ("D", "S"))
PARTITIONS = {
    "DO|QOS": frozenset(("D", "O")),
    "DS|OQO": frozenset(("D", "S")),
    "DQO|OS": frozenset(("D", "QO")),
}
CLASSES = {
    "AB": ("DO|QOS", "DS|OQO"),
    "ALT_A": ("DO|QOS", "DQO|OS"),
    "ALT_B": ("DS|OQO", "DQO|OS"),
}
PAGE_RE = re.compile(r"^<(?P<folio>f\d+[rv]\d?)>\s*<!\s*(?P<meta>.*?)>", re.I)
LOCUS_RE = re.compile(r"^<(?P<folio>f\d+[rv]\d?)\.(?P<num>\d+),(?P<code>[^>]+)>\s*(?P<text>.*)$", re.I)
META_RE = re.compile(r"\$(?P<key>[A-Z])=(?P<value>[^\s>]+)")
ALT_RE = re.compile(r"\[([^:\]]+):([^\]]+)\]")
CURLY_RE = re.compile(r"\{([^}]*)\}")
TAG_RE = re.compile(r"<[^>]*>")
HIGH_RE = re.compile(r"@\d+;")
TOKEN_RE = re.compile(r"^[a-z]+$")


@dataclass(frozen=True)
class Line:
    witness: str
    folio: str
    locus: str
    line_number: int
    code: str
    text_raw: str
    text_clean: str
    tokens: tuple[str, ...]
    uncertain: bool
    alternatives: bool
    uncertain_space: bool
    language: str
    hand: str
    illustration: str
    section: str


@dataclass(frozen=True)
class Record:
    witness: str
    folio: str
    locus: str
    family: str
    remainder: str
    position: float
    position_bin: int
    initial: bool
    terminal: bool
    language: str
    hand: str
    section: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def section_of(folio: str) -> str:
    m = re.match(r"f(\d+)", folio)
    n = int(m.group(1)) if m else 0
    if folio == "f1r": return "text_only"
    if 1 <= n <= 66: return "herbal"
    if 67 <= n <= 73: return "astronomical"
    if 75 <= n <= 84: return "biological"
    if 85 <= n <= 86: return "cosmological"
    if 87 <= n <= 102: return "pharmaceutical"
    if 103 <= n <= 116: return "recipes"
    return "other"


def normalize_ivtff(raw: str, strict: bool) -> tuple[str, bool, bool, bool]:
    alternatives = bool(ALT_RE.search(raw))
    uncertain_space = "<->" in raw
    high = bool(HIGH_RE.search(raw))
    text = raw.replace("<->", ".")
    text = ALT_RE.sub(lambda m: m.group(1), text)
    text = HIGH_RE.sub("?", text)
    text = CURLY_RE.sub(lambda m: m.group(1), text)
    text = TAG_RE.sub("", text)
    uncertain = "?" in text or high
    text = text.lower().replace("!", "?")
    text = re.sub(r"[.,:;/|]+", ".", text)
    text = re.sub(r"\.+", ".", text).strip(".")
    if strict and (uncertain or alternatives or uncertain_space):
        return "", uncertain, alternatives, uncertain_space
    toks = []
    for tok in text.split("."):
        tok = re.sub(r"[^a-z?]", "", tok)
        if not tok or "?" in tok or not TOKEN_RE.match(tok):
            continue
        toks.append(tok)
    return ".".join(toks), uncertain, alternatives, uncertain_space


def parse_ivtff(path: Path, witness: str, strict: bool = True) -> list[Line]:
    lines: list[Line] = []
    page_meta: dict[str, dict[str, str]] = {}
    current_meta: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        pm = PAGE_RE.match(raw_line)
        if pm:
            current_meta = {m.group("key"): m.group("value") for m in META_RE.finditer(pm.group("meta"))}
            page_meta[pm.group("folio").lower()] = dict(current_meta)
            continue
        lm = LOCUS_RE.match(raw_line)
        if not lm:
            continue
        code = lm.group("code")
        if "P" not in code or code.startswith("=") or "X" in code:
            continue
        folio = lm.group("folio").lower()
        meta = page_meta.get(folio, current_meta)
        clean, uncertain, alternatives, uncertain_space = normalize_ivtff(lm.group("text"), strict)
        if not clean:
            continue
        tokens = tuple(t for t in clean.split(".") if t)
        if not tokens:
            continue
        num = int(lm.group("num"))
        lines.append(Line(
            witness=witness, folio=folio, locus=f"{folio}:{num}", line_number=num,
            code=code, text_raw=lm.group("text"), text_clean=clean, tokens=tokens,
            uncertain=uncertain, alternatives=alternatives, uncertain_space=uncertain_space,
            language=meta.get("L", "?"), hand=meta.get("H", "?"),
            illustration=meta.get("I", "?"), section=section_of(folio),
        ))
    return lines


def family_parse(token: str, mode: str) -> tuple[str, str] | None:
    if token.startswith("qo"): return "QO", token[2:] or "<EMPTY>"
    if token.startswith("q"):
        return ("QO", token[1:] or "<EMPTY>") if mode == "broad_q" else None
    if token.startswith("she"): return "S", token[3:] or "<EMPTY>"
    if token.startswith("sh"):
        return ("S", token[2:] or "<EMPTY>") if mode != "strict_she" else None
    if token.startswith("s"):
        return ("S", token[1:] or "<EMPTY>") if mode != "strict_she" else None
    if token.startswith("d"): return "D", token[1:] or "<EMPTY>"
    if token.startswith("o"): return "O", token[1:] or "<EMPTY>"
    return None


def records(lines: Sequence[Line], mode: str, bins: int = 5) -> list[Record]:
    out: list[Record] = []
    for line in lines:
        n = len(line.tokens)
        for i, token in enumerate(line.tokens):
            parsed = family_parse(token, mode)
            if not parsed: continue
            fam, rem = parsed
            pos = 0.5 if n <= 1 else i / (n - 1)
            out.append(Record(
                line.witness, line.folio, line.locus, fam, rem, pos,
                min(bins - 1, int(pos * bins)), i == 0, i == n - 1,
                line.language, line.hand, line.section,
            ))
    return out


def clip(p: float) -> float:
    return min(1-EPS, max(EPS, p))


def bit_value(fam: str, partition: str) -> int:
    return 0 if fam in PARTITIONS[partition] else 1


def _inc(table: dict, key, y: int) -> None:
    row = table.setdefault(key, [0, 0])
    row[0] += 1
    row[1] += y


def _sub(total: dict, held: dict, key) -> tuple[int, int]:
    a = total.get(key, (0, 0))
    b = held.get(key, (0, 0))
    return a[0] - b[0], a[1] - b[1]


@dataclass
class CountIndex:
    partition: str
    group_of: Callable[[Record], str]
    global_count: list[int]
    group_global: dict[str, list[int]]
    pos: dict[int, list[int]]
    group_pos: dict[str, dict[int, list[int]]]
    meta: dict[tuple[str, str, str], list[int]]
    group_meta: dict[str, dict[tuple[str, str, str], list[int]]]
    rem: dict[str, list[int]]
    group_rem: dict[str, dict[str, list[int]]]
    joint: dict[tuple[str, int], list[int]]
    group_joint: dict[str, dict[tuple[str, int], list[int]]]


def build_index(recs: Sequence[Record], partition: str, group: Callable[[Record], str]) -> CountIndex:
    idx = CountIndex(partition, group, [0, 0], {}, {}, {}, {}, {}, {}, {}, {}, {})
    for r in recs:
        g = group(r)
        y = bit_value(r.family, partition)
        idx.global_count[0] += 1
        idx.global_count[1] += y
        _inc(idx.group_global, g, y)
        _inc(idx.pos, r.position_bin, y)
        _inc(idx.group_pos.setdefault(g, {}), r.position_bin, y)
        mk = (r.section, r.language, r.hand)
        _inc(idx.meta, mk, y)
        _inc(idx.group_meta.setdefault(g, {}), mk, y)
        _inc(idx.rem, r.remainder, y)
        _inc(idx.group_rem.setdefault(g, {}), r.remainder, y)
        jk = (r.remainder, r.position_bin)
        _inc(idx.joint, jk, y)
        _inc(idx.group_joint.setdefault(g, {}), jk, y)
    return idx


def indexed_probability(idx: CountIndex, target: Record, held_group: str, *, alpha: float = 4.0, use_remainder: bool = True) -> float:
    hg = idx.group_global.get(held_group, (0, 0))
    n = idx.global_count[0] - hg[0]
    ones = idx.global_count[1] - hg[1]
    p_global = (ones + alpha) / (n + 2 * alpha)

    def smooth(pair: tuple[int, int], prior: float) -> float:
        count, one_count = pair
        return (one_count + alpha * prior) / (count + alpha)

    pos_pair = _sub(idx.pos, idx.group_pos.get(held_group, {}), target.position_bin)
    p_pos = smooth(pos_pair, p_global) if pos_pair[0] else p_global
    mk = (target.section, target.language, target.hand)
    meta_pair = _sub(idx.meta, idx.group_meta.get(held_group, {}), mk)
    p_meta = smooth(meta_pair, p_global) if meta_pair[0] else p_global
    if not use_remainder:
        return clip((p_pos + p_meta + p_global) / 3)
    rem_pair = _sub(idx.rem, idx.group_rem.get(held_group, {}), target.remainder)
    p_rem = smooth(rem_pair, p_global) if rem_pair[0] else p_global
    jk = (target.remainder, target.position_bin)
    joint_pair = _sub(idx.joint, idx.group_joint.get(held_group, {}), jk)
    p_joint = smooth(joint_pair, (p_rem + p_pos) / 2) if joint_pair[0] else (p_rem + p_pos) / 2
    return clip((2 * p_joint + p_rem + p_pos + p_meta + p_global) / 6)


def cv_scores(recs: Sequence[Record], group: Callable[[Record], str], use_remainder: bool = True) -> dict[str, float]:
    if not recs:
        return {name: float("inf") for name in CLASSES}
    indices = {part: build_index(recs, part, group) for part in PARTITIONS}
    part_loss = {part: 0.0 for part in PARTITIONS}
    for r in recs:
        held = group(r)
        for part, idx in indices.items():
            p = indexed_probability(idx, r, held, use_remainder=use_remainder)
            y = bit_value(r.family, part)
            part_loss[part] -= math.log(p if y else 1-p)
    return {cls: sum(part_loss[part] for part in parts) / len(recs) for cls, parts in CLASSES.items()}


def cross_witness(recs: Sequence[Record], use_remainder: bool = True) -> dict[str, dict[str, float]]:
    if len({r.witness for r in recs}) < 2:
        return {}
    return {f"train_other_test_{w}": _cross_one(recs, w, use_remainder=use_remainder) for w in sorted({r.witness for r in recs})}


def _cross_one(recs: Sequence[Record], held_witness: str, *, use_remainder: bool) -> dict[str, float]:
    group = lambda r: r.witness
    indices = {part: build_index(recs, part, group) for part in PARTITIONS}
    test = [r for r in recs if r.witness == held_witness]
    part_loss = {part: 0.0 for part in PARTITIONS}
    for r in test:
        for part, idx in indices.items():
            p = indexed_probability(idx, r, held_witness, use_remainder=use_remainder)
            y = bit_value(r.family, part)
            part_loss[part] -= math.log(p if y else 1-p)
    return {cls: sum(part_loss[part] for part in parts) / max(1, len(test)) for cls, parts in CLASSES.items()}


def rank(scores: Mapping[str,float], cls: str="AB") -> int:
    return 1 + sum(v < scores[cls]-1e-12 for k,v in scores.items() if k!=cls)


def family_metrics(recs: Sequence[Record]) -> dict:
    by=defaultdict(list)
    for r in recs: by[r.family].append(r)
    return {f:{"n":len(xs), "mean_position":sum(x.position for x in xs)/len(xs), "initial_rate":sum(x.initial for x in xs)/len(xs), "terminal_rate":sum(x.terminal for x in xs)/len(xs)} for f,xs in sorted(by.items()) if xs}


def same_remainder_order(recs: Sequence[Record]) -> dict[str,dict]:
    grouped=defaultdict(lambda:defaultdict(list))
    for r in recs: grouped[r.remainder][r.family].append(r.position)
    out={}
    for a,b in (("S","QO"),("S","O"),("S","D"),("QO","O"),("QO","D"),("O","D")):
        vals=[]
        for fams in grouped.values():
            if a in fams and b in fams:
                vals.append((sum(fams[a])/len(fams[a])) < (sum(fams[b])/len(fams[b])))
        out[f"{a}<{b}"]={"remainders":len(vals),"rate":sum(vals)/len(vals) if vals else None}
    return out


def align(zl: Sequence[Line], it: Sequence[Line]) -> dict:
    a={x.locus:x for x in zl}; b={x.locus:x for x in it}; common=sorted(set(a)&set(b))
    exact=sum(a[k].text_clean==b[k].text_clean for k in common)
    token_exact=sum(a[k].tokens==b[k].tokens for k in common)
    return {"zl_lines":len(zl),"it_lines":len(it),"common_loci":len(common), "exact_clean_line_rate":exact/len(common) if common else None, "exact_token_sequence_rate":token_exact/len(common) if common else None}


def q_scope(lines: Sequence[Line]) -> dict:
    q=qo=0
    for l in lines:
        for t in l.tokens:
            if t.startswith("q"):
                q+=1; qo+=t.startswith("qo")
    return {"q_initial":q,"qo_initial":qo,"qo_given_q":qo/q if q else None}


def source_report(path: Path, expected: str|None) -> dict:
    actual=sha256(path)
    return {"path":str(path),"bytes":path.stat().st_size,"sha256":actual, "expected_sha256":expected,"hash_verified":actual==expected if expected else None}


def render_md(report: dict) -> str:
    s=report["summary"]
    lines=["# Voynich × BR21 V3 dual-witness report","",f"**Disposition:** `{s['disposition']}`","", f"ZL strict paragraph lines: **{report['alignment']['zl_lines']:,}**  ", f"IT strict paragraph lines: **{report['alignment']['it_lines']:,}**  ", f"Common loci: **{report['alignment']['common_loci']:,}**  ", "", "## Factorization ranks", "", "| parser | corpus | AB rank | winner |", "|---|---:|---:|---|"]
    for mode,m in report["models"].items():
        for corpus,x in m.items():
            if not isinstance(x,dict) or "scores" not in x: continue
            winner=min(x["scores"],key=x["scores"].get)
            lines.append(f"| {mode} | {corpus} | {x['ab_rank']} | {winner} |")
    lines += ["", "## Obligations", ""] + [f"- {o}" for o in s["obligations"]]
    return "\n".join(lines)+"\n"


def run(args) -> dict:
    zl_path=Path(args.zl); it_path=Path(args.it)
    sources={"ZL3b":source_report(zl_path,args.zl_sha),"IT":source_report(it_path,args.it_sha)}
    zl=parse_ivtff(zl_path,"ZL3b",strict=True); it=parse_ivtff(it_path,"IT",strict=True)
    models={}; stable=[]
    for mode in ("minimal","strict_she","broad_q"):
        rz=records(zl,mode); ri=records(it,mode); rp=rz+ri
        corp={}
        for name,rr in (("ZL3b",rz),("IT",ri),("pooled",rp)):
            scores=cv_scores(rr,lambda r:r.folio)
            position_only=cv_scores(rr,lambda r:r.folio,use_remainder=False)
            corp[name]={"records":len(rr),"folios":len({r.folio for r in rr}),"scores":scores, "position_only_scores":position_only,"ab_rank":rank(scores), "family_metrics":family_metrics(rr),"same_remainder_order":same_remainder_order(rr)}
        corp["cross_witness"]={"scores":cross_witness(rp)}
        models[mode]=corp
        stable.append(all(corp[n]["ab_rank"]==1 for n in ("ZL3b","IT","pooled")))
    obligations=[]
    if not sources["ZL3b"]["hash_verified"]: obligations.append("ZL3b hash mismatch.")
    if args.it_sha and not sources["IT"]["hash_verified"]: obligations.append("IT hash mismatch.")
    if not all(stable): obligations.append("AB factorization is not first in every witness/parser chart.")
    if models["strict_she"]["pooled"]["family_metrics"].get("S",{}).get("n",0)<100: obligations.append("Strict-S family support remains sparse.")
    disposition="NEAR" if not obligations and all(stable) else "AMBIG" if all(stable) else "HOLD"
    report={"schema":"VBR.DUAL_WITNESS.v3","sources":sources,"alignment":align(zl,it), "q_scope":{"ZL3b":q_scope(zl),"IT":q_scope(it)},"models":models, "summary":{"disposition":disposition,"all_parser_witness_ab_first":all(stable),"obligations":obligations}}
    payload=json.dumps(report,sort_keys=True,separators=(",",":"))
    report["receipt_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    return report


def self_test() -> None:
    sample='''#=IVTFF EvaT 1.7\n<f1r> <! $I=T $L=A $H=1>\n<f1r.1,@P0> sain.qoain.oain.dain\n<f1r.2,+P0> sol.qool.ool.dol\n'''
    p=Path(".vbr_test_ivtff.txt"); p.write_text(sample)
    try:
        ls=parse_ivtff(p,"X",strict=True); rs=records(ls,"minimal")
        assert len(ls)==2 and len(rs)==8
        assert AB_SIGNATURE == (("D","O"),("D","S"))
        assert q_scope(ls)["qo_given_q"]==1.0
        scores=cv_scores(rs,lambda r:r.folio)
        assert set(scores)==set(CLASSES)
    finally: p.unlink(missing_ok=True)


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--zl"); ap.add_argument("--it"); ap.add_argument("--zl-sha"); ap.add_argument("--it-sha")
    ap.add_argument("--json",default="v3_report.json"); ap.add_argument("--markdown",default="v3_report.md")
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    if args.self_test: self_test(); print("self-test: PASS"); return 0
    if not args.zl or not args.it: ap.error("--zl and --it are required")
    report=run(args)
    Path(args.json).write_text(json.dumps(report,indent=2,sort_keys=True))
    Path(args.markdown).write_text(render_md(report))
    print(json.dumps(report["summary"],indent=2))
    return 0

if __name__=="__main__": raise SystemExit(main())
