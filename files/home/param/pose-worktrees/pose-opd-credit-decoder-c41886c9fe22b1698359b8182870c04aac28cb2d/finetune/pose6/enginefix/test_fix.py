#!/usr/bin/env python3
"""pose6 ENGINEFIX proof (CPU only): the decp_bb_off -> alpha 0 patch for the DEPLOYED monoplex engine (aloe-pose-monoplex-infer/ppm/fast_ppm.py)
and for pose-s2p/infer/ppm/fast_ppm.py.
  D  line diff: the patch removes exactly the two alpha lines of ckpt_info (local / dcp) and adds their replacements + the _link_alpha helper;
     no other line changes.
  R  every deployed model: ckpt_info(original) == ckpt_info(patched) (the whole returned dict: kind, step, dims, pose / vad / decp_bb flags,
     alpha, shapes, side; build_model and FastPPM read nothing else from it and every other line is unchanged -> same engine, same bytes);
     no decp_bb_off recorded anywhere (meta cfg / train_cfg.json / its cfg).
  S  synthetic records with decp_bb_off 1: patched -> alpha 0; the original -> the knob (the bug), both formats.
Results -> enginefix/test_fix.json."""
import ast
import difflib
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

DEP = "/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer/ppm"
S2P = "/data/home/amartyavarma/ppm/pose-s2p/infer/ppm"
FIX = "/data/home/amartyavarma/ppm/pose6/enginefix"
SCR = f"/scratch/amartyavarma/pose6-enginefix/{os.environ.get('SLURM_JOB_ID', 'x')}"
CKS = {"monoplex export-357 (genweb worker.py GENWEB_CK default)": "/data/home/amartyavarma/ppm/aloe-pose-mono-8020/exports/export-357",
       "Rosie export-18 (genweb rosie_unit.sh GENWEB_CK)": "/data/home/amartyavarma/ppm/aloe-pose-rosie/exports-local/export-18",
       "duplex export-498 (/data original; served from node /scratch copies)": "/data/home/amartyavarma/ppm/aloe-posedup-live/ck/export-498-dcp",
       "duplex export-498 (slinky-18/-10 /scratch serving copy)": "/scratch/amartyavarma/aloe-posedup-live/export-498-dcp"}
sys.path.insert(0, DEP)
for k in ("PPM_S2P", "PPM_G0COND"):
    os.environ.pop(k, None)
import torch  # noqa: E402

RES = {"host": os.uname().nodename, "t_start": time.strftime("%Y-%m-%d %H:%M:%S")}
for fn in sorted(os.listdir(DEP)):
    if fn.endswith(".py") and fn != "fast_ppm.py":
        assert hashlib.md5(open(f"{DEP}/{fn}", "rb").read()).digest() == hashlib.md5(open(f"{S2P}/{fn}", "rb").read()).digest(), fn


def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    md = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(md)
    return md


MODS = {"deployed": (load("fp_dep_orig", f"{DEP}/fast_ppm.py"), load("fp_dep_fix", f"{FIX}/deployed/fast_ppm.py")),
        "pose-s2p": (load("fp_s2p_orig", f"{S2P}/fast_ppm.py"), load("fp_s2p_fix", f"{FIX}/pose-s2p/fast_ppm.py"))}
RES["md5"] = {f"{k}/{w}": hashlib.md5(open(p, "rb").read()).hexdigest() for k, (o, f) in MODS.items() for w, p in (("orig", o.__file__), ("fix", f.__file__))}


def log(*a):
    print("[enginefix]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ D line diff
RES["D"] = {}
for k, (o, f) in MODS.items():
    a, b = open(o.__file__).read().splitlines(), open(f.__file__).read().splitlines()
    rem, add = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != "equal":
            rem += a[i1:i2]
            add += b[j1:j2]
    rem_ok = sorted(x.strip() for x in rem) == sorted(['decp_bb_alpha=IP.decp_bb_alpha_of(c, meta["step"]) if bb else None, **dims)',
                                                         *[x.strip() for x in rem if x.strip().startswith('decp_bb_alpha=float(side.get("decp_bb_alpha", 1.0))')]])
    calls = [x for x in add if x.strip().startswith("_refuse_pose6_keys(ci)")]
    helper = [x for x in add if "decp_bb_alpha=_link_alpha" not in x and x not in calls]
    top = [type(nd).__name__ + ":" + (getattr(nd, "name", None) or nd.targets[0].id) for nd in ast.parse("\n".join(helper)).body]
    RES["D"][k] = {"removed": rem, "n_added": len(add), "replacements": [x.strip() for x in add if "decp_bb_alpha=_link_alpha" in x],
                   "added_calls": [x.strip()[:40] for x in calls], "added_top_level_defs": top,
                   "pass": len(rem) == 2 and rem_ok and len([x for x in add if "decp_bb_alpha=_link_alpha" in x]) == 2 and len(calls) == 1
                   and top == ["FunctionDef:_link_alpha", "Assign:POSE6_KEYS", "FunctionDef:_refuse_pose6_keys"]}
    log(f"D {k}: {json.dumps(RES['D'][k])}")


# ------------------------------------------------------------------------------------------------ R deployed models
def recs(ci):
    """every record the engines read alpha from + whether decp_bb_off appears in any of them."""
    if ci["kind"] == "local":
        c = ci["meta"]["cfg"]
        return {"meta.cfg decp keys": {k: c[k] for k in c if "decp" in k}, "decp_bb_off_anywhere": "decp_bb_off" in c}
    sd = ci["side"]
    sc = sd.get("cfg") or {}
    return {"train_cfg.json decp keys": {k: sd[k] for k in sd if "decp" in k}, "train_cfg.json cfg decp keys": {k: sc[k] for k in sc if "decp" in k},
            "decp_bb_off_anywhere": ("decp_bb_off" in sd) or ("decp_bb_off" in sc)}


def duplex_alpha(ci):
    """aloe-posedup-live/live_ppm_pose.py ckpt_info's alpha rule (lines 198 / 215) with ppm/code/infer_ppm.decp_bb_alpha_of (lines 87-92),
    re-implemented here (importing the live module would load its whole run code path)."""
    c = ci["meta"]["cfg"] if ci["kind"] == "local" else (ci["side"].get("cfg") or ci["side"])
    a, f0, fn = float(c.get("decp_bb_alpha", 1.0)), int(c.get("decp_bb_fade_from", -1)), int(c.get("decp_bb_fade_steps", 0))
    if f0 >= 0 and fn > 0:
        a *= min(1.0, max(0.0, 1.0 - (int(ci["step"]) - f0) / fn))
    return a


RES["R"] = {}
for name, path in CKS.items():
    if not os.path.exists(path):
        RES["R"][name] = {"path": path, "skipped": f"not on this node ({RES['host']})"}
        log(f"R {name}: not on this node")
        continue
    r = {"path": path}
    for k, (o, f) in MODS.items():
        co, cf = o.ckpt_info(path), f.ckpt_info(path)
        try:
            f._refuse_pose6_keys(cf)
            gpass = True
        except RuntimeError as e:
            gpass = f"RAISED: {e}"
        r[k] = {"identical_ckpt_info": co == cf, "alpha_orig": co["decp_bb_alpha"], "alpha_fix": cf["decp_bb_alpha"], "kind": co["kind"],
                "step": co["step"], "decp_bb": co["decp_bb"], "pose6_key_guard_passes": gpass}
        ci = co
    r["records"] = recs(ci)
    r["duplex_engine_alpha_today"] = duplex_alpha(ci)
    if ci["kind"] == "dcp":
        r["train_cfg_md5"] = hashlib.md5(open(f"{path}/train_cfg.json", "rb").read()).hexdigest()
        r["metadata_md5"] = hashlib.md5(open(f"{path}/.metadata", "rb").read()).hexdigest()
    r["pass"] = (all(r[k]["identical_ckpt_info"] and r[k]["alpha_orig"] == r[k]["alpha_fix"] and r[k]["pose6_key_guard_passes"] is True for k in MODS)
                 and not r["records"]["decp_bb_off_anywhere"])
    RES["R"][name] = r
    log(f"R {name}: {json.dumps(r, default=str)}")

# ------------------------------------------------------------------------------------------------ S synthetic decp_bb_off records
import torch.distributed.checkpoint as dcp  # noqa: E402


def tiny_dcp(path, side):
    os.makedirs(path, exist_ok=True)
    t = lambda *s: torch.zeros(*s, dtype=torch.bfloat16)  # noqa: E731
    sd = {"layers.0.w": t(2, 2), "tower_s.layers.0.w": t(2, 2), "dec_s.model.layers.0.w": t(2, 2), "head.proj.G": t(64, 8),
          "tower_s.text_emb.weight": t(16, 4), "tower_p.w": t(2, 2), "decp_bb.weight": t(4, 8)}
    try:
        dcp.save(sd, checkpoint_id=path, no_dist=True)
    except TypeError:
        dcp.save(sd, checkpoint_id=path)
    json.dump(side, open(f"{path}/train_cfg.json", "w"))
    open(f"{path}/COMPLETE", "w").write("1")


def tiny_local(path, cfg):
    os.makedirs(path, exist_ok=True)
    torch.save({"decp_bb.weight": torch.zeros(4, 8), "tower_p.w": torch.zeros(2, 2)}, f"{path}/rank0.pt")
    torch.save({"dec_s.w": torch.zeros(2)}, f"{path}/rep.pt")
    json.dump({"format": "ppm-local-bf16-shards-v1", "step": 5, "world": 1, "cfg": cfg}, open(f"{path}/meta.json", "w"))
    open(f"{path}/COMPLETE", "w").write("5")


dims = {"layers": 1, "tower_layers": 1, "dec_layers": 1, "proj_r": 64, "tower_vocab": 16}
CASES = {"dcp 31101 side (alpha 0 + off 1)": ("dcp", {"step": 1, "decp_bb": 1, "decp_bb_alpha": 0.0, "decp_bb_off": 1}, 0.0, 0.0),
         "dcp side alpha 1 + off 1 (FT export side today + off)": ("dcp", {"step": 1, "decp_bb": 1, "decp_bb_alpha": 1.0, "decp_bb_off": 1}, 1.0, 0.0),
         "dcp side alpha 1, no off (= export-498 / today)": ("dcp", {"step": 1, "decp_bb": 1, "decp_bb_alpha": 1.0}, 1.0, 1.0),
         "dcp side alpha 1 + cfg {decp_bb_off 1} (FT export: flag only in the nested training cfg)":
             ("dcp", {"step": 1, "decp_bb": 1, "decp_bb_alpha": 1.0, "cfg": {"decp_bb_alpha": 1.0, "decp_bb_off": 1}}, 1.0, 0.0),
         "local cfg alpha 1 + off 1 (local copy of a 31101 save / FT local export, link off)": ("local", dict(dims, decp_bb_alpha=1.0, decp_bb_off=1), 1.0, 0.0),
         "local cfg alpha 1, no off (= export-357 / Rosie)": ("local", dict(dims, decp_bb_alpha=1.0), 1.0, 1.0)}
RES["S"] = {}
for i, (name, (fmt, rec, want_o, want_f)) in enumerate(CASES.items()):
    p = f"{SCR}/{i}"
    (tiny_dcp if fmt == "dcp" else tiny_local)(p, rec)
    r = {}
    for k, (o, f) in MODS.items():
        r[k] = {"alpha_orig": o.ckpt_info(p)["decp_bb_alpha"], "alpha_fix": f.ckpt_info(p)["decp_bb_alpha"]}
    r["pass"] = all(r[k]["alpha_orig"] == want_o and r[k]["alpha_fix"] == want_f for k in MODS)
    r["expected (orig, fix)"] = [want_o, want_f]
    RES["S"][name] = r
    log(f"S {name}: {json.dumps(r)}")
# G  synthetic checkpoints carrying the pose6 keys: the patched build_model raises (naming the pose6 engine) before building anything;
#    the original has no guard (its ckpt_info succeeds and build_model would silently drop the keys)
def tiny_dcp_keys(path, extra):
    os.makedirs(path, exist_ok=True)
    t = lambda *s: torch.zeros(*s, dtype=torch.bfloat16)  # noqa: E731
    sd = {"layers.0.w": t(2, 2), "tower_s.layers.0.w": t(2, 2), "dec_s.model.layers.0.w": t(2, 2), "head.proj.G": t(64, 8),
          "tower_s.text_emb.weight": t(16, 4), "tower_p.w": t(2, 2), "decp_bb.weight": t(4, 8), **{k: t(*v) for k, v in extra.items()}}
    try:
        dcp.save(sd, checkpoint_id=path, no_dist=True)
    except TypeError:
        dcp.save(sd, checkpoint_id=path)
    json.dump({"step": 1, "decp_bb": 1, "decp_bb_alpha": 0.0, "decp_bb_off": 1}, open(f"{path}/train_cfg.json", "w"))
    open(f"{path}/COMPLETE", "w").write("1")


def tiny_local_keys(path, extra):
    os.makedirs(path, exist_ok=True)
    torch.save({"decp_bb.weight": torch.zeros(4, 8), "tower_p.w": torch.zeros(2, 2), **{k: torch.zeros(*v) for k, v in extra.items()}}, f"{path}/rank0.pt")
    torch.save({"dec_s.w": torch.zeros(2)}, f"{path}/rep.pt")
    json.dump({"format": "ppm-local-bf16-shards-v1", "step": 5, "world": 1, "cfg": dims}, open(f"{path}/meta.json", "w"))
    open(f"{path}/COMPLETE", "w").write("5")


GK = {"inp.g0 only": {"inp.g0.weight": (5120, 8)}, "s_emb only": {"tower_p.s_emb.weight": (2048, 2560)},
      "both": {"inp.g0.weight": (5120, 8), "tower_p.s_emb.weight": (2048, 2560)}}
RES["G"] = {}
for j, (gname, extra) in enumerate(GK.items()):
    for fmt in ("dcp", "local"):
        p_ = f"{SCR}/g{j}{fmt}"
        (tiny_dcp_keys if fmt == "dcp" else tiny_local_keys)(p_, extra)
        r = {}
        for k, (o_, f_) in MODS.items():
            co = o_.ckpt_info(p_)
            want_raise = (k == "deployed") or ("inp.g0.weight" in extra)    # pose-s2p serves s_emb itself
            try:
                if want_raise:
                    f_.build_model(p_, [torch.device("cpu")])              # raises at its first line after ckpt_info (nothing built)
                else:
                    f_._refuse_pose6_keys(f_.ckpt_info(p_))                 # (no tiny full-width build: the guard alone)
                got = "no exception (guard passes)"
            except RuntimeError as e:
                got = str(e)
            r[k] = {"orig_ckpt_info_ok (no guard)": co is not None and not hasattr(o_, "_refuse_pose6_keys"), "patched_build_model": got[:200],
                    "pass": (("pose6 engine ppm/pose6/infer/ppm/fast_ppm.py" in got) == want_raise)}
        r["pass"] = all(v["pass"] for v in r.values())
        RES["G"][f"{gname}/{fmt}"] = r
        log(f"G {gname}/{fmt}: {json.dumps(r)}")
RES["pass"] = (all(v["pass"] for v in RES["D"].values()) and all(v["pass"] for v in RES["G"].values()) and all(v.get("pass", True) for v in RES["R"].values())
               and all(v["pass"] for v in RES["S"].values()))
RES["t_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(RES, open(f"{FIX}/test_fix.json", "w"), indent=1, default=str)
log(f"pass {RES['pass']}")
