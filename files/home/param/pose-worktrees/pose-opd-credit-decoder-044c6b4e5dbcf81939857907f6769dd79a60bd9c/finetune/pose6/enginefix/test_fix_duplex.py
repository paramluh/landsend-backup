#!/usr/bin/env python3
"""pose6 ENGINEFIX proof for the DUPLEX engine aloe-posedup-live/live_ppm_pose.py (CPU only; its own process: it imports the run's code path).
  D  line diff: exactly its two ckpt_info alpha lines (198 local / 215 dcp) replaced + the _link_alpha helper added; nothing else.
  R  every model this engine is configured to load (SWAP_TO.txt / LOADED.json of every instance, demo.sbatch / worker_dup.py / stage-lb CK,
     CK_LABELS.json): ckpt_info(original) == ckpt_info(patched) and no decp_bb_off recorded (paths absent on this node are listed as such).
  S  synthetic records (argv 'S'; node /scratch only): decp_bb_off 1 (top level, local cfg, nested side cfg) -> patched alpha 0, original = the knob.
usage: test_fix_duplex.py R[,S,D]  -> enginefix/duplex/test_fix_duplex_<host>.json"""
import difflib
import hashlib
import importlib.util
import json
import os
import sys
import time

LIVE = "/data/home/amartyavarma/ppm/aloe-posedup-live"
FIX = "/data/home/amartyavarma/ppm/pose6/enginefix"
ST = (sys.argv[1] if len(sys.argv) > 1 else "R,S,D").split(",")
SCR = f"/scratch/amartyavarma/pose6-enginefix/{os.environ.get('SLURM_JOB_ID', 'x')}-duplex"
sys.path.insert(0, LIVE)
import torch  # noqa: E402


def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    md = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(md)
    return md


O = load("lpp_orig", f"{LIVE}/live_ppm_pose.py")


def load_as_live(name, src_path):
    """the PATCHED source executed as if it sat at the live path (module-level reads such as DEFAULTS.json resolve next to the live file)."""
    import types
    md = types.ModuleType(name)
    md.__file__ = f"{LIVE}/live_ppm_pose.py"
    sys.modules[name] = md
    exec(compile(open(src_path).read(), src_path, "exec"), md.__dict__)
    md.__src__ = src_path
    return md


F = load_as_live("lpp_fix", f"{FIX}/duplex/live_ppm_pose.py")
RES = {"host": os.uname().nodename, "t_start": time.strftime("%Y-%m-%d %H:%M:%S"),
       "md5": {"orig": hashlib.md5(open(O.__file__, "rb").read()).hexdigest(), "fix": hashlib.md5(open(F.__src__, "rb").read()).hexdigest()}}


def log(*a):
    print("[enginefix-duplex]", *a, flush=True)


if "D" in ST:
    a, b = open(O.__file__).read().splitlines(), open(F.__src__).read().splitlines()
    rem, add = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != "equal":
            rem += a[i1:i2]; add += b[j1:j2]
    import ast
    rep = [x for x in add if "decp_bb_alpha=" in x]
    calls = [x for x in add if x.strip().startswith("_refuse_pose6_keys(ci)")]
    top = [type(nd).__name__ + ":" + (getattr(nd, "name", None) or nd.targets[0].id)
           for nd in ast.parse("\n".join(x for x in add if "decp_bb_alpha=" not in x and x not in calls)).body]
    RES["D"] = {"removed": rem, "replacements": rep, "n_added": len(add), "added_calls": [x.strip()[:40] for x in calls], "added_top_level_defs": top,
                "pass": len(rem) == 2 and all("IP.decp_bb_alpha_of(" in x for x in rem) and len(rep) == 2 and all("_link_alpha(path, " in x for x in rep)
                and len(calls) == 1 and top == ["FunctionDef:_link_alpha", "Assign:POSE6_KEYS", "FunctionDef:_refuse_pose6_keys"]}
    log(f"D {json.dumps(RES['D'])}")


def configured():
    """every checkpoint path the duplex engine is configured to load."""
    out = {}
    for root, dirs, files in os.walk(LIVE):
        if root.count(os.sep) - LIVE.count(os.sep) >= 2:              # instance state dirs sit at depth <= 2 (stage-lb/inst*)
            dirs[:] = []
        dirs[:] = [d for d in dirs if d not in ("runs", "render", "logs", "out", "openings", "openings-sam", "people_extra") and ".bak" not in d
                   and "pagebackup" not in d]
        for fn in files:
            if fn == "SWAP_TO.txt":
                for ln in open(os.path.join(root, fn)):
                    if ln.strip() and not ln.startswith("#"):
                        out.setdefault(ln.strip(), []).append(os.path.relpath(os.path.join(root, fn), LIVE))
            elif fn == "LOADED.json":
                try:
                    out.setdefault(json.load(open(os.path.join(root, fn)))["ck"], []).append(os.path.relpath(os.path.join(root, fn), LIVE))
                except (ValueError, KeyError, OSError):
                    pass
    for k in json.load(open(f"{LIVE}/CK_LABELS.json")):
        out.setdefault(k, []).append("CK_LABELS.json")
    for p_, src in (("/scratch/amartyavarma/aloe-posedup-live/export-498-dcp", "demo.sbatch CK default / genweb worker_dup.py GENWEB_DUP_CK default"),
                    (f"{LIVE}/ck/export-498-dcp", "stage-lb/switch_1027/restore.sh posedup-lb-P CK"),
                    ("/scratch/amartyavarma/pdsamples/export-498", "posefix/exact sbatches")):
        out.setdefault(p_, []).append(src)
    return out


def recs(ci):
    if ci["kind"] == "local":
        c = ci["meta"]["cfg"]
        return {"meta.cfg decp keys": {k: c[k] for k in c if "decp" in k}, "decp_bb_off_anywhere": "decp_bb_off" in c}
    sd = ci["side"]
    sc = sd.get("cfg") or {}
    return {"side decp keys": {k: sd[k] for k in sd if "decp" in k}, "side.cfg decp keys": {k: sc[k] for k in sc if "decp" in k},
            "decp_bb_off_anywhere": ("decp_bb_off" in sd) or ("decp_bb_off" in sc)}


if "R" in ST:
    RES["R"] = {}
    for path, srcs in sorted(configured().items()):
        if not os.path.isdir(path) or not (os.path.isfile(f"{path}/COMPLETE")):
            RES["R"][path] = {"sources": srcs, "skipped": f"not a COMPLETE checkpoint on {RES['host']}" + ("" if os.path.isdir(path) else " (absent)")}
            log(f"R {path}: skipped ({RES['R'][path]['skipped']})")
            continue
        co, cf = O.ckpt_info(path), F.ckpt_info(path)
        try:
            F._refuse_pose6_keys(cf)
            gpass = True
        except RuntimeError as e:
            gpass = f"RAISED: {e}"
        r = {"sources": srcs, "kind": co["kind"], "step": co["step"], "identical_ckpt_info": co == cf, "alpha_orig": co["decp_bb_alpha"],
             "alpha_fix": cf["decp_bb_alpha"], "pose6_key_guard_passes": gpass, **recs(co)}
        if co["kind"] == "dcp":
            r["train_cfg_md5"] = hashlib.md5(open(f"{path}/train_cfg.json", "rb").read()).hexdigest() if os.path.isfile(f"{path}/train_cfg.json") else None
        r["pass"] = r["identical_ckpt_info"] and r["alpha_orig"] == r["alpha_fix"] and not r["decp_bb_off_anywhere"] and r["pose6_key_guard_passes"] is True
        RES["R"][path] = r
        log(f"R {path}: {json.dumps(r, default=str)}")

if "S" in ST:
    import torch.distributed.checkpoint as dcp

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
    CASES = {"dcp 31101 side (alpha 0 + off 1, no cfg)": ("dcp", {"step": 1, "decp_bb": 1, "decp_bb_alpha": 0.0, "decp_bb_off": 1}, 0.0, 0.0),
             "dcp side + cfg {alpha 1, off 1} (FT export link off)": ("dcp", {"step": 1, "decp_bb": 1, "decp_bb_alpha": 1.0,
                                                                              "cfg": {"decp_bb_alpha": 1.0, "decp_bb_off": 1}}, 1.0, 0.0),
             "dcp side alpha 1 + off 1, cfg {alpha 1} (off only at top level)": ("dcp", {"step": 1, "decp_bb": 1, "decp_bb_alpha": 1.0, "decp_bb_off": 1,
                                                                                          "cfg": {"decp_bb_alpha": 1.0}}, 1.0, 0.0),
             "dcp side + cfg {alpha 1} (= export-498)": ("dcp", {"step": 1, "decp_bb": 1, "decp_bb_alpha": 1.0, "cfg": {"decp_bb_alpha": 1.0}}, 1.0, 1.0),
             "local cfg alpha 1 + off 1": ("local", dict(dims, decp_bb_alpha=1.0, decp_bb_off=1), 1.0, 0.0),
             "local cfg alpha 1 (no off)": ("local", dict(dims, decp_bb_alpha=1.0), 1.0, 1.0)}
    RES["G"] = {}
    for j, (gname, extra) in enumerate({"inp.g0 only": {"inp.g0.weight": (5120, 8)}, "s_emb only": {"tower_p.s_emb.weight": (2048, 2560)}}.items()):
        for fmt in ("dcp", "local"):
            p_ = f"{SCR}/g{j}{fmt}"
            if fmt == "dcp":
                os.makedirs(p_, exist_ok=True)
                t = lambda *s_: torch.zeros(*s_, dtype=torch.bfloat16)  # noqa: E731
                sd = {"layers.0.w": t(2, 2), "tower_s.layers.0.w": t(2, 2), "dec_s.model.layers.0.w": t(2, 2), "head.proj.G": t(64, 8),
                      "tower_s.text_emb.weight": t(16, 4), "tower_p.w": t(2, 2), "decp_bb.weight": t(4, 8), **{k: t(*v) for k, v in extra.items()}}
                try:
                    dcp.save(sd, checkpoint_id=p_, no_dist=True)
                except TypeError:
                    dcp.save(sd, checkpoint_id=p_)
                json.dump({"step": 1, "decp_bb": 1, "decp_bb_alpha": 0.0, "decp_bb_off": 1}, open(f"{p_}/train_cfg.json", "w"))
                open(f"{p_}/COMPLETE", "w").write("1")
            else:
                os.makedirs(p_, exist_ok=True)
                torch.save({"decp_bb.weight": torch.zeros(4, 8), "tower_p.w": torch.zeros(2, 2), **{k: torch.zeros(*v) for k, v in extra.items()}},
                           f"{p_}/rank0.pt")
                torch.save({"dec_s.w": torch.zeros(2)}, f"{p_}/rep.pt")
                json.dump({"format": "ppm-local-bf16-shards-v1", "step": 5, "world": 1, "cfg": dims}, open(f"{p_}/meta.json", "w"))
                open(f"{p_}/COMPLETE", "w").write("5")
            co = O.ckpt_info(p_)
            try:
                F.build_model(p_, [torch.device("cpu")])
                got = "no exception"
            except RuntimeError as e:
                got = str(e)
            except Exception as e:
                got = f"guard passed ({type(e).__name__} later)"
            RES["G"][f"{gname}/{fmt}"] = {"orig_ckpt_info_ok (no guard)": co is not None and not hasattr(O, "_refuse_pose6_keys"),
                                          "patched_build_model": got[:200], "pass": "pose6 engine ppm/pose6/infer/ppm/fast_ppm.py" in got}
            log(f"G {gname}/{fmt}: {json.dumps(RES['G'][f'{gname}/{fmt}'])}")
    RES["S"] = {}
    for i, (name, (fmt, rec, wo, wf)) in enumerate(CASES.items()):
        p_ = f"{SCR}/{i}"
        (tiny_dcp if fmt == "dcp" else tiny_local)(p_, rec)
        ao, af = O.ckpt_info(p_)["decp_bb_alpha"], F.ckpt_info(p_)["decp_bb_alpha"]
        RES["S"][name] = {"alpha_orig": ao, "alpha_fix": af, "expected (orig, fix)": [wo, wf], "pass": ao == wo and af == wf}
        log(f"S {name}: {json.dumps(RES['S'][name])}")
RES["pass"] = all(RES.get("D", {"pass": True})["pass"] for _ in [0]) and all(v.get("pass", True) for v in RES.get("R", {}).values()) and \
    all(v["pass"] for v in RES.get("S", {}).values()) and all(v["pass"] for v in RES.get("G", {}).values())
RES["t_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(RES, open(f"{FIX}/duplex/test_fix_duplex_{RES['host']}.json", "w"), indent=1, default=str)
log(f"pass {RES['pass']}")
