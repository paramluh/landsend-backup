#!/usr/bin/env python3
"""pose5 D, audit S9: the trainer's START EVAL (train_ppm.main's evaluate(step)) with the new flags, on CPU, tiny config, real eval micros.

evaluate() is a closure inside main(); this test takes its EXACT source from the code under test (ast of train_ppm.main -> the nested
'def evaluate'), and runs it with the trainer's own globals plus main()'s locals rebuilt here the way main() builds them:
  ev / evw / ebase / epool  every eval family main() builds: pose (pose_micro_ppm with main's pcfg: + g0pos / g0f under G0COND), speech,
                            speech_A, gemma, chat, speech_s3_A/B (evalsets/speech_s3 pickles), gemma_s3_A/B (gemma-txtdup-v1 eval packs,
                            main's lane rule); node-local stores replaced by their Weka copies of the same format (pose: plan-1 fresh rows,
                            FreshRows = PoseRows' dict; speech / speech_A: speech_s3 lanes B / A; gemma: gemma-txtdup-v1 eval; chat: split 2)
  cdp                       codep_s3q.build on those pose rows + codep_micro, then (G0COND) train_ppm.codep_g0 = main()'s exact calls
  xdp                       xdep_micro on the speech_s3 rows (silent / absent) + the silence codes, main()'s dict
Variants (env as the trainer reads it): ORIG off | COPY off | COPY G0 + S2P cb0 | COPY G0 + S2P full, every new tensor at its zero init.
PASS: evaluate() returns (no assert anywhere, every value finite) and every metric (time keys aside) equals ORIG's bit for bit.
usage: test_eval.py --code ORIG|COPY --g0 0|1 --s2p none|cb0|full --out <json> [--base <orig json>]"""
import argparse
import ast
import json
import os
import pickle
import sys
import textwrap
import time
import zlib

import numpy as np
import torch

sys.path.insert(0, "/data/home/amartyavarma/ppm/pose5/code-811-g0/tests/g0")
import g0_cpu as H  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--g0", default="0")
ap.add_argument("--s2p", default="none")
ap.add_argument("--out", required=True)
ap.add_argument("--base", default="")
ap.add_argument("--port", type=int, default=29880)
ap.add_argument("--n", type=int, default=2)
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
code = {"ORIG": H.ORIG, "COPY": H.COPY}[a.code]
os.environ["PPM_G0COND"] = a.g0
if a.s2p != "none":
    os.environ["PPM_S2P"] = a.s2p
else:
    os.environ.pop("PPM_S2P", None)
os.environ["PPM_XDEP_ROWS"] = str(a.n)
sys.path.insert(0, code)
import train_ppm as T  # noqa: E402

M, D, CH, CDP = T.M, T.D, T.CH, T.CDP
assert os.path.dirname(os.path.abspath(T.__file__)) == code
G0, S2P = bool(getattr(T, "G0", False)), getattr(T, "S2P", None)
assert G0 == (a.g0 == "1") and S2P == (None if a.s2p == "none" else a.s2p)
H.cpu_patches()
torch.cuda.synchronize = lambda *x, **k: None
dist = H.init_dist(a.port)
dev = torch.device("cpu")
CFG = T.CFG
dcfg = CFG["data"]
pcfg = dict(dcfg, g0cond=1) if G0 else dcfg                         # = main()
E_ = a.n

# ---- model (tiny, seeded per name: identical shared tensors in every variant; new tensors zero)
c32, c4, c06 = H.tiny_cfgs(M)
kw = ({"g0cond": True} if G0 else {}) | ({"s2p": S2P} if S2P else {})
model, _ = M.build(c32, c4, c06, dev, H.log, partial=True, proj_r=64, tower_vocab=M.TOWER_VOCAB, init=False, dec_ac=False,
                   dec_mode="replicate", pose_head=False, decp_bb=True, **kw)
H.seeded_init(M, model, c4, c06)
model.decp_bb_on = not T.BB_OFF
rot = M.Rot(c32, dev)
dos = {"dec_s": T.DecOpt(model.dec_s, "dec_s"), "dec_p": T.DecOpt(model.dec_p, "dec_p")}

# ---- eval families (main()'s structure; Weka copies of the node-local stores)
FP = T._fresh_pose_module()
import pyarrow.parquet as pq  # noqa: E402

SRC = "/data/home/amartyavarma/data/new-pose-llm-data/new-pose-llm-data/jobs"
t = pq.ParquetFile(f"{H.ORIG}/fresh/scan/slinky-0.parquet").read_row_group(0, columns=["file", "row", "segment_id", "video_id", "parent_clip_id", "segment_index", "n", "sam_ok"]).to_pydict()
_ex, k = {}, []
for i in range(len(t["file"])):
    if t["file"][i] not in _ex:
        _ex[t["file"][i]] = os.path.exists(f"{SRC}/{t['file'][i]}")
    if t["sam_ok"][i] and _ex[t["file"][i]]:
        k.append(i)
        if len(k) == 400:
            break
files = sorted({t["file"][i] for i in k}); fidx = {f: j for j, f in enumerate(files)}
clips = [{"video_id": t["video_id"][i], "parent_clip_id": t["parent_clip_id"][i], "segment_index": int(t["segment_index"][i]),
          "segment_id": t["segment_id"][i], "n": int(t["n"][i])} for i in k]
rows = FP.B.pack(FP.B.units_of(clips), np.random.default_rng(np.random.SeedSequence([FP.PACK_SEED, zlib.crc32(b"slinky-0"), 0])))
z = {"seg_file": np.array([fidx[t["file"][i]] for i in k], np.int32), "seg_row": np.array([t["row"][i] for i in k], np.int32),
     "seg_n": np.array([t["n"][i] for i in k], np.int32), "row_seg": np.array([j for r in rows for j in r], np.int32),
     "row_off": np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64),
     "row_len": np.array([sum(int(t["n"][k[j]]) + FP.B.SLOT + FP.B.GAP for j in r) for r in rows], np.int64), "row_shard": np.zeros(len(rows), np.int32)}
fr = FP.FreshRows({"files": files, "segment_ids": [t["segment_id"][i] for i in k], "src_root": SRC, "host": "slinky-0", "n_rows": len(rows), "z": z}, cache_files=8)
prow = [fr[i] for i in (6, 8)][:E_]                                   # two short-clip-count rows (10 / 16 clips)
S3 = CFG["eval_s3_dir"]
ld = lambda o, i: pickle.load(open(f"{S3}/{o}-{i:03d}.pkl", "rb"))  # noqa: E731
gst = D.TXD.Store(CFG["eval_gemma_s3_root"]); gcfg = dict(dcfg, txt_cap=int(gst.cap))
cp = CH.multi_split_packs([2], dcfg["txt_cap"], CFG["chat_seed"])
ev = {"pose": [D.packed(D.pose_micro_ppm(r, pcfg, 0), pcfg) for r in prow],
      "speech": [ld("B", i) for i in range(E_)],
      "gemma": [D.TextDS(CFG["eval_gemma_s3_root"], "eval", list(range(E_)), dcfg)[i] for i in range(E_)],
      "chat": [CH.ChatDS([cp[i]], dcfg)[0] for i in range(E_)],
      "speech_A": [ld("A", i) for i in range(E_)]}
evw = {kk: [1.0] * len(v) for kk, v in ev.items()}
ebase = {"pose": "pose", "speech": "speech", "gemma": "gemma", "chat": "chat", "speech_A": "speech"}
epool = {}
for o in ("A", "B"):
    ev[f"speech_s3_{o}"] = [ld(o, i) for i in range(E_)]; evw[f"speech_s3_{o}"] = [1.0] * E_; ebase[f"speech_s3_{o}"] = "speech"
epool["speech_s3"] = ("speech_s3_A", "speech_s3_B")
for o, fl in (("A", 1), ("B", 0)):
    mic = []
    for p_ in range(E_):
        ids_ = gst.pack("eval", p_); recs_ = [gst.get(j) for j in ids_]
        mic.append(D.packed(D.text_micro(recs_, gcfg, sws=[((int(gst.eval_ptr[p_]) + kk) % 2) ^ fl for kk in range(len(recs_))]), gcfg))
    ev[f"gemma_s3_{o}"] = mic; evw[f"gemma_s3_{o}"] = [1.0] * E_; ebase[f"gemma_s3_{o}"] = "gemma"
epool["gemma_s3"] = ("gemma_s3_A", "gemma_s3_B")
cdp = None
if CFG["codep"]:                                                    # = main(), on the pose eval rows
    pdc_ = {"new_word": dcfg["new_word"], "pose_delay": dcfg["pose_delay"], "light_seed": dcfg["light_seed"]}
    cdp = CDP.build(prow, pdc_, 0, CFG["codep_frames"], CFG["codep_seed"], CFG["codep_light_frac"], CFG["codep_light_frac"])
    cdp["micros"] = [T.codep_micro(cdp, v, dcfg) for v in CDP.VARIANTS]
    if G0:
        T.codep_g0(cdp, prow)
xdp = None
if CFG["xdep"] and CFG["eval_s3_dir"]:                              # = main() (world 1: every item on this rank)
    nx_ = E_
    sil_ = np.asarray(json.load(open(T.SIL_CODES))["seq"], np.int64)
    items_ = [(c_, o, i) for c_ in CFG["xdep_conds"] for o in ("A", "B") for i in range(nx_)]
    xdp = {"n": nx_, "conds": list(CFG["xdep_conds"]), "w": [1.0] * len(items_), "cond": [it[0] for it in items_],
           "mic": [T.xdep_micro(ld(it[1], it[2]), it[0], sil_) for it in items_],
           "sub": {f"speech_s3_{o}": [1.0] * E_ for o in ("A", "B")}}
H.log(f"eval families {[(kk, len(v)) for kk, v in ev.items()]}; codep {len(cdp['micros']) if cdp else 0} micros; xdep {len(xdp['mic']) if xdp else 0}; "
      f"pose micros carry g0: {['g0pos' in m for m in ev['pose']]}; codep g0: {['g0pos' in m for m in cdp['micros']] if cdp else None}")

# ---- the trainer's evaluate(), source taken from main()
src = open(T.__file__).read()
mainf = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main")
evf = next(n for n in ast.walk(mainf) if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
seg = textwrap.dedent(ast.get_source_segment(src, evf))
ns = dict(vars(T))
ns.update(model=model, ev=ev, evw=evw, ebase=ebase, epool=epool, cdp=cdp, xdp=xdp, dev=dev, rot=rot, dos=dos, emit=lambda d, s: None)
exec(compile(seg, f"{T.__file__}:main.evaluate", "exec"), ns)
t0 = time.time()
d = ns["evaluate"](17850)
H.log(f"evaluate(17850) returned {len(d)} metrics in {time.time() - t0:.0f}s")
res = {"code": code, "g0": a.g0, "s2p": a.s2p, "metrics": {kk: float(v).hex() for kk, v in d.items()}, "metrics_float": d,
       "evaluate_source_sha": __import__("hashlib").sha256(seg.encode()).hexdigest()[:16], "PASS": True}
if a.base:
    b = json.load(open(a.base))["metrics"]
    keys = sorted(kk for kk in b if "time_s" not in kk)
    diff = [kk for kk in keys if res["metrics"].get(kk) != b[kk]]
    res.update(compared=len(keys), differ=diff, missing=[kk for kk in keys if kk not in res["metrics"]],
               extra=[kk for kk in res["metrics"] if kk not in b])
    res["PASS"] = not diff and not res["missing"] and not res["extra"]
    H.log(f"vs base {a.base}: {len(keys)} metrics compared, {len(diff)} differ {diff[:8]}; extra {res['extra'][:6]}")
json.dump(res, open(a.out, "w"), indent=0, default=str)
H.log(("PASS" if res["PASS"] else "FAIL"), "->", a.out)
dist.destroy_process_group()
