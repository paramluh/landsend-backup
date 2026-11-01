#!/usr/bin/env python3
"""pose6 kit (B3, 10-27): verify a FINAL consolidated FT export (the trainer's export_weights: bf16 DCP + train_cfg.json + COMPLETE), CPU only,
read only. Exit 0 = every check passes, 1 = any fails (each printed).
  1. COMPLETE present and == the expected step (default 357); .metadata present; train_cfg.json step == COMPLETE.
  2. key set + full shapes == the FT model's table (kit/ref/model_keys_export357.json = job 24272's export-357, 2,150 keys) EXACTLY, + the
     --mods extra keys (s2pg0: tower_p.s_emb.weight / inp.g0.weight); every tensor bf16.
  3. every value finite (all tensors read in ~6 GB batches; counts non-finite per key).
  4. train_cfg.json cfg: init / run_tag / export path printed; --expect-init / --expect-cfg k=v checks (e.g. decp_bb_off absent: decp_bb_off=<absent>).
usage: verify_export.py <export dir> [--step 357] [--mods none|s2pg0] [--expect-init PATH] [--expect-cfg key=value ...]"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed.checkpoint as dcp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preflight as PF  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--step", default="357")
ap.add_argument("--mods", default="none", choices=("none", "s2pg0"))
ap.add_argument("--expect-init", default="")
ap.add_argument("--expect-cfg", nargs="*", default=[])
ap.add_argument("--batch-gb", type=float, default=6.0)
a = ap.parse_args()
t0 = time.time()
P = a.path.rstrip("/")
fails = []


def chk(cond, msg):
    print(("  ok: " if cond else "  FAIL: ") + msg, flush=True)
    if not cond:
        fails.append(msg)


print(f"== verify_export {P} (mods {a.mods}, step {a.step})", flush=True)
chk(os.path.isfile(f"{P}/COMPLETE") and os.path.isfile(f"{P}/.metadata"), "COMPLETE + .metadata present (consolidated DCP)")
if fails:
    sys.exit(1)
st = open(f"{P}/COMPLETE").read().strip()
chk(st == a.step, f"COMPLETE = {st} (want {a.step})")
tc = json.load(open(f"{P}/train_cfg.json")) if os.path.isfile(f"{P}/train_cfg.json") else {}
chk(bool(tc), "train_cfg.json present")
cfg = tc.get("cfg", {})
chk(str(tc.get("step")) == st, f"train_cfg.json step {tc.get('step')} == COMPLETE")
print(f"     run {tc.get('run')} | init {cfg.get('init')} | export {cfg.get('export')} | s2p {cfg.get('s2p', '<absent>')} | g0cond {cfg.get('g0cond', '<absent>')} "
      f"| decp_bb_off {cfg.get('decp_bb_off', '<absent>')} | export_steps {cfg.get('export_steps')} | save_every_min {cfg.get('save_every_min')}", flush=True)
if a.expect_init:
    chk(cfg.get("init") == a.expect_init, f"cfg init {cfg.get('init')} == {a.expect_init}")
for kv in a.expect_cfg:
    k, v = kv.split("=", 1)
    got = cfg.get(k, "<absent>")
    chk(str(got) == v, f"cfg {k} = {got!r} (want {v})")
# ---- 2. keys / shapes / dtypes
ref = json.load(open(PF.REF))["shapes"]
want = dict(ref)
if a.mods == "s2pg0":
    h_tower = ref["tower_p.norm.weight"][0] if "tower_p.norm.weight" in ref else None
    h_trunk = [v[0] for k, v in sorted(ref.items()) if k.startswith("layers.0.") and k.endswith("input_layernorm.weight")][0]
    want["tower_p.s_emb.weight"] = [2048, h_tower]
    want["inp.g0.weight"] = [h_trunk, 8]
tab, dt, rep, info = PF.table_dcp(P)
miss, extra = sorted(set(want) - set(tab)), sorted(set(tab) - set(want))
shp = sorted(k for k in set(want) & set(tab) if list(want[k]) != list(tab[k]))
nbf = sorted(k for k, v in dt.items() if v != "torch.bfloat16")
chk(not miss and not extra and not shp, f"key set / shapes == the FT model's ({len(tab)} keys vs {len(want)}; missing {miss[:4]}, extra {extra[:4]}, shape {shp[:4]})")
chk(not nbf, f"all bf16 (non-bf16: {nbf[:4]})")
# ---- 3. all finite
md = dcp.FileSystemReader(P).read_metadata().state_dict_metadata
keys, bad, nb, batch, bb = sorted(md), {}, 0, [], 0


def flush():
    global batch, bb, nb
    if not batch:
        return
    sd = {k: torch.empty(tuple(md[k].size), dtype=md[k].properties.dtype) for k in batch}
    dcp.load(sd, checkpoint_id=P, no_dist=True)
    for k, t in sd.items():
        n = int((~torch.isfinite(t.float())).sum()) if t.numel() else 0
        if n:
            bad[k] = n
        nb += t.numel() * t.element_size()
    batch, bb = [], 0


for k in keys:
    batch.append(k)
    bb += math.prod(md[k].size) * 2
    if bb >= a.batch_gb * 2 ** 30:
        flush()
flush()
chk(not bad, f"all {len(keys)} tensors finite ({nb / 1e9:.1f} GB read; non-finite: {dict(list(bad.items())[:4])})")
print(f"== RESULT {'PASS' if not fails else 'FAIL'} ({len(fails)} failed) in {time.time() - t0:.0f}s", flush=True)
sys.exit(0 if not fails else 1)
