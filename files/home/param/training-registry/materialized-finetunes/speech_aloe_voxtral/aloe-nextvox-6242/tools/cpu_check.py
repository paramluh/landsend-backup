"""CPU-only pre-launch check (no GPU, no training): import the trainer with the run env (top-level asserts: arch == next.env, env keys),
the resolved-config diff vs j23081, the VOX grouping, and il_check + the no-user-words rule on a few real v2vox micros (rank 0's step-0
item + first eval item). usage (compute node, CPU): RANK=0 WORLD_SIZE=64 LOCAL_RANK=0 LOCAL_WORLD_SIZE=8 ALOE_LNODES='0;1;2;3;4;5;6;7'
QQA_RUN=... python -I cpu_check.py"""
import os
import sys

sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-nextvox-6242/code")
import numpy as np  # noqa: E402
import train_aloe_nextvox as T  # noqa: E402

T.log("[cpu-check] trainer imported: arch", T.CFG["arch"])
n, bad = T.cfg_diff_nextvox()
M = T.M
for name, want in (("vadapter.mlp.linear1.weight", "vadapter"), ("vgain.theta", "vgain"), ("cb0_head.weight", "tower_s"), ("inp.emb.0.weight", "trunk"),
                   ("inp.aud.1.weight", "audio"), ("head.spec_out", "spec"), ("head.proj.D", "proj"), ("layers.3.layer.mlp.up_proj.weight", "trunk")):
    got = M.group_of(name)
    assert got == want, (name, got, want)
assert T.group_of("inp.aud.0.weight") is None, "inp.aud.0 must be frozen"
print("[cpu-check] groups ok; inp.aud.0 frozen", flush=True)
A = T.A
PL = A.make_plan(64, roots=A.SRC_STORES)              # node-local stores are staged by the sbatch pre-flight; read the /data source here
st = A.Stores(roots=A.SRC_STORES)
for R in (0, 1, 63):
    tr, ev = A.rank_items(PL, R, 64)
    for it, cfg in ((tr[0], T.CFG["data"]), (ev[0], T.CFG["eval_data"])):
        m = A.as_micro(st.conv(it, cfg, R, audio=False))
        T.il_check(m)
        P, nn = (int(x) for x in np.asarray(m["vin"]).reshape(2))
        ida = np.asarray(m["ids"])[0]
        print(f"[cpu-check] rank {R} item {it}: T {len(ida)} P {P} latent frames {nn} (T - P - CD = {len(ida) - P - T.D.CD}); user lane non-PAD ids "
              f"{int((ida != T.D.PAD_ID).sum())}; present_A sum {float(np.asarray(m['present'])[0].sum())}; il_check ok", flush=True)
print("[cpu-check] PASS", flush=True)
