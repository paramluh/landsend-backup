"""aloe-ft-32-samsep CPU dry check (no GPU, no launch): the trainer module's resolved CFG built from run.env exactly as rank 0 of the world-14
job builds it, its own startup diff vs 26232 (abort rule) + the report-only diff vs 26467, and the plan / layout / micro facts the run asserts
at start (plan digest, counts, steps, slots / dummies, A FLAC presence of every item, a few micros + the [same-as-v2] row compare).
usage (compute node, env sourced from run.env): python -I cfg_diff_dry.py <code dir> <out dir>"""
import hashlib
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
OUT = sys.argv[2]
os.environ.update(RANK="0", WORLD_SIZE="14", LOCAL_RANK="0", LOCAL_WORLD_SIZE="8", COTRAIN_CFG_DRY="1")
os.environ["QQA_RUN"] = os.environ["QQA_RUN"] + "-jDRY"
os.environ["ALOE_FLAC_ROOT"] = os.environ["ALOE_SHARED_FLAC"]          # dry node: the shared copy the per-host symlink points to
import train_samsep as T  # noqa: E402

T.LOGDIR = OUT
A = T.A
print("PROD", T.PROD, "ALOE_ONLY", T.ALOE_ONLY, "aloe ranks", T.NA, "slots", T.GMAX, "rank hosts", T.RANK_HOST, "offload", T.OFFLOAD)
n, bad = T.cfg_diff_26232()
rows = T.cfg_diff_report(os.environ["ALOE_REF_CFG_26467"], "26467")
print("cfg diff vs 26232:", n, "bad", bad, "| vs 26467:", len(rows))
# plan facts (from the /data sam-sep stores; the run builds the same plan from the node-local copies, digests asserted equal)
PL = A.make_plan(64, roots=A.SRC_STORES)
dg = hashlib.sha256(json.dumps(PL["items"]).encode()).hexdigest()[:16]
facts = {"plan_digest": dg, "expect": os.environ["QQA_REF_PLAN_DIGEST"], "items": len(PL["items"]), "eval": len(PL["eval"]), "steps": PL["steps"],
         "n_first": PL["n_first"], "dummies_last_step": PL["steps"] * 64 - len(PL["items"]), "slots": T.GMAX,
         "pad_slot_dummies_per_step": T.GMAX * T.NA - 64}
assert dg == facts["expect"] and len(PL["items"]) == int(os.environ["ALOE_EXPECT_TRAIN_MICROS"]) and len(PL["eval"]) == int(os.environ["ALOE_EXPECT_EVAL_MICROS"])
assert PL["steps"] == int(os.environ["QQA_STEPS"])
# every logical rank's A FLAC (train + eval items + the dummy fallback) present with its index size; mimidec count
miss, nm, seen = [], 0, set()
for it in PL["items"] + PL["eval"]:
    k, r, sw = it
    rows_ = A.vox_index()[int(k)][int(r)]
    a = rows_["A"]["assistant" if int(sw) else "user"]
    nm += a["kind"] != "gcs"
    if a["flac"] in seen:
        continue
    seen.add(a["flac"])
    p = f"{A.FLAC_ROOT}/{a['flac']}"
    if not os.path.exists(p) or (a["kind"] == "gcs" and os.path.getsize(p) != int(a["flac_bytes"])):
        miss.append(a["flac"])
facts.update(a_tracks=len(seen), a_missing=len(miss), a_mimidec_items=nm)
# physical layout: rank i runs logical ranks i + 14 s
facts["layout"] = {i: [i + 14 * s for s in range(T.GMAX) if i + 14 * s < 64] for i in range(14)}
# a few micros through the trainer's data path (+ the [same-as-v2] compare vs the reference row of the old stores)
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("data_aloe_v2_22383", "/data/home/amartyavarma/ppm/aloe-v2-ft/code/data_aloe_v2.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
rm = json.load(open(os.environ["ALOE_SAMSEP_ROW_MAP"]))
OON = {k: {int(v): int(o) for o, v in rm["map"][f"m1n{k}"]["new_of_old"].items()} for k in range(4)}
st, rst = A.Stores(roots=A.SRC_STORES), R.Stores(roots=R.SRC_STORES)
import numpy as np  # noqa: E402
chk = []
for L, it in [(0, PL["items"][0]), (13, PL["items"][13]), (63, PL["items"][63]), (5, PL["eval"][5])]:
    cfg = T.CFG["data"] if it in PL["items"][:64] else T.CFG["eval_data"]
    m = st.conv(it, cfg, L, audio=True)
    mr = rst.conv((it[0], OON[it[0]][it[1]], it[2]), cfg, L)
    P = int(m["P"])
    c = {"item": it, "row_key_eq": m["row_key"] == mr["row_key"], "P": P == int(mr["P"]), "ids_B": bool(np.array_equal(m["ids"][1], mr["ids"][1])),
         "code_mask_B": bool(np.array_equal(m["depth"][1] >= 0, mr["depth"][1] >= 0)), "codes_B_differ": bool(not np.array_equal(m["depth"][1], mr["depth"][1])),
         "targets": all(np.array_equal(np.asarray(m[k]), np.asarray(mr[k])) for k in ("sel", "y", "word", "pad", "eob", "nw", "light")),
         "cb0_mask": bool(np.array_equal(np.asarray(m["cb0"]) >= 0, np.asarray(mr["cb0"]) >= 0)), "A_depth_all_neg": bool((m["depth"][0] < 0).all()),
         "voice_clip": int(m["voice_clip"]), "wav_s": round(len(m["wav"]) / 24000.0, 2), "aref_kind": m["aref"]["kind"], "gain_db": m["aref"]["gain_db"]}
    chk.append(c)
facts["micros"] = chk
json.dump(facts, open(f"{OUT}/dry_facts.json", "w"), indent=1, default=str)
print(json.dumps(facts, default=str)[:3000])
print("DRY", "PASS" if not bad and not miss and all(c["row_key_eq"] and c["P"] and c["ids_B"] and c["code_mask_B"] and c["targets"] and c["cb0_mask"]
                                                     and c["A_depth_all_neg"] for c in chk) else "FAIL")
