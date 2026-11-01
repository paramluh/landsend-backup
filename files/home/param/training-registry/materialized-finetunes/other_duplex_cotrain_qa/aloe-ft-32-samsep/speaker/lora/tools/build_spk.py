"""speaker FT build (10-24, CPU, idempotent): account + logical world + epochs / steps knobs -> the speaker plan and everything the trainer's
PROD checks read, under speaker/data/<acct8>/ (nothing of 29345 touched; no audio fetched or re-made):
  plan_lw<LW>_<digest>.json      spk_plan.build output (items, steps, standard eval, speaker eval, clip pool, stats, digests)
  RAWGAIN_<digest>.json          = the reference RAWGAIN entries (aloe-v2vox-raw/plan/RAWGAIN.json, 26232's; unchanged) of every A track this
                                   plan reads (train + both evals), plan_digest = this plan's (data_aloe_v2vox.rawgain asserts it)
  flac/                          shared A-FLAC root: <session> -> symlink into the reference GCS-verified copy aloe-v2vox-raw/flac (the same
                                   files 29345 reads); _ledger/DONE_spk_<digest>_node<l>.json + .tsv per logical node l (8 logical ranks each):
                                   every track its ranks read, size == index flac_bytes == reference ledger, generation equal
  summary_lw<LW>_<digest>.json   counts, the placeholder schedule (warmup / export steps), the env values to set
Also writes speaker/ref_cfg_29345.json (29345's logged resolved config, for the trainer's report-only diff) once.
usage: python -I build_spk.py <acct> <lworld> [epochs (default 1)] [steps (default auto)] [eval_frac (default 0.04)]"""
import json
import os
import sys
import time

SPK = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker"
LORA = f"{SPK}/lora"
sys.path.insert(0, f"{LORA}/code/lib"); sys.path.insert(0, f"{LORA}/code")
import spk_plan as SP  # noqa: E402

INDEX = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/data/plan/M1VOX_INDEX_samsep.json"
REF_FLAC = "/data/home/amartyavarma/ppm/aloe-v2vox-raw/flac"
REF_RAWGAIN = "/data/home/amartyavarma/ppm/aloe-v2vox-raw/plan/RAWGAIN.json"
LOG29345 = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/aloesamsep-29345.log"
acct, LW = sys.argv[1], int(sys.argv[2])
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
steps = None if len(sys.argv) <= 4 or sys.argv[4] == "auto" else int(sys.argv[4])
efrac = float(sys.argv[5]) if len(sys.argv) > 5 else 0.04
ecalls = int(sys.argv[6]) if len(sys.argv) > 6 else 0
rfrac = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
every = int(sys.argv[8]) if len(sys.argv) > 8 else 10
t0 = time.time()
# ---- 29345's resolved config (its first log line carries 'cfg {json}')
rc = f"{SPK}/ref_cfg_29345.json"
if not os.path.exists(rc):
    for ln in open(LOG29345):
        if "ppm-aloe-samsep-vox17127-j29345: world 64" in ln and "; cfg {" in ln:
            json.dump(json.loads(ln.split("; cfg ", 1)[1]), open(rc, "w"), indent=1)
            break
    assert os.path.exists(rc), "29345 cfg line not found"
PL = SP.build(acct, LW, epochs=epochs, steps=steps, eval_frac=efrac, eval_calls=ecalls, replay_frac=rfrac)
dg = PL["plan_digest"]
OUT = f"{LORA}/data/{acct[:8]}"
os.makedirs(f"{OUT}/flac/_ledger", exist_ok=True)
pf = f"{OUT}/plan_lw{LW}_{dg}.json"
json.dump({k: v for k, v in PL.items() if not k.startswith("_")}, open(pf, "w"))
idx = json.load(open(INDEX))
assert idx["digests"] == SP.SAMSEP_DIGESTS
rows = {int(k): v for k, v in idx["rows"].items()}
led_ref = {}
for l in range(8):
    for ln in open(f"{REF_FLAC}/_ledger/v2vox_node{l}.tsv"):
        f = ln.rstrip("\n").split("\t")
        led_ref[f[0]] = f
rg = json.load(open(REF_RAWGAIN))
assert rg["plan_digest"] == "ba2d70729ea437ab"


def track(it):
    k, r, sw = it[:3]
    e = rows[int(k)][int(r)]
    assert e["row"] == int(r)
    return e["A"]["assistant" if int(sw) else "user"]


used, per = {}, {}
for l in range(-(-LW // 8)):                                        # 10-25 fix: ceil (a logical world < 8 or not a multiple of 8, e.g. Rosie lw6)
    its = set()
    for R in range(8 * l, min(8 * l + 8, LW)):
        tr, ev = PL["items"][R::LW], PL["eval"][R::LW]
        its |= {tuple(x) for x in tr if x is not None} | {tuple(x) for x in ev} | {tuple(x) for x in PL["eval_spk"][R::LW]}
    its.add(tuple(PL["items"][0]))                                  # the dummy fallback item
    tr_ = {}
    for it in sorted(its):
        t = track(it)
        ref = led_ref.get(t["flac"])
        assert t["kind"] == "gcs" and ref is not None, ("A track not a verified reference FLAC", it, t["flac"], t["kind"])
        assert os.path.getsize(f"{REF_FLAC}/{t['flac']}") == int(t["flac_bytes"]) == int(ref[1]) and int(ref[2]) == int(t["generation"]), (t["flac"], ref)
        g = rg["rows"][t["flac"]]
        assert g["kind"] == t["kind"] and abs(float(g["gain_mimi_db"]) - float(t["gain_db"])) < 1e-12 and float(g["gain_raw_db"]) == 0.0, (t["flac"], g)
        tr_[t["flac"]] = t; used[t["flac"]] = t
    with open(f"{OUT}/flac/_ledger/spk_{dg}_lw{LW}_node{l}.tsv", "w") as f:
        for rel in sorted(tr_):
            f.write("\t".join(led_ref[rel]) + "\n")
    json.dump({"node": l, "plan_digest": dg, "lworld": LW, "account": acct, "items": len(its), "gcs_files": len(tr_), "gcs_bytes": sum(int(x["flac_bytes"]) for x in tr_.values()),
               "mimidec": 0, "physical_host": "shared /data (symlinks into the reference copy)",
               "verified": f"size == index flac_bytes == reference ledger (crc32c + md5 verified vs GCS by {REF_FLAC}/_ledger); generation equal",
               "reference_root": REF_FLAC, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, open(f"{OUT}/flac/_ledger/DONE_spk_{dg}_lw{LW}_node{l}.json", "w"), indent=1)
    per[l] = (len(its), len(tr_))
for top in sorted({rel.split("/")[0] for rel in used}):
    dst = f"{OUT}/flac/{top}"
    if os.path.islink(dst):
        assert os.readlink(dst) == f"{REF_FLAC}/{top}"
    else:
        os.symlink(f"{REF_FLAC}/{top}", dst)
for rel, t in used.items():
    assert os.path.getsize(f"{OUT}/flac/{rel}") == int(t["flac_bytes"])
rg2 = {k: v for k, v in rg.items() if k != "rows"}
rg2.update(plan_digest=dg, rows={rel: rg["rows"][rel] for rel in sorted(used)},
           spk={"derived_from": REF_RAWGAIN, "ref_plan_digest": rg["plan_digest"], "account": acct, "rule": "entries unchanged, subset = tracks this plan reads"})
rgf = f"{OUT}/RAWGAIN_{dg}.json"
json.dump(rg2, open(rgf, "w"), indent=1)
sch = SP.schedule_defaults(PL["steps"])
sch["export_steps"] = [s for s in range(every, PL["steps"], every)]   # LoRA run: export every EXPORT_EVERY steps (knob; main: ~10)
warm = max(1, sch["warmup"])                                         # sched_s3q's WSD asserts 0 < decay start (= warmup) < steps
summ = {"account": acct, "lworld": LW, "plan_file": pf, "plan_digest": dg, "eval_digest": PL["eval_digest"], "eval_spk_digest": PL["eval_spk_digest"],
        "steps": PL["steps"], "epochs": PL["epochs"], "train_micros": PL["n_train_micros"], "dummies_last_step": PL["steps"] * LW - PL["n_train_micros"], "replay": PL.get("replay"), "convs_heldout_list": PL["convs_heldout"],
        "eval_micros": len(PL["eval"]), "eval_spk_micros": len(PL["eval_spk"]), "n_train_rows": PL["n_train_rows"], "convs": PL["convs"],
        "convs_heldout": PL["convs_heldout"], "stats": PL["stats"], "a_tracks": len(used), "ledgers": per, "rawgain": rgf, "flac_root": f"{OUT}/flac",
        "placeholder_schedule": {"warmup_rule": round(10 * PL["steps"] / 306), "warmup": warm, "export_steps": sch["export_steps"],
                                 "decay": f"linear from step {warm} to 0 at step {PL['steps']}"},
        "env": {"ALOE_SPK_ACCOUNT": acct, "ALOE_LOGICAL_WORLD": LW, "ALOE_SPK_EPOCHS": epochs, "ALOE_SPK_STEPS": steps or "auto",
                "ALOE_SPK_EVAL_FRAC": efrac, "ALOE_SPK_EVAL_CALLS": ecalls, "ALOE_REPLAY_FRAC": rfrac, "QQA_STEPS": PL["steps"], "QQA_WARMUP": warm, "QQA_EXPORT_STEPS": ",".join(map(str, sch["export_steps"])),
                "QQA_REF_PLAN_DIGEST": dg, "ALOE_RAWGAIN_PLAN_DIGEST": dg, "ALOE_RAWGAIN": rgf, "ALOE_SPK_PLAN": pf, "ALOE_SHARED_FLAC": f"{OUT}/flac",
                "ALOE_EXPECT_TRAIN_MICROS": PL["n_train_micros"], "ALOE_EXPECT_EVAL_MICROS": len(PL["eval"]), "ALOE_SPK_EXPECT_EVAL_MICROS": len(PL["eval_spk"])},
        "time_s": round(time.time() - t0, 1)}
json.dump(summ, open(f"{OUT}/summary_lw{LW}_{dg}.json", "w"), indent=1)
print(json.dumps({k: summ[k] for k in ("plan_digest", "steps", "epochs", "train_micros", "dummies_last_step", "eval_micros", "eval_spk_micros", "a_tracks",
                                        "ledgers", "placeholder_schedule")}))
for k, v in summ["env"].items():
    print(f"{k}={v}")
