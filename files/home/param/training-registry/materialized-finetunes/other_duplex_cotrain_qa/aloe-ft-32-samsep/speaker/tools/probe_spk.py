"""speaker FT probe (10-24, CPU): per candidate account -> rows / conversations / hours / flipped share / held-out / steps per epoch at 16 and 64
rows per step, B code frames (her lane's coded frames in the rows), the A-side (partner) FLAC coverage for the speaker items (M1VOX_INDEX_samsep
entry of the A side + file present in the reference FLAC copy with the index size + a RAWGAIN entry with the same gain), and the manifest
gap (her none-bucket calls that are not in the stores). usage: python -I probe_spk.py <out json> <acct> [<acct> ...]"""
import json
import os
import sys

HERE = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/code"
sys.path.insert(0, f"{HERE}/lib"); sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import spk_plan as SP  # noqa: E402
import s3rows as R3  # noqa: E402

INDEX = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/data/plan/M1VOX_INDEX_samsep.json"
REF_FLAC = "/data/home/amartyavarma/ppm/aloe-v2vox-raw/flac"
REF_RAWGAIN = "/data/home/amartyavarma/ppm/aloe-v2vox-raw/plan/RAWGAIN.json"
OUT = sys.argv[1]
idx = json.load(open(INDEX))
assert idx["digests"] == SP.SAMSEP_DIGESTS
rg = json.load(open(REF_RAWGAIN))["rows"]
st = [R3.Rows(R3.load_node(f"{SP.STORES_SRC}/m1n{k}")) for k in range(4)]
res = {}
for acct in sys.argv[2:]:
    rows = SP.speaker_rows(acct)
    p16 = SP.build(acct, 16)
    p64 = SP.build(acct, 64)
    assert p16["n_train_rows"] == p64["n_train_rows"] and p16["convs_heldout"] == p64["convs_heldout"]
    trs = {(k, r) for k, r, _ in p16["items"]}
    cov = {"items": 0, "gcs": 0, "mimidec": 0, "missing_file": [], "size_mismatch": [], "no_rawgain": [], "gain_mismatch": [], "a_spk_mismatch": [],
           "tracks": set()}
    fr = {"train_B_frames": 0, "train_B_coded_frames": 0, "eval_B_frames": 0, "eval_B_coded_frames": 0, "max_row_frames": 0}
    for x in rows:
        k, r, sw = x["k"], x["r"], x["swap"]
        aside = "assistant" if sw else "user"
        e = idx["rows"][str(k)][r]
        assert e["row"] == r and e["row_key"] == x["row_key"]
        a = e["A"][aside]
        row = st[k][r]
        assert row["row_key"] == x["row_key"]
        cov["items"] += 1
        cov[a["kind"]] += 1
        cov["tracks"].add(a["flac"])
        if a["a_spk"] != row[f"{aside}_speaker_dir"]:
            cov["a_spk_mismatch"].append(x["row_key"])
        p = f"{REF_FLAC}/{a['flac']}"
        if not os.path.exists(p):
            cov["missing_file"].append(a["flac"])
        elif a["kind"] == "gcs" and os.path.getsize(p) != int(a["flac_bytes"]):
            cov["size_mismatch"].append(a["flac"])
        g = rg.get(a["flac"])
        if g is None:
            cov["no_rawgain"].append(a["flac"])
        elif g["kind"] != a["kind"] or abs(float(g["gain_mimi_db"]) - float(a["gain_db"])) >= 1e-12:
            cov["gain_mismatch"].append(a["flac"])
        c = np.asarray(row[f"codes_{x['side']}"])
        n = int(row["n_frames"])
        coded = int((c[:, :n] >= 0).all(0).sum())
        tag = "train" if (k, r) in trs else "eval"
        fr[f"{tag}_B_frames"] += n
        fr[f"{tag}_B_coded_frames"] += coded
        fr["max_row_frames"] = max(fr["max_row_frames"], n)
    cov["tracks"] = len(cov["tracks"])
    man = SP.manifest()
    none_calls = sorted(rk for rk, m in man.items() if m["bucket"] == "none" and acct in m["acc"].values())
    in_st = {x["conv"] for x in rows}
    xb = sorted(rk for rk, m in man.items() if m["bucket"] != "none" and acct in m["acc"].values())
    res[acct] = {"stats": p16["stats"], "convs": p16["convs"], "convs_heldout": p16["convs_heldout"], "heldout_from_std_eval": p16["convs_heldout_from_std_eval"],
                 "n_train_rows": p16["n_train_rows"], "n_eval_spk_rows": p16["n_eval_spk_rows"], "steps_1ep_lw16": p16["steps"], "steps_1ep_lw64": p64["steps"],
                 "plan_digest_lw16": p16["plan_digest"], "plan_digest_lw64": p64["plan_digest"], "a_flac": cov, "frames": fr,
                 "manifest_none_calls": len(none_calls), "manifest_none_not_in_stores": sorted(set(none_calls) - in_st),
                 "manifest_xbleed_calls": len(xb), "actor_ids": sorted({man[c]["actor"][L] for c in in_st for L in ("00", "01") if man[c]["acc"][L] == acct})}
    s = p16["stats"]
    print(f"{acct[:8]} convs {p16['convs']} rows {s['all']['rows']} (train {s['train']['rows']} / eval_spk {s['eval_spk']['rows']} in {len(p16['convs_heldout'])} calls, "
          f"{len(p16['convs_heldout_from_std_eval'])} from the std eval split) flipped {s['all']['flipped_rows']} hours {s['all']['hours']} "
          f"steps/epoch lw16 {p16['steps']} lw64 {p64['steps']} | B frames train {fr['train_B_frames']} coded {fr['train_B_coded_frames']} | A FLAC: "
          f"{cov['gcs']} gcs {cov['mimidec']} mimidec, missing {len(cov['missing_file'])} size {len(cov['size_mismatch'])} rawgain {len(cov['no_rawgain'])} "
          f"gain {len(cov['gain_mismatch'])} a_spk {len(cov['a_spk_mismatch'])}; manifest none calls {len(none_calls)}, not in stores "
          f"{len(set(none_calls) - in_st)}", flush=True)
json.dump(res, open(OUT, "w"), indent=1, default=list)
