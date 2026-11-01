"""CLEAN-TARGET preflight (10-30, raw-codes pose-duplex): every row of the sam-sep stores comes from the handover "none" crossbleed bucket, so
both tracks should be clean by construction. Verify it per lane from the handover copy itself:
  DATA/files/none/<src_conversation>/w00.<speaker_dir>.xbleed.json  must say  acoustic_crossbleed false AND transcript_crossbleed false
(speaker == the row's <side>_speaker_dir). Every FIRST-CHUNK row x both sides of the 4 stores is read (so any plan / orientation is covered);
then the run's plan (data_aloe_posedup.make_plan with the env knobs, PLAN_FULL_STEPS steps) is checked: every used item's TARGET (B) side
(train items of every epoch + the eval list) must be a clean lane. The partner (A) side is reported too (info).
usage: python -I -B clean_target_check.py <code dir> <out json>   (env = the run env; POSEDUP_CLEAN_TARGET is ignored here)
Out: format posedup-clean-target-v1, store_digests (= the sam-sep stores'), clean_lanes [[store, row, side]], flagged / missing lanes, the plan
summary, VERDICT. The trainer's startup assert reads it via POSEDUP_CLEAN_TARGET=<out json>."""
import datetime
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

CODE, OUT = sys.argv[1], sys.argv[2]
os.environ.pop("POSEDUP_CLEAN_TARGET", None)
sys.path.insert(0, CODE)
A = __import__("data_aloe_posedup")
assert os.path.dirname(os.path.abspath(A.__file__)) == os.path.abspath(CODE), (A.__file__, CODE)
DATA = os.environ.get("HANDOVER_DATA", "/data/home/amartyavarma/data/aloe-handover-full/DATA")
LW = 64
S = int(os.environ.get("PLAN_FULL_STEPS", "426"))


def lanes():
    import pyarrow.parquet as pq
    out = []
    for k, root in enumerate(A.STORES):
        t = pq.read_table(f"{root}/rows.parquet", columns=["row", "row_key", "chunk_index", "src_conversation", "user_speaker_dir",
                                                            "assistant_speaker_dir", "split"]).to_pydict()
        for i, r in enumerate(t["row"]):
            if int(t["chunk_index"][i]) != 0:
                continue
            for sd in ("user", "assistant"):
                out.append((k, int(r), sd, t["src_conversation"][i], t[f"{sd}_speaker_dir"][i], t["row_key"][i], t["split"][i]))
    return out


def read(l):
    k, r, sd, conv, spk, rk, split = l
    f = f"{DATA}/files/none/{conv}/w00.{spk}.xbleed.json"
    if not os.path.exists(f):
        return {"lane": [k, r, sd], "row_key": rk, "file": f, "status": "missing"}
    j = json.load(open(f))
    ac, tc = j.get("acoustic_crossbleed"), j.get("transcript_crossbleed")
    ok = j.get("speaker") == spk and ac is False and tc is False
    return {"lane": [k, r, sd], "row_key": rk, "file": f, "speaker": j.get("speaker"), "acoustic_crossbleed": ac, "transcript_crossbleed": tc,
            "status": "clean" if ok else "flagged"}


if __name__ == "__main__":
    L = lanes()
    with ThreadPoolExecutor(int(os.environ.get("NTHREADS", "48"))) as ex:
        res = list(ex.map(read, L))
    clean = {tuple(x["lane"]) for x in res if x["status"] == "clean"}
    PL = A.make_plan(LW, steps=S)
    used = [tuple(x) for x in PL["items0"]] + [tuple(x) for x in PL["eval"]]
    ub = [(int(x[0]), int(x[1]), A.b_side(x)) for x in used]
    ua = [(int(x[0]), int(x[1]), "assistant" if A.b_side(x) == "user" else "user") for x in used]
    bad_b = [list(x) for x in ub if x not in clean]
    out = {"format": "posedup-clean-target-v1", "built": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z", "data_root": DATA,
           "rule": "w00.<speaker_dir>.xbleed.json: speaker == the row's <side>_speaker_dir, acoustic_crossbleed false, transcript_crossbleed false",
           "stores": list(A.STORES), "store_digests": list(A.A2.DIGESTS),
           "lanes": {"first_chunk_lanes": len(res), "clean": len(clean), "flagged": sum(x["status"] == "flagged" for x in res),
                     "missing": sum(x["status"] == "missing" for x in res)},
           "plan": {"steps": PL["steps"], "digest": A.plan_digest(PL)[0], "items": A.ITEMS, "band_mask": int(A.POSE_BAND_MASK),
                    "train_items_epoch0": len(PL["items0"]), "eval_items": len(PL["eval"]), "used_items": len(used),
                    "target_B_clean": sum(x in clean for x in ub), "target_B_not_clean": len(bad_b),
                    "partner_A_clean": sum(x in clean for x in ua)},
           "not_clean_target_items": bad_b[:200],
           "not_clean_lanes": [x for x in res if x["status"] != "clean"][:500],
           "clean_lanes": sorted([list(x) for x in clean])}
    out["VERDICT"] = "PASS" if not bad_b else "FAIL"
    txt = json.dumps(out)
    open(OUT, "w").write(txt)
    print(json.dumps({k: v for k, v in out.items() if k not in ("clean_lanes", "not_clean_lanes")}, indent=1))
    print("WROTE", OUT, "sha256", hashlib.sha256(txt.encode()).hexdigest(), out["VERDICT"])
