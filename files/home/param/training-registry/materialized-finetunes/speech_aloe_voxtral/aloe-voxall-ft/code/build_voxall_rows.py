"""aloe-voxall-ft DATA BUILD (10-16): = build_voxB_rows.py (aloe-vox-ft) on the ALL-TRACKS selection (build_allB_core.qualifying: every
conversation of the Aloe handover set, every speaker track a B candidate, bleed into B dropped; no pose). Conversations that were in
aloe-vox-ft's plan keep their store index (sticky: node k's existing A FLACs stay useful); new ones balanced by B items (LPT + swaps of
non-sticky conversations only) -> 8 conversation-sharded stores. A FLAC refs + vox_index exactly as aloe-vox-ft.
--- original doc ---
aloe-vox-ft DATA BUILD (10-16): the aloe-pose-ft rows (build_poseB_rows.py, UNCHANGED row maths: same qualifying B tracks, same rows,
words, personas, pose, 2% eval split) re-partitioned into 8 CONVERSATION-SHARDED stores balanced by B ITEMS (one store per training node),
plus channel A's AUDIO REFERENCE per row (A = the partner of each qualifying B track; channel A is ONLY its Voxtral latents in this run).

PLAN (python -I build_voxB_rows.py plan --out DIR): build_poseB_rows.qualifying() (the pose FT's selection, counts asserted = 18,093 B tracks
  / 28,048 B items 27,517 train / 531 eval) -> every conversation to one of 8 stores; train and eval balanced separately by B ITEMS
  (= used rows x qualifying sides): LPT then pairwise swaps until max train items per store <= ceil(27,517 / 8) = 3,440 (-> 430 steps of
  8 ranks x 8 nodes, = the pose FT's 430). Store k is trained ONLY by node k's 8 ranks (data_aloe_vox), so node k needs ONLY store k and
  the A FLACs of store k's conversations (each A track lives on exactly one node).
  A TRACKS: for each qualifying B speaker, A = the other speaker: gs://aloe-processed-datasets/stage2-handover-20261008/wlt/<ses>/<seg>/
  SPEAKER_0N.flac (manifest.jsonl uri / bytes / generation) + the recorded Mimi LUFS gain of that speaker (w00.mimi.json
  mimi[SPEAKER_0N].lufs_gain_db: the gain the Mimi codes were made with = audio_s4q.ffmpeg_row_audio's rule for handover rows).
BUILD (... build --out DIR --store K --workers W): build_poseB_rows.build_conv per conversation (identical rows) -> DIR/pK (s3-duplex-rows-v1
  + pose add-on, = the pose FT store format) + vox_index.json: per row, per data side that is channel A of some item of the row
  (the other side qualifies as B): {flac (path relative to the FLAC root), gain_db, frame0 (= 7500 x chunk), n_frames, flac_bytes}.
usage: python -I build_voxB_rows.py plan --out DIR ; python -I build_voxB_rows.py build --out DIR --store K --workers 16"""
import argparse
import collections
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "lib"))
import build_allB_core as BP  # noqa: E402   (all-tracks selection, pose FT row maths otherwise)

B, F = BP.B, BP.F
B.VERSION = "aloe-voxall-ft-v1"
OLD_PLAN = "/data/home/amartyavarma/ppm/aloe-vox-ft/plan/PLAN.json"
NST = 8
FLAC_PREFIX = "gs://aloe-processed-datasets/stage2-handover-20261008/wlt/"
OTHER = {"SPEAKER_00": "SPEAKER_01", "SPEAKER_01": "SPEAKER_00"}
EXPECT = {"B tracks": 18093, "B rows used": 28048, "B rows used train": 27517, "B rows used eval": 531}


def flac_meta():
    out = {}
    for line in open(f"{BP.DATA}/_meta/manifest.jsonl"):
        r = json.loads(line)
        for spk in BP.SPK:
            k = spk.lower()
            u = r.get(f"wlt_{k}_flac_uri")
            if u:
                out[(r["session_id"], r["seg"], spk)] = {"uri": u, "bytes": int(r[f"wlt_{k}_flac_bytes"]),
                                                          "generation": int(r[f"wlt_{k}_flac_generation"])}
    return out


def balance(convs, S, cap, sticky):
    """sticky conversations pinned to their old store; the rest LPT by items, then pairwise swaps of NON-sticky conversations
    (largest bin <-> smallest) until max <= cap (or no improving swap)."""
    load = [0] * S
    bins = [[] for _ in range(S)]
    for c in convs:
        if c["conv"] in sticky:
            k = sticky[c["conv"]]; bins[k].append(c); load[k] += c["items"]
    for c in sorted([c for c in convs if c["conv"] not in sticky], key=lambda c: (-c["items"], c["conv"])):
        k = min(range(S), key=lambda k: (load[k], k)); bins[k].append(c); load[k] += c["items"]
    for _ in range(100000):
        hi = max(range(S), key=lambda k: load[k]); lo = min(range(S), key=lambda k: load[k])
        if load[hi] <= cap or load[hi] - load[lo] <= 1:
            break
        gap = load[hi] - load[lo]
        best = None
        for a in [x for x in bins[hi] if x["conv"] not in sticky]:
            for b in [None] + [x for x in bins[lo] if x["conv"] not in sticky]:
                d = a["items"] - (b["items"] if b else 0)
                if 0 < d < gap and (best is None or abs(gap - 2 * d) < abs(gap - 2 * best[2])):
                    best = (a, b, d)
        if best is None:
            break
        a, b, d = best
        bins[hi].remove(a); bins[lo].append(a)
        if b is not None:
            bins[lo].remove(b); bins[hi].append(b)
        load[hi] -= d; load[lo] += d
    return bins, load


def plan(a):
    convs, st = BP.qualifying()
    print("[plan] selection", json.dumps({k: (round(v, 1) if isinstance(v, float) else v) for k, v in sorted(st.items())}), flush=True)
    sticky = {c["conv"]: int(c["store"]) for c in json.load(open(OLD_PLAN))["convs"]}
    fm = flac_meta()
    for c in convs:
        c["items"] = c["nrow"] * len(c["b_ok"])
        c["a_tracks"] = []
        mj = json.load(open(f"{c['dir']}/w00.mimi.json"))
        for spk in c["b_ok"]:
            a_spk = OTHER[spk]
            m = fm[(c["session"], c["seg"], a_spk)]
            assert m["uri"] == f"{FLAC_PREFIX}{c['session']}/{c['seg']}/{a_spk}.flac", m["uri"]
            mm = mj["mimi"][a_spk]
            assert int(mm["n_frames"]) == int(c["T"]) and int(mm["audio_sr"]) == 24000 and float(mj["clip"]["start_sec"]) == 0.0, (c["conv"], a_spk)
            c["a_tracks"].append({"b_spk": spk, "a_spk": a_spk, "uri": m["uri"], "bytes": m["bytes"], "generation": m["generation"],
                                  "rel": f"{c['session']}/{c['seg']}/{a_spk}.flac", "gain_db": float(mm["lufs_gain_db"]),
                                  "lufs_status": mm["lufs_status"], "audio_s": float(mm["audio_s"])})
    out = {"convs": [], "stores": NST}
    loads = {}
    for sp in ("train", "eval"):
        cs = [c for c in convs if c["split"] == sp]
        tot = sum(c["items"] for c in cs)
        cap = -(-tot // NST)
        bins, load = balance(cs, NST, cap, sticky)
        loads[sp] = load
        for k, b in enumerate(bins):
            for c in b:
                c["store"] = k
        print(f"[plan] {sp}: {len(cs)} conversations, {tot} B items -> per store {load} (cap {cap})", flush=True)
    print(f"[plan] train items {sum(loads['train'])} max/store {max(loads['train'])} cap {-(-sum(loads['train']) // NST)}", flush=True)
    out["convs"] = sorted(convs, key=lambda c: c["conv"])
    per = collections.defaultdict(lambda: collections.Counter())
    for c in convs:
        p = per[c["store"]]
        p["convs"] += 1; p["a_tracks"] += len(c["a_tracks"]); p["a_bytes"] += sum(t["bytes"] for t in c["a_tracks"])
        p["a_hours"] += sum(t["audio_s"] for t in c["a_tracks"]) / 3600
        p[f"items_{c['split']}"] += c["items"]
    out["per_store"] = {k: dict(v) for k, v in sorted(per.items())}
    out["stats"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in st.items()}
    out["stats"]["a_tracks"] = sum(len(c["a_tracks"]) for c in convs)
    out["stats"]["a_bytes"] = sum(t["bytes"] for c in convs for t in c["a_tracks"])
    out["stats"]["a_hours"] = round(sum(t["audio_s"] for c in convs for t in c["a_tracks"]) / 3600, 1)
    out["stats"]["all_flac_tracks"] = len(fm); out["stats"]["all_flac_bytes"] = sum(m["bytes"] for m in fm.values())
    out["loads"] = loads
    out["built"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(f"{a.out}/PLAN.json", "w"), indent=0)
    print("[plan] stats", json.dumps(out["stats"]), flush=True)
    for k, v in out["per_store"].items():
        print(f"[plan] store {k}: {json.dumps({x: (round(y, 1) if isinstance(y, float) else y) for x, y in v.items()})}", flush=True)


def build(a):
    import numpy as np
    P = json.load(open(a.plan))
    assert P["stores"] == NST
    # build_poseB_rows.build reads <out>/PLAN.json (convs with 'store'); its tmp dir is redirected to this run's scratch
    os.makedirs(a.out, exist_ok=True)
    pj = f"{a.out}/PLAN.json"
    if not os.path.exists(pj):
        json.dump({"convs": P["convs"], "stores": NST}, open(pj + ".tmp", "w")); os.replace(pj + ".tmp", pj)
    t0 = time.time()
    BP.build(a)
    out = f"{a.out}/p{a.store}"
    import pyarrow.parquet as pq
    t = pq.read_table(f"{out}/rows.parquet").to_pydict()
    convs = {c["conv"]: c for c in P["convs"] if c["store"] == a.store}
    keys = t["row_key"]
    vi = []
    nf_col = t["n_frames"]
    for r, rk in enumerate(keys):
        parts = rk.split("/")                                     # ALOE/s2handover-all/<ses>/<seg>/cKK
        ses, seg, ck = parts[2], parts[3], int(parts[4][1:])
        c = convs[f"{ses}/{seg}"]
        ent = {}
        for tr in c["a_tracks"]:
            side_a = BP.SIDE_OF[tr["a_spk"]]                          # data side of channel A for the item whose B = tr["b_spk"]
            ent[side_a] = {"flac": tr["rel"], "gain_db": tr["gain_db"], "frame0": F.ROW_FRAMES * ck, "n_frames": int(nf_col[r]),
                           "flac_bytes": tr["bytes"], "generation": tr["generation"], "a_spk": tr["a_spk"]}
        vi.append({"row": r, "row_key": rk, "A": ent})
    z = np.load(f"{out}/pose_index.npz")
    for r, e in enumerate(vi):                                     # A entry exists exactly for the sides that are A of a qualifying B item
        for j, side in enumerate(F.SIDES):                         # b_ok col 0 = user qualifies as B -> A = assistant
            other = F.SIDES[1 - j]
            assert bool(z["b_ok"][r, j]) == (other in e["A"]), (e["row_key"], side, e["A"].keys())
    json.dump({"format": "aloe-vox-index-v1", "store": f"p{a.store}", "rows": vi,
               "flac_root_note": "relative to the node-local FLAC root (aloe_vox.env ALOE_FLAC_ROOT)"}, open(f"{out}/vox_index.json", "w"))
    print(f"[p{a.store}] vox_index.json: {len(vi)} rows, {sum(len(e['A']) for e in vi)} A refs ({time.time() - t0:.0f}s)", flush=True)
    open(f"{out}/VOX_DONE", "w").write(str(len(vi)))


def _build_src(code):
    """the pose builder's build() source (tmp dir redirected) as a standalone function definition."""
    i = code.index("\ndef build(a):")
    j = code.index("\ndef main():")
    return code[i + 1:j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "build"))
    ap.add_argument("--out", required=True); ap.add_argument("--plan", default="/data/home/amartyavarma/ppm/aloe-voxall-ft/plan/PLAN.json")
    ap.add_argument("--store", type=int, default=0); ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tmp", default="/scratch/amartyavarma/aloe-voxall-build")
    a = ap.parse_args()
    plan(a) if a.cmd == "plan" else build(a)


if __name__ == "__main__":
    main()
