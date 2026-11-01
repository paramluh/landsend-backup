"""pose6 / B2 (10-27): G0COND frame-0 conditioning inputs for EVERY Aloe monoplex pose item (aloe-pose-mono-8020: 27,517 train + 531 eval).

ITEMS = data_aloe_pose.splits_v2 under ALOE_DATA=aloe-mono (aloe-pose-mono-8020/code/data_aloe_pose.py:401-430), re-implemented here line
for line and asserted against tests/plan_mono_check.json (item counts + frames): one item per (store k, row r, swap s) with
b_ok[r, 1 - s] and n_frames >= MIN_CLIP (= 2, :368); swap 0 -> B = data 'assistant' = SPEAKER_00, swap 1 -> B = 'user' = SPEAKER_01.
Item frame f = store row frame f = conversation frame 7500 * chunk_index + f (build_poseB_rows.py build_conv; data_aloe_pose.pose_arrays
puts it at P + pose_delay + f, no crop / offset).

SOURCE (on disk, read only): the B track's pose.json (aloe-handover-full DATA/<src_dir>/w00.<SPEAKER>.pose.json, the file build_poseB_rows.py
asserted and tokenised from). Its 'normalization' block = the tokenizer's frame-0 anchor of the WHOLE cut (tokenizer_call 'one call over the
whole cut'): root0_yx_normalized, shoulder_width0_normalized, orig_h, orig_w, rotation, frame0_pose_yxc_pixels [133 x 3].

DEFINITIONS = TRAINING'S, executed from the training sources' own text (no re-typing):
  frame0 fields: the frame-0 block of pose5/code-811-s2p/fresh/fresh_pose.py decode_new (lines 131-141, = build_pose_rows_sam.py 136-146),
    exec'd verbatim with P = {global_state: [[root0_y, root0_x, width0]] (the stored anchor, = fresh's constant per-frame global_state),
    frame0_pose, orig_h, orig_w} and B.JOINTS from build_pose_rows_sam.py:42.
  f: data_ppm.g0_feat (pose5/code-811-s2p/data_ppm.py:257-275, + G0_DIM / G0_FIELDS / g0_problem :243-254), exec'd from that file.
DECISION (pose6 10-27): chunk_index > 0 items get the cut-frame-0 anchor too (the anchor their tokens are relative to); column
anchor_is_item_frame0 marks them.

usage: python -I build_g0aloe.py [--workers N] [--no-verify-tokens] [--limit-tracks N (test only; writes to --out)] [--out DIR]"""
import argparse
import ast
import collections
import hashlib
import json
import os
import sys
import textwrap
import time
import types

import numpy as np

STORE_DIR = "/data/home/amartyavarma/ppm/aloe-pose-ft/stores"
NST = 8
MIN_CLIP = 2                                                                   # data_aloe_pose.py:368
ROW_FRAMES = 7500
PLAN_CHECK = "/data/home/amartyavarma/ppm/aloe-pose-mono-8020/tests/plan_mono_check.json"
DATA_PPM = "/data/home/amartyavarma/ppm/pose5/code-811-s2p/data_ppm.py"
FRESH = "/data/home/amartyavarma/ppm/pose5/code-811-s2p/fresh/fresh_pose.py"
SAM = "/data/home/amartyavarma/qwen-1ch-data/pose-s3-samdata/scripts/build_pose_rows_sam.py"
BUILDER = "/data/home/amartyavarma/ppm/aloe-pose-ft/code/build_poseB_rows.py"
DATA_ALOE = "/data/home/amartyavarma/ppm/aloe-pose-mono-8020/code/data_aloe_pose.py"
TOKENIZER_POSE = "InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0"          # build_poseB_rows.py TOKENIZER_POSE
NORM = "frame0-root-frame0-horizontal-shoulder-width-v1"                         # build_poseB_rows.py NORM
SIDE_SPK = {"assistant": "SPEAKER_00", "user": "SPEAKER_01"}                    # build_poseB_rows.py SIDE_OF (inverted)
SIDES = ("user", "assistant")                                                    # store pose_index b_ok column order


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------ training definitions, from their own text
def load_defs():
    ns = {"np": np, "sys": sys}
    src = open(DATA_PPM).read()
    got = []
    for node in ast.parse(src).body:
        nm = node.name if isinstance(node, ast.FunctionDef) else \
            (node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if nm in ("G0_DIM", "G0_FIELDS", "g0_feat", "g0_problem"):
            exec(compile(ast.get_source_segment(src, node), DATA_PPM, "exec"), ns); got.append(nm)
    assert sorted(got) == sorted(["G0_DIM", "G0_FIELDS", "g0_feat", "g0_problem"]) and ns["G0_DIM"] == 8, got
    sam = open(SAM).read()
    jn = [n for n in ast.parse(sam).body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "JOINTS"]
    assert len(jn) == 1
    jns = {}
    exec(compile(ast.get_source_segment(sam, jn[0]), SAM, "exec"), jns)
    joints = jns["JOINTS"]
    assert joints == list(range(13)) + list(range(91, 133)), "JOINTS"
    lines = open(FRESH).read().split("\n")
    i0 = [i for i, l in enumerate(lines) if l.startswith("    frame0_ok, H, W, root, width, j55 = False, 0.0, 0.0,")]
    assert len(i0) == 1, i0
    i0 = i0[0]
    i1 = next(i for i in range(i0, i0 + 20) if lines[i].strip() == "frame0_ok = False" and lines[i - 1].strip().startswith("except Exception"))
    block = textwrap.dedent("\n".join(lines[i0:i1 + 1]))
    assert block.count("\n") == 10 and "np.allclose(gs[0], [*root, width], atol=1e-4)" in block, block
    code = compile(block, FRESH, "exec")
    return ns, joints, code, {"fresh_block_lines": f"{i0 + 1}-{i1 + 1}", "fresh_block_md5": hashlib.md5(block.encode()).hexdigest(),
                              "fresh_block": block}


DEFS, JOINTS, FRESH_CODE, FRESH_INFO = load_defs()
g0_feat, g0_problem, G0_FIELDS = DEFS["g0_feat"], DEFS["g0_problem"], DEFS["G0_FIELDS"]


def frame0_fields(N):
    """pose.json normalization block -> the clip record's frame0 fields, by fresh_pose.decode_new's own frame-0 block."""
    P = {"global_state": [[N.get("root0_yx_normalized", [None, None])[0], N.get("root0_yx_normalized", [None, None])[1],
                           N.get("shoulder_width0_normalized")]],
         "frame0_pose": N.get("frame0_pose_yxc_pixels"), "orig_h": N.get("orig_h"), "orig_w": N.get("orig_w")}
    loc = {"np": np, "P": P, "B": types.SimpleNamespace(JOINTS=JOINTS)}
    exec(FRESH_CODE, loc)
    root, width = loc["root"], loc["width"]
    try:
        anc = np.asarray(P["global_state"], np.float32).reshape(-1, 3)[0]
        diff = float(np.max(np.abs(anc.astype(np.float64) - np.array([root[0], root[1], width], np.float64))))
    except Exception:  # noqa: BLE001
        anc, diff = np.full(3, np.nan, np.float32), float("nan")
    rec = {"orig_h": loc["H"], "orig_w": loc["W"], "frame0_root": [float(root[0]), float(root[1])], "frame0_width": float(width),
           "frame0_ok": bool(loc["frame0_ok"])}
    return rec, anc, diff


# ------------------------------------------------------------------------------------------------ items (= splits_v2, aloe-mono)
def items():
    import pyarrow.parquet as pq
    out = {"train": [], "eval": []}
    skipped = collections.Counter()
    cols = ["row", "split", "n_frames", "row_key", "chunk_index", "frame_start", "src_dir", "src_frames", "user_speaker_dir",
            "assistant_speaker_dir", "src_conversation"]
    for k in range(NST):
        root = f"{STORE_DIR}/p{k}"
        t = pq.read_table(f"{root}/rows.parquet", columns=cols).to_pydict()
        assert t["row"] == list(range(len(t["row"]))), root
        z = np.load(f"{root}/pose_index.npz")
        b_ok, sp = z["b_ok"], z["split"]
        assert len(b_ok) == len(t["row"]) and [("eval" if x else "train") for x in sp] == t["split"], root
        for i, (r, s, n, key) in enumerate(zip(t["row"], t["split"], t["n_frames"], t["row_key"])):
            for sw in (0, 1):                                  # swap 0 -> B = assistant (b_ok col 1); swap 1 -> B = user (col 0)
                if b_ok[r, 1 - sw]:
                    if int(n) < MIN_CLIP:
                        skipped[s] += 1
                        continue
                    side = "assistant" if sw == 0 else "user"
                    spk = t[f"{side}_speaker_dir"][i]
                    assert spk == SIDE_SPK[side], (key, side, spk)
                    ch, fs = int(t["chunk_index"][i]), int(t["frame_start"][i])
                    assert fs == ROW_FRAMES * ch and key.endswith(f"/c{ch:02d}") and \
                        int(n) == min(int(t["src_frames"][i]) - fs, ROW_FRAMES), (key, ch, fs, n, t["src_frames"][i])
                    out[s].append({"store": k, "row": int(r), "swap": sw, "n_frames": int(n), "row_key": key, "b_side": side,
                                   "item_key": f"{key}|{side}", "chunk_index": ch, "frame_start": fs, "src_dir": t["src_dir"][i],
                                   "src_frames": int(t["src_frames"][i]), "speaker": spk, "conversation": t["src_conversation"][i]})
    return out, skipped


# ------------------------------------------------------------------------------------------------ one B track (all its items)
_MM = {}


def store_pose(k, side):
    if (k, side) not in _MM:
        _MM[(k, side)] = np.load(f"{STORE_DIR}/p{k}/rows-00000-of-00001.pose_{side}.npy", mmap_mode="r")
    return _MM[(k, side)]


def do_track(job):
    (src_dir, spk), its, verify = job
    res = {"src_dir": src_dir, "speaker": spk, "items": [x["item_key"] for x in its], "problem": "", "rec": None}
    pj = f"{src_dir}/w00.{spk}.pose.json"
    try:
        j = json.load(open(pj))
    except Exception as e:  # noqa: BLE001
        res["problem"] = f"pose.json unreadable: {type(e).__name__}: {e}"
        return res
    T = its[0]["src_frames"]
    chk = []
    if j.get("status") != "ok" or j.get("speaker") != spk:
        chk.append(f"status/speaker {j.get('status')}/{j.get('speaker')}")
    if not str(j.get("tokenizer", "")).startswith(TOKENIZER_POSE) or float(j.get("token_rate_hz", 0)) != 12.5:
        chk.append(f"tokenizer/rate {j.get('tokenizer')} {j.get('token_rate_hz')}")
    N = j.get("normalization") or {}
    if not str(N.get("recipe", "")).startswith(NORM):
        chk.append(f"recipe {N.get('recipe')}")
    if list(j.get("pose_tokens_shape") or []) != [32, T]:
        chk.append(f"pose_tokens_shape {j.get('pose_tokens_shape')} != [32, {T}]")
    if chk:
        res["problem"] = "pose.json contract: " + "; ".join(chk)
        return res
    rec, anc, diff = frame0_fields(N)
    res.update(rec=rec, anchor=[float(x) for x in anc], diff=diff, rotation=N.get("rotation"), recipe=N.get("recipe"),
               tokenizer_call=j.get("tokenizer_call"))
    if verify:                                                # the tokens this pose.json describes == the store's B pose on every item
        import io
        raw = open(f"{src_dir}/{j['pose_tokens_file']}", "rb").read()
        if hashlib.sha256(raw).hexdigest() != j["pose_tokens_sha256"]:
            res["problem"] = "pose tokens sha256 != pose.json"
            return res
        p = np.load(io.BytesIO(raw))
        if p.shape != (32, T):
            res["problem"] = f"pose tokens shape {p.shape} != (32, {T})"
            return res
        bad = []
        for x in its:
            sp = store_pose(x["store"], x["b_side"])[x["row"]]
            n, fs = x["n_frames"], x["frame_start"]
            if not (np.array_equal(np.asarray(sp[:, :n]), p[:, fs:fs + n].astype(np.int16)) and (np.asarray(sp[:, n:]) < 0).all()):
                bad.append(x["item_key"])
        res["tokens_verified"] = len(its) - len(bad)
        if bad:
            res["problem"] = f"store pose != tokens file on {bad}"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-verify-tokens", action="store_true")
    ap.add_argument("--limit-tracks", type=int, default=0)
    ap.add_argument("--out", default="/data/home/amartyavarma/ppm/pose6/g0aloe")
    a = ap.parse_args()
    t0 = time.time()
    its, skipped = items()
    ref = json.load(open(PLAN_CHECK))
    cnt = {s: len(v) for s, v in its.items()}
    frames = {s: sum(x["n_frames"] for x in v) for s, v in its.items()}
    assert cnt == {"train": ref["n_train_items"], "eval": ref["n_eval_items"]}, (cnt, ref["n_train_items"], ref["n_eval_items"])
    assert frames == {"train": ref["train_frames"], "eval": ref["eval_frames"]}, (frames, ref["train_frames"], ref["eval_frames"])
    print(f"[items] {cnt} frames {frames} skipped<{MIN_CLIP} {dict(skipped)} ({time.time() - t0:.1f}s)", flush=True)
    allit = [dict(x, split=s, idx=i) for s in ("train", "eval") for i, x in enumerate(its[s])]
    tracks = collections.OrderedDict()
    for x in allit:
        tracks.setdefault((x["src_dir"], x["speaker"]), []).append(x)
    jobs = [(k, v, not a.no_verify_tokens) for k, v in tracks.items()]
    if a.limit_tracks:
        jobs = jobs[:a.limit_tracks]
        assert a.out != "/data/home/amartyavarma/ppm/pose6/g0aloe", "--limit-tracks is a test: give a scratch --out"
    print(f"[tracks] {len(tracks)} B tracks for {len(allit)} items; running {len(jobs)} with {a.workers} workers, "
          f"verify_tokens={not a.no_verify_tokens}", flush=True)
    import multiprocessing as mp
    res = {}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(do_track, jobs, chunksize=8)):
            res[(r["src_dir"], r["speaker"])] = r
            if (i + 1) % 2000 == 0 or i + 1 == len(jobs):
                print(f"[tracks] {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    # ---------------------------------------------------------------------------------------- per-item table
    rows, F, OK = [], [], []
    for x in allit:
        r = res.get((x["src_dir"], x["speaker"]))
        if r is None:                                            # --limit-tracks only
            continue
        rec = r["rec"] if (r["rec"] is not None and not r["problem"]) else None
        clip = dict(rec) if rec else {k: None for k in G0_FIELDS}
        f, ok = g0_feat(clip)
        prob = g0_problem(clip) if rec else ("SOURCE: " + r["problem"])
        rows.append({"split": x["split"], "idx": x["idx"], "store": x["store"], "row": x["row"], "swap": x["swap"], "n_frames": x["n_frames"],
                     "row_key": x["row_key"], "b_side": x["b_side"], "item_key": x["item_key"], "chunk_index": x["chunk_index"],
                     "frame_start": x["frame_start"], "anchor_is_item_frame0": x["chunk_index"] == 0, "conversation": x["conversation"],
                     "speaker": x["speaker"], "pose_json": f"{x['src_dir']}/w00.{x['speaker']}.pose.json",
                     "orig_h": clip["orig_h"], "orig_w": clip["orig_w"], "frame0_root": clip["frame0_root"],
                     "frame0_width": clip["frame0_width"], "frame0_ok": clip["frame0_ok"],
                     "rotation": None if r.get("rotation") is None else int(r["rotation"]), "anchor_stored": r.get("anchor"), "anchor_recompute_maxdiff": r.get("diff"),
                     "f": [float(v) for v in f], "g0_ok": bool(ok), "problem": prob})
        F.append(f); OK.append(ok)
    F = np.stack(F).astype(np.float32) if F else np.zeros((0, 8), np.float32)
    os.makedirs(a.out, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pa.schema([("split", pa.string()), ("idx", pa.int32()), ("store", pa.int8()), ("row", pa.int32()), ("swap", pa.int8()),
                        ("n_frames", pa.int32()), ("row_key", pa.string()), ("b_side", pa.string()), ("item_key", pa.string()),
                        ("chunk_index", pa.int16()), ("frame_start", pa.int32()), ("anchor_is_item_frame0", pa.bool_()),
                        ("conversation", pa.string()), ("speaker", pa.string()), ("pose_json", pa.string()),
                        ("orig_h", pa.float64()), ("orig_w", pa.float64()), ("frame0_root", pa.list_(pa.float64())),
                        ("frame0_width", pa.float64()), ("frame0_ok", pa.bool_()), ("rotation", pa.int32()),
                        ("anchor_stored", pa.list_(pa.float64())), ("anchor_recompute_maxdiff", pa.float64()),
                        ("f", pa.list_(pa.float32())), ("g0_ok", pa.bool_()), ("problem", pa.string())])
    tbl = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(tbl, f"{a.out}/g0aloe.parquet.tmp")
    os.replace(f"{a.out}/g0aloe.parquet.tmp", f"{a.out}/g0aloe.parquet")
    np.savez(f"{a.out}/g0aloe_f.npz.tmp.npz", f=F, g0_ok=np.asarray(OK, bool), split=np.array([r["split"] == "eval" for r in rows], np.int8),
             idx=np.array([r["idx"] for r in rows], np.int32), store=np.array([r["store"] for r in rows], np.int8),
             row=np.array([r["row"] for r in rows], np.int32), swap=np.array([r["swap"] for r in rows], np.int8),
             item_key=np.array([r["item_key"] for r in rows]))
    os.replace(f"{a.out}/g0aloe_f.npz.tmp.npz", f"{a.out}/g0aloe_f.npz")
    # ---------------------------------------------------------------------------------------- re-read check: f == g0_feat(fields) byte-exact
    back = pq.read_table(f"{a.out}/g0aloe.parquet").to_pylist()
    for r_, f_ in zip(back, F):
        f2, ok2 = g0_feat({k: r_[k] for k in G0_FIELDS})
        assert np.array_equal(np.asarray(r_["f"], np.float32), f2) and np.array_equal(f2, f_) and ok2 == r_["g0_ok"], r_["item_key"]
    z = np.load(f"{a.out}/g0aloe_f.npz")
    assert np.array_equal(z["f"], F)
    # ---------------------------------------------------------------------------------------- coverage
    def stats(sel):
        sel = list(sel)
        d = [r["anchor_recompute_maxdiff"] for r in sel if r["anchor_recompute_maxdiff"] is not None]
        return {"items": len(sel), "all_fields": sum(r["frame0_ok"] is not None for r in sel), "g0_ok": sum(r["g0_ok"] for r in sel),
                "frame0_ok_false": sum(r["frame0_ok"] is False for r in sel), "missing_source": sum(r["frame0_ok"] is None for r in sel),
                "anchor_recompute_maxdiff_max": float(np.nanmax(d)) if d else None}
    cov = {"built_pt": None, "out": a.out, "verify_tokens": not a.no_verify_tokens, "limit_tracks": a.limit_tracks,
           "items_expected": cnt, "tracks": len(tracks), "tracks_run": len(jobs),
           "tracks_tokens_verified_items": int(sum(r.get("tokens_verified", 0) for r in res.values())),
           "by_split": {s: stats(r for r in rows if r["split"] == s) for s in ("train", "eval")},
           "by_split_chunk": {f"{s}_{c}": stats(r for r in rows if r["split"] == s and (r["chunk_index"] == 0) == (c == "c0"))
                              for s in ("train", "eval") for c in ("c0", "cgt0")},
           "c0_anchor_recompute_maxdiff_max": stats(r for r in rows if r["chunk_index"] == 0)["anchor_recompute_maxdiff_max"],
           "all_anchor_recompute_maxdiff_max": stats(rows)["anchor_recompute_maxdiff_max"],
           "rotation_hist": dict(collections.Counter(str(r["rotation"]) for r in rows)),
           "orig_hw_hist_items": dict(collections.Counter(f"{r['orig_h']}x{r['orig_w']}" for r in rows).most_common(20)),
           "f_min": F.min(0).tolist() if len(F) else None, "f_max": F.max(0).tolist() if len(F) else None,
           "f_digest_blake2b16": hashlib.blake2b(F.tobytes(), digest_size=16).hexdigest(),
           "missing_items": [r["item_key"] for r in rows if r["frame0_ok"] is None],
           "missing_problems": sorted({r["problem"] for r in rows if r["frame0_ok"] is None})[:50],
           "frame0_ok_false_items": [[r["item_key"], r["problem"]] for r in rows if r["frame0_ok"] is False],
           "sources_md5": {p: md5(p) for p in (DATA_PPM, FRESH, SAM, BUILDER, DATA_ALOE, PLAN_CHECK)},
           "fresh_block_lines": FRESH_INFO["fresh_block_lines"], "fresh_block_md5": FRESH_INFO["fresh_block_md5"],
           "wall_s": round(time.time() - t0, 1), "host": os.uname().nodename}
    big = [[r["item_key"], r["anchor_recompute_maxdiff"]] for r in rows if r["frame0_ok"] is not None and
           not (r["anchor_recompute_maxdiff"] is not None and r["anchor_recompute_maxdiff"] <= 1e-4)]   # > 1e-4 or NaN
    cov["anchor_diff_gt_1e-4_items"] = big
    cov["c0_anchor_assert_1e-4"] = not any(k.split("|")[0].endswith("/c00") for k, _ in big)
    import subprocess
    cov["built_pt"] = subprocess.run(["date", "+%F %H:%M %Z"], env={"TZ": "America/Los_Angeles"}, capture_output=True, text=True).stdout.strip()
    json.dump(cov, open(f"{a.out}/COVERAGE.json", "w"), indent=1)
    print("[coverage]", json.dumps({k: cov[k] for k in ("by_split", "by_split_chunk", "c0_anchor_recompute_maxdiff_max",
                                                        "all_anchor_recompute_maxdiff_max", "rotation_hist", "tracks_tokens_verified_items",
                                                        "f_digest_blake2b16", "wall_s")}), flush=True)
    print(f"[done] {len(rows)} items -> {a.out} ({time.time() - t0:.0f}s)", flush=True)
    n_bad = len(cov["missing_items"])
    if n_bad:
        print(f"[FAIL-LOUD] {n_bad} items have NO frame-0 source (COVERAGE.json missing_items)", file=sys.stderr, flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
