"""pose6 A2: G0COND frame-0 features f for INFERENCE, from the playground's opening / prefill sources (contract: pose6/briefs/CONTRACT.md).

    g0_from_opening(opening_dir, kind) -> np.float32[8]       kind in KINDS = ("openings", "openings-sam", "clip")
    g0_record(opening_dir, kind)       -> (clip record {frame0_root, frame0_width, orig_h, orig_w, frame0_ok}, provenance dict)

f = data_ppm.g0_feat(record): the TRAINING function, byte-copied below and asserted byte-equal to
pose5/code-811-s2p/data_ppm.py's at import (G0SRC_TRAIN_DATA_PPM overrides the path). The record is built with the TRAINING builder's frame-0
formula (fresh_pose.decode_new, code-811-s2p/fresh/fresh_pose.py:131-141 == build_pose_rows_sam.py:136-146):
    j55 = frame0_pose[JOINTS] / [orig_h, orig_w, 1] (float32; frame0_pose = COCO-WholeBody 133 x (y, x, conf) in source pixels)
    root = (j55[5, :2] + j55[6, :2]) / 2 (shoulder midpoint, y/H, x/W), width = |j55[5, 1] - j55[6, 1]| (horizontal shoulder width / W)
    frame0_ok = finite, 133 joints, H > 0, W > 0, width > 0 and the source's stored normalised frame-0 state == (root, width) at atol 1e-4
whenever the source has frame-0 pixels; otherwise the source's stored normalised values (only the v2pure 'openings' fields, which ARE
that builder's output). Fail loud: any missing field / file / ambiguous source -> G0SourceError. f = 0 only where training gives 0
(frame0_ok false; a WARNING goes to stderr, as the trainer's). Sources per kind: pose6/infer/G0SRC.md.

usage: python -I g0src.py --coverage [--json OUT]    (every opening of every kind; proofs + coverage table)
"""
import ast
import functools
import glob
import inspect
import json
import os
import sys

import numpy as np

TRAIN_DATA_PPM = os.environ.get("G0SRC_TRAIN_DATA_PPM", "/data/home/amartyavarma/ppm/pose5/code-811-s2p/data_ppm.py")
GENWEB = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb"
PROMPTS = "/data/home/amartyavarma/video/demo-data/prompts"
V2PURE_ITEMS = "/data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure/*/items.parquet"   # Aloe pose release v2 (orig_h / orig_w per item)
FT_STORES = "/data/home/amartyavarma/ppm/aloe-pose-ft/stores"                               # the handover pose stores (mono8020 / Rosie FT data)
KINDS = ("openings", "openings-sam", "clip")
JOINTS = list(range(13)) + list(range(91, 133))   # build_pose_rows_sam.JOINTS: the pose tokenizer's 55 joints of COCO-WholeBody 133
ATOL_OK = 1e-4                                     # the builders' frame0_ok tolerance (stored state vs derived)
NORM_RECIPE = "frame0-root-frame0-horizontal-shoulder-width-v1"

G0_DIM = 8
G0_FIELDS = ("frame0_root", "frame0_width", "orig_h", "orig_w", "frame0_ok")


class G0SourceError(RuntimeError):
    pass


# ---------------------------------------------------------------------------- byte copy of the training function (asserted below)
def g0_feat(clip):
    """G0COND: one pose clip's frame-0 conditioning features (computed in float64, stored float32) -> (f [G0_DIM], ok).
    r = frame0_root (y, x; image-normalised: y / orig_h, x / orig_w), w = frame0_width (horizontal shoulder width, x / orig_w), H, W =
    orig_h, orig_w (the source video's pixel size, parquet pose struct; every clip record of the fresh plans and of PPM's samdata stores):
      f = [r_y, r_x, log w, -r_y / w, (1 - r_y) / w, -r_x / w, (1 - r_x) / w, log(H / W)]
    (f[3:7] = the distances from the root to the top / bottom / left / right frame edges in shoulder widths = pose_s3q.global_features'
    edge term e; f[7] = the aspect ratio, log so portrait / landscape are symmetric). Invalid frame 0 (frame0_ok false, a non-finite value,
    w <= 0, H <= 0 or W <= 0) -> f = 0 (ok False): with g_proj bias-free the clip's SLOT input is then exactly today's."""
    try:
        ry, rx = (float(x) for x in clip["frame0_root"])
        w, H, W = float(clip["frame0_width"]), float(clip["orig_h"]), float(clip["orig_w"])
        ok = bool(clip.get("frame0_ok", True)) and all(np.isfinite([ry, rx, w, H, W])) and w > 0 and H > 0 and W > 0
    except (KeyError, TypeError, ValueError):
        ok = False
    if not ok:
        return np.zeros(G0_DIM, np.float32), False
    f = np.array([ry, rx, np.log(w), -ry / w, (1.0 - ry) / w, -rx / w, (1.0 - rx) / w, np.log(H / W)], np.float64)
    return f.astype(np.float32), True


def _assert_same_as_training(path=TRAIN_DATA_PPM):
    """g0_feat above == the training file's (byte for byte), and G0_DIM / G0_FIELDS equal; else raise."""
    src = open(path).read()
    tree = ast.parse(src)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "g0_feat"]
    if len(fn) != 1:
        raise G0SourceError(f"{path}: expected one g0_feat, found {len(fn)}")
    train_txt = ast.get_source_segment(src, fn[0])
    if train_txt != inspect.getsource(g0_feat).rstrip("\n"):
        raise G0SourceError(f"g0src.g0_feat differs from the training g0_feat in {path}")
    consts = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
              if isinstance(t, ast.Name) and t.id in ("G0_DIM", "G0_FIELDS")}
    if consts != {"G0_DIM": G0_DIM, "G0_FIELDS": G0_FIELDS}:
        raise G0SourceError(f"G0_DIM / G0_FIELDS differ from {path}: {consts}")
    return train_txt


_TRAIN_G0FEAT_TXT = _assert_same_as_training()


def _req(d, key, where):
    if not isinstance(d, dict) or key not in d or d[key] is None:
        raise G0SourceError(f"{where}: missing field '{key}'")
    return d[key]


def _jload(path):
    if not os.path.isfile(path):
        raise G0SourceError(f"missing file {path}")
    with open(path) as f:
        return json.load(f)


def _st_load(path):
    from safetensors.numpy import load_file
    if not os.path.isfile(path):
        raise G0SourceError(f"missing file {path}")
    return load_file(path)


# ---------------------------------------------------------------------------- the training builder's frame-0 record
def frame0_record(f0_px, H, W, state=None):
    """fresh_pose.decode_new's frame-0 fields (fresh/fresh_pose.py:131-141), with `state` = the source's stored normalised [root_y, root_x,
    width] in the role of global_state[0] (None = no stored state to check). Returns the clip-record fields g0_feat reads."""
    frame0_ok, root, width = False, np.zeros(2, np.float32), 0.0
    try:
        f0 = np.asarray(f0_px, np.float32).reshape(-1, 3)
        H, W = float(H), float(W)
        gs = None if state is None else np.asarray(state, np.float32).reshape(-1)
        if f0.shape == (133, 3) and np.isfinite(f0).all() and H > 0 and W > 0 and (gs is None or (gs.shape == (3,) and np.isfinite(gs).all())):
            j55 = f0[JOINTS] / np.array([H, W, 1.0], np.float32)
            root, width = (j55[5, :2] + j55[6, :2]) / 2, abs(j55[5, 1] - j55[6, 1])
            frame0_ok = bool(width > 0) and (gs is None or np.allclose(gs, [*root, width], atol=ATOL_OK))
    except (TypeError, ValueError):
        frame0_ok = False
    return {"frame0_root": [float(root[0]), float(root[1])], "frame0_width": float(width), "orig_h": float(H), "orig_w": float(W),
            "frame0_ok": bool(frame0_ok)}


# ---------------------------------------------------------------------------- orig_h / orig_w sources
@functools.lru_cache(maxsize=1)
def _v2pure_orig():
    """recording id -> {(orig_h, orig_w)} over every Aloe pose release v2 item (v2pure cache items.parquet; 370,978 items, ~1.4 s)."""
    import pyarrow.parquet as pq
    fs = sorted(glob.glob(V2PURE_ITEMS))
    if not fs:
        raise G0SourceError(f"no v2pure items at {V2PURE_ITEMS}")
    out = {}
    for f in fs:
        t = pq.read_table(f, columns=["rid", "orig_h", "orig_w"]).to_pydict()
        for r, h, w in zip(t["rid"], t["orig_h"], t["orig_w"]):
            out.setdefault(r, set()).add((h, w))
    return out


def v2pure_orig_hw(recording_id):
    s = _v2pure_orig().get(recording_id)
    if not s:
        raise G0SourceError(f"recording {recording_id}: not in the Aloe pose release v2 items ({V2PURE_ITEMS}) -> no orig_h / orig_w")
    if len(s) != 1:
        raise G0SourceError(f"recording {recording_id}: several orig sizes {sorted(s)}")
    (h, w), = s
    if h is None or w is None:
        raise G0SourceError(f"recording {recording_id}: orig_h / orig_w null in the v2 items")
    return float(h), float(w)


def handover_pose_json_record(pose_json, state_check=None, where=""):
    """aloe-handover-full w00.<SPEAKER>.pose.json (schema aloe-handover-pose-v1) -> the frame-0 record (pixels + orig from its
    'normalization' block; the stored root0_yx_normalized / shoulder_width0_normalized play global_state[0])."""
    pj = _jload(pose_json)
    if pj.get("status") != "ok":
        raise G0SourceError(f"{pose_json}: status {pj.get('status')!r}")
    nm = _req(pj, "normalization", pose_json)
    if not str(_req(nm, "recipe", pose_json)).startswith(NORM_RECIPE):
        raise G0SourceError(f"{pose_json}: normalization recipe {nm['recipe']!r}")
    stored = [*(float(x) for x in _req(nm, "root0_yx_normalized", pose_json)), float(_req(nm, "shoulder_width0_normalized", pose_json))]
    rec = frame0_record(_req(nm, "frame0_pose_yxc_pixels", pose_json), _req(nm, "orig_h", pose_json), _req(nm, "orig_w", pose_json), stored)
    if state_check is not None and not np.allclose(np.asarray(state_check, np.float64), stored, atol=1e-6, rtol=0):
        raise G0SourceError(f"{where}: frame-0 state {list(state_check)} != {pose_json} normalization {stored}")
    return rec, {"pose_json": pose_json, "stored_state": stored, "rotation": nm.get("rotation"), "source_video": pj.get("source_video")}


@functools.lru_cache(maxsize=16)
def _store_rows(store):
    import pyarrow.parquet as pq
    p = f"{FT_STORES}/{store}/rows.parquet"
    if not os.path.isfile(p):
        raise G0SourceError(f"missing file {p}")
    return pq.read_table(p, columns=["row_key", "row", "src_dir", "user_speaker_dir", "assistant_speaker_dir"]).to_pydict()


def handover_row_lanes(row_key, store=None, row=None):
    """FT-store row -> (src_dir, {side: speaker_dir}); searches every store when `store` is None; exactly one match or raise."""
    hits = []
    for st in ([store] if store else sorted(os.path.basename(d) for d in glob.glob(f"{FT_STORES}/p*") if os.path.isdir(d))):
        t = _store_rows(st)
        for i, k in enumerate(t["row_key"]):
            if k == row_key and (row is None or int(t["row"][i]) == int(row)):
                hits.append((st, t["src_dir"][i], {"user": t["user_speaker_dir"][i], "assistant": t["assistant_speaker_dir"][i]}))
    if len(hits) != 1:
        raise G0SourceError(f"row {row_key} (store {store}, row {row}): {len(hits)} matches in {FT_STORES}")
    return hits[0]


# ---------------------------------------------------------------------------- per kind
def _openings(d):
    m = _jload(f"{d}/meta.json")
    where = f"{d}/meta.json"
    f0 = _req(_req(_req(m, "sources", where), "pose_tokens", where), "frame0", where)
    fl = f0.get("fields")
    if isinstance(fl, dict) and "orig_h" in fl:                         # v2pure_first_window: v2 items row (build_sam_cache_v2pure = same formula)
        for k in G0_FIELDS + ("frame0_joints55",):
            _req(fl, k, where + " frame0.fields")
        rec = {k: fl[k] for k in G0_FIELDS}
        j = np.asarray(fl["frame0_joints55"], np.float32).reshape(55, 3)   # recompute root / width from the stored normalised joints: same?
        root, width = (j[5, :2] + j[6, :2]) / 2, abs(j[5, 1] - j[6, 1])
        if not np.allclose([*root, width], [*fl["frame0_root"], fl["frame0_width"]], atol=1e-6, rtol=0):
            raise G0SourceError(f"{where}: frame0 fields != their own joints55")
        it = _v2_items_row(_req(f0, "items_parquet", where), _req(f0, "key", where))
        for k in G0_FIELDS:                                             # == the v2 items row the opening was cut from
            a, b = it[k], fl[k]
            if not (a == b if isinstance(a, bool) else np.allclose(np.asarray(a, np.float64), np.asarray(b, np.float64), atol=0, rtol=0)):
                raise G0SourceError(f"{where}: field {k} {b} != items row {a}")
        return rec, {"variant": "v2pure fields (= v2 items row)", "items_parquet": f0["items_parquet"], "key": f0["key"]}
    if "path" in f0:                                                    # pkg_chunk0: the avatar package's own chunk-0 frame 0 (llm_inputs)
        st = _st_load(f0["path"])
        for k in ("frame0_pose", "global_state"):
            _req(st, k, f0["path"])
        rid = _req(m, "recording_id", where)
        pm = _jload(os.path.join(os.path.dirname(f0["path"]), "meta.json"))
        if pm.get("recording_id") != rid:
            raise G0SourceError(f"{where}: recording {rid} != package {pm.get('recording_id')}")
        H, W = v2pure_orig_hw(rid)
        rec = frame0_record(st["frame0_pose"], H, W, st["global_state"].reshape(-1)[:3])
        return rec, {"variant": "package chunk-0 llm_inputs (frame0_pose px + global_state) + orig from v2 items of the recording",
                     "path": f0["path"], "recording_id": rid, "stored_state": [float(x) for x in st["global_state"].reshape(-1)[:3]]}
    if isinstance(fl, dict) and "row_key" in m:                         # Rosie: handover store row -> her lane's pose.json
        st_, src, lanes = handover_row_lanes(m["row_key"], _req(m, "store", where), _req(m, "row", where))
        side = _req(m, "side", where)
        state = [*(float(x) for x in _req(fl, "frame0_root", where)), float(_req(fl, "frame0_width", where))]
        rec, prov = handover_pose_json_record(f"{src}/w00.{lanes[side]}.pose.json", state, where)
        return rec, dict(prov, variant="handover store row -> pose.json", row_key=m["row_key"], store=st_, side=side)
    raise G0SourceError(f"{where}: frame0 has neither v2 fields with orig_h/orig_w, nor a package path, nor a handover row_key")


@functools.lru_cache(maxsize=64)
def _v2_items_parquet(path):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=["key", *G0_FIELDS]).to_pydict()


def _v2_items_row(path, key):
    if not os.path.isfile(path):
        raise G0SourceError(f"missing file {path}")
    t = _v2_items_parquet(path)
    ii = [i for i, k in enumerate(t["key"]) if k == key]
    if len(ii) != 1:
        raise G0SourceError(f"{path}: {len(ii)} rows with key {key}")
    return {k: t[k][ii[0]] for k in G0_FIELDS}


def _openings_sam(d):
    m = _jload(f"{d}/meta.json")
    where = f"{d}/meta.json"
    pt = _req(_req(m, "sources", where), "pose_tokens", where)
    fl = _req(_req(pt, "frame0", where), "fields", where)
    state = [*(float(x) for x in _req(fl, "frame0_root", where)), float(_req(fl, "frame0_width", where))]
    tok = _req(pt, "path", where)
    if not tok.endswith(".pose_tokens.npy"):
        raise G0SourceError(f"{where}: pose_tokens path {tok} is not a handover .pose_tokens.npy")
    rec, prov = handover_pose_json_record(tok[:-len(".pose_tokens.npy")] + ".pose.json", state, where)
    for k in ("orig_h", "orig_w"):
        if float(_req(fl, k, where)) != rec[k]:
            raise G0SourceError(f"{where}: {k} {fl[k]} != pose.json {rec[k]}")
    return rec, dict(prov, variant="handover pose.json (sibling of the pose_tokens.npy)")


def _clip(d):
    pt_path = f"{d}/gt/pose_tokens.safetensors"
    st = _st_load(pt_path)
    gs = _req(st, "global_state", pt_path).reshape(-1)[:3]
    m = _jload(f"{d}/meta.json")
    if "frame0_pose" in st:                                             # demo-data avatar package (v2 studio chunk store)
        rid = _req(m, "recording_id", f"{d}/meta.json")
        H, W = v2pure_orig_hw(rid)
        rec = frame0_record(st["frame0_pose"], H, W, gs)
        return rec, {"variant": "package gt/pose_tokens (frame0_pose px + global_state) + orig from v2 items of the recording",
                     "path": pt_path, "recording_id": rid, "stored_state": [float(x) for x in gs]}
    if "row_key" in m:                                                  # people_extra (Rosie): handover row; the lane whose pose.json == gs
        st_, src, lanes = handover_row_lanes(m["row_key"])
        hits = []
        for side, lane in sorted(lanes.items()):
            pj = f"{src}/w00.{lane}.pose.json"
            try:
                hits.append((side,) + handover_pose_json_record(pj, gs, pt_path))
            except G0SourceError:
                continue
        if len(hits) != 1:
            raise G0SourceError(f"{pt_path}: global_state {list(gs)} matches {len(hits)} lanes of {m['row_key']}")
        side, rec, prov = hits[0]
        return rec, dict(prov, variant="handover store row -> the lane whose pose.json state == global_state", row_key=m["row_key"],
                         store=st_, side=side)
    raise G0SourceError(f"{pt_path}: no frame0_pose and no handover row_key in {d}/meta.json -> no orig_h / orig_w source")


_BY_KIND = {"openings": _openings, "openings-sam": _openings_sam, "clip": _clip}


def g0_record(opening_dir, kind):
    if kind not in KINDS:
        raise G0SourceError(f"kind {kind!r} not in {KINDS}")
    d = os.path.abspath(str(opening_dir).rstrip("/"))
    if not os.path.isdir(d):
        raise G0SourceError(f"missing dir {d}")
    rec, prov = _BY_KIND[kind](d)
    for k in G0_FIELDS:
        _req(rec, k, f"{d} ({kind}) record")
    return rec, dict(prov, kind=kind, dir=d)


def g0_from_opening(opening_dir, kind):
    """-> np.float32[8] = training g0_feat of the opening's frame-0 record; raises G0SourceError on any missing field / source."""
    rec, prov = g0_record(opening_dir, kind)
    f, ok = g0_feat(rec)
    if not ok:
        print(f"[g0src] WARNING {prov['dir']} ({kind}): frame 0 invalid ({rec}) -> f = 0, as training", file=sys.stderr, flush=True)
    assert f.dtype == np.float32 and f.shape == (G0_DIM,)
    return f


# ---------------------------------------------------------------------------- coverage + proofs (python -I g0src.py --coverage)
def _all_sources():
    out = []
    for d in sorted(glob.glob(f"{GENWEB}/openings/*/")):
        if os.path.isfile(f"{d}meta.json") and os.path.isfile(f"{d}pose_tokens.safetensors"):
            out.append(("openings", d[:-1]))
    for d in sorted(glob.glob(f"{GENWEB}/openings-sam/*/")):
        if os.path.isfile(f"{d}meta.json") and os.path.isfile(f"{d}pose_tokens.safetensors"):
            out.append(("openings-sam", d[:-1]))
    for root in (PROMPTS, f"{GENWEB}/people_extra"):
        for d in sorted(glob.glob(f"{root}/*/")):
            if os.path.isfile(f"{d}gt/pose_tokens.safetensors"):
                out.append(("clip", d[:-1]))
    return out


def coverage():
    import time
    ns = {"np": np}
    exec(_TRAIN_G0FEAT_TXT, ns)                                         # the training text itself, run independently of the copy
    rows, fails = [], []
    for kind, d in _all_sources():
        t0 = time.time()
        try:
            rec, prov = g0_record(d, kind)
            f = g0_from_opening(d, kind)
            f_train, ok = ns["g0_feat"](rec)
            assert f.tobytes() == f_train.tobytes(), "copy != training text"
            st = prov.get("stored_state")
            drift = None if st is None else float(np.max(np.abs(np.asarray([*rec["frame0_root"], rec["frame0_width"]], np.float64) - st)))
            rows.append({"kind": kind, "id": os.path.basename(d), "ok": bool(ok), "f": [float(x) for x in f], "record": rec,
                         "max_abs_derived_minus_stored": drift, "variant": prov.get("variant"), "src": prov.get("pose_json") or
                         prov.get("path") or prov.get("items_parquet"), "ms": round(1e3 * (time.time() - t0), 1)})
        except G0SourceError as e:
            fails.append({"kind": kind, "id": os.path.basename(d), "error": str(e)})
    return rows, fails


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.coverage:
        rows, fails = coverage()
        for r in rows:
            print(f"{r['kind']:13s} {r['id']:32s} ok={int(r['ok'])} H,W={r['record']['orig_h']:.0f},{r['record']['orig_w']:.0f} "
                  f"f=[{', '.join(f'{x:.5f}' for x in r['f'])}] drift={r['max_abs_derived_minus_stored']} {r['ms']} ms  {r['variant']}")
        for x in fails:
            print("FAIL", x)
        print(f"TOTAL {len(rows)} ok, {len(fails)} failed; training g0_feat text == copy: yes ({TRAIN_DATA_PPM})")
        if a.json:
            json.dump({"rows": rows, "fails": fails}, open(a.json, "w"), indent=1)
