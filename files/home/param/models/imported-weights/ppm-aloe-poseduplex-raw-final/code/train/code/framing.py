"""FRAMING_COND (10-29, agent framingprep for main -1e; prep only, nothing launched). Shared, byte-identical copy in every trainer code dir
that has the switch (pose6/code_bandmask, poseduplex/code_bandmask, aloe-v2pure-main-17850/code_framing) and in the inference patch dir.

WHY   posev3 pose tokens are body-relative: (pos - root0) / sw0, root0 = the frame-0 shoulder midpoint (y / H, x / W), sw0 = the frame-0
      horizontal shoulder width / W (pose.global_state). The model's only pose input is pose_emb(pdepth), so it never sees where the frame
      edges are, while hand visibility in the data is set by the frame edge (genweb/checks/hands_page).
WHAT  per pose TRACK (one store row side = one speaker's pose lane) the three framing numbers root_y, root_x, sw (the tokenizer's frame-0
      anchor, image-normalised exactly as the tokens are) -> features
        f_raw = [root_y, root_x, sw, top, bottom, left, right]   (+ log(H / W) only if ASPECT, see below)
        top = root_y / sw, bottom = (1 - root_y) / sw, left = root_x / sw, right = (1 - root_x) / sw
      = the four frame edges in TOKEN units: a decoded joint at rel (y, x) = (pos - root0) / sw0 lies inside the frame iff
      -top <= y <= bottom and -left <= x <= right (pose.decode / tokens_to_latents: pos = root0 + sw0 * rel, canvas = [0, 1) x [0, 1) of the
      source frame in (y / H, x / W); y and x are each normalised by their own image side, sw by W, so the edge formulas hold for any aspect).
      f = (f_raw - mean) / std (float64 maths, stored float32) with FIXED data stats from FRAMING_STATS.json (train pose tracks).
MODEL model_ppm FramingMLP: Linear(F -> 256, bias) -> SiLU -> Linear(256 -> 5120, bias), the LAST layer zero-initialised (weight + bias);
      its output for a track is added to the TRUNK input at every position that carries that track's pose codes (pdepth[p, 0] >= 0, i.e.
      exactly where pose_emb(pdepth) adds), next to pose_emb. Step 0 = the base model exactly. Towers untouched.
MICRO keys (only when FRAMING_COND=1): frpos [M] int64 (packed positions with pose), fridx [M] int64 (row of frf for each), frf [K, F] float32
      (one standardised feature row per packed item / track).
ASPECT the source aspect orig_h / orig_w: the 4th raw number only if it varies in the training data (FRAMING_STATS.json 'aspect': the stats
      builder decides from the data and records the histogram). Constant aspect -> not a feature (a constant has no information and std 0)."""
import hashlib
import json
import os

import numpy as np

SIDE_FORMAT = "pose-framing-side-v1"
STATS_FORMAT = "pose-framing-stats-v1"
RAW_COLS = ("root_y", "root_x", "sw", "orig_h", "orig_w")         # side array columns (float64 [R, 5]); NaN when there is no source
STATUS = {0: "ok", 1: "no pose on this side in the store", 2: "pose but no source pose.json / frame-0 record", 3: "pose but ambiguous source",
          4: "pose but invalid / inconsistent frame-0 values", 5: "pose but the store tokens != the source tokens"}
BASE_FEAT = ("root_y", "root_x", "sw", "top", "bottom", "left", "right")
HIDDEN_MID = 256


def feat_names(aspect):
    return BASE_FEAT + (("log_aspect_h_over_w",) if aspect else ())


def raw_features(root_y, root_x, sw, orig_h=None, orig_w=None, aspect=False):
    """-> float64 [F] unstandardised features (module doc). Raises on invalid values (never silent)."""
    ry, rx, w = float(root_y), float(root_x), float(sw)
    assert np.isfinite([ry, rx, w]).all() and w > 0, ("framing: invalid frame-0 values", ry, rx, w)
    f = [ry, rx, w, ry / w, (1.0 - ry) / w, rx / w, (1.0 - rx) / w]
    if aspect:
        H, W = float(orig_h), float(orig_w)
        assert np.isfinite([H, W]).all() and H > 0 and W > 0, ("framing: invalid frame size", H, W)
        f.append(np.log(H / W))
    return np.asarray(f, np.float64)


def standardise(f_raw, stats):
    """f_raw [.., F] float64 -> float32 (f_raw - mean) / std with the stats' FIXED vectors."""
    mu, sd = np.asarray(stats["mean"], np.float64), np.asarray(stats["std"], np.float64)
    assert np.asarray(f_raw).shape[-1] == len(mu) == len(sd) and (sd > 0).all(), ("framing stats / feature width", np.asarray(f_raw).shape, len(mu))
    return ((np.asarray(f_raw, np.float64) - mu) / sd).astype(np.float32)


def load_stats(path, sha=None):
    """FRAMING_STATS.json -> dict; format / names / widths asserted; sha256 of the file asserted when given."""
    raw = open(path, "rb").read()
    if sha:
        assert hashlib.sha256(raw).hexdigest() == sha, ("FRAMING_STATS.json sha256 != FRAMING_STATS_SHA", path)
    s = json.loads(raw)
    assert s["format"] == STATS_FORMAT, ("framing stats format", path, s.get("format"))
    assert tuple(s["feat"]) == feat_names(bool(s["aspect"])) and len(s["mean"]) == len(s["std"]) == len(s["feat"]), ("framing stats names", path)
    s["sha256"] = hashlib.sha256(raw).hexdigest()
    s["path"] = os.path.abspath(path)
    return s


def track_features(rec, stats):
    """one side-array record (RAW_COLS values) -> standardised float32 [F]."""
    ry, rx, w, H, W = (float(x) for x in rec)
    return standardise(raw_features(ry, rx, w, H, W, bool(stats["aspect"])), stats)


class SideStore:
    """one store's framing side arrays: <root>/<store>/FRAMING.json + framing_<side>.npy float64 [R, 5] + framing_status_<side>.npy int8 [R].
    FRAMING.json format / dataset / store / store digest (== the run's store) / rows / sha256 of every array asserted at open."""

    def __init__(self, root, store, dataset, store_digest, rows=None):
        d = f"{root}/{store}"
        meta = json.load(open(f"{d}/FRAMING.json"))
        assert meta["format"] == SIDE_FORMAT and meta["dataset"] == dataset and meta["store"] == store, ("framing side meta", d, meta.get("format"))
        assert meta["store_digest"] == store_digest, ("framing side arrays built for another store", d, meta["store_digest"], store_digest)
        assert tuple(meta["cols"]) == RAW_COLS, ("framing side columns", d)
        self.rows = int(meta["rows"])
        assert rows is None or int(rows) == self.rows, ("framing side rows != store rows", d, self.rows, rows)
        self.v, self.s = {}, {}
        for sd in ("user", "assistant"):
            for kind, f in (("v", f"{d}/framing_{sd}.npy"), ("s", f"{d}/framing_status_{sd}.npy")):
                assert hashlib.sha256(open(f, "rb").read()).hexdigest() == meta["sha256"][os.path.basename(f)], ("framing side sha256", f)
                a = np.load(f)
                if kind == "v":
                    assert a.dtype == np.float64 and a.shape == (self.rows, len(RAW_COLS)), ("framing side array", f, a.dtype, a.shape)
                    self.v[sd] = a
                else:
                    assert a.dtype == np.int8 and a.shape == (self.rows,), ("framing status array", f, a.dtype, a.shape)
                    self.s[sd] = a
        self.meta = meta

    def get(self, r, side):
        """-> (RAW_COLS record float64 [5], status int); the caller decides what a non-zero status means (fail loud on a pose track)."""
        return self.v[side][int(r)], int(self.s[side][int(r)])


def pose_positions(pdepth):
    """the positions where pose_emb adds (pdepth[:, 0] >= 0) -> int64 [n]."""
    return np.flatnonzero(np.asarray(pdepth)[:, 0] >= 0).astype(np.int64)


def micro_arrays(pdepths, feats, F):
    """packed items -> (frpos [M] int64, fridx [M] int64, frf [K, F] float32). pdepths: each item's pdepth [T_k, 32] in pack order (offsets =
    cumulative T_k, = data_qqa.pack's segment offsets); feats: each item's standardised [F] float32 or None (no pose track: frf row of zeros
    and no positions; asserted: such an item carries no pose)."""
    pos, idx, ff, o = [], [], [], 0
    for k, (pd, f) in enumerate(zip(pdepths, feats)):
        p = pose_positions(pd)
        if f is None:
            assert len(p) == 0, ("framing: an item with pose positions but no framing features", k, len(p))
            f = np.zeros(F, np.float32)
        f = np.asarray(f, np.float32)
        assert f.shape == (F,) and np.isfinite(f).all(), ("framing features", k, f.shape)
        pos.append(p + o)
        idx.append(np.full(len(p), k, np.int64))
        ff.append(f)
        o += int(len(pd))
    return (np.concatenate(pos).astype(np.int64) if pos else np.zeros(0, np.int64),
            np.concatenate(idx).astype(np.int64) if idx else np.zeros(0, np.int64), np.stack(ff).astype(np.float32))


def cfg_view(stats, side_root, dataset):
    """the resolved config block CFG['framing_cond'] (only when FRAMING_COND=1): everything inference needs to rebuild f exactly."""
    return {"on": 1, "feat": list(stats["feat"]), "aspect": int(bool(stats["aspect"])), "mean": [float(x) for x in stats["mean"]],
            "std": [float(x) for x in stats["std"]], "stats": stats["path"], "stats_sha256": stats["sha256"], "side_root": side_root,
            "dataset": dataset, "mlp": f"Linear({len(stats['feat'])}->{HIDDEN_MID}) SiLU Linear({HIDDEN_MID}->hidden), last layer zero-init",
            "where": "trunk input at every position with pdepth[:, 0] >= 0 (next to pose_emb), the track's own framing"}
