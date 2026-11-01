"""Local HF video scanning and strict nine-slot reference-bank selection.

The measurement bundle comes from ``measurements.initialize_measurements`` and
contains the archived audit/audit59/teeth/allteeth/pscan modules. This module
does no downloading and never changes archived code or existing bank data.

Track dictionaries use: track, uuid, bank, recording_id, speaker, session_id,
video_path, source (the HF video URI), usable_chunks (integer 300-second chunk
indices), and excluded_ranges (inclusive seconds). Optional allowed_ranges
limits timestamps further. Every returned source retains its exact frame index.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd

from upstream import checked_thumbnail

FPS, CHUNK_FRAMES, STEP = 25, 7500, 12
W, H, CW, CH = 512, 960, 256, 480
CAP, GAP_SECONDS, THUMB_MAD = 20, 10.0, 6.0
POSE_PREFIX = "conv-4.3k-5mins-muon-neutrino-256x480-v1"
SLOTS = ("front", "face_image_left", "face_image_right", "mouth_open",
         "hands_visible", "hands_interacting", "hair_clothing_01",
         "hair_clothing_02", "hair_clothing_03")


def contact(keypoints: np.ndarray) -> bool:
    """Archived v8 strict contact, on the native 256x480 pose canvas."""
    a, b = keypoints[91:112], keypoints[112:133]

    def observed(hand):
        mask = ((hand[:, 2] > 0.3) & (hand[:, 0] >= 0) &
                (hand[:, 0] < 256) & (hand[:, 1] >= 0) & (hand[:, 1] < 480))
        return mask, mask.sum() >= 12 and hand[:, 2].mean() > 0.3

    ca, oa = observed(a)
    cb, ob = observed(b)
    if not (oa and ob):
        return False
    palm = (np.linalg.norm(a[5, :2] - a[17, :2]) +
            np.linalg.norm(b[5, :2] - b[17, :2])) / 2
    tolerance = min(3.0, 0.12 * palm)
    distances = np.linalg.norm(a[ca, None, :2] - b[None, cb, :2], axis=2)
    return bool((distances.min(1) <= tolerance).sum() >= 2 and
                (distances.min(0) <= tolerance).sum() >= 2)


def contact_gate(keypoints: np.ndarray, valid: np.ndarray, index: int) -> bool:
    """Any three consecutive contact frames within i-2..i+2, as archived."""
    flags = [0 <= j < len(keypoints) and bool(valid[j]) and contact(keypoints[j])
             for j in range(index - 2, index + 3)]
    return any(all(flags[start:start + 3]) for start in range(3))


def measure_video_frame(image: np.ndarray, keypoints: np.ndarray, modules,
                        gate: bool = False) -> dict[str, Any]:
    """Exact archived face/hands/teeth measurements and permissive prefilters.

    Keypoints here have already been scaled to the 512x960 RGB canvas. Final
    strict slot predicates are applied by score_candidates, not these prefilters.
    """
    row = dict(modules.audit.measure(image, keypoints))
    row["gate"] = bool(gate)
    row["kp"] = keypoints.reshape(-1).tolist()
    if not row.get("mp_ok", False):
        return row
    if not (row["blinkL"] > 0.5 and row["blinkR"] > 0.5) and row["blur"] >= 40:
        _measure_hands(image, keypoints, row, modules.audit59)
    if (abs(row["yaw"]) <= 20 and row["blur"] >= 40 and
            row["hand_in_face"] < 3 and row["oof"] == 0 and row["gap"] >= 0.02):
        side = 1.25 * max(row["bx1"] - row["bx0"], row["by1"] - row["by0"])
        cx, cy = (row["bx0"] + row["bx1"]) / 2, (row["by0"] + row["by1"]) / 2
        crop = cv2.resize(modules.pscan.crop(image, cx, cy, side)[0],
                          (288, 288), interpolation=cv2.INTER_CUBIC)
        landmarks = modules.allteeth.landmarks(crop)
        if landmarks is not None:
            try:
                row.update(t_ok=True, **modules.allteeth.rows(crop, landmarks))
            except (ValueError, IndexError, ZeroDivisionError, cv2.error) as exc:
                # Preserve archived fail-closed behaviour, but retain the error.
                row["t_ok"] = False
                row["teeth_error"] = f"{type(exc).__name__}: {exc}"
    row["thumb"] = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
                             (32, 60), interpolation=cv2.INTER_AREA).reshape(-1).tolist()
    return row


def _measure_hands(image, keypoints, row, audit59):
    hands = audit59.detect_hands(image)
    row["n_hands"] = len(hands)
    face_box = (row["bx0"], row["by0"], row["bx1"], row["by1"])
    regions = audit59.regions(face_box, keypoints)
    for index, (_, points) in enumerate(hands):
        in_frame = ((points[:, 0] >= 0) & (points[:, 0] < W) &
                    (points[:, 1] >= 0) & (points[:, 1] < H))
        row.update({f"h{index}_inframe": int(in_frame.sum()),
                    f"h{index}_palm": float(np.linalg.norm(points[5] - points[17])),
                    f"h{index}_ext": audit59.n_ext(points),
                    f"h{index}_sharp": audit59.sharp(image, points),
                    f"h{index}_face_ov": float(audit59.inside(points, [face_box]).mean())})
        counts = {k: int(audit59.inside(points, boxes).sum()) for k, boxes in regions.items()}
        tips = points[[4, 8, 12, 16, 20]]
        tip_counts = {k: int(audit59.inside(tips, boxes).sum()) for k, boxes in regions.items()}
        best = max(counts, key=lambda k: (counts[k], tip_counts[k]))
        row[f"h{index}_target"] = best if counts[best] >= 4 else "none"
    if len(hands) == 2:
        a, b = hands[0][1], hands[1][1]
        row["hh_dist"] = float(np.linalg.norm(a[:, None] - b[None], axis=2).min()) / max(
            1e-3, (row["h0_palm"] + row["h1_palm"]) / 2)


def load_pose_chunk(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read only known safe numeric safetensors types; no pickle/torch needed."""
    data = Path(path).read_bytes()
    header_length = int.from_bytes(data[:8], "little")
    if not 2 <= header_length <= 1024 * 1024 or 8 + header_length > len(data):
        raise ValueError(f"Invalid safetensors header: {path}")
    header = json.loads(data[8:8 + header_length])
    dtypes = {"F32": np.dtype("<f4"), "BOOL": np.dtype("?")}
    arrays = {}
    for key in ("keypoints", "valid"):
        entry = header[key]
        dtype = dtypes[entry["dtype"]]
        start, end = entry["data_offsets"]
        shape = entry["shape"]
        if start < 0 or end > len(data) - 8 - header_length or end - start != int(np.prod(shape)) * dtype.itemsize:
            raise ValueError(f"Invalid {key} tensor offsets: {path}")
        arrays[key] = np.frombuffer(data, dtype=dtype, count=int(np.prod(shape)),
                                   offset=8 + header_length + start).reshape(shape)
    keypoints, valid = arrays["keypoints"], arrays["valid"]
    if keypoints.ndim != 3 or keypoints.shape[1:] != (133, 3) or valid.shape != (len(keypoints),):
        raise ValueError(f"Expected [n,133,3] keypoints and [n] valid: {path}")
    if len(keypoints) > CHUNK_FRAMES:
        raise ValueError(f"Pose chunk exceeds 300 seconds: {path}")
    return keypoints, valid


def timestamp_allowed(track: Mapping[str, Any], seconds: float) -> bool:
    if any(float(start) <= seconds <= float(end) for start, end in track.get("excluded_ranges", ())):
        return False
    allowed = track.get("allowed_ranges")
    return allowed is None or any(float(start) <= seconds <= float(end) for start, end in allowed)


def attach_corpus_exclusions(track: Mapping[str, Any], corpus) -> dict:
    """Attach the same exclusion/allow-list windows used for upstream images."""
    result = dict(track)
    name = result["track"]
    result["excluded_ranges"] = sorted(set(
        interval for ranges in (corpus.ranges_eval, corpus.ranges_blackbands, corpus.ranges_lighting)
        for interval in ranges.get(name, ())))
    if result.get("cohort") == "eval-unseen":
        result["allowed_ranges"] = corpus.ranges_excess.get(name, [])
    return result


def validate_video(path: str | Path) -> dict:
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
                             "-of", "json", str(path)], capture_output=True, text=True, check=True)
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
    if (stream["width"], stream["height"]) != (W, H) or numerator != FPS * denominator:
        raise ValueError(f"Expected 512x960 at 25fps: {path}: {stream}")
    return stream


def _decode_frames(video_path: str | Path, chunk: int, expression: str):
    """Yield selected RGB frames with bounded memory and surfaced ffmpeg errors."""
    command = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-ss", str(300 * chunk),
               "-t", "300", "-i", str(video_path), "-vf", f"select='{expression}'",
               "-vsync", "0", "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            while True:
                data = process.stdout.read(W * H * 3)
                if not data:
                    break
                if len(data) != W * H * 3:
                    raise RuntimeError(f"Truncated RGB frame from {video_path}, chunk {chunk}")
                yield np.frombuffer(data, np.uint8).reshape(H, W, 3)
            status = process.wait()
            if status:
                errors.seek(0)
                raise RuntimeError(f"ffmpeg failed ({status}) for {video_path}: {errors.read().decode(errors='replace')}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()


def scan_chunk(track: Mapping[str, Any], chunk: int, pose_path: str | Path,
               modules) -> pd.DataFrame:
    """Measure every twelfth local frame of one allowed 300-second chunk."""
    chunk = int(chunk)
    if chunk not in {int(k) for k in track["usable_chunks"]}:
        raise ValueError(f"Chunk {chunk} is not allowed for {track['track']}")
    if "excluded_ranges" not in track:
        raise ValueError("Use attach_corpus_exclusions before scanning so eval margins are enforced")
    if str(track["source"]).startswith("gs://"):
        raise ValueError("Fresh scan provenance must identify the HF source, not a GCS URI")
    keypoints, valid = load_pose_chunk(pose_path)
    rows = []
    for count, image in enumerate(_decode_frames(track["video_path"], chunk, f"not(mod(n\\,{STEP}))")):
        local_index = count * STEP
        frame_index = chunk * CHUNK_FRAMES + local_index
        seconds = frame_index / FPS
        if (local_index >= len(keypoints) or not valid[local_index] or
                not timestamp_allowed(track, seconds)):
            continue
        points = keypoints[local_index] * np.array([2, 2, 1], np.float32)
        if not np.isfinite(points).all():
            continue
        row = {key: track[key] for key in ("track", "uuid", "recording_id", "speaker", "session_id")}
        row.update(bank=track.get("bank", "bank"), cohort=track.get("cohort", ""),
                   chunk=chunk, frame_index=frame_index, time_s=seconds, pose_valid=True,
                   source=f"{str(track['source']).split('#frame=')[0]}#frame={frame_index}",
                   candidate_id=f"{track['track']}#frame={frame_index}",
                   in_eval=False, in_bb=False, in_light=False)
        row.update(measure_video_frame(image, points, modules,
                                       gate=contact_gate(keypoints, valid, local_index)))
        rows.append(row)
    return pd.DataFrame(rows)


def scan_track(track: Mapping[str, Any], pose_paths: Mapping[int, str | Path],
               modules, output_path: str | Path | None = None) -> pd.DataFrame:
    validate_video(track["video_path"])
    pose_paths = {int(chunk): path for chunk, path in pose_paths.items()}
    chunks = []
    for chunk in sorted(map(int, track["usable_chunks"])):
        if chunk not in pose_paths:
            raise FileNotFoundError(f"Missing HF pose chunk {track['track']}/{chunk:06d}")
        chunks.append(scan_chunk(track, chunk, pose_paths[chunk], modules))
    result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        result.to_parquet(temporary, index=False)
        temporary.replace(output_path)
    return result


def choose_scan_tracks(tracks, upstream_selected: pd.DataFrame,
                       max_tracks: int | None = 2) -> list[dict]:
    """Rank tracks of actors with an empty slot by usable chunks, then track.

    ``None`` returns the full ranking so the HF planner can resolve sources
    before imposing the two-track limit, as the archived target planner does.
    """
    if max_tracks is not None and max_tracks < 1:
        raise ValueError("max_tracks must be positive or None")
    if isinstance(tracks, pd.DataFrame):
        tracks = tracks.to_dict("records")
    present = defaultdict(set)
    for row in upstream_selected.to_dict("records"):
        present[(row.get("bank", "bank"), row["uuid"])].add(int(row["property"]))
    grouped = defaultdict(list)
    for row in tracks:
        row = dict(row)
        chunks = row["usable_chunks"]
        if isinstance(chunks, str):
            chunks = json.loads(chunks)
        row["usable_chunks"] = sorted(set(map(int, chunks)))
        if row["usable_chunks"] and len(present[(row.get("bank", "bank"), row["uuid"])]) < 9:
            grouped[(row.get("bank", "bank"), row["uuid"])].append(row)
    return [row for key in sorted(grouped) for row in
            sorted(grouped[key], key=lambda row: (-len(row["usable_chunks"]), row["track"]))[:max_tracks]]


def build_scan_download_plan(chosen_tracks: list[dict], video_index: Mapping,
                             source_root: str | Path, corpus,
                             bucket: str = "InternalCan/aloe-conversationa-data") -> tuple[list[dict], list[tuple[str, Path]]]:
    """Resolve chosen tracks into exact HF video/pose keys and local destinations.

    The returned pairs can be passed directly to hf_sources.download_pairs.
    Missing or ambiguous video entries fail before any download; receipts and
    content verification remain the source downloader's responsibility.
    """
    try:
        from .hf_sources import resolve_video
    except ImportError:
        from hf_sources import resolve_video
    source_root = Path(source_root)
    planned, downloads = [], []
    for original in chosen_tracks:
        track = attach_corpus_exclusions(original, corpus)
        video = resolve_video(track["recording_id"], track["speaker"], video_index,
                              original_blob=track.get("original_blob"))
        name = track["track"]
        video_path = source_root / "videos" / name / "full.mp4"
        track["video_path"] = str(video_path)
        track["source"] = f"hf://buckets/{bucket}/{video['path']}"
        track["video_source_metadata"] = dict(video)
        track["pose_paths"] = {}
        track["pose_sources"] = {}
        downloads.append((video["path"], video_path))
        for chunk in sorted(map(int, track["usable_chunks"])):
            suffix = f"recordings/{name}/chunks/{chunk:06d}/poses.safetensors"
            key = f"{POSE_PREFIX}/{suffix}"
            path = source_root / "poses" / suffix
            track["pose_paths"][chunk] = str(path)
            track["pose_sources"][chunk] = f"hf://buckets/{bucket}/{key}"
            downloads.append((key, path))
        planned.append(track)
    return planned, downloads


def selector_features(keypoints):
    face = keypoints[23:91]
    nose, eye1, eye2, jaw0, jaw16 = face[30], face[36], face[45], face[0], face[16]

    def asymmetry(a, b):
        a, b = (a, b) if a[0] < b[0] else (b, a)
        da, db = abs(nose[0] - a[0]), abs(b[0] - nose[0])
        return (da - db) / (da + db + 1e-6)

    body, left_eye, right_eye = keypoints[0], keypoints[1], keypoints[2]
    return [asymmetry(eye1, eye2), asymmetry(jaw0, jaw16),
            (nose[0] - (jaw0[0] + jaw16[0]) / 2) / (abs(jaw16[0] - jaw0[0]) + 1e-6),
            (nose[0] - (eye1[0] + eye2[0]) / 2) / (abs(eye2[0] - eye1[0]) + 1e-6),
            (body[0] - (left_eye[0] + right_eye[0]) / 2) / (abs(left_eye[0] - right_eye[0]) + 1e-6),
            face[:, 2].mean(), min(nose[2], eye1[2], eye2[2])]


def _design(features, sign):
    features = np.asarray(features, float)
    directional = sign * features[:, :5]
    return np.c_[directional, features[:, 5:], directional ** 2, np.ones(len(features))]


def fit_selector_proxy(fit_score_path: str | Path) -> np.ndarray:
    fit = pd.read_parquet(fit_score_path)
    fit = fit[fit.cat.isin(["face_image_left", "face_image_right"])].dropna()
    signs = np.where(fit.cat == "face_image_left", -1, 1)[:, None]
    features = fit[["eye", "jaw", "jaw_c", "eye_c", "body", "conf", "minc"]].values
    return np.linalg.lstsq(_design(features, signs), fit.score.values, rcond=None)[0]


def face_crop_box(keypoints) -> tuple[int, int, int, int] | None:
    face = keypoints[23:91]
    face = face[face[:, 2] > 0.3]
    if len(face) < 10:
        return None
    x0, y0 = face[:, :2].min(0)
    x1, y1 = face[:, :2].max(0)
    y0 -= 0.35 * (y1 - y0)
    scale = max(1.0, (x1 - x0) / CW, (y1 - y0) / CH)
    if scale > 1.1:
        return None
    width = int(np.ceil(CW * scale - 1e-9))
    for width in (width, width + 2):
        width = min(width, 282)
        height = int(round(width * CH / CW))
        left = int(round(min(max((x0 + x1) / 2 - width / 2, 0), W - width)))
        top = int(round(min(max((y0 + y1) / 2 - height / 2, 0), H - height)))
        if x0 >= left and y0 >= top and x1 <= left + width and y1 <= top + height:
            return left, top, width, height
    return None


def score_candidates(candidates: pd.DataFrame, fit_score_path: str | Path | None = None,
                     proxy_weights: np.ndarray | None = None) -> pd.DataFrame:
    """Apply all current strict sinks.md rules; missing measurements fail closed."""
    frames = candidates.reset_index(drop=True).copy()
    if frames.empty:
        return frames
    if frames.duplicated(["track", "frame_index"]).any():
        raise ValueError("Duplicate candidate track/frame pairs")
    points = np.stack([np.asarray(k, np.float32).reshape(133, 3) for k in frames.kp])
    if proxy_weights is None:
        if fit_score_path is None:
            raise ValueError("fit_score_path or proxy_weights is required")
        proxy_weights = fit_selector_proxy(fit_score_path)
    features = np.array([selector_features(k) for k in points])
    frames["score2"] = _design(features, -1) @ proxy_weights
    frames["score3"] = _design(features, 1) @ proxy_weights

    def number(name, default=np.nan):
        return pd.to_numeric(frames.get(name, pd.Series(default, index=frames.index)), errors="coerce").fillna(default)

    def flag(name, default=False):
        return frames.get(name, pd.Series(default, index=frames.index)).fillna(default).astype(bool)

    def target(index):
        return frames.get(f"h{index}_target", pd.Series("none", index=frames.index))

    valid = flag("mp_ok") & flag("pose_valid", True)
    for name in ("in_eval", "in_bb", "in_light"):
        valid &= ~flag(name)
    eyes = ~((number("blinkL") > 0.5) & (number("blinkR") > 0.5))
    eyes_full = ~((number("blinkL") > 0.6) & (number("blinkR") > 0.6))
    eyes &= number("blinkL").notna() & number("blinkR").notna()
    eyes_full &= number("blinkL").notna() & number("blinkR").notna()
    sharp = number("blur") >= 60
    hand = number("hand_in_face") == 0
    cut = ((number("oof") == 0) & (number("bx0") >= 2) & (number("by0") >= 2) &
           (number("bx1") <= W - 2) & (number("by1") <= H - 2))
    face_ok = valid & eyes & sharp
    two = number("n_hands") == 2
    inframe = (number("h0_inframe", 0) >= 20) & (number("h1_inframe", 0) >= 20)
    no_face_hand = ~((number("h0_face_ov", 0) > 0.3) | (number("h1_face_ov", 0) > 0.3))
    hand0, hand1 = number("h0_sharp", 0), number("h1_sharp", 0)
    sharp_hand = (hand0 >= 80) | (hand1 >= 80)
    clothing = target(0).eq("clothing") | target(1).eq("clothing")
    hair = target(0).eq("hair") | target(1).eq("hair")
    predicates = {
        1: valid & (number("yaw").abs() < 10) & (pd.concat([number("blinkL"), number("blinkR")], axis=1).max(axis=1) < 0.35) & eyes & hand & sharp & cut,
        2: valid & (frames.score2 >= 0.3) & number("yaw").between(-50, -20) & eyes & hand & sharp & cut,
        3: valid & (frames.score3 >= 0.3) & number("yaw").between(20, 50) & eyes & hand & sharp & cut,
        4: valid & flag("t_ok") & (number("up") >= 1) & (number("lo") >= 1) & (number("span") >= 0.4) & eyes_full & hand & sharp & (number("yaw").abs() <= 15) & (number("oof") == 0),
        5: face_ok & two & inframe & (hand0 >= 80) & (hand1 >= 80) & (number("h0_ext", 0) >= 3) & (number("h1_ext", 0) >= 3) & (number("hh_dist", 0) > 0.15) & no_face_hand,
        6: face_ok & flag("gate") & two & inframe & (np.minimum(hand0, hand1) >= 60) & no_face_hand,
        7: face_ok & hair & sharp_hand,
        8: face_ok & clothing & sharp_hand & no_face_hand,
        9: face_ok & clothing & sharp_hand & no_face_hand,
    }
    ranks = {1: number("blur"), 2: frames.score2, 3: frames.score3, 4: number("jaw"),
             5: np.minimum(hand0, hand1), **{p: np.maximum(hand0, hand1) for p in (6, 7, 8, 9)}}
    boxes = {}
    for index in frames.index:
        if any(predicates[prop].iloc[index] for prop in (1, 2, 3, 4)):
            boxes[index] = face_crop_box(points[index])
    frames["face_crop"] = [boxes.get(index) for index in frames.index]
    for prop in range(1, 10):
        frames[f"pass{prop}"] = predicates[prop] & (frames.face_crop.notna() if prop <= 4 else True)
        frames[f"rank{prop}"] = ranks[prop]
    return frames


def select_scanned(candidates: pd.DataFrame, upstream_selected: pd.DataFrame,
                   fit_score_path: str | Path | None = None,
                   proxy_weights: np.ndarray | None = None) -> pd.DataFrame:
    """Per-track greedy dedup then actor/slot round-robin append, cap 20.

    Ten-second track separation, actor-wide MAD<6 rejection against upstream
    and scan images, shared slots 7-9, and one assignment per scanned frame.
    Existing upstream indices are retained; returned rows contain only additions.
    iids are intentionally left to the caller's global deterministic final merge.
    """
    frames = score_candidates(candidates, fit_score_path, proxy_weights)
    if frames.empty:
        return pd.DataFrame()
    previous, upstream_thumbnails, upstream_hair = defaultdict(list), defaultdict(list), defaultdict(list)
    upstream_count = defaultdict(int)
    for row in upstream_selected.to_dict("records"):
        track = row.get("track", f"{row['recording_id']}/speaker_{int(row['speaker'])}")
        prop = int(row["property"])
        previous[(track, prop)].append(float(row["time_s"]))
        key = row.get("bank", "bank"), row["uuid"], prop
        upstream_count[key] += 1
        upstream_thumbnails[key].append(checked_thumbnail(row.get("thumb")))
        if prop >= 7:
            upstream_hair[key[:2]].append((row['recording_id'], float(row['time_s'])))
    if any(count > CAP for count in upstream_count.values()):
        raise ValueError("Upstream selection already exceeds per-slot cap")
    thumbnails = {}
    for index in frames.index[frames[[f"pass{p}" for p in range(1, 10)]].any(axis=1)]:
        thumbnails[index] = checked_thumbnail(frames.loc[index, "thumb"])
    lists = {}
    for track, group in frames.groupby("track", sort=True):
        used = set()
        accepted = {p: [] for p in range(1, 10)}
        for prop in range(1, 10):
            indices = group.index[group[f"pass{prop}"]]
            indices = frames.loc[indices].sort_values(f"rank{prop}", ascending=False, kind="stable", na_position="last").index
            siblings = (7, 8, 9) if prop >= 7 else (prop,)
            old = [time for sibling in siblings for time in previous[(track, sibling)]]
            for index in indices:
                if len(accepted[prop]) >= CAP:
                    break
                if index in used:
                    continue
                seconds = frames.loc[index, "time_s"]
                peers = [peer for sibling in siblings for peer in accepted[sibling]]
                if any(abs(seconds - frames.loc[peer, "time_s"]) < GAP_SECONDS for peer in peers) or any(abs(seconds - old_time) < GAP_SECONDS for old_time in old):
                    continue
                if any(np.abs(thumbnails[index] - thumbnails[peer]).mean() < THUMB_MAD for peer in peers):
                    continue
                actor = frames.loc[index]
                if prop >= 7 and any(recording == actor.recording_id and abs(seconds - old_time) < 2
                                     for recording, old_time in upstream_hair[(actor.get('bank', 'bank'), actor.uuid)]):
                    continue
                if any(np.abs(thumbnails[index] - old_thumbnail).mean() < THUMB_MAD
                       for sibling in siblings
                       for old_thumbnail in upstream_thumbnails[(actor.get('bank', 'bank'), actor.uuid, sibling)]):
                    continue
                accepted[prop].append(index)
                used.add(index)
            lists[(track, prop)] = accepted[prop]
    result = []
    if "bank" not in frames:
        frames["bank"] = "bank"
    for (bank, uuid), group in frames.groupby(["bank", "uuid"], sort=True):
        tracks = sorted(group.track.unique())
        accepted = defaultdict(list)
        for prop in range(1, 10):
            count = upstream_count[(bank, uuid, prop)]
            room = CAP - count
            remaining = [list(lists.get((track, prop), [])) for track in tracks]
            output = []
            while room > 0 and any(remaining):
                heads = [(float(frames.loc[items[0], f"rank{prop}"]), k) for k, items in enumerate(remaining) if items]
                heads = [(rank if np.isfinite(rank) else -np.inf, k) for rank, k in heads]
                for _, k in sorted(heads, reverse=True):
                    if room <= 0:
                        break
                    frame_id = remaining[k].pop(0)
                    siblings = (7, 8, 9) if prop >= 7 else (prop,)
                    if any((prop >= 7 and frames.loc[frame_id, 'recording_id'] == frames.loc[peer, 'recording_id']
                            and abs(frames.loc[frame_id, 'time_s'] - frames.loc[peer, 'time_s']) < 2)
                           or np.abs(thumbnails[frame_id] - thumbnails[peer]).mean() < THUMB_MAD
                           for sibling in siblings for peer in accepted[sibling]):
                        continue
                    output.append(frame_id)
                    accepted[prop].append(frame_id)
                    room -= 1
            for index, frame_id in enumerate(output, start=count):
                row = frames.loc[frame_id].to_dict()
                row.update(property=prop, slot=SLOTS[prop - 1], index=index,
                           rank_metric=row[f"rank{prop}"],
                           sel_score=row[f"score{prop}"] if prop in (2, 3) else np.nan,
                           origin="video-scan")
                row["cx0"], row["cy0"], row["cw"], row["ch"] = row["face_crop"] if prop <= 4 else (0, 0, W, H)
                row["scale"] = CW / row["cw"] if prop <= 4 else 0.5
                for key in ("kp", "thumb", "face_crop", *[f"pass{p}" for p in range(1, 10)], *[f"rank{p}" for p in range(1, 10)]):
                    row.pop(key, None)
                result.append(row)
    selected = pd.DataFrame(result)
    selected.attrs["passing_frames_per_property"] = {p: int(frames[f"pass{p}"].sum()) for p in range(1, 10)}
    selected.attrs["dedup_kept_per_property"] = {p: sum(len(v) for (_, q), v in lists.items() if q == p) for p in range(1, 10)}
    return selected


def prepare_crop(image: np.ndarray, row: Mapping[str, Any]) -> np.ndarray:
    """Exact uint8 pre-encoding crop: native face or antialiased shrink, body mean."""
    import torch
    import torch.nn.functional as functional
    tensor = torch.from_numpy(image.copy())
    if int(row["property"]) >= 5:
        # Archived refs_lib.downsample: mean in float32, nearest-even torch.round.
        output = tensor.float().reshape(CH, 2, CW, 2, 3).mean(dim=(1, 3)).round().clamp(0, 255)
        return output.byte().numpy()
    left, top, width, height = (int(row[key]) for key in ("cx0", "cy0", "cw", "ch"))
    crop = tensor[top:top + height, left:left + width].permute(2, 0, 1).float()[None]
    if width != CW:
        crop = functional.interpolate(crop, size=(CH, CW), mode="bilinear", antialias=True, align_corners=False).round().clamp(0, 255)
    if tuple(crop.shape) != (1, 3, CH, CW):
        raise ValueError(f"Invalid final crop dimensions: {tuple(crop.shape)}")
    return crop[0].byte().permute(1, 2, 0).numpy()


def decode_selected_crops(selected: pd.DataFrame, tracks_by_name: Mapping[str, Mapping],
                          output_dir: str | Path, *, prepare: bool = False) -> pd.DataFrame:
    """Decode exact selected frames to lossless RGB PNGs for the common encoder.

    Default: image_path points at the full 512x960 frame, prepared_image=False;
    the encoder applies crop metadata once. With prepare=True, output is already
    256x480 and prepared_image=True explicitly prevents a second crop/downsample.
    """
    from PIL import Image
    if selected.empty:
        return selected.copy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = selected.copy()
    result["image_path"] = ""
    result["prepared_image"] = bool(prepare)
    temporary = result.assign(_chunk=result.frame_index.astype("int64") // CHUNK_FRAMES)
    for (track_name, chunk), rows in temporary.groupby(["track", "_chunk"], sort=True):
        local_indices = sorted(set(int(frame) % CHUNK_FRAMES for frame in rows.frame_index))
        expression = "+".join(f"eq(n\\,{frame})" for frame in local_indices)
        stream = _decode_frames(tracks_by_name[track_name]["video_path"], int(chunk), expression)
        decoded = 0
        for position, image in enumerate(stream):
            if position >= len(local_indices):
                raise RuntimeError("ffmpeg decoded more selected frames than requested")
            frame_index = int(chunk) * CHUNK_FRAMES + local_indices[position]
            for index, row in rows[rows.frame_index == frame_index].iterrows():
                destination = output_dir / str(row["uuid"]) / f"property_{int(row['property'])}" / f"{int(row['index']):03d}.png"
                destination.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(prepare_crop(image, row) if prepare else image).save(destination)
                result.loc[index, "image_path"] = str(destination)
            decoded += 1
        if decoded != len(local_indices):
            raise RuntimeError(f"Missing decoded frames for {track_name} chunk {chunk}: {decoded}/{len(local_indices)}")
    return result
