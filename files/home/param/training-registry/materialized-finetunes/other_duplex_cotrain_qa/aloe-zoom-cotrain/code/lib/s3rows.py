"""Stage-3 DUPLEX row store (SAM podcast episodes -> one row per 10-min chunk, two Mimi channels). Format + loader.

ONE ROW = one 7,500-frame (600 s, 12.5 Hz) chunk k of one episode: frames [7500k, min(7500(k+1), T)) of BOTH speakers' whole-episode
codes.npy (frame 0 = chunk_00 start, so row start time t_row = chunk_00.global_start_sec + 600 k). Roles exactly as the data labels them
(speaker.json role user / assistant; no swapping). Rows shorter than MIN_FRAMES (60 s) are dropped (only an episode's last chunk can be).

PER NODE (flat, memory-mappable; mirrors stage-2 tts128m/flat: NODE.json + per-shard DONE / index.npz / codes / words.arrow):
  NODE.json                        node, shards, rows, frames, ids_blake2b per shard, totals, per-dataset rows / hours, format constants
  rows-SSSSS.codes_user.npy        int16 [R, 32, 7500]  user codes, row-major per row; frames >= n_frames are -1 (PAD)
  rows-SSSSS.codes_assistant.npy   int16 [R, 32, 7500]  assistant codes, same layout
  rows-SSSSS.index.npz             length (= n_frames) int32 [R], chunk_index, frame_start, dataset (0 A / 1 B / 2 ALOE), n_words_*, n_pieces_*
  rows-SSSSS.words.arrow           one record per row: provenance (row_key, dataset, dataset_folder, podcast_id, episode_id, chunk_index,
                                   frame_start, frame_end, n_frames, {user,assistant}_{speaker_dir,speaker}) + per channel
                                   words_<side> list<struct{text, start, end (s, ROW-relative, float64), start_frame, end_frame (int32,
                                   floor(t * 12.5 + 1e-6) = speech_q.lane's rule), in_overlap, truncated}>, piece_ids_<side> list<int32>
                                   (Qwen3-32B tokenizer, split_special_tokens=True, all < 151643; speech_q.word_pieces), piece_offsets_<side>
                                   list<int32> (len n_words + 1; word j owns piece_ids[off[j]:off[j+1]])
  rows-SSSSS.DONE                  shard json (rows, frames, episodes, ids_blake2b, stats, bad episodes)
  rows.jsonl / rows.parquet        node-level provenance index: one line per row (row = node row id, shard, row_in_shard, provenance)
The words / pieces / offsets plug straight into speech_q.lane(words, piece_ids, piece_offsets, n) (word "start" is row-relative)."""
import hashlib
import json
import math
import os

import numpy as np

DEPTH, CBS, FPS, ROW_FRAMES, MIN_FRAMES = 32, 2048, 12.5, 7500, 750
# 10-11 (stage-4 gen finetune: QA calls of 30-100 s, one row per call): S3_MIN_FRAMES overrides the row-length floor check_row enforces
# (read path too: data_s3q / data_s4q call check_row on every row). Unset = 750 = unchanged for every other store / run.
MIN_FRAMES = int(os.environ.get("S3_MIN_FRAMES", MIN_FRAMES))
assert 1 <= MIN_FRAMES <= ROW_FRAMES, MIN_FRAMES
CTRL_LO = 151643
PAD_CODE = -1
SIDES = ("user", "assistant")
DATASETS = ("A", "B", "ALOE")       # ALOE (10-08): akv13/aloe-duplex-ft-sysprompts finetune rows (qwen-1ch-data/aloe-ft-prep/aloe_to_s3rows.py); appended, A/B indices unchanged
FORMAT = "s3-duplex-rows-v1"


def frame_of(t):
    """speech_q.lane's floor rule: frame = floor(t * 12.5 + 1e-6)."""
    return math.floor(float(t) * FPS + 1e-6)


def shard_digest(row_keys, n_frames):
    h = hashlib.blake2b(digest_size=16)
    for k in row_keys:
        h.update(k.encode()); h.update(b"\0")
    h.update(np.asarray(n_frames, np.int32).tobytes())
    return h.hexdigest()


def load_node(root):
    """speech_q.load_node's contract on the duplex store -> dict(shards, length int32 [N] (= n_frames), shard_of int16, row_of int32,
    digest, node, flat_root)."""
    meta = json.load(open(f"{root}/NODE.json"))
    assert meta.get("format") == FORMAT, meta.get("format")
    shards = list(meta["shards"]); assert shards == sorted(shards)
    L, S, R = [], [], []
    for k, s in enumerate(shards):
        d = json.load(open(f"{root}/{s}.DONE"))
        z = np.load(f"{root}/{s}.index.npz")
        ln = z["length"].astype(np.int32)
        assert len(ln) == d["rows"] == meta["rows"][k] and int(ln.sum()) == d["frames"], (s, len(ln), d["rows"])
        L.append(ln); S.append(np.full(len(ln), k, np.int16)); R.append(np.arange(len(ln), dtype=np.int32))
    h = hashlib.blake2b(digest_size=16)
    for s, dg in zip(shards, meta["ids_blake2b"]):
        h.update(s.encode()); h.update(dg.encode())
    z = lambda a: np.concatenate(a) if a else np.zeros(0, np.int32)  # noqa: E731
    return {"shards": shards, "length": z(L), "shard_of": z(S), "row_of": z(R), "digest": h.hexdigest(), "node": meta["node"],
            "flat_root": root, "meta": meta}


class Rows:
    """lazy per-worker random access: rows[i] -> dict(codes_user [32, 7500] int16 (mmap view), codes_assistant, n_frames, provenance,
    words_<side>, piece_ids_<side> (int64 np), piece_offsets_<side> (int32 np))."""

    def __init__(self, node):
        self.node, self.cu, self.ca, self.tb = node, {}, {}, {}

    def _open(self, k):
        if k not in self.cu:
            import pyarrow as pa
            import pyarrow.ipc as ipc
            s, root = self.node["shards"][k], self.node["flat_root"]
            self.cu[k] = np.load(f"{root}/{s}.codes_user.npy", mmap_mode="r")
            self.ca[k] = np.load(f"{root}/{s}.codes_assistant.npy", mmap_mode="r")
            self.tb[k] = ipc.open_file(pa.memory_map(f"{root}/{s}.words.arrow", "r")).read_all()

    def __len__(self):
        return len(self.node["length"])

    def __getitem__(self, i):
        k, r = int(self.node["shard_of"][i]), int(self.node["row_of"][i])
        self._open(k)
        rec = self.tb[k].slice(r, 1).to_pylist()[0]
        out = {"codes_user": self.cu[k][r], "codes_assistant": self.ca[k][r], "row": int(i)}
        for s in SIDES:
            out[f"piece_ids_{s}"] = np.asarray(rec.pop(f"piece_ids_{s}"), np.int64)
            out[f"piece_offsets_{s}"] = np.asarray(rec.pop(f"piece_offsets_{s}"), np.int32)
        out.update(rec)
        assert out["n_frames"] == int(self.node["length"][i])
        return out


def check_row(row, strict=True):
    """the per-row invariants (also asserted by the builder before writing). Returns a list of violations (empty = ok)."""
    bad = []
    n = int(row["n_frames"])
    if not (MIN_FRAMES <= n <= ROW_FRAMES): bad.append(f"n_frames {n}")
    for s in SIDES:
        c = np.asarray(row[f"codes_{s}"])
        if c.shape != (DEPTH, ROW_FRAMES): bad.append(f"{s} codes shape {c.shape}")
        v = c[:, :n]
        if v.size and (int(v.min()) < 0 or int(v.max()) >= CBS): bad.append(f"{s} codes out of [0,2047] in valid frames")
        if n < ROW_FRAMES and not (c[:, n:] == PAD_CODE).all(): bad.append(f"{s} pad not -1")
        ws = row[f"words_{s}"]; dur = n / FPS
        st = [w["start"] for w in ws]
        if any(b < a for a, b in zip(st, st[1:])): bad.append(f"{s} words not sorted")
        for w in ws:
            if not (0.0 <= w["start"] < dur and w["start"] <= w["end"] <= dur + 1e-9):
                bad.append(f"{s} word outside row {w}"); break
            if w["start_frame"] != frame_of(w["start"]) or w["end_frame"] != min(frame_of(w["end"]), n): bad.append(f"{s} frame rule {w}"); break
        pid, off = np.asarray(row[f"piece_ids_{s}"]), np.asarray(row[f"piece_offsets_{s}"])
        if len(off) != len(ws) + 1 or off[0] != 0 or off[-1] != len(pid) or (np.diff(off) < 0).any(): bad.append(f"{s} offsets")
        if len(pid) and (int(pid.max()) >= CTRL_LO or int(pid.min()) < 0): bad.append(f"{s} piece id >= {CTRL_LO}")
        for f in ("speaker_dir", "speaker"):
            if not row.get(f"{s}_{f}"): bad.append(f"provenance {s}_{f}")
    for f in ("row_key", "dataset", "dataset_folder", "podcast_id", "episode_id"):
        if not row.get(f): bad.append(f"provenance {f}")
    if row.get("dataset") not in DATASETS: bad.append("dataset")
    if row["frame_start"] != ROW_FRAMES * row["chunk_index"] or row["frame_end"] != row["frame_start"] + n: bad.append("frame_start/end")
    if strict and bad:
        raise AssertionError(f"{row.get('row_key')}: {bad[:5]}")
    return bad
