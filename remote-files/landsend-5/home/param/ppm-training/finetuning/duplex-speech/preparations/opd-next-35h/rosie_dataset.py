"""Whole-conversation Rosie data using the completed Aloe SFT's speech layout.

Assistant targets use SAM-isolated Mimi codes where delivered (set A), otherwise
original Mimi codes (set B). The user always enters as the original waveform at
raw gain through the frozen Voxtral encoder. No row is split, filtered or cropped.
The delivery has one whole conversation per session, so there are no later sibling
chunks for the Aloe voice-clip rule: every row gets its empty five-token prefix.
"""
from functools import lru_cache
import json
import os
from pathlib import Path
import sys

import numpy as np

PRODUCTION = Path(os.environ.get("OPD_SFT_PRODUCTION", "/home/param/pose-repo/finetune/duplex_distill/production"))
for _p in (PRODUCTION, PRODUCTION / "lib"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import data_s3q as DS
import speech_q as SP
import sysprompt as SPR
import audio_s4q as AU

if SP.CD != 2:
    raise ValueError("The OPD finishing recipe requires PPM_CD=2")

DEFAULT_MANIFEST = Path("/home/param/datasets/rosie-stamp-conversations-v1/training-conversations.jsonl")
DEFAULT_TOKENIZER = Path("/home/param/models/ppm-811-17850-base-bf16/configs/Qwen3-32B")
DEFAULT_VOXTRAL = Path("/home/param/models/aloe-ft-vox17127-nopersona/assets/voxtral-realtime")
FFMPEG = Path("/home/param/pose-repo/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2")


def load_records(manifest=DEFAULT_MANIFEST):
    records = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    keys = [r["row_id"] for r in records]
    if not records or len(set(keys)) != len(keys):
        raise ValueError("Expected nonempty, unique whole-conversation records")
    for r in records:
        if int(r["n_frames"]) < 2 or not Path(r["conversation_dir"]).is_dir():
            raise ValueError(f"Invalid conversation record: {r['row_id']}")
    return records


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer
    path = Path(os.environ.get("ROSIE_TOKENIZER", str(DEFAULT_TOKENIZER)))
    if not (path / "tokenizer.json").is_file():
        alternate = Path("/home/param/models/imported-weights/ppm-811-17850-base-bf16/configs/Qwen3-32B")
        if (alternate / "tokenizer.json").is_file():
            path = alternate
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


@lru_cache(maxsize=216)
def _words(directory):
    doc = json.loads((Path(directory) / "words.json").read_text())
    out = {}
    for role in ("user", "assistant"):
        words = sorted((dict(w) for w in doc["words"] if w["role"] == role), key=lambda w: float(w["start"]))
        pieces = SP.word_pieces(tokenizer(), words)
        out[role] = (words, np.asarray([x for ps in pieces for x in ps], np.int64),
                     np.asarray([0] + list(np.cumsum([len(ps) for ps in pieces])), np.int64))
    if sum(len(v[0]) for v in out.values()) != len(doc["words"]):
        raise ValueError(f"Unexpected word role in {directory}")
    return out


def _source_row(record):
    directory = Path(record["conversation_dir"])
    meta = json.loads((directory / "conversation.json").read_text())
    n = int(record["n_frames"])
    if meta["roles"]["assistant"]["name"] != "Rosie Stamp" or meta["mimi"]["T"] != n:
        raise ValueError(f"Role/frame mismatch for {record['row_id']}")
    if meta["roles"]["assistant"]["lane"] != record["assistant_source_lane"]:
        raise ValueError(f"Assistant lane mismatch for {record['row_id']}")
    has_sam = bool(record["has_sam"])
    stream = "sam_isolated" if has_sam else "original"
    row = {"row_key": record["row_id"], "n_frames": n}
    words_by_role = _words(str(directory))
    for side, prefix, selected in (("user", "user", "original"), ("assistant", "Rosie_Stamp", stream)):
        codes = np.load(directory / f"{prefix}.{selected}.codes.npy", allow_pickle=False)
        if codes.shape != (32, n) or codes.dtype != np.int16 or codes.min() < 0 or codes.max() >= 2048:
            raise ValueError(f"Invalid {side} Mimi codes for {record['row_id']}")
        words, ids, offsets = words_by_role[side]
        if any(float(w["start"]) < 0 or float(w["start"]) >= n / 12.5 for w in words):
            raise ValueError(f"Word outside whole conversation: {record['row_id']}")
        row[f"codes_{side}"] = codes
        row[f"words_{side}"] = words
        row[f"piece_ids_{side}"] = ids
        row[f"piece_offsets_{side}"] = offsets
    return row, stream


def load_raw_batch(record, epoch=0, seed=20261011, *, audio=True):
    """Return the ordinary CPU PPM speech batch, one entire conversation.

    Epoch does not change the row or its labels. The trainer owns shuffled order,
    global token normalizers and all tensor device transfers.
    """
    del epoch
    row, stream = _source_row(record)
    sp = SPR.build_prefix(row, [{"kind": "text", "track": 1, "ids": []}], {"new_word": True})
    p, n = int(sp["P"]), int(row["n_frames"])
    if p != 5:
        raise ValueError("The empty Aloe system block must have five positions")
    cfg = {"new_word": True, "eob_gap_s": 0.0, "wpid": "both", "light_seed": int(seed),
           "row_frames": p + n, "frame_light": True, "no_crop": True}
    arrays = DS.row_arrays(row, cfg, rank=0, row_id=0, node_name="rosie", light_frac=1.0, sp=sp, swap=0)
    target = arrays[1]
    b = {"kind": "speech", **{k: arrays[k] for k in ("ids", "depth", "present", "pid")},
         **{k: target[k] for k in ("sel", "y", "word", "pad", "eob", "nw", "cb0", "light")},
         "P": p, "segs": [p + n + SP.CD], "vin": np.asarray([p, n], np.int64),
         "row_key": row["row_key"], "n_text": float(len(target["sel"])), "dummy": False,
         "own": np.asarray([[len(target["sel"]), n, n]], np.float64), "voice_clip": 0, "voice_drawn": 0,
         "assistant_codes_stream": stream}
    # Exact Aloe v2vox channel A: no text, Mimi codes or presence in the
    # conversation. Only the trainable adapter will inject frozen user latents.
    b["ids"][0, p:] = SP.PAD_ID
    b["depth"][0] = -1
    b["present"][0, p:] = 0.0
    b["aref"] = json.dumps({"row_key": row["row_key"], "n": n, "side": "user", "gain_db": 0.0,
                             "stream": "original", "gain_mode": "raw"})
    if audio:
        ffmpeg = Path(os.environ.get("S4_FFMPEG", str(FFMPEG)))
        if not ffmpeg.is_file():
            raise FileNotFoundError(f"Pinned FFmpeg executable missing: {ffmpeg}")
        AU.FFMPEG_EXE = str(ffmpeg)
        b["wav"] = AU.ffmpeg_row_audio({"flac": str(Path(record["conversation_dir"]) / "user.original.flac"),
                                        "frame0": 0, "n_frames": n, "row_key": row["row_key"]}, gain_db=0.0)
        if b["wav"].shape != (1920 * n,) or not np.isfinite(b["wav"]).all():
            raise ValueError(f"Invalid waveform for {row['row_key']}")
    assert len(b["pid"]) == p + n + 2
    assert int((b["cb0"] >= 0).sum()) == n and len(b["light"]) == n
    assert np.array_equal(b["light"], np.arange(p + 2, p + 2 + n))
    assert (b["cb0"][:p + 2] < 0).all() and (b["depth"][0] < 0).all()
    assert (b["ids"][0] == SP.PAD_ID).all() and (b["present"][0] == 0).all()
    return b


def make_front(device):
    """Frozen production Voxtral frontend, bf16 and outside FSDP."""
    path = Path(os.environ.get("ROSIE_VOXTRAL", str(DEFAULT_VOXTRAL)))
    if not (path / "audio_tower.safetensors").is_file():
        alternate = Path("/home/param/models/imported-weights/aloe-ft-vox17127-nopersona/assets/voxtral-realtime")
        if (alternate / "audio_tower.safetensors").is_file():
            path = alternate
    return AU.make_front({"s4_audio": "flac", "s4_vox_model": str(path), "s4_enc_hidden": 1280}, device)


def encode(front, batch):
    """Encode the whole user waveform with the unchanged production frontend."""
    import torch
    ref = json.loads(batch["aref"])
    lat = front.encode([torch.from_numpy(np.ascontiguousarray(batch["wav"]))], [ref])
    if lat.shape != (1, 4 * ref["n"], 1280) or not torch.isfinite(lat).all():
        raise ValueError(f"Invalid Voxtral latents for {ref['row_key']}")
    return lat


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Materialize the actual longest Rosie CPU batch")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    records = load_records(args.manifest)
    record = max(records, key=lambda r: int(r["n_frames"]))
    batch = load_raw_batch(record)
    print(json.dumps({"row_id": record["row_id"], "rows": len(records), "n_frames": record["n_frames"],
                      "sequence_length": len(batch["pid"]), "prefix": batch["P"],
                      "text_targets": len(batch["sel"]), "text_pieces": int(batch["word"].sum()),
                      "cb0_frames": int((batch["cb0"] >= 0).sum()), "depth_frames": len(batch["light"]),
                      "assistant_codes_stream": batch["assistant_codes_stream"], "wav_samples": len(batch["wav"])}, indent=2))
