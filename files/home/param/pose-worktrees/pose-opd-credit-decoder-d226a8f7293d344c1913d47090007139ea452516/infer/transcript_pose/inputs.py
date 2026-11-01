"""Inference inputs built with the exact transcript-training data helpers."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "finetune/transcript_pose"
LEGACY = ROOT / "finetune/xabi-main-17850"
FPS, CD, DEPTH, CBS = 12.5, 2, 32, 2048
PAD, NW, EOB, CTRL_LO = 151670, 151671, 151645, 151643
OFF = np.arange(DEPTH, dtype=np.int64) * CBS


def training_data():
    # Literal assignments only: the archived run.env is not shell code to run.
    for line in (LEGACY / "run.env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key] = value
    sys.path.insert(0, str(TRAIN))
    import data_transcript_pose as D
    return D


def codes_file(path):
    """Read local [32, frames] codec IDs, never global offset IDs."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.numpy import load_file
        obj = load_file(str(path))
        x = obj["codes"] if "codes" in obj else obj[next(iter(obj))]
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as obj:
            x = obj["codes"] if "codes" in obj else obj[obj.files[0]]
    else:
        x = np.load(path, allow_pickle=False)
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[0] != DEPTH or not np.issubdtype(x.dtype, np.integer):
        raise ValueError(f"Expected integer [32, frames] codes: {path}: {x.shape}")
    if not x.size or x.min() < 0 or x.max() >= CBS:
        raise ValueError(f"Expected nonempty local codec IDs in [0, 2047]: {path}")
    return x.astype(np.int64)


def word_records(path):
    obj = json.loads(Path(path).read_text())
    obj = obj["words"] if isinstance(obj, dict) else obj
    words = []
    for x in obj:
        if isinstance(x, dict):
            words.append([float(x["start"]), float(x.get("end", x["start"])), str(x["text"])])
        elif isinstance(x[0], str):
            words.append([float(x[1]), float(x[2]), x[0]])
        else:
            words.append([float(x[0]), float(x[1]), str(x[2])])
    if any(not math.isfinite(s) or not math.isfinite(e) for s, e, _ in words):
        raise ValueError("Word times must be finite seconds")
    return words


def voice_prefix(D, clip):
    segs = []
    if clip is not None:
        if clip.shape[1] < 38:
            raise ValueError("A voice reference must contain at least 38 Mimi frames (about 3 s)")
        clip = clip[:, :125]
        segs = [dict(kind="speech", track=1, w0=0, L=clip.shape[1],
                     src_row={"codes_assistant": clip}, src_side="assistant",
                     side="assistant", blank_lane=True)]
    if not segs:
        segs = [dict(kind="text", track=1, ids=[])]
    return D.legacy().D.SPR.build_prefix({}, segs, {"new_word": True}), clip


def replace_system(m, sp):
    old, new = int(m["P"]), int(sp["P"])
    out = dict(m, P=new)
    for k in ("ids", "depth", "present"):
        out[k] = np.concatenate([sp[k], m[k][:, old:]], axis=1)
    out["pdepth"] = np.concatenate([np.full((new, DEPTH), -1, np.int64), m["pdepth"][old:]])
    return out


def make_inputs(args, voice=None):
    """Return sanitized engine input, provenance and the tokenizer, without CUDA.

    Only explicitly requested prefill media survive. Cached rows supply words and
    deterministic voice conditioning, never ground-truth generation media.
    """
    D = training_data()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    custom = any((args.words, args.lane, args.transcript is not None, args.transcript_file))
    frames = args.frames if args.frames is not None else (round(args.seconds * FPS) if args.seconds is not None else None)
    if frames is not None and frames < 2:
        raise ValueError("At least two generation frames are required")
    compact = args.recipe == "compact-all"
    text_mode = getattr(args, "text_mode", None)
    if text_mode not in (None, "forced", "native"):
        raise ValueError("text_mode must be forced or native")
    if text_mode == "native" and args.recipe == "aligned-media":
        raise ValueError("aligned-media requires its supplied frame-aligned text lane")
    if text_mode == "native" and getattr(args, "force_text", False):
        raise ValueError("--force-text conflicts with --text-mode native")
    forced_text = (text_mode == "forced" if text_mode else not compact or bool(getattr(args, "force_text", False)))
    text_mode = "forced" if forced_text else "native"
    upfront = args.recipe != "xabi-original"
    if custom:
        if args.clip_index is not None or args.start_frame or args.prefill_frames:
            raise ValueError("Custom text cannot be combined with cache clip selection or media prefill")
        if compact and not (args.transcript is not None or args.transcript_file or args.words):
            raise ValueError("compact-all needs --transcript, --transcript-file, or --words")
        if compact and forced_text and not args.words:
            raise ValueError("Custom compact-all --force-text needs timestamped --words; a plain transcript does not specify a lane")
        if not compact and not (args.words or args.lane):
            raise ValueError("Framewise-text recipes need --words with timestamps or --lane")
        words_raw = word_records(args.words) if args.words else []
        given_lane = np.load(args.lane, allow_pickle=False) if args.lane else None
        if given_lane is not None:
            if given_lane.ndim != 1 or not np.issubdtype(given_lane.dtype, np.integer):
                raise ValueError("--lane must be a one-dimensional integer .npy array")
            if frames is not None and frames != len(given_lane):
                raise ValueError("--frames/--seconds must equal the supplied lane's length")
            frames = len(given_lane)
        if frames is None:
            if compact or not words_raw:
                raise ValueError("Supply --frames or --seconds; this model has no trained duration/stop head")
            frames = max(2, math.ceil(max(max(s, e) for s, e, _ in words_raw) * FPS))
        words = D.clean_words(words_raw, frames)
        lane = np.full(frames, PAD, np.int64)
        if given_lane is not None:
            lane = given_lane.astype(np.int64)
        elif words:
            SP = D.legacy().D.SP
            pieces = SP.word_pieces(tokenizer, words)
            offsets = np.concatenate([[0], np.cumsum([len(x) for x in pieces])])
            wif = np.zeros(frames, np.int64)
            lane = SP.lane(words, [x for p in pieces for x in p], offsets, frames, wif)
            SP.insert_new_word(lane, wif)
        if np.any(~((lane >= 0) & ((lane < CTRL_LO) | np.isin(lane, [PAD, NW, EOB])))):
            raise ValueError("Lane contains IDs outside lexical tokens/PAD/NEW_WORD/EOB")
        text = (Path(args.transcript_file).read_text() if args.transcript_file else args.transcript)
        text = " ".join(str(text).split()) if text is not None else " ".join(w["text"] for w in words)
        sp, voice = voice_prefix(D, voice)
        P, N = int(sp["P"]), int(sp["P"]) + frames + CD
        m = dict(P=P, ids=np.full((2, N), PAD, np.int64), depth=np.full((2, N, DEPTH), -1, np.int64),
                 present=np.zeros((2, N), np.float32), pdepth=np.full((N, DEPTH), -1, np.int64),
                 swap=0, row_key="custom-transcript", item=[-1, -1, 0])
        for k in ("ids", "depth", "present"):
            m[k][:, :P] = sp[k]
        m["ids"][1, P:P + frames], m["present"][1, P:] = lane, 1.0
        tokens = np.asarray(tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True), np.int64) if compact else None
        if tokens is not None and (np.any(tokens < 0) or np.any(tokens >= CTRL_LO)):
            raise ValueError("Compact transcript contains a non-text token")
        micro = D.transform_micro(m, 0, frames, transcript=upfront, prefix_ids=tokens)
        info = dict(input="custom", transcript=text, words=words, source_span=None,
                    voice_provenance={"mode": "explicit" if args.voice_clip else "empty", "source": str(args.voice_clip) if args.voice_clip else None})
    else:
        data = D.CompactXabi(args.cache, args.tokenizer, tokenizer=tokenizer)
        index = data.split_indices["eval"][0] if args.clip_index is None else args.clip_index
        if not 0 <= index < len(data.clips):
            raise ValueError(f"--clip-index outside [0, {len(data.clips) - 1}]")
        c = data.clips[index]
        start = args.start_frame
        end = c.n if frames is None else start + frames
        if not 0 <= start < end <= c.n or end - start < 2:
            raise ValueError(f"Invalid source slice [{start}, {end}) for {c.n}-frame clip")
        frames = end - start
        original = data.micro((index, 0, c.n), mode="original")
        if args.voice_clip or args.no_voice_clip:
            sp, voice = voice_prefix(D, voice)
            original = replace_system(original, sp)
        else:
            p = int(original["P"])
            ii = np.flatnonzero(original["depth"][1, :p, 0] >= 0)
            voice = (original["depth"][1, ii] - OFF).T if len(ii) else None
        tokens = data.compact_tokens(index, start, end) if compact else None
        micro = D.transform_micro(original, start, end, transcript=upfront, prefix_ids=tokens)
        words = D.clean_words(data._shard(c.shard)[0]["words"][c.cache_index], c.n)
        words = [dict(w, start=w["start"] - start / FPS, end=w["end"] - start / FPS) for w in words
                 if (w["end"] > start / FPS and w["start"] < end / FPS) or
                 (w["end"] == w["start"] and start / FPS <= w["start"] < end / FPS)]
        provenance = json.loads(original.get("prov", "{}"))
        if args.voice_clip or args.no_voice_clip:
            provenance = {"mode": "explicit" if args.voice_clip else "empty override",
                          "source": str(args.voice_clip) if args.voice_clip else None}
        info = dict(input="cache", cache=str(Path(args.cache).resolve()), clip_index=index, row_key=c.row_key,
                    cache_key=c.cache_key, split=c.split, source_span=[start, end],
                    transcript=" ".join(w["text"] for w in words), words=words,
                    voice_provenance=provenance)
    P, npf = int(micro["P"]), int(args.prefill_frames)
    if not 0 <= npf < frames:
        raise ValueError("--prefill-frames must be nonnegative and shorter than the generated window")
    total = P + frames + CD
    if total > 7500:
        raise ValueError(f"Input needs {total} positions; training limit is 7500. Split the transcript/window.")
    inst = {k: micro[k].copy() for k in ("ids", "depth", "present", "pdepth", "pid")}
    inst.update(P=P, a_mode="empty")
    # f counts the delayed training lane. A prefill of zero means genuinely no
    # ground-truth audio/pose or framewise words in native text mode.
    f0 = npf + CD if npf else 0
    inst["depth"][1, P + CD + npf:] = -1
    inst["pdepth"][P + CD + npf:] = -1
    if not forced_text:
        inst["ids"][1, P + f0:] = PAD
    else:
        inst["text_force"] = inst["ids"][1].copy()
    assert np.all(inst["depth"][1, P + CD + npf:] == -1)
    assert np.all(inst["pdepth"][P + CD + npf:] == -1)
    info.update(recipe=args.recipe, frames=frames, seconds=frames / FPS, P=P,
                conditioning_P=int(micro["conditioning_P"]), transcript_span=micro["transcript_span"],
                prefill_frames=npf, f0=f0, code_delay=CD, total_positions=total,
                voice_clip_frames=0 if voice is None else int(voice.shape[1]),
                force_text=forced_text, text_mode=text_mode,
                source_transcript=info["transcript"],
                supplied_transcript=info["transcript"] if upfront or forced_text else None,
                upfront_transcript=bool(upfront),
                text_generation="forced aligned lane" if forced_text else "sampled; initial frame PAD when no prefill",
                transcript_end_token=D.TRANSCRIPT_END if upfront else None)
    return inst, info, tokenizer, voice
