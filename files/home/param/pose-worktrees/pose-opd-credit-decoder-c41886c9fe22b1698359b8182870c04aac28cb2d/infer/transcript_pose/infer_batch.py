#!/usr/bin/env python3
"""One checkpoint, one GPU pair, one or two genuinely batched clips.

--batch-json is a list of jobs with clip_index, voice_clip, seed, frames, out,
and optional start_frame. Inputs retain each recipe's training prefix. The
explicit voice reference is supplied for every recipe. Text defaults to forced;
native mode samples framewise text for compact-all and xabi-original.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from infer import arguments, engine_modules, sampling
from inputs import CD, CBS, DEPTH, FPS, OFF, PAD, ROOT, codes_file, make_inputs


def prepare(a):
    jobs = json.loads(a.batch_json.read_text())
    if not isinstance(jobs, list) or not 1 <= len(jobs) <= 2:
        raise ValueError("--batch-json must contain one or two jobs")
    prepared = []
    for job in jobs:
        argv = ["--recipe", a.recipe, "--text-mode", a.text_mode, "--clip-index", str(job["clip_index"]),
                "--start-frame", str(job.get("start_frame", 0)), "--frames", str(job["frames"]),
                "--seed", str(job["seed"]), "--out", str(job["out"]),
                "--voice-clip", str(job["voice_clip"]), "--devs", a.devs]
        for key in ("ckpt", "cache", "tokenizer"):
            if getattr(a, key) is not None:
                argv += ["--" + key, str(getattr(a, key))]
        for key in ("text_temp", "text_topk", "audio_temp", "pose_temp", "audio_topk", "pose_topk"):
            argv += ["--" + key.replace("_", "-"), str(getattr(a, key))]
        args = arguments(argv)
        voice = codes_file(args.voice_clip)
        inst, info, tok, voice = make_inputs(args, voice)
        assert info["f0"] == 0 and info["voice_clip_frames"] > 0
        info["voice_codes_sha256"] = hashlib.sha256(np.ascontiguousarray(voice, dtype=np.int64).tobytes()).hexdigest()
        info["framewise_lane_sha256"] = (hashlib.sha256(np.ascontiguousarray(inst["text_force"][info["P"]:], dtype=np.int64).tobytes()).hexdigest()
                                         if info["force_text"] else None)
        prepared.append((args, inst, info, tok, voice))
    if len({str(x[0].out.resolve()) for x in prepared}) != len(prepared):
        raise ValueError("Each batch row needs a distinct output directory")
    for args, inst, info, tok, voice in prepared:
        args.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out / "inputs.npz", **{k: v for k, v in inst.items() if isinstance(v, (int, np.ndarray))})
        np.save(args.out / "voice_clip_mimi.npy", voice)
        (args.out / "input.json").write_text(json.dumps(info, indent=2) + "\n")
    return prepared


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--batch-json", type=Path, required=True)
    p.add_argument("--recipe", choices=("aligned-media", "compact-all", "xabi-original"), required=True)
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--cache", type=Path)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--devs", default="0,1")
    p.add_argument("--text-mode", choices=("forced", "native"), default="forced", help="Native samples text for compact-all and xabi-original; aligned-media requires forced")
    p.add_argument("--text-temp", type=float, default=1.0)
    p.add_argument("--text-topk", type=int, default=0, help="Text top-k; 0 disables filtering (maximum 256)")
    p.add_argument("--audio-temp", type=float, default=1.0, help="Shared CB0/residual audio temperature")
    p.add_argument("--pose-temp", type=float, default=1.0, help="Shared CB0/residual pose temperature")
    p.add_argument("--audio-topk", type=int, default=0, help="Shared CB0/residual audio top-k; 0 disables filtering")
    p.add_argument("--pose-topk", type=int, default=0, help="Shared CB0/residual pose top-k; 0 disables filtering")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--no-graphs", action="store_true")
    p.add_argument("--no-audio-decode", action="store_true")
    a = p.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["PPM_CD"] = str(CD)
    fa = ROOT / ".cache/kernels/vllm-flash-attn3"
    if fa.exists():
        os.environ.setdefault("FA3_DIR", str(fa))
    began = time.time()
    rows = prepare(a)
    if a.prepare_only:
        print(json.dumps([x[2] for x in rows], indent=2), flush=True)
        return

    import torch
    from safetensors.numpy import save_file
    args0 = rows[0][0]
    FP, IP = engine_modules(args0.tokenizer)
    IP._TOK[0] = rows[0][3]
    if (args0.ckpt / "meta.json").exists():
        cfg = json.loads((args0.ckpt / "meta.json").read_text()).get("cfg", {})
        if cfg.get("recipe") not in (None, a.recipe):
            raise ValueError(f"Checkpoint recipe {cfg['recipe']} does not match --recipe {a.recipe}")
    params = sampling(args0)
    for args, *_ in rows:
        row_params = sampling(args)
        clean, bad = FP.validate(row_params)
        if bad or clean != row_params or row_params != params:
            raise ValueError(f"Invalid or inconsistent batch sampling parameters: {bad}")
    devs = [torch.device(f"cuda:{int(x)}") for x in a.devs.split(",")]
    torch.cuda.set_device(devs[-1])
    n_max = max(x[2]["frames"] for x in rows)
    n_min = min(x[2]["frames"] for x in rows)
    p_max = max(x[2]["P"] for x in rows)
    with torch.inference_mode():
        m, ldev, cfgs, ck = FP.build_model(str(args0.ckpt), devs)
        if not ck["pose"] or not ck["decp_bb"] or ck["decp_bb_alpha"] != 1.0:
            raise ValueError(f"Expected the Xabi pose architecture with D33 alpha=1: {ck}")
        eng = FP.FastPPM(m, ldev, cfgs, devs, n_inst=len(rows), cfg_rows=False,
                         max_s=n_max / FPS, p_max=p_max, mimi_stream=False,
                         per_instance_rng=True, forced_text_only=rows[0][2]["force_text"])
        if not a.no_graphs:
            eng.capture(f=CD, P=p_max)
        t0 = time.time()
        eng.start([x[1] for x in rows], [x[0].seed for x in rows], params, f0=0)
        t1 = time.time()
        for f in range(n_max + CD):
            # Match the single-clip native path: final CD slots are PAD while
            # flushing media. -1 preserves sampling for longer teammates.
            force = None
            if a.text_mode == "native" and f >= n_min:
                force = {"lane": np.asarray([PAD if f >= x[2]["frames"] else -1 for x in rows], np.int64)}
            eng.step(f, graphs=not a.no_graphs, cold_pad=not rows[0][2]["force_text"], force=force)
        for d in devs:
            torch.cuda.synchronize(d)
        gen_s = time.time() - t1
        outputs = []
        for r, (args, inst, info, tok, voice) in enumerate(rows):
            n, P = info["frames"], info["P"]
            result = eng.readback(r, P + n + CD)
            audio = (result["dep_b"][P + CD:P + CD + n] - OFF).T
            pose = (result["pdep"][P + CD:P + CD + n] - OFF).T
            lane = result["lane_b"][P:P + n]
            for codes in (audio, pose):
                if codes.shape != (DEPTH, n) or codes.min() < 0 or codes.max() >= CBS:
                    raise RuntimeError("Generation did not produce all requested media frames")
            if info["force_text"] and not np.array_equal(result["lane_b"], inst["ids"][1]):
                raise RuntimeError("Framewise text differs from the supplied lane")
            if not info["force_text"] and lane[0] != PAD:
                raise RuntimeError("Native text must retain the initial PAD bootstrap")
            if not np.array_equal(result["lane_b"][:P], inst["ids"][1, :P]):
                raise RuntimeError("Conditioning changed during generation")
            if not np.all(result["lane_b"][P + n:P + n + CD] == PAD):
                raise RuntimeError("Final delayed text slots must be PAD")
            if not np.all(result["lane_a"] == PAD) or not np.all(result["dep_a"] == -1):
                raise RuntimeError("Monoplex inference unexpectedly wrote channel A")
            words = IP.lane_words(result["lane_b"][:P + n], P, P, P)
            np.savez_compressed(args.out / "tokens.npz", audio_codes=audio.astype(np.int16),
                                pose_codes=pose.astype(np.int16), lane=lane, prefill_frames=np.int64(0))
            save_file({"codes": np.ascontiguousarray(audio.astype(np.int16))}, str(args.out / "mimi.safetensors"))
            save_file({"codes": np.ascontiguousarray(pose.astype(np.int64))}, str(args.out / "pose_tokens.safetensors"))
            (args.out / "words.json").write_text(json.dumps(dict(words=words, lane=lane.tolist(),
                text=" ".join(w["text"] for w in words), supplied_transcript=info["supplied_transcript"],
                source_transcript=info["source_transcript"]), indent=2) + "\n")
            meta = dict(**info, checkpoint=ck, sampling=eng.params, seed=args.seed,
                        generated_lane_sha256=hashlib.sha256(np.ascontiguousarray(lane, dtype=np.int64).tobytes()).hexdigest(),
                        batch_size=len(rows), batch_row=r, batch_clip_indices=[x[2]["clip_index"] for x in rows],
                        rng_policy="per-instance per-modality generators; row-wise [1,vocab] multinomial; " + ("no text sampling" if info["force_text"] else "native text sampling"),
                        prefill_seconds=t1 - t0, generation_seconds=gen_s,
                        ms_per_step=1000 * gen_s / (n_max + CD),
                        ground_truth_generation_media=False, fixed_duration=True,
                        peak_gib={str(d): torch.cuda.max_memory_allocated(d) / 2**30 for d in devs})
            outputs.append((args, audio, voice, meta))
        del eng, m
        gc.collect()
        for d in devs:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        if not a.no_audio_decode:
            mimi = IP.Mimi(devs[-1])
            for args, audio, voice, meta in outputs:
                wav = mimi.decode(audio)
                IP.write_wav(str(args.out / "audio.wav"), wav)
                meta["audio_stats"] = IP.audio_stats(wav)
                IP.write_wav(str(args.out / "voice_clip.wav"), mimi.decode(voice))
        for args, audio, voice, meta in outputs:
            meta["wall_seconds"] = time.time() - began
            (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
            print(f"Completed {a.recipe} batch row {meta['batch_row']}: {meta['frames']} frames -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
