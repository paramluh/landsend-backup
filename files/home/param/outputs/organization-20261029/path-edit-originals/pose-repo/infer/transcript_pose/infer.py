#!/usr/bin/env python3
"""Generate audio and pose with transcript-conditioned fine-tunes or the Xabi baseline."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from inputs import CD, CBS, DEPTH, FPS, OFF, PAD, ROOT, codes_file, make_inputs

ENGINE = ROOT / "infer/engine"


def arguments(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--recipe", choices=("aligned-media", "compact-all", "xabi-original"), required=True)
    p.add_argument("--ckpt", type=Path, help="Complete local BF16 export or consolidated DCP; inferred from recipe if omitted")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=Path("/home/param/datasets/xabi-v2pure-cache"))
    p.add_argument("--tokenizer", type=Path, default=Path("/home/param/models/ppm-811-17850-base-bf16/configs/Qwen3-32B"))
    p.add_argument("--clip-index", type=int, help="Global CompactXabi clip index; defaults to first held-out clip")
    p.add_argument("--start-frame", type=int, default=0)
    duration = p.add_mutually_exclusive_group()
    duration.add_argument("--frames", type=int, help="Number of 12.5 Hz media frames")
    duration.add_argument("--seconds", type=float)
    text = p.add_mutually_exclusive_group()
    text.add_argument("--words", type=Path, help="Timestamped words JSON; required for aligned custom text unless --lane is used")
    text.add_argument("--lane", type=Path, help="Integer .npy frame-aligned text lane, including PAD/NEW_WORD")
    text.add_argument("--transcript", help="Ordinary full transcript for compact-all")
    text.add_argument("--transcript-file", type=Path)
    voice = p.add_mutually_exclusive_group()
    voice.add_argument("--voice-clip", type=Path, help="Same-speaker 3–10 s WAV or local Mimi-code .npy/.npz/.safetensors")
    voice.add_argument("--no-voice-clip", action="store_true", help="Use the trained empty system block")
    p.add_argument("--prefill-frames", type=int, default=25,
                   help="Source audio/pose/text prefix in 12.5 Hz frames; default 25 (2 seconds), explicit 0 disables")
    p.add_argument("--force-text", action="store_true", help="Also supply the framewise lane to compact-all; its ordinary transcript prefix is unchanged")
    p.add_argument("--text-mode", choices=("forced", "native"), help="Explicit framewise text policy; defaults to compact native and other recipes forced")
    p.add_argument("--prepare-only", action="store_true", help="Write/validate exact inputs without loading the PPM model")
    p.add_argument("--devs", default="0,1", help="CUDA devices visible in this process")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--text-temp", type=float)
    p.add_argument("--audio-temp", type=float)
    p.add_argument("--pose-temp", type=float)
    p.add_argument("--text-topk", type=int, default=0)
    p.add_argument("--audio-topk", type=int, default=0)
    p.add_argument("--pose-topk", type=int, default=0)
    p.add_argument("--pose-light-topk", type=int, help="Pose cb1–31 top-k; defaults to --pose-topk")
    p.add_argument("--pose-audio-guidance", type=float, default=0.0,
                   help="All-pose logits = audio + strength * (audio - silence); independent full silence pass (0 disables)")
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--no-graphs", action="store_true", help="Eager decode for debugging")
    p.add_argument("--no-audio-decode", action="store_true", help="Write Mimi codes without loading the Mimi decoder")
    p.add_argument("--frame0", type=Path, help=".npy or .safetensors frame0_pose [133,3], y/x/conf pixels on a 480x256 canvas")
    p.add_argument("--package", type=Path, help="Video identity package with llm_inputs frame0, sink_latent and meta; no implicit media prefill")
    p.add_argument("--pose-latents", action="store_true", help="Decode pose and encode Electron pose latents; requires frame0 and even --frames")
    p.add_argument("--skeleton-video", action="store_true", help="Render a 25 fps skeleton preview with the generated audio")
    p.add_argument("--video-sample", action="store_true", help="Build the diffusion sample directory; requires --package; implies --pose-latents")
    a = p.parse_args(argv)
    if not np.isfinite(a.pose_audio_guidance) or not 0 <= a.pose_audio_guidance <= 5:
        p.error("--pose-audio-guidance must be finite and between 0 and 5")
    if a.ckpt is None:
        a.ckpt = Path("/home/param/models/transcript-pose") / a.recipe / "export-200"
    if a.video_sample:
        if not a.package:
            p.error("--video-sample requires --package with frame-zero geometry and a sink latent")
        a.pose_latents = True
    if a.out.exists() and any(a.out.iterdir()):
        p.error("--out must be empty; choose a new output directory")
    return a


def engine_modules(tokenizer_path):
    # Data preparation uses legacy CPU modules; its sys.path entries must not
    # select the training model when importing the inference implementation.
    sys.path.insert(0, str(ENGINE / "ppm"))
    import fast_ppm as FP
    import ppm_io as IP
    FP.M.QWEN32 = str(tokenizer_path)
    return FP, IP


def sampling(a):
    tt, at, pt = (a.temp if v is None else v for v in (a.text_temp, a.audio_temp, a.pose_temp))
    strength = getattr(a, "pose_audio_guidance", 0.0)
    light_topk = getattr(a, "pose_light_topk", None)
    return dict(text_temp=tt, text_topk=a.text_topk, text_topp=1.0, min_p=a.min_p, min_p_rel=0,
                freq_pen=0.0, text_rep=1.0, allow_eob=0, free_text=0, word_greedy=0,
                cb0_temp=at, cb0_topk=a.audio_topk, light_temp=at, light_topk=a.audio_topk,
                audio_rep_pen=1.0, pose_temp=pt, pose_topk=a.pose_topk,
                pose_light_temp=pt, pose_light_topk=a.pose_topk if light_topk is None else light_topk,
                pose_cfg_on=int(strength > 0), pose_cfg_s=1.0 + strength,
                pose_cfg_cb0=1, pose_cfg_light=1, pose_cfg_speaking_only=0,
                pose_rep_cb0=1.0, pose_rep_light=1.0, pose_rep_win=0, adaptive_rep_on=0)


def guidance_silence(a, IP, devs, p_max, n_max):
    """Encode actual zero-waveform audio before allocating the large PPM model."""
    if not a.pose_audio_guidance:
        return None
    import torch
    frames = int(p_max) + int(n_max) + 8  # FastPPM's required p_max + n_fr.
    mimi = IP.Mimi(devs[-1])
    silence = mimi.encode(np.zeros(frames * 1920, dtype=np.float32))[:, :frames]
    del mimi
    gc.collect()
    with torch.cuda.device(devs[-1]):
        torch.cuda.empty_cache()
    if silence.shape != (DEPTH, frames) or silence.min() < 0 or silence.max() >= CBS:
        raise RuntimeError("Mimi did not encode the full digital-silence conditioning sequence")
    return np.ascontiguousarray(silence, dtype=np.int64)


def guidance_metadata(a, IP, silence, engine, row=0):
    if silence is None:
        return {"enabled": False}
    np.save(a.out / "guidance_silence_mimi.npy", silence)
    audit = engine.guidance_audit(row)
    return dict(enabled=True, strength=a.pose_audio_guidance, engine_scale=1.0 + a.pose_audio_guidance,
                formula="audio + strength * (audio - silence)", scope="all pose codebooks",
                silence_source="Mimi encoding of digital silence", silence_sample_rate=24000,
                silence_codec=f"{IP.MIMI_REPO}@{IP.MIMI_REV}", silence_frames=silence.shape[1],
                silence_codes_sha256=hashlib.sha256(silence.tobytes()).hexdigest(),
                full_model_silence_pass=True, shared_text_and_pose=True, audit=audit)


def continuation_force(instances, infos, f):
    """Finish delayed known media without exposing extra native-text frames."""
    prefills = {int(info["prefill_frames"]) for info in infos}
    if len(prefills) != 1:
        raise ValueError("Batch rows must have the same prefill duration")
    npf = next(iter(prefills))
    force = {}
    if npf and CD <= f < npf + CD:
        force["codes"] = np.stack([inst["depth"][1, info["P"] + f] for inst, info in zip(instances, infos)])
        force["pose"] = np.stack([inst["pdepth"][info["P"] + f] for inst, info in zip(instances, infos)])
    if any(f >= info["frames"] for info in infos):
        force["lane"] = np.asarray([PAD if f >= info["frames"] else -1 for info in infos], np.int64)
    return force or None


def validate_prefill(result, inst, info):
    P, npf = info["P"], info["prefill_frames"]
    if not np.array_equal(result["lane_b"][:P + npf], inst["ids"][1, :P + npf]):
        raise RuntimeError("Source text/prefix changed during generation")
    span = slice(P + CD, P + CD + npf)
    if (not np.array_equal(result["dep_b"][span], inst["depth"][1, span])
            or not np.array_equal(result["pdep"][span], inst["pdepth"][span])):
        raise RuntimeError("Source audio/pose prefill changed during generation")


def read_frame0(a):
    path = a.frame0 or (a.package / "llm_inputs.safetensors" if a.package else None)
    if path is None:
        return None
    if path.suffix == ".safetensors":
        from safetensors.numpy import load_file
        frame0 = load_file(str(path))["frame0_pose"]
    else:
        frame0 = np.load(path, allow_pickle=False)
    frame0 = np.asarray(frame0, np.float32)
    if frame0.shape != (133, 3) or not np.isfinite(frame0).all():
        raise ValueError("frame0 must be finite [133,3] y,x,confidence on the 480x256 canvas")
    if abs(float(frame0[5, 1] - frame0[6, 1])) <= 0:
        raise ValueError("frame0 shoulders must have nonzero horizontal separation for pose normalization")
    return frame0


def main():
    a = arguments()
    began = time.time()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["PPM_CD"] = str(CD)
    fa = ROOT / ".cache/kernels/vllm-flash-attn3"
    if fa.exists():
        os.environ.setdefault("FA3_DIR", str(fa))
    voice = None
    if a.voice_clip:
        if a.voice_clip.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"):
            # CPU encoding keeps input preparation usable without reserving GPUs.
            _, IP = engine_modules(a.tokenizer)
            voice = IP.Mimi("cpu").encode(IP.read_wav24(str(a.voice_clip)))
        else:
            voice = codes_file(a.voice_clip)
    inst, info, tokenizer, voice = make_inputs(a, voice)
    n, P, f0 = info["frames"], info["P"], info["f0"]
    frame0 = read_frame0(a)
    if a.pose_latents or a.skeleton_video:
        if frame0 is None:
            raise ValueError("Pose rendering requires --frame0 or --package")
        if n % 2:
            raise ValueError("Pose/video export requires an even media frame count; specify --frames to avoid codec trimming")
    if a.video_sample:
        for name in ("sink_latent.safetensors", "refs", "meta.json"):
            if not (a.package / name).exists():
                raise ValueError(f"Video identity package is missing {name}: {a.package}")
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "inputs.npz", **{k: v for k, v in inst.items() if isinstance(v, (int, np.ndarray))})
    (a.out / "input.json").write_text(json.dumps(info, indent=2) + "\n")
    if voice is not None:
        np.save(a.out / "voice_clip_mimi.npy", voice)
    if frame0 is not None:
        np.save(a.out / "frame0_pose.npy", frame0)
    if a.prepare_only:
        print(json.dumps(info, indent=2), flush=True)
        return

    import torch
    from safetensors.numpy import save_file
    FP, IP = engine_modules(a.tokenizer)
    IP._TOK[0] = tokenizer
    if (a.ckpt / "meta.json").exists():
        cfg = json.loads((a.ckpt / "meta.json").read_text()).get("cfg", {})
        if cfg.get("recipe") not in (None, a.recipe):
            raise ValueError(f"Checkpoint recipe {cfg['recipe']} does not match --recipe {a.recipe}")
    params = sampling(a)
    clean, bad = FP.validate(params)
    if bad or clean != params:
        raise ValueError(f"Invalid sampling parameters: {bad}")
    devs = [torch.device(f"cuda:{int(x)}") for x in a.devs.split(",")]
    torch.cuda.set_device(devs[-1])
    with torch.inference_mode():
        silence = guidance_silence(a, IP, devs, P, n)
        m, ldev, cfgs, ck = FP.build_model(str(a.ckpt), devs)
        if not ck["pose"] or not ck["decp_bb"] or ck["decp_bb_alpha"] != 1.0:
            raise ValueError(f"Expected the Xabi pose architecture with D33 alpha=1: {ck}")
        # Prefix length is not capped at the old demo's 2048; aligned text can
        # occupy nearly half of the full 7500-position training context.
        eng = FP.FastPPM(m, ldev, cfgs, devs, n_inst=1, cfg_rows=silence is not None,
                         max_s=n / FPS, p_max=P, mimi_stream=False, silence=silence,
                         per_instance_rng=True, forced_text_only=info["force_text"])
        if not a.no_graphs:
            eng.capture(f=CD, P=P)
        t0 = time.time()
        eng.start([inst], a.seed, params, f0=f0)
        if any(eng.params[k] != v for k, v in params.items()):
            raise RuntimeError("Effective inference parameters differ from the requested settings")
        t1 = time.time()
        for f in range(f0, n + CD):
            # Training ends with CD PAD text slots while the last media frames
            # are flushed. The compact lane is otherwise sampled independently.
            force = continuation_force([inst], [info], f)
            eng.step(f, graphs=not a.no_graphs, cold_pad=not info["force_text"], force=force)
        result = eng.readback(0, P + n + CD)
        gen_s = time.time() - t1
        audio = (result["dep_b"][P + CD:P + CD + n] - OFF).T
        pose = (result["pdep"][P + CD:P + CD + n] - OFF).T
        lane = result["lane_b"][P:P + n]
        for codes in (audio, pose):
            if codes.shape != (DEPTH, n) or codes.min() < 0 or codes.max() >= CBS:
                raise RuntimeError("Generation did not produce all requested media frames")
        if not np.all(result["lane_a"] == PAD) or not np.all(result["dep_a"] == -1):
            raise RuntimeError("Monoplex inference unexpectedly wrote channel A")
        if info["force_text"] and not np.array_equal(result["lane_b"], inst["ids"][1]):
            raise RuntimeError("Framewise text differs from the supplied aligned lane")
        validate_prefill(result, inst, info)
        words = IP.lane_words(result["lane_b"][:P + n], P, P, P + f0)
        np.savez_compressed(a.out / "tokens.npz", audio_codes=audio.astype(np.int16), pose_codes=pose.astype(np.int16),
                            lane=lane, prefill_frames=np.int64(a.prefill_frames))
        save_file({"codes": np.ascontiguousarray(audio.astype(np.int16))}, str(a.out / "mimi.safetensors"))
        save_file({"codes": np.ascontiguousarray(pose.astype(np.int64))}, str(a.out / "pose_tokens.safetensors"))
        (a.out / "words.json").write_text(json.dumps(dict(words=words, lane=lane.tolist(),
            text=" ".join(w["text"] for w in words), supplied_transcript=info["supplied_transcript"],
            source_transcript=info["source_transcript"]), indent=2) + "\n")
        meta = dict(**info, checkpoint=ck, sampling=eng.params, seed=a.seed,
                    audio_guidance=guidance_metadata(a, IP, silence, eng),
                    generated_lane_sha256=hashlib.sha256(np.ascontiguousarray(lane, dtype=np.int64).tobytes()).hexdigest(),
                    rng_policy="per-instance per-modality generators; row-wise [1,vocab] multinomial",
                    prefill_seconds=t1 - t0, generation_seconds=gen_s,
                    ms_per_step=gen_s * 1000 / (n + CD - f0),
                    ground_truth_generation_media=False, fixed_duration=True,
                    peak_gib={str(d): torch.cuda.max_memory_allocated(d) / 2**30 for d in devs})
        del eng, m
        gc.collect()
        for d in devs:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
        if not a.no_audio_decode:
            mimi = IP.Mimi(devs[-1])
            wav = mimi.decode(audio)
            IP.write_wav(str(a.out / "audio.wav"), wav)
            meta["audio_stats"] = IP.audio_stats(wav)
            if voice is not None:
                IP.write_wav(str(a.out / "voice_clip.wav"), mimi.decode(voice))
            del mimi
            torch.cuda.empty_cache()
        if a.pose_latents or a.skeleton_video:
            from render import render_outputs
            with torch.cuda.device(devs[0]):
                meta.update(render_outputs(a, pose, frame0))
    meta["wall_seconds"] = time.time() - began
    (a.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Completed {a.recipe}: {n} frames ({n / FPS:.2f} s) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
