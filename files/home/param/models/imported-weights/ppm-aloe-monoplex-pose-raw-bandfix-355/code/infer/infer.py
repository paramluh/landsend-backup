#!/usr/bin/env python3
"""Pose (+ speech + words) inference with the PPM 32B MONOPLEX pose fine-tune (aloe-pose-mono-8020, step 357). See README.md.

  python infer.py --ckpt CKPT_DIR --package PROMPT_DIR --out OUT_DIR            # (a) a studio prompt package
  python infer.py --ckpt CKPT_DIR --row ROW.pkl --out OUT_DIR                    # (b) an Aloe-format row (trainer micro)

One speaker on channel B (speech + words + pose), channel A empty. Positions [0, P + F0) are teacher-forced (system block = voice clip
[+ persona], then the first --prefill-s seconds of real audio, pose and words; F0 = prefill frames + code delay 2), then:
  --mode all   words, Mimi audio and pose are all sampled to --seconds (default; tonight's studio eval path)
  --mode pose  words and audio are teacher-forced to the given track for the whole window; only the pose is sampled
Outputs in OUT_DIR: audio.wav, voice_clip.wav, words.json, tokens.npz, pose_tokens.safetensors, meta.json
  [+ pose_latents_electron_v3.safetensors with --pose-latents; + video_sample/ (pose-demo-website demo-v3 SAMPLE_DIR) with --video-sample]."""
import argparse
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ppm"))
os.environ.setdefault("PPM_CD", "2")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------------------------- data licences (asserted)
# Every data file shipped in this repo and every artefact fetched at run time, with its licence. Commercial use must be allowed.
COMMERCIAL_OK = {"Apache-2.0", "CC-BY-4.0", "BSD-3-Clause", "internal (Canopy Labs)"}
BUNDLED_DATA = {"ppm/configs/Qwen3-32B": "Apache-2.0",          # Qwen/Qwen3-32B config.json + tokenizer files (weights NOT shipped)
                "ppm/configs/Qwen3-4B": "Apache-2.0"}           # Qwen/Qwen3-4B config.json (tower architecture)
RUNTIME_DATA = {"InternalCan/ppm-aloe-pose-mono-8020": "internal (Canopy Labs)",       # the checkpoint
                "kyutai/mimi": "CC-BY-4.0",                                             # Mimi codec
                "kernels-community/vllm-flash-attn3": "BSD-3-Clause",                   # FA3 kernels
                "InternalCan/pose-tokenizer-32cb-200k-noshift": "internal (Canopy Labs)",  # --pose-latents only
                "vanshjjw/electron-vae": "internal (Canopy Labs)"}                      # --pose-latents only
assert all(v in COMMERCIAL_OK for v in {**BUNDLED_DATA, **RUNTIME_DATA}.values()), "every data source must be commercially licensable"
assert all(os.path.isdir(os.path.join(HERE, k)) for k in BUNDLED_DATA), "bundled data dirs missing"

import numpy as np  # noqa: E402

NPF_PKG = 25                                   # a prompt package's llm_inputs prefill: 25 frames = 2.0 s at 12.5 Hz
CLIP_MIN, CLIP_MAX = 38, 125                   # voice clip length in Mimi frames (3-10 s), the training rule (data_ppm voice_clips)
STEP = 357


def log(*a):
    print(f"[infer {time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_codes(path, key=None):
    """[32, F] int64 local codes from .safetensors ('codes' or the only tensor) / .npy / .npz."""
    if path.endswith(".safetensors"):
        from safetensors.numpy import load_file
        d = load_file(path)
        x = d[key or ("codes" if "codes" in d else next(iter(d)))]
    elif path.endswith(".npz"):
        d = np.load(path)
        x = d[key or ("codes" if "codes" in d else d.files[0])]
    else:
        x = np.load(path)
    x = np.asarray(x).astype(np.int64)
    assert x.ndim == 2 and x.shape[0] == 32 and x.min() >= 0 and x.max() < 2048, (path, x.shape)
    return x


def audio_codes(path, mimi):
    """a .wav (any rate, mono-mixed, Mimi-encoded here) or saved Mimi codes -> [32, F] local codes."""
    if path.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
        import ppm_io as IP
        return mimi.encode(IP.read_wav24(path))
    return load_codes(path)


def read_words(path):
    """words file -> [[start_s, end_s, text], ...] sorted by start. Accepts a list of [start, end, text] / [text, start, end] /
    {"start", "end", "text"} entries, or {"words": [...]}."""
    w = json.load(open(path))
    w = w["words"] if isinstance(w, dict) else w
    out = []
    for x in w:
        if isinstance(x, dict):
            out.append([float(x["start"]), float(x.get("end", x["start"])), str(x["text"])])
        elif isinstance(x[0], str):
            out.append([float(x[1]), float(x[2]), x[0]])
        else:
            out.append([float(x[0]), float(x[1]), str(x[2])])
    return sorted(out, key=lambda x: x[0])


def lane_from_words(words, n, first_word_space=False):
    """[[start, end, text]] (seconds from frame 0) -> lane [n]: the training rows' placement (speech_q.word_pieces + lane +
    insert_new_word): first piece on frame floor(start x 12.5), later pieces on the following frames, NEW_WORD on the PAD frame before."""
    import ppm_io as IP
    ws = [{"text": str(t), "start": float(s), "end": float(e)} for s, e, t in words if s >= 0]
    pre = [{"text": "x"}] if first_word_space else []  # window starts mid-track: its first word keeps the leading space it has in the row
    pieces = IP.SP.word_pieces(IP.tok(), pre + ws)[len(pre):]
    pid, offs = [], [0]
    for p in pieces:
        pid += p; offs.append(len(pid))
    wif = np.zeros(n, np.int64)
    lane = IP.SP.lane(ws, pid, offs, n, wif)
    IP.SP.insert_new_word(lane, wif)
    return lane                                        # (without first_word_space this is ppm_io.words_lane exactly)


def system_block(clip, persona):
    """voice clip codes [32, L] local (or None) + persona text (or None) -> sysprompt.build_prefix dict (B = the only channel)."""
    import sysprompt as SPR
    segs = []
    if clip is not None:
        L = int(clip.shape[1])
        segs.append({"kind": "speech", "track": 1, "w0": 0, "L": L, "src_row": {"codes_assistant": clip}, "src_side": "assistant",
                     "side": "assistant", "blank_lane": True})
    if persona:
        import persona_text_s3q as PT
        ids = [PT.SOP] + PT.encode(PT.block_text({"persona": persona}, ["persona"])) + [PT.EOP]
        segs.append({"kind": "text", "track": 1, "ids": ids})
    if not segs:
        segs = [{"kind": "text", "track": 1, "ids": []}]                   # the EMPTY system block (5 positions)
    sp = SPR.build_prefix({}, segs, {"new_word": True})
    assert 5 <= sp["P"] <= 2000, ("system prompt length outside the training cap", sp["P"])
    return sp


def assemble(sp, lane, audio, pose, n, npf_pose):
    """system block + conversation -> trainer-layout arrays (global code ids, code delay CD = 2 for Mimi and pose).
    lane [>= n + CD] B's words lane; audio [32, Fa] local Mimi codes of frames 0..Fa-1 (Fa <= n); pose [32, >= npf_pose] local pose codes."""
    import ppm_io as IP
    CD, OFF, PAD = IP.CD, IP.OFF, IP.PAD
    P = int(sp["P"])
    T = P + n + CD
    ids = np.full((2, T), PAD, np.int64)
    depth = np.full((2, T, 32), -1, np.int64)
    present = np.zeros((2, T), np.float32)
    pdepth = np.full((T, 32), -1, np.int64)
    ids[:, :P], depth[:, :P], present[:, :P] = sp["ids"], sp["depth"], sp["present"]
    ids[1, P:] = lane[:n + CD]
    present[1, P:] = 1.0
    fa = min(int(audio.shape[1]), n)
    depth[1, P + CD:P + CD + fa] = audio[:, :fa].T + OFF
    pdepth[P + CD:P + CD + npf_pose] = pose[:, :npf_pose].T + OFF
    return {"P": P, "ids": ids, "depth": depth, "present": present, "pdepth": pdepth, "pid": np.arange(T)}


def package_inputs(a, mimi, n_req, npf):
    """(a) studio prompt package (pose-demo-website demo-data/prompts/<id>) -> (row dict, n, info)."""
    from safetensors.numpy import load_file
    pk = a.package.rstrip("/")
    meta = json.load(open(f"{pk}/meta.json"))
    li = load_file(f"{pk}/llm_inputs.safetensors")
    pa, pp = li["prefill_audio"].astype(np.int64), li["prefill_pose"].astype(np.int64)
    assert pa.shape == (32, NPF_PKG) and pp.shape == (32, NPF_PKG), (pa.shape, pp.shape)
    gt_a = f"{pk}/gt/mimi_local.safetensors"
    gt_p = f"{pk}/gt/pose_tokens.safetensors"
    if npf > NPF_PKG:                                   # longer prefill: the package's ground-truth tracks (12 s), checked against llm_inputs
        ga, gp = load_codes(gt_a), load_codes(gt_p)
        assert np.array_equal(ga[:, :NPF_PKG], pa) and np.array_equal(gp[:, :NPF_PKG], pp), "package gt/ != llm_inputs prefill"
        assert npf <= min(ga.shape[1], gp.shape[1]), ("prefill longer than the package's ground truth", npf, ga.shape[1])
        pa, pp = ga[:, :npf], gp[:, :npf]
    # voice clip (system block)
    clip, clip_src = None, None
    if not a.no_voice_clip:
        cands = [a.voice_clip] if a.voice_clip else [f"{pk}/{f}" for f in ("voice_clip_mimi.safetensors", "voice_clip.wav")]
        cands = [c for c in cands if c and os.path.exists(c)]
        assert cands, ("no voice clip: pass --voice-clip WAV|CODES (3-10 s of this speaker; tonight: a later chunk of the same recording), "
                       "put voice_clip.wav / voice_clip_mimi.safetensors in the package, or --no-voice-clip")
        clip, clip_src = audio_codes(cands[0], mimi), cands[0]
        assert clip.shape[1] >= CLIP_MIN, ("voice clip shorter than 3 s (training clips: 3-10 s)", clip.shape[1] / 12.5)
        clip = clip[:, :CLIP_MAX]                                              # <= 10 s, as the training rule
    # words
    wpath = a.words or f"{pk}/words.json"
    assert os.path.exists(wpath) or a.no_prefill_words, ("no word timings: pass --words JSON ([[start_s, end_s, text], ...] in package time) or "
                                                         "put words.json in the package (or --no-prefill-words for a PAD lane)", wpath)
    words = read_words(wpath) if os.path.exists(wpath) else []
    # window + audio track
    if a.mode == "pose":
        src = a.force_audio or gt_a
        audio = audio_codes(src, mimi)
        assert np.array_equal(audio[:, :npf], pa[:, :npf]) or a.force_audio, "forced audio != the package prefill audio"
        n = min(n_req, int(audio.shape[1]))
        assert n > npf, ("forced audio shorter than the prefill", audio.shape[1], npf)
        if not words or words[-1][0] < 0.8 * n / 12.5:
            log(f"WARNING --mode pose: the word timings end at {words[-1][0] if words else 0:.1f} s of a {n / 12.5:.1f} s window "
                "(forced lane is PAD after the last word)")
    else:
        audio, src, n = pa, f"{pk}/llm_inputs.safetensors prefill_audio", n_req
    lane = lane_from_words(words, n + 2, a.first_word_space)
    sp = system_block(clip, a.persona)
    x = assemble(sp, lane, audio if a.mode == "pose" else pa[:, :npf], pp, n, npf)
    f0 = None
    if "frame0_pose" in li:
        f0 = li["frame0_pose"].astype(np.float32)
    info = {"input": "package", "package": pk, "identity": meta.get("prompt"), "category": meta.get("category"), "uuid": meta.get("uuid"),
            "ref_seed_key": meta.get("ref_seed_key"), "voice_clip": clip_src, "voice_clip_frames": None if clip is None else int(clip.shape[1]),
            "words_file": wpath if words else None, "audio_track": src, "persona": a.persona or None}
    return x, n, info, f0, clip


def row_inputs(a, mimi, n_req, npf):
    """(b) Aloe-format row: the trainer micro (P, ids [2, T], depth [2, T, 32], present [2, T], pdepth [T, 32]; global code ids, -1 none)."""
    import ppm_io as IP
    CD = IP.CD
    if a.row.endswith(".npz"):
        z = np.load(a.row, allow_pickle=False)
        x = {k: z[k] for k in ("ids", "depth", "present", "pdepth")}
        x["P"] = int(z["P"])
    else:
        x = pickle.load(open(a.row, "rb"))
    P = int(x["P"])
    ids, depth, present, pdepth = (np.asarray(x[k]) for k in ("ids", "depth", "present", "pdepth"))
    assert ids.shape[0] == 2 and depth.shape[1:] == (ids.shape[1], 32) and pdepth.shape == (ids.shape[1], 32), "not an Aloe-format row"
    assert (ids[0] == IP.PAD).all() and (depth[0] < 0).all() and (present[0] == 0).all(), "monoplex model: channel A must be empty"
    n_av = int(ids.shape[1]) - P - CD
    nreal_a = int((depth[1, P + CD:, 0] >= 0).sum())
    nreal_p = int((pdepth[P + CD:, 0] >= 0).sum())
    assert nreal_p >= npf and nreal_a >= npf, ("row has less real audio / pose than the prefill", nreal_a, nreal_p, npf)
    n = min(n_req, n_av if a.mode == "all" else nreal_a)
    clip = None
    cpos = np.flatnonzero(depth[1, :P, 0] >= 0)
    if len(cpos):
        clip = (depth[1, cpos] - IP.OFF).T
    if a.voice_clip or a.persona or a.no_voice_clip:          # re-prompt: replace the row's own system block
        clip_new = None if a.no_voice_clip else (audio_codes(a.voice_clip, mimi) if a.voice_clip else clip)
        if clip_new is not None:
            clip_new = clip_new[:, :CLIP_MAX]
        sp = system_block(clip_new, a.persona)
        clip = clip_new
    else:
        sp = {"P": P, "ids": ids[:, :P], "depth": depth[:, :P], "present": present[:, :P]}
    audio = (depth[1, P + CD:P + CD + n] - IP.OFF).T if a.mode == "pose" else (depth[1, P + CD:P + CD + npf] - IP.OFF).T
    assert (audio >= 0).all(), "row audio has gaps inside the window"
    pose = (pdepth[P + CD:P + CD + npf] - IP.OFF).T
    xo = assemble(sp, ids[1, P:P + n + CD], audio, pose, n, npf)
    info = {"input": "row", "row": a.row, "row_key": x.get("row_key"), "row_prompt": "own" if sp.get("layout") is None else "rebuilt",
            "voice_clip_frames": None if clip is None else int(clip.shape[1]), "persona": a.persona or None,
            "audio_track": "the row's own Mimi codes", "words": "the row's own lane"}
    return xo, n, info, None, clip


class LazyMimi:
    """ppm_io.Mimi on `dev`, constructed on first use."""

    def __init__(self, dev):
        self.dev, self.m = dev, None

    def get(self):
        if self.m is None:
            import ppm_io as IP
            self.m = IP.Mimi(self.dev)
        return self.m

    def encode(self, x):
        return self.get().encode(x)

    def decode(self, c):
        return self.get().decode(c)


def params_of(a):
    t = a.temp
    tt, at, pt = (t if v is None else v for v in (a.text_temp, a.audio_temp, a.pose_temp))
    p = {"text_temp": tt, "text_topk": int(a.text_topk), "text_topp": 1.0, "min_p": float(a.min_p), "min_p_rel": 0, "freq_pen": 0.0,
            "text_rep": 1.0, "word_greedy": 0,   # 10-28 USER ORDER: no lane grammar knobs (allow_eob / free_text gone)
            "cb0_temp": at, "cb0_topk": int(a.audio_topk), "light_temp": at, "light_topk": int(a.audio_topk), "audio_rep_pen": 1.0,
            "pose_temp": pt, "pose_topk": int(a.pose_topk), "pose_light_temp": pt, "pose_light_topk": int(a.pose_topk),
            "pose_cfg_on": int(a.pose_cfg > 0), "pose_cfg_s": float(a.pose_cfg) if a.pose_cfg > 0 else 3.0, "pose_cfg_cb0": 1, "pose_cfg_light": 1,
            "pose_rep_cb0": 1.0, "pose_rep_light": 1.0, "pose_rep_win": 0, "adaptive_rep_on": 0, "audio_rep_speaking": 1.0,
            "audio_rep_silent": 1.0, "text_rep_speaking": 1.0, "text_rep_silent": 1.0, "pose_rep_cb0_speaking": 1.0,
            "pose_rep_cb0_silent": 1.0, "pose_rep_light_speaking": 1.0, "pose_rep_light_silent": 1.0}
    # PACKAGE PATCH 10-30: the old combined pose_rep* keys are aliases since fast_ppm's 10-26 cb0 / light split; validate() expands them,
    # so the old dict failed `clean == PARAMS`. Same values (1.0 = off). --params JSON then overrides any FP.PARAMS key (playground settings).
    if getattr(a, "params", None):
        import json as _json
        p.update(_json.load(open(a.params)) if a.params.endswith(".json") else _json.loads(a.params))
    return p


# the sampling of the 10-19 studio eval videos (gen_studio_nocfg.py PARAMS): the defaults must reproduce it exactly
TONIGHT = {"text_temp": 1.0, "text_topk": 0, "text_topp": 1.0, "min_p": 0.0, "min_p_rel": 0, "freq_pen": 0.0, "text_rep": 1.0,
           "word_greedy": 0,   # 10-28 USER ORDER: no lane grammar knobs (allow_eob / free_text gone)
           "cb0_temp": 1.0, "cb0_topk": 30, "light_temp": 1.0, "light_topk": 30, "audio_rep_pen": 1.0,
           "pose_temp": 1.0, "pose_topk": 30, "pose_light_temp": 1.0, "pose_light_topk": 30,
           "pose_cfg_on": 0, "pose_cfg_s": 3.0, "pose_cfg_cb0": 1, "pose_cfg_light": 1,
           "pose_rep_cb0": 1.0, "pose_rep_light": 1.0, "pose_rep_win": 0, "adaptive_rep_on": 0, "audio_rep_speaking": 1.0,
           "audio_rep_silent": 1.0, "text_rep_speaking": 1.0, "text_rep_silent": 1.0, "pose_rep_cb0_speaking": 1.0,
           "pose_rep_cb0_silent": 1.0, "pose_rep_light_speaking": 1.0, "pose_rep_light_silent": 1.0}   # PACKAGE PATCH: split keys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="checkpoint dir: the consolidated DCP export-357 (see README: fetch_checkpoint.sh)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--package", help="studio prompt package dir (meta.json, llm_inputs.safetensors, sink_latent.safetensors, refs/, gt/)")
    g.add_argument("--row", help="Aloe-format row (.pkl / .npz trainer micro: P, ids, depth, present, pdepth)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("all", "pose"), default="all",
                    help="all: sample words + audio + pose after the prefill (default); pose: force words + audio, sample pose only")
    ap.add_argument("--prefill-s", type=float, default=2.0, help="teacher-forced prefill in seconds (default 2.0 = 25 frames)")
    ap.add_argument("--seconds", type=float, default=60.0, help="total length incl. the prefill (default 60)")
    ap.add_argument("--voice-clip", help="voice clip for the system block: wav (3-10 s, Mimi-encoded here) or Mimi codes [32, L]")
    ap.add_argument("--no-voice-clip", action="store_true", help="no voice clip (default: voice clip ON)")
    ap.add_argument("--persona", default=None, help="persona text for the system block (default: OFF)")
    ap.add_argument("--words", help="package input: word timings JSON (default <package>/words.json)")
    ap.add_argument("--no-prefill-words", action="store_true", help="package input without word timings: PAD lane in the prefill")
    ap.add_argument("--first-word-space", action="store_true", help="package window starts mid-track: tokenise its first word with a space")
    ap.add_argument("--force-audio", help="--mode pose with a package: the audio track to force (wav or codes; default <package>/gt/mimi_local)")
    ap.add_argument("--temp", type=float, default=1.0, help="temperature of every stream (default 1)")
    ap.add_argument("--text-temp", type=float, default=None)
    ap.add_argument("--audio-temp", type=float, default=None)
    ap.add_argument("--pose-temp", type=float, default=None)
    ap.add_argument("--audio-topk", type=int, default=30, help="Mimi cb0 + cb1-31 top-k (default 30)")
    ap.add_argument("--pose-topk", type=int, default=30, help="pose cb0 + cb1-31 top-k (default 30)")
    ap.add_argument("--text-topk", type=int, default=0, help="text top-k (default 0 = off)")
    ap.add_argument("--min-p", type=float, default=0.0, help="text min-p (default 0 = off)")
    ap.add_argument("--pose-cfg", type=float, default=0.0, help="pose CFG scale (default 0 = OFF; >0 = silent-speech uncond, e.g. 3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params", default=None, help="PACKAGE PATCH: JSON file (.json) or JSON string overriding any fast_ppm PARAMS key, "
                    "e.g. config/playground_sampling.json (the playground's monoband355 settings)")
    ap.add_argument("--devs", default="0,1", help="CUDA devices (default 0,1; the 32B bf16 model needs 2 x 80 GB)")
    ap.add_argument("--pose-latents", action="store_true", help="also write pose_latents_electron_v3.safetensors (package input)")
    ap.add_argument("--video-sample", action="store_true", help="also write video_sample/ for pose-demo-website demo-v3/video.py (package)")
    a = ap.parse_args()
    assert os.environ.get("GEMM_PRECISION", "bf16") == "bf16", "bf16 only"
    if a.video_sample:
        a.pose_latents = True
    assert not (a.pose_latents and a.row), "--pose-latents / --video-sample need a package (frame0_pose, sink, refs)"
    os.makedirs(a.out, exist_ok=True)
    T00 = time.time()
    import torch
    import ppm_io as IP
    import fast_ppm as FP
    CD, FPS, OFF, PAD, DEPTH, CBS = FP.CD, FP.FPS, FP.OFF, FP.PAD, FP.DEPTH, FP.CBS
    npf = int(round(a.prefill_s * FPS))
    n_req = int(round(a.seconds * FPS))
    assert 1 <= npf < n_req, (npf, n_req)
    F0 = npf + CD
    PARAMS = params_of(a)
    devs = [torch.device(f"cuda:{int(x)}") for x in a.devs.split(",")]
    torch.cuda.set_device(devs[-1])
    with torch.inference_mode():
        mimi = LazyMimi(devs[-1])          # built on first use: after the model unless a wav input needs it (= the reference order)
        x, n, info, frame0, clip = (package_inputs if a.package else row_inputs)(a, mimi, n_req, npf)
        P = int(x["P"])
        assert x["ids"].shape[1] == P + n + CD
        log(f"input {info['input']}: P {P} (system block), window {n} frames = {n / FPS:.2f} s, prefill {npf} frames, mode {a.mode}")
        m, ldev, cfgs, binfo = FP.build_model(a.ckpt, devs)
        mimi.get()
        assert binfo["pose"] and binfo["decp_bb"], ("not the monoplex pose checkpoint", binfo)
        if binfo["step"] != STEP:
            log(f"WARNING checkpoint step {binfo['step']} != {STEP} (this README describes step {STEP})")
        max_s = n / FPS + 2
        p_max = 2048
        nfr = int(round(max_s * FPS)) + 8
        sil = mimi.encode(np.zeros((p_max + nfr) * 1920, np.float32))[:, :p_max + nfr]
        eng = FP.FastPPM(m, ldev, cfgs, devs, n_inst=1, cfg_rows=True, max_s=max_s, mimi_stream=False, silence=sil)
        eng.capture()
        clean, bad = FP.validate(PARAMS)
        assert not bad and clean == PARAMS, (bad, clean)
        nsteps = n + CD - F0
        inst = dict(x, a_mode="empty")
        CF = None
        if a.mode == "pose":
            inst["text_force"] = np.asarray(x["ids"][1], np.int64)
            CF = np.asarray(x["depth"][1][P + F0:P + n + CD], np.int64)
            assert CF.shape == (nsteps, DEPTH) and (CF >= 0).all(), ("forced audio must cover the window", CF.shape)
        t_load = time.time() - T00
        t0 = time.time()
        eng.start([inst], a.seed, PARAMS, f0=F0)
        assert all(eng.params[k] == v for k, v in PARAMS.items()), "engine params != requested"
        t1 = time.time()
        for i, f in enumerate(range(F0, F0 + nsteps)):
            eng.step(f, cold_pad=False, force=None if CF is None else {"codes": CF[i:i + 1]})
        S = eng.readback(0, P + n + CD)
        gen_s = time.time() - t1
        log(f"prefill {t1 - t0:.1f}s; {nsteps} steps in {gen_s:.1f}s ({1e3 * gen_s / nsteps:.1f} ms / step); load + capture {t_load:.0f}s")
        lane_b, dep_b, pdep = S["lane_b"], S["dep_b"], S["pdep"]
        # ---------------------------------------------------------------------------------------- checks
        assert (S["lane_a"][:P + n + CD] == PAD).all() and (S["dep_a"][:P + n + CD] < 0).all(), "channel A was written"
        c1 = (dep_b[P + CD:P + CD + n] - OFF).T
        p1 = (pdep[P + CD:P + CD + n] - OFF).T
        for c in (c1, p1):
            assert c.shape == (DEPTH, n) and c.min() >= 0 and c.max() < CBS, (c.shape, c.min(), c.max())
        pa = (x["depth"][1][P + CD:P + CD + npf] - OFF).T
        pp = (x["pdepth"][P + CD:P + CD + npf] - OFF).T
        assert (c1[:, :npf] == pa).all() and (p1[:, :npf] == pp).all(), "prefill codes != input"
        assert np.array_equal(lane_b[:P + F0], x["ids"][1][:P + F0]), "prefill lane != input"
        checks = {"channel_A_empty": True, "prefill_audio_pose_lane_equal_input": True}
        if a.mode == "pose":
            eq = lane_b[:P + n + CD] == x["ids"][1][:P + n + CD]
            eqa = dep_b[:P + n + CD] == x["depth"][1][:P + n + CD]
            assert eq.all() and eqa.all(), ("forced lane / audio != input", int((~eq).sum()), int((~eqa).sum()))
            checks.update(forced_lane_equal_input_positions=int(eq.size), forced_audio_equal_input_positions=int(eqa.size))
        # ---------------------------------------------------------------------------------------- outputs
        del eng
        w1 = mimi.decode(c1)
        IP.write_wav(f"{a.out}/audio.wav", w1)
        if clip is not None:
            IP.write_wav(f"{a.out}/voice_clip.wav", mimi.decode(clip))
        words = IP.lane_words(lane_b[:P + n], P, P, P + F0)
        gen = [w for w in words if not w["prompt"]]
        json.dump({"words": words, "prefill_s": npf / FPS, "prefill_lane_frames": F0, "text_prefill": " ".join(w["text"] for w in words if w["prompt"]),
                   "text_after_prefill": " ".join(w["text"] for w in gen), "lane": lane_b[P:P + n].tolist()}, open(f"{a.out}/words.json", "w"), indent=0)
        np.savez_compressed(f"{a.out}/tokens.npz", audio_codes=c1.astype(np.int16), pose_codes=p1.astype(np.int16), lane=lane_b[P:P + n],
                            prefill_frames=np.int64(npf))
        from safetensors.numpy import save_file
        save_file({"codes": np.ascontiguousarray(p1.astype(np.int64))}, f"{a.out}/pose_tokens.safetensors")
        save_file({"codes": np.ascontiguousarray(c1.astype(np.int16))}, f"{a.out}/mimi.safetensors")
    meta = {"model": "PPM 32B monoplex pose FT (aloe-pose-mono-8020)", "ckpt": a.ckpt, "step": binfo["step"], **info, "mode": a.mode,
            "P": P, "n_frames": n, "seconds": round(n / FPS, 2), "prefill_frames": npf,
            "generated": "words, audio, pose" if a.mode == "all" else "pose (words + audio forced)", "gen_words_after_prefill": len(gen),
            "sampling": PARAMS, "sampling_equals_10_19_studio_eval": PARAMS == TONIGHT, "rep_pen": None, "seed": a.seed, "dtype": "bf16",
            "engine": "fast_ppm (CUDA graphs, bf16; 1 row + 1 pose-CFG uncond row, unused when CFG is off)", "audio_stats": IP.audio_stats(w1[npf * 1920:]),
            "pose_gen_cb0_unique": int(len(np.unique(p1[0, npf:]))), "checks": checks, "ms_per_step": round(1e3 * gen_s / nsteps, 1),
            "load_capture_s": round(t_load, 1), "wall_s": None, "peak_gib": {str(d): round(torch.cuda.max_memory_allocated(d) / 2**30, 1) for d in devs},
            "made": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)            # the LLM outputs are complete here
    if a.pose_latents:
        del m
        torch.cuda.empty_cache()
        assert frame0 is not None, "package has no frame0_pose"
        sys.path.insert(0, HERE)
        import posetok
        from safetensors.torch import save_file as save_t
        lat = posetok.tokens_to_latents(torch.from_numpy(p1.astype(np.int64)), torch.from_numpy(frame0))
        save_t(lat, f"{a.out}/pose_latents_electron_v3.safetensors")
        meta["pose_latents"] = {"shape": list(lat["latents"].shape), "n_frames": int(lat["n_frames"]), "finite": bool(torch.isfinite(lat["latents"].float()).all())}
        log(f"pose latents {tuple(lat['latents'].shape)}")
        if a.video_sample:
            import shutil
            vs = f"{a.out}/video_sample"
            os.makedirs(vs, exist_ok=True)
            shutil.copyfile(f"{a.package}/sink_latent.safetensors", f"{vs}/sink_latent.safetensors")
            if not os.path.exists(f"{vs}/refs"):
                os.symlink(os.path.abspath(f"{a.package}/refs"), f"{vs}/refs")
            shutil.copyfile(f"{a.out}/pose_latents_electron_v3.safetensors", f"{vs}/pose_latents_electron_v3.safetensors")
            shutil.copyfile(f"{a.out}/mimi.safetensors", f"{vs}/mimi.safetensors")
            json.dump({"uuid": info["uuid"], "ref_seed_key": info["ref_seed_key"], "identity": info["identity"], "frames": n},
                      open(f"{vs}/meta.json", "w"), indent=1)
            meta["video_sample"] = vs
    meta["wall_s"] = round(time.time() - T00, 1)
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    log(f"done in {meta['wall_s']:.0f}s -> {a.out}: {len(gen)} words after the prefill: {' '.join(w['text'] for w in gen)[:240]}")


if __name__ == "__main__":
    main()
