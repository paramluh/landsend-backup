"""PPM pose server: request -> engine instance (re-imported by the worker for every request, so prompt changes need no model reload).
System prompt = the live Aloe demo's builder (aloe-m1vox-live/prompts_ppm.build_prefix = the trainer's sysprompt.build_prefix with B-only
segments; SELFTEST: the clip's training-path row prefix is reproduced exactly): B's voice clip (rule = the clip's training-rule clip taken from
its row; none; or an uploaded voice wav: mono 24 kHz, -23 LUFS, Mimi, 3-10 s) + optional persona text (the demo's persona format, rendered by
the training renderer between <sop>/<eop>; empty = no persona). Conversation prefill:
  clip   : the clip's real Mimi + pose for frames 0..PF-1, the store row's word lane for frames 0..PF+CD-1 (training-path row);
  upload : the uploaded wav (first prefill_s; mono 24 kHz, -23 LUFS, Mimi) with the word lane PAD (no ASR); pose = the uploaded pose npz
           (prefill_pose [32, >= PF], global_state [3], frame0_pose [133, 3], optional sink_latent [48, 30, 16] for video) or, without one,
           the pose is generated from frame 0 while the upload's audio is teacher-forced for the prefill frames ('force' plan).
GT FORCING (10-23, request gt_force; clip prompts with the rule voice clip and no persona only): B's word lane and Mimi codes are teacher-forced
  from the first generated frame on to the clip's GT (rows/<id>.gt12.npz: the package's GT Mimi + its timed GT words placed by the trainer's
  lane rule, 12 s), via the engine's TEXT FORCING (inst text_force) and AUDIO FORCING (inst audio_force); only the pose is sampled. Every
  generated frame must be covered by the GT (else ValueError)."""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-m1vox-live")
import prompts_ppm as PR  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

CD, DEPTH, CBS, PAD = 2, 32, 2048, PR.D.PAD_ID
OFF = np.arange(DEPTH, dtype=np.int64) * CBS
DEFAULT_GS = np.array([0.42, 0.5, 0.33], np.float32)        # frame-0 root (y, x) + shoulder width for stick figures of pose-less uploads


def wav_codes(path, mimi, max_s=None):
    """wav -> mono 24 kHz float32, -23 LUFS (pyloudnorm, as the demo's uploads / the chunk gain), -> Mimi codes [32, F]."""
    import soundfile as sf
    import pyloudnorm as pyln
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 24000:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr), 24000)
        x = resample_poly(x, 24000 // g, int(sr) // g).astype(np.float32)
    if max_s:
        x = x[:int(max_s * 24000)]
    m = pyln.Meter(24000)
    ld = m.integrated_loudness(x) if len(x) > 24000 * 0.5 else -70
    if np.isfinite(ld) and ld > -70:
        x = pyln.normalize.loudness(x, ld, -23.0)
    x = np.clip(x, -0.995, 0.995).astype(np.float32)
    n = len(x) // 1920
    codes = np.asarray(mimi.encode(x[:n * 1920]))[:, :n]
    return codes.astype(np.int64), x


def rule_clip(x):
    """the training-rule voice clip of a row pkl (its prefix's clip codes) -> a prompts_ppm clip dict (audio codes only, blank lane)."""
    P = int(x["P"])
    pos = np.flatnonzero(x["depth"][1, :P, 0] >= 0)
    if not len(pos):
        return None
    codes = (x["depth"][1, pos] - OFF).T
    return PR.upload_clip(codes)


def build(req, d, mimi, clipinfo):
    """-> (inst dict for FastPPM.start, plan dict). plan: f0, n, force (None or {'codes': [PF, 32] global, 'lane': [PF + CD]}), info."""
    PF, NG = int(req["frames_prefill"]), int(req["frames_gen"])
    n = PF + NG
    info = {"prompt": req.get("prompt", "clip")}
    x = None
    if info["prompt"] == "clip":
        x = pickle.load(open(clipinfo["row"] + ".pkl", "rb"))
    # ---- system prompt
    vc = req.get("voice_clip", "rule")
    clip = None
    if vc == "rule":
        if x is None:
            raise ValueError("voice_clip 'rule' needs a clip prompt (uploads: none or upload)")
        clip = rule_clip(x)
    elif vc == "upload":
        vcodes, _ = wav_codes(f"{d}/voice.wav", mimi, max_s=10.0)
        if vcodes.shape[1] < 38:
            raise ValueError("uploaded voice clip shorter than 3 s")
        clip = PR.upload_clip(vcodes[:, :125])
    persona = (req.get("persona") or "").strip()
    spx, pinfo = PR.build_prefix(persona, clip)
    P = int(spx["P"])
    if x is not None and vc == "rule" and not persona:                 # SELFTEST: = the training-path row's prefix exactly
        for k in ("ids", "depth", "present"):
            assert np.array_equal(np.asarray(spx[k]), np.asarray(x[k])[:, :P]) and P == int(x["P"]), ("prefix != the row's", k)
        info["prefix_equals_training_row"] = True
    info.update(P=P, persona_tokens=pinfo.get("persona_tokens"), persona_cut=pinfo.get("persona_cut"), voice_clip=vc,
                voice_clip_frames=pinfo.get("clip_frames"))
    Tn = P + n + CD
    ids = np.full((2, Tn), PAD, np.int64)
    depth = np.full((2, Tn, DEPTH), -1, np.int64)
    present = np.zeros((2, Tn), np.float32)
    pdepth = np.full((Tn, DEPTH), -1, np.int64)
    ids[:, :P], depth[:, :P], present[:, :P] = np.asarray(spx["ids"]), np.asarray(spx["depth"]), np.asarray(spx["present"])
    present[1, P:] = 1.0
    force = None
    if info["prompt"] == "clip":
        pd = clipinfo["prompt_dir"]
        gm = load_file(f"{pd}/gt/mimi_local.safetensors")["codes"].astype(np.int64)
        gp = load_file(f"{pd}/gt/pose_tokens.safetensors")["codes"].astype(np.int64)
        if PF > min(gm.shape[1], gp.shape[1]):
            raise ValueError(f"prefill {PF} frames > the clip's {min(gm.shape[1], gp.shape[1])} real frames")
        Px = int(x["P"])
        nl = x["ids"].shape[1] - Px
        if PF + CD > nl:
            raise ValueError("prefill longer than the row's word lane")
        L = min(n + CD, nl)
        ids[1, P:P + L] = x["ids"][1, Px:Px + L]
        depth[1, P + CD:P + CD + PF] = gm[:, :PF].T + OFF
        pdepth[P + CD:P + CD + PF] = gp[:, :PF].T + OFF
        li = load_file(f"{pd}/llm_inputs.safetensors")
        info.update(global_state=li["global_state"].astype(np.float32), frame0_pose=li["frame0_pose"].astype(np.float32),
                    sink=f"{pd}/gt/video_latents.safetensors", row_key=x.get("row_key"), held_out=clipinfo["held_out"], f0=PF + CD)
    else:
        ac, _ = wav_codes(f"{d}/upload.wav", mimi)
        if ac.shape[1] < PF:
            raise ValueError(f"uploaded audio has {ac.shape[1]} frames < prefill {PF}")
        info.update(held_out=None, row_key=None)
        if os.path.exists(f"{d}/upload_pose.npz"):
            z = np.load(f"{d}/upload_pose.npz")
            pp = z["prefill_pose"].astype(np.int64)
            if pp.shape[0] != DEPTH or pp.shape[1] < PF:
                raise ValueError("upload pose: prefill_pose must be [32, >= prefill frames]")
            depth[1, P + CD:P + CD + PF] = ac[:, :PF].T + OFF
            pdepth[P + CD:P + CD + PF] = pp[:, :PF].T + OFF
            info.update(global_state=z["global_state"].astype(np.float32), frame0_pose=z["frame0_pose"].astype(np.float32) if "frame0_pose" in z else None,
                        f0=PF + CD)
            if "sink_latent" in z and "frame0_pose" in z:
                from safetensors.numpy import save_file
                save_file({"latents": z["sink_latent"].astype(np.float32).reshape(1, 48, 30, 16)}, f"{d}/upload_sink.safetensors")
                info["sink"] = f"{d}/upload_sink.safetensors"
            else:
                info["sink"] = None
        else:                                                            # no pose: generate it from frame 0, force the upload's audio
            force = {"codes": ac[:, :PF].T + OFF, "lane": np.full(PF + CD, PAD, np.int64)}
            info.update(global_state=DEFAULT_GS, frame0_pose=None, sink=None, f0=0, pose_from_frame0=True)
        info["upload_frames"] = int(ac.shape[1])
    info.update(n=n, PF=PF, NG=NG)
    inst = dict(P=P, ids=ids, depth=depth, present=present, pdepth=pdepth, a_mode="empty")
    if req.get("gt_force"):
        if info["prompt"] != "clip" or vc != "rule" or persona:
            raise ValueError("gt_force: clip prompt with the rule voice clip and no persona only (the row's own prefix)")
        eng = sys.modules.get("fast_ppm_srv")
        assert eng is not None and "AUDIO FORCING" in (eng.__doc__ or ""), "gt_force: the worker's engine predates AUDIO FORCING (rebuild it)"
        g = np.load(clipinfo["row"] + ".gt12.npz")
        lane, gd = g["lane"].astype(np.int64), g["depth"].astype(np.int64)
        f0 = info["f0"]
        a, b = P + f0, P + n + CD                                        # generated positions
        assert int(g["P"]) == P and len(lane) >= b and gd.shape[0] >= b and gd.shape[1] == DEPTH, ("gt12 arrays", g["P"], P, lane.shape, gd.shape)
        assert np.array_equal(lane[:a], ids[1, :a]), "gt12 lane != the request's prefix + prefill lane"
        if (gd[a:b] < 0).any() or (lane[a:b] < 0).any():
            raise ValueError(f"gt_force: GT covers {int((gd[a:b] >= 0).all(1).sum())} of the {b - a} generated frames (shorten gen_s)")
        ids[1, a:b] = lane[a:b]                                          # the request's lane = GT over the generated span too (records only)
        af = np.full((len(ids[1]), DEPTH), -1, np.int64)
        af[a:b] = gd[a:b]
        inst.update(text_force=ids[1].copy(), audio_force=af)
        info["gt_force"] = {"positions": [a, b], "src": clipinfo["row"] + ".gt12.npz"}
    return inst, {"force": force, "info": info}


def selftest(clips):
    """every clip's rule prefix == its training-path row prefix (CPU)."""
    out = []
    for c in clips:
        x = pickle.load(open(c["row"] + ".pkl", "rb"))
        spx, _ = PR.build_prefix("", rule_clip(x))
        P = int(spx["P"])
        ok = P == int(x["P"]) and all(np.array_equal(np.asarray(spx[k]), np.asarray(x[k])[:, :P]) for k in ("ids", "depth", "present"))
        out.append((c["id"], ok, P))
    return out
