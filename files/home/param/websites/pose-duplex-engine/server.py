#!/usr/bin/env python3
"""ALOE POSE-DUPLEX LIVE DEMO server (aloe-posedup-live, 10-25): ppm/aloe-m1vox-live/server.py (copied; every behaviour kept: sessions, the
non-preempting 'test' client, presets with sam-sep codes, voice clip only / no persona text by default, hot swap between sessions, 45 s
dead-session expiry, max 600 s, adaptive PAD bias, per-frame ws 'f' messages) serving the pose-duplex Aloe FT (job 30074: SAM speech duplex +
the assistant's pose on channel B) with live_ppm_pose.LivePPM (pose branch on the first GPU). Additions:
  * every ws 'f' message also carries "pose": the assistant's 32 pose codes generated this step (local ids; code frame "cf" = f - 2, null
    before) and "cf"; 'ready' carries "session" (the session id = its sessions/<id> dir = the feed's session) and "feed" / "person";
  * FEED for videort (feed.py; endpoint = first line of {STATE}/FEED_URL.txt, else env PPMLIVE_FEED_URL, else ws://slinky-10:8990/feed,
    re-read at every session start; 'off' = disabled): start / one 'f' per code frame (pose + Mimi codes + the frame's int16 audio, base64) /
    end, never blocking the session; every message also in sessions/<id>/feed.jsonl;
  * optional avatar person id ("person" in the start message; server default = {STATE}/PERSON.txt or env PPMLIVE_PERSON; default absent)
    -> the feed's start; api/presets 'avatar_url' = {STATE}/AVATAR_URL.txt or env PPMLIVE_AVATAR_URL (the page's Video avatar iframe);
  * defaults = PPMLIVE_DEFAULTS=nopersona (DEFAULTS.json = aloe-m1vox-live/NOPERSONA_DEFAULTS.json) + pose temperature 1 / top-k 30.
  * 10-25 POSE CFG (live_ppm_pose module doc; = fast_ppm's pose CFG): schema group 'Pose CFG' after 'Pose sampling': pose_cfg_on (bool, default
    off), pose_cfg_s (1-6, default 3), pose_cfg_cb0 / pose_cfg_light (bool, default on), all live (the uncond row always runs); its hint carries
    the measured per-frame latency with pose CFG on (start-up smoke, then refreshed by every session with >= 100 pose-CFG frames) and flags it
    when above the 80 ms real-time budget; every frame message carries pcf (pose CFG on that frame, 0/1) and pcs (its strength, 1 when off);
    session.npz records pcf per frame. Independent of the audio CFG (separate uncond row): both can be on.
----- the copied server's doc -----
LIVE VOICE DEMO server for the Voxtral-A Aloe fine-tune (aloe-m1vox-ft, job 22335): PPM (Qwen3-32B backbone + 4B speech tower + 0.6B
decoder), channel A = the caller's live mic through the frozen Voxtral Realtime encoder + the run's adapter x s, channel B (words + Mimi)
generated and streamed back. One session at a time on 2 GPUs. = qwen-1ch/infer/s4live-asr/server.py's protocol / session loop / hot swap /
non-preempting test path, with the PPM engine (live_ppm.LivePPM) and the Aloe prompt builder (prompts_ppm).

HTTP  GET /?token=...           the page (index.html, read from disk per request)
      GET /api/presets?token=   held-out Aloe personas / B voice clips, checkpoint, state, control schema
      POST /api/clip?token=     upload a voice clip for B (-> audio-only clip, -23 LUFS, Mimi codes, 3-10 s)
WS    /ws?token=...             client -> server: JSON {"type": "start", "client", "preset", "persona", "clip", params...} | {"type": "set", ...} |
                                {"type": "stop"} | {"type": "ping", "t"}; binary = int16 LE mono 24 kHz mic PCM (any chunking)
                                server -> client: JSON status / ready / per-frame {"type": "f", f, tok, ws, ctl, pp, pn, ms, lag, ps, pd} /
                                settings / done; binary = int16 B audio, 1920 samples per frame
The front proxy (proxy.py) owns the public port and injects the token; the plain URL needs none.
HOT SWAP: a watcher polls the candidate exports (env PPMLIVE_SWAP_TO: comma list, newest COMPLETE marker wins); a newer one is loaded
in-process between sessions (the proxy stays up; the page shows 'model loading').
TEST CLIENTS: only client 'browser' may take over a running session; any other client is refused while a session is active.
No ASR anywhere (standing order): the transcript is B's own word lane.
10-17 (aloe-v2vox hot swap): CK_LABELS.json (re-read per call) gives a checkpoint its page label and prompt mode; prompt 'voice_clip_only' (the
aloe-v2 / v2vox training prompt) = B's voice clip and NO text by default (an empty persona box no longer falls back to the preset's persona);
the run's last step comes from the checkpoint's own train cfg; s from the checkpoint's own run metrics (live_ppm.run_metrics_for); start-up
asserts the v2 prompt selftest (prompts_ppm.selftest_v2) next to the m1vox one.
10-17 ADAPTIVE REPETITION PENALTY (live_ppm.ADAPT_KEYS, live, default off): schema group 'Adaptive repetition penalty' (after the audio
rep-pen rows); every frame message carries spk (B SPEAKING: a word piece in its last speak_window_frames frames), aon (adaptive on), arp / trp
(the audio / text rep-pen in effect that frame); session.npz records spk / arp / trp per frame.
10-17 FACTORISED TEXT SAMPLING (live_ppm module doc; default ON): schema rows factorised_text (bool) + cat_temp at the top of 'Text sampling'
(group hint), every text knob's hint says it acts on the word choice only; min-p default 0.05 (absolute) with factorised, 0 in the joint
fallback (the engine moves min-p to the mode's default on a switch unless it was changed).
10-18 PAD BIAS (live_ppm module doc; live, default 0, range -10..+2): schema row pad_bias 'PAD bias (negative = speak more)' right after
cat_temp at the top of 'Text sampling' (step 0.1); the start-up smoke's all-knobs pass also sets pad_bias -1.
10-24 ADAPTIVE PAD BIAS (live_ppm module doc; live, default off in every profile): schema rows adaptive_pad_on (bool), pad_silent_s (s),
pad_bias_silent, pad_bias_other right after pad_bias in 'Text sampling' (FACT_KEYS); every frame message carries pon (adaptive PAD bias on,
0/1, as used by that frame), sil (seconds since B's last spoken lane token: word piece or NEW_WORD; counts from the first conversation frame)
and pb (the PAD bias in effect that frame, nats: the static pad_bias when off); session.npz records pon / sil / pb per frame; the start-up
smoke's all-knobs pass also turns it on (threshold 1 s, so the 40 silent frames exercise both states)."""
import argparse
import asyncio
import gc
import hmac
import json
import os
import queue
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
# 10-26 (user via main): version badge + changelog. CHANGELOG.json (this dir) is appended by every deploy (tools/changelog_add.py);
# /api/changelog serves it from disk plus what THIS process loaded at start (code hashes, start time, checkpoint).
def _sha8(f):
    import hashlib
    try:
        with open(f, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:8]
    except OSError:
        return None
LOADED = {"server_sha": _sha8(os.path.abspath(__file__)), "engine_sha": _sha8(f"{HERE}/live_ppm_pose.py"),
          "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
STATE = os.environ.get("PPMLIVE_STATE_DIR") or HERE   # 10-17 instance 2: per-instance LOADED.json / SWAP_TO.txt / sessions (default = HERE = instance 1)
sys.path.insert(0, HERE)
import live_ppm_pose as LP  # noqa: E402   (aloe-posedup-live: live_ppm + the channel-B pose branch)
import feed as FD  # noqa: E402
import prompts_ppm as PR  # noqa: E402
import prefill_clip as PC  # noqa: E402   (10-27: the shared voice-clip-window prefill, identical with the speech twin)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from aiohttp import web, WSMsgType  # noqa: E402

import base64  # noqa: E402
import re  # noqa: E402
import secrets  # noqa: E402
import importlib  # noqa: E402
import a_level as AL  # noqa: E402   (s4live's A-input level rule: -23 LUFS from the first voiced frame + limiter)
AL.LIVE = True

VS = LP.VS
log = LP.log
SPF = VS.SPF24
LOOK = VS.LOOK24
# 10-25 (user stalls = the quick tunnel's upstream going silent for 10-30 s while the browser kept sending): SESSION RESUME. A session survives
# its socket for RESUME_S; the page's upload Worker, after 2 s without model frames while it is sending (or on an unexpected close), opens a
# new socket (alternating between the page's own tunnel and the ALT_URLS.txt tunnels, e.g. a second cloudflared over http2) and sends
# {type:'resume', session, rtoken}; the session then continues on it (the old socket is closed, its late audio ignored).
RESUME_S = 20.0
REAL_CLIENTS = ("browser", "playground")   # 10-26 busy fix: real users (the demo page, the playground's live mic) may replace a detached session
def alt_urls():
    try:
        return [l.strip().rstrip("/") for l in open(f"{HERE}/ALT_URLS.txt") if l.strip().startswith("https://")][:4]
    except OSError:
        return []
HEARTBEAT_S = 45.0                                 # 10-25 main: 45 s (was 15 s from 10-24; transient upload stalls ended calls); the page pings every 2 s


def ck_meta(ck):
    """10-17: {label, prompt} of a checkpoint from CK_LABELS.json ({abs path: {"label": str, "prompt": "voice_clip_only" | "persona"}}, re-read
    on every call); unlisted checkpoints: no label, prompt 'persona' (= the m1vox prompt: the preset persona text + its clip)."""
    try:
        j = json.load(open(f"{HERE}/CK_LABELS.json"))
    except (OSError, ValueError):
        j = {}
    e = j.get(os.path.abspath(str(ck).rstrip("/")), {}) if ck else {}
    # aloe-posedup-live: unlisted checkpoints get THIS model family's prompt (voice clip only, no text) and sam-sep preset codes
    if not e and ck:
        e = {"label": "Aloe pose-duplex FT step " + str(ck).rstrip("/").split("-")[-1], "default_preset": "aloe-1-1648-0"}   # = DEFAULTS.json preset
    return {"label": e.get("label"), "prompt": e.get("prompt", "voice_clip_only"), "preset_codes": e.get("preset_codes", "samsep"),
            "default_preset": e.get("default_preset"), "extra_presets": e.get("extra_presets")}


def ck_last_step(M):
    """the loaded run's last step: the checkpoint's own trainer cfg 'steps' for an aloe trainer export, else PPMLIVE_LAST_STEP."""
    if M is None:
        return LAST_STEP
    rt, ts = M.info.get("run_tag") or "", M.info.get("total_steps")
    return int(ts) if ts and "aloe" in rt else LAST_STEP
SWAP_TO = [x for x in os.environ.get("PPMLIVE_SWAP_TO", "").split(",") if x]
LAST_STEP = int(os.environ.get("PPMLIVE_LAST_STEP", "366"))
HOST_PARAMS = {
    "a_lufs_on":     (int, 0, 1, 0, "A loudness match on: your mic reaches the model at the -23 LUFS training level from your first voiced words (fast acquisition), then tracks the BS.1770 integrated level; peaks limited to 0.995 (training never exceeded it). 10-17: default OFF (mic audio reaches the model exactly as captured)"),
    "a_target_lufs": (float, -40.0, -10.0, -23.0, "A loudness target (training recipe: -23 LUFS)"),
    "a_slew_db_s":   (float, 0.5, 20.0, 3.0, "max gain change speed (dB/s) after the first ~1 s of voiced speech (which is acquired instantly)"),
    "a_gain_db":     (float, -20.0, 20.0, 0.0, "manual gain on your mic audio (dB), used only when the loudness match is off"),
    "limit_on":         (int, 0, 1, 0, "soft peak limiter on B's audio before int16 on; off (default, 10-17) = bypassed: B's audio goes straight to int16, plain clipping only for out-of-range samples"),
    "limit_ceil":       (float, 0.5, 1.0, 0.98, "soft peak limiter ceiling on B's audio before int16"),
    "limit_attack_ms":  (float, 0.5, 10.0, 2.0, "limiter attack ramp (ms)"),
    "limit_release_ms": (float, 10.0, 500.0, 50.0, "limiter release (ms)"),
}
START_PARAMS = {
    "seed":    (int, 0, 2 ** 31 - 1, 0, "sampling seed (0 = random each Start)"),
    "max_s":   (float, 30.0, 600.0, 600.0, "max session length (s)"),
    # 10-25 user: "add prefill video as an inference param": the first S s of the avatar video are the actor's REAL video latents (videort,
    # = infer_gt.py --prefill-s); sent in the feed start; videort clamps to the person's real video and snaps to whole latents (0.16 s)
    # 10-27 user: "add a toggle called prefill which determines if to prefill the assistant side and a way to supply an amount and it always
    # prefills from the same clip selected as the voice": prefill (on/off) + prefill_s (the amount); prefill_src 3 = the voice clip window
    "prefill_on": (int, 0, 1, 1, "Prefill (default on): prefill the assistant (B) side for the first 'Prefill (s)' seconds from the SAME clip selected as the voice (its codes, its words lane, and its pose where the clip has pose). Off = no assistant prefill at all. Applies on Start."),
    "prefill_s": (float, 0.0, 30.0, 2.0, "Prefill (s): the amount. With the default source (the voice clip) the window is the first S seconds of the selected voice clip, capped at the clip's length (10 s for 125-frame clips); frames = 2 x round(S / 0.16) + 2 (2 s -> 26 frames). Prefill video (s): the first S seconds of the avatar are the actor's real video AND the model's pose stream is the actor's real pose (Prefill pose), then generation continues. Default 2 s (snaps to 1.92); 0 = none. Steps of one video latent (0.16 s); capped at what the actor's video allows. Applies on Start."),   # 10-25 user: 600 s (10 min = the Aloe training row length; was 300)
    # 10-25 user via main ("prefill", posefix): the same S seconds also force B's POSE codes to the actor's own real pose tokens
    # (gt/pose_tokens.safetensors, frame-aligned with the real video), so the model continues from the actor's real pose history
    # 10-25 user via main ("prefill" = the real pose + words + audio of the prefill seconds; from the conversation START when available)
    "prefill_audio": (int, 0, 1, 1, "Prefill audio (default on): during the prefill seconds B's voice is the actor's real recording (Mimi codes forced), so the model hears and continues it. Applies on Start."),
    "prefill_words": (int, 0, 1, 1, "Prefill words (default on): during the prefill seconds B's word lane is the actor's real words with their timings (no timings on disk for this actor -> an empty lane, flagged). Applies on Start."),
    "prefill_src": (int, 0, 3, 3, "Prefill source: 3 = the voice clip's own window (default, 10-27 user: always the same clip selected as the voice); 0 = auto = the actor's conversation START (the recording's opening: pose, voice, words; plus the real video where a t = 0 video exists) whenever it is on disk, else the avatar's own 12 s clip; 1 = clip; 2 = start"),   # 10-27: 3 = voice clip window (prefill_clip.py)
    "pose_prefill": (int, 0, 1, 1, "Prefill pose too (default on): during the 'Prefill video' seconds the model's pose stream is the actor's own real pose (same frames as the real video), then the model continues from it. Off = video-only prefill (the model's pose starts from nothing). Applies on Start."),
    "a_g0_db": (float, -30.0, 30.0, 0.0, "A gain before your first voiced speech (dB). 0 = automatic (the median level of earlier browser mics). From the first voiced frame on, the gain is the estimated -23 LUFS training gain"),
}
HOST_DEFAULTS = {k: v[3] for k, v in HOST_PARAMS.items()}
SHARED_SCHEMA = os.path.join(HERE, "live_params_schema.json")
EXTRA_GROUPS = {**{k: "Pose CFG" for k in LP.POSE_CFG_KEYS},
                "text_rep": "Text sampling", "text_rep_win": "Text sampling",
                "word_greedy": "Text sampling", "a_target_lufs": "A input", "a_slew_db_s": "A input", "a_g0_db": "A input",
                **{k: "Adaptive repetition penalty" for k in LP.ADAPT_KEYS}, **{k: "Pose sampling" for k in LP.POSE_KEYS}}
POSE_CFG_LABELS = {"pose_cfg_on": "Pose CFG", "pose_cfg_s": "Pose CFG strength", "pose_cfg_cb0": "Pose CFG on cb0", "pose_cfg_light": "Pose CFG on cb1-31",
                   "pose_cfg_speaking_only": "Pose CFG only while speaking", "pose_cfg_keep_mic": "Pose CFG keeps your mic (uncond hears it)"}
POSE_CFG_BOOL = ("pose_cfg_on", "pose_cfg_cb0", "pose_cfg_light", "pose_cfg_speaking_only", "pose_cfg_keep_mic")
POSE_CFG_GROUP_HINT = ("Classifier-free guidance on the assistant's pose (default off): an uncond stream that hears digital silence (every Mimi "
                       "code of both sides, voice clip included, replaced by the codes of digital silence; your mic replaced by digital zero) "
                       "runs every frame next to the real one, and the pose logits become uncond + strength x (cond - uncond) "
                       "(FastPPM's pose CFG = the SAM CFG-3 recipe; strength 3 = the released setting). Changes apply live at the next frame. "
                       "Independent of the audio CFG: both can be on.")
POSE_LABELS = {"pose_temp": "Pose cb0 temperature", "pose_topk": "Pose cb0 top-k", "pose_light_temp": "Pose cb1-31 temperature",
               "pose_light_topk": "Pose cb1-31 top-k", "pose_rep": "Pose cb0 repetition penalty", "pose_rep_light": "Pose cb1-31 repetition penalty",
               "pose_rep_win": "Pose repetition window (frames)"}
FEED_MODEL = "posedup"
PERSON_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


VIDEO_PEOPLE = os.environ.get("PPMLIVE_VIDEO_PEOPLE", "/home/param/datasets/pose-demo/prompts")


def add_video_people(pres):
    """10-25 user (via main): the 20 video people packaged for the avatar (videort's demo-data/prompts/<NN>_<8 hex>) as B voice presets
    'Person NN': voice clip = that person's own voice = the first 10 s (training clip_max 125 frames) of the same 12 s chunk the avatar's real
    video comes from, as its Mimi codes gt/mimi_local (kyutai/mimi, chunk-gain -23 LUFS, native sliding-window attention = the training /
    upload Mimi recipe; no decode + re-encode), no persona; preset id = the person id (VOICE_PERSON.json maps it to that same person)."""
    from safetensors.numpy import load_file
    n, vroot = 0, openings_root()                                     # 10-26: the openings root file, read at load for the voice clips
    for d in sorted(os.listdir(VIDEO_PEOPLE)) if os.path.isdir(VIDEO_PEOPLE) else []:
        m = re.match(r"^(\d\d)_[0-9a-f]{8}$", d)
        if pres.get(d) is not None or not os.path.isfile(f"{VIDEO_PEOPLE}/{d}/gt/mimi_local.safetensors"):
            continue
        vf, vkey = voice_clip_file(d, vroot)                          # 10-26 user: the SAM-isolated voice clip when the root has a ready package
        try:
            codes = np.asarray(load_file(vf)["codes"], np.int64)
            codes = codes if vf.endswith("/voice_clip.safetensors") else codes[:, :125]   # samsep's voice_clip is the trainer's clip as-is
            assert codes.shape[0] == 32 and codes.shape[1] >= 38 and codes.min() >= 0 and codes.max() < 2048, codes.shape
        except Exception as e:  # noqa: BLE001
            log(f"video person {d} skipped: {type(e).__name__} {str(e)[:120]}")
            continue
        clip = PR.upload_clip(codes)
        clip.update(src="video_person", src_key=vkey, src_file=vf)
        label = f"Person {m.group(1)}" if m else d.replace("_", " ")
        pres.items.append({"id": d, "name": label, "label": label, "persona": "", "prose": "", "clip": clip,
                           "row_key": f"VIDEO/{d}", "group": "Video people (the avatar is this person)"})
        n += 1
    log(f"PRESETS + {n} video people from {VIDEO_PEOPLE}; voice clips: openings root {vroot}")


GENWEB = "/home/param/websites/pose-playground"
# 10-26 USER (via main): duplex live prefill + voice clip use SAM-isolated codes, matching training. The openings root comes from a FILE,
# {STATE}/OPENINGS_ROOT.txt (one path, e.g. genweb/openings-sam), read per call for the prefill and at load for the voice clips; absent / empty /
# not a dir = genweb/openings (today). Per person the fallback is WHOLE-PACKAGE only (samsep: different conversations, never mix roots): a
# person whose package in that root is not ready (meta.json is written last = ready) uses genweb/openings for ALL its files.
OPENINGS_DEFAULT = f"{GENWEB}/openings"
OPENINGS_FILE = f"{STATE}/OPENINGS_ROOT.txt"
PKG_FILES = ("mimi_local.safetensors", "pose_tokens.safetensors", "meta.json")


def openings_root():
    """the configured openings root (OPENINGS_ROOT.txt), else genweb/openings."""
    try:
        r = open(OPENINGS_FILE).read().strip()
        if r and os.path.isdir(r):
            return os.path.abspath(r)
    except OSError:
        pass
    return OPENINGS_DEFAULT


def pkg_ready(root, person):
    return bool(person) and all(os.path.isfile(f"{root}/{person}/{f}") for f in PKG_FILES)


def opening_root(person, root=None):
    """the root serving this person's whole opening package: the configured root when its package is ready, else genweb/openings."""
    r = root or openings_root()
    return r if pkg_ready(r, person) else OPENINGS_DEFAULT


def opening_summary(person, oroot):
    """SUMMARY.json[person] of the serving root only (never mixed across roots)."""
    try:
        v = json.load(open(f"{oroot}/SUMMARY.json")).get(person)
        return v if isinstance(v, dict) else None
    except (OSError, ValueError):
        return None


def voice_clip_file(d, root):
    """-> (codes file, src_key): <root>/<d>/voice_clip.safetensors when root is not genweb/openings and d's package there is ready, else
    demo-data/prompts/<d>/gt/mimi_local.safetensors (today: its first 125 frames)."""
    if root != OPENINGS_DEFAULT and pkg_ready(root, d) and os.path.isfile(f"{root}/{d}/voice_clip.safetensors"):
        return f"{root}/{d}/voice_clip.safetensors", f"OPENINGS:{os.path.basename(root)}/{d}/voice_clip"
    return f"{VIDEO_PEOPLE}/{d}/gt/mimi_local.safetensors", f"VIDEO/{d}"


def prefill_inputs(person, pf, src_mode, want_pose, want_audio, want_words, tok):
    """10-25 FULL PREFILL (posefix; live_ppm_pose module doc): -> (pose [N, 32] local | None, {"lane", "codes"} | None, info dict).
    Source: src_mode 2 / auto with genweb/openings/<person>/ present = the conversation START (mimi_local, pose_tokens, words.json of the
    recording's opening); 1 / auto otherwise = the avatar's 12 s clip (demo-data gt/ + genweb/words/<person>/words.json). N = video-latent
    aligned (2 x round(pf / 0.16) + 2, capped by the data). Lane = the training placement of the timed words over frames 0..N+1."""
    from safetensors.numpy import load_file
    info = {"src": None, "frames": 0, "streams": [], "notes": []}
    if not person or pf <= 0:
        return None, None, info
    oroot = opening_root(person)                                     # 10-26: per call, whole package from one root
    op = f"{oroot}/{person}"
    start_ok = all(os.path.isfile(f"{op}/{f}") for f in ("mimi_local.safetensors", "pose_tokens.safetensors"))
    if src_mode == 2 or (src_mode == 0 and start_ok):   # 10-25 user: auto = the conversation START whenever the actor's opening is on disk
        if not start_ok:
            info["notes"].append("no conversation-start package for this actor: no prefill"); return None, None, info
        base, wf, info["src"] = op, f"{op}/words.json", "conversation start"
        info["openings_root"] = oroot                                   # 10-26: the root that served this call (session.json prefill)
        if oroot != openings_root():
            info["notes"].append(f"no ready package under {openings_root()}: whole package from {oroot}")
        try:
            info["t0"] = float((opening_summary(person, oroot) or {}).get("t0") or 0.0)
        except (TypeError, ValueError):
            info["t0"] = 0.0
    else:
        base = None
        for root in (VIDEO_PEOPLE, "/home/param/datasets/pose-demo/actors"):
            if os.path.isdir(f"{root}/{person}/gt"):
                base = f"{root}/{person}/gt"; break
        if base is None:
            info["notes"].append("no clip package for this actor: no prefill"); return None, None, info
        wf, info["src"] = f"{GENWEB}/words/{person}/words.json", "avatar clip"
    pose = np.asarray(load_file(f"{base}/pose_tokens.safetensors")["codes"], np.int64)
    aud = np.asarray(load_file(f"{base}/mimi_local.safetensors")["codes"], np.int64)
    N = min(pose.shape[1], aud.shape[1], 2 * int(round(pf / 0.16)) + 2)
    info["frames"] = N
    pp = pose[:, :N].T.copy() if want_pose else None
    pre = {}
    if want_audio:
        pre["codes"] = aud[:, :N].T.copy()
    if want_words or want_audio:
        lane = np.full(N + LP.CD, LP.PAD, np.int64)
        if want_words and os.path.isfile(wf):
            w = json.load(open(wf)); w = w["words"] if isinstance(w, dict) else w
            ws = []
            for x in w:
                if isinstance(x, dict):
                    ws.append({"text": str(x.get("text", x.get("w"))), "start": float(x.get("start", x.get("t0"))), "end": float(x.get("end", x.get("t1", 0)))})
                else:
                    ws.append({"text": str(x[2]), "start": float(x[0]), "end": float(x[1])})
            ws = [x for x in sorted(ws, key=lambda x: x["start"]) if x["start"] >= 0]
            pieces = LP.SP.word_pieces(tok, ws)
            pid, offs = [], [0]
            for pc in pieces:
                pid += pc; offs.append(len(pid))
            nl = N + LP.CD + 25                                           # longer than the window: a word starting right after it puts
            wif = np.zeros(nl, np.int64)                                  # its NEW_WORD on the window's last frame (= the training lane)
            lane = np.asarray(LP.SP.lane(ws, pid, offs, nl, wif), np.int64)
            LP.SP.insert_new_word(lane, wif)
            lane = lane[:N + LP.CD]
            info["streams"].append("words")
        elif want_words:
            info["notes"].append("no word timings on disk for this actor: empty (PAD) word lane during the prefill")
        pre["lane"] = lane
    if want_pose:
        info["streams"].insert(0, "pose")
        gs = opening_frame0_gs(person, oroot) if info["src"] == "conversation start" else None
        if gs is not None and N > 0:                                      # the continued tokens are relative to the OPENING window's frame 0
            info["frame0_global_state"] = gs
    if want_audio:
        info["streams"].append("audio")
    log(f"prefill: {person} {info['src']}" + (f" [{info['openings_root']}]" if info.get("openings_root") else "") + f" {N} frames ({N / 12.5:.2f} s), streams {info['streams']} {info['notes']}"
        + (f" frame0_global_state {info['frame0_global_state']}" if info.get("frame0_global_state") else ""))
    return pp, (pre or None), info


def clips_labelled(pr):
    """10-27 main: the voice-clip list with each entry labelled by its true source (SAM clip / avatar clip / bank clip) + its length."""
    out = []
    for c in pr.clips_public():
        p = pr.get(c["id"][7:]) if str(c.get("id", "")).startswith("preset:") else None
        if p is not None:
            c = dict(c, label=f"{p['name']} · {p['clip']['L'] / 12.5:.1f} s · {clip_source_label(p)}", src=clip_source_label(p).split(" (")[0],
                     seconds=round(p["clip"]["L"] / 12.5, 2), has_pose=bool(str(p["clip"].get("src_key", "")).startswith("OPENINGS:") or p["clip"].get("src") == "video_person"))
        out.append(c)
    return out


def clip_codes_of(clip):
    """10-27: the voice clip's codes [32, L] int64 exactly as sysprompt.build_prefix slices them (src_row codes_<src_side>[:, w0:w0 + L])."""
    if not clip:
        return None
    try:
        side = clip.get("src_side", clip.get("side"))
        w0, L = int(clip.get("w0", 0)), int(clip["L"])
        return np.asarray(clip["src_row"][f"codes_{side}"])[:, w0:w0 + L].astype(np.int64)
    except (KeyError, TypeError, ValueError):
        return None


def clip_source_label(p):
    """10-27 main: label each voice by its source (what the prefill / voice clip actually use)."""
    k = str((p.get("clip") or {}).get("src_key") or "")
    if k.startswith("OPENINGS:openings-raw/"):                       # 10-30 pdxhost: RAW-codes model (33111): genweb/openings-raw = raw-lane codes
        return "raw-codes clip from the training conversation (prefill = the same clip)"
    if k.startswith("OPENINGS:"):
        return "SAM clip from the training conversation (prefill = the same clip)"
    if (p.get("clip") or {}).get("src") == "video_person":
        return "avatar clip = the audio of the avatar's own video (prefill = the same clip, with its pose)"
    return "bank clip (prefill = the same clip, audio only: no pose / words for this clip)"


def opening_frame0_gs(person, oroot=None):
    """10-26 posefix (for main): an opening packaged from a recording WINDOW (meta sources.pose_tokens.frame0.fields) was tokenized against
    that window's frame 0, not the avatar package's. -> [root_y, root_x, width] in pose.global_state's units (y / H, x / W, shoulder width in
    x units; checked equal to llm_inputs global_state for every package) | None for openings that ARE the package (frame0 = its llm_inputs)."""
    try:
        f = json.load(open(f"{oroot or opening_root(person)}/{person}/meta.json"))["sources"]["pose_tokens"]["frame0"]["fields"]
        return [float(f["frame0_root"][0]), float(f["frame0_root"][1]), float(f["frame0_width"])]
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def opening_has_video(person):
    """genweb/openings/SUMMARY.json[person]["video"]: the avatar package IS the recording's opening (chunk 000000 at t = 0), so the real
    video of the prefill seconds matches the opening (aloedemo 10-25: persons 09, 10, 12, 13, 14, 15, 19)."""
    oroot = opening_root(person)
    v = opening_summary(person, oroot)
    if not isinstance(v, dict):
        return False
    if oroot != OPENINGS_DEFAULT:                                      # 10-26 samsep: SAM openings are OTHER conversations than the avatar's
        return bool(v.get("video")) and v.get("mode") == "pkg_chunk0"  # package: real video only if that root says it IS the package chunk 0
    return bool(v.get("video"))


def pose_prefill_codes(person, pf):
    """10-25 POSE PREFILL (posefix; live_ppm_pose module doc): the avatar person's own real pose tokens for the video prefill window ->
    [N, 32] local ids or None. videort prefills video latents 1..n (n = pf / 0.16 whole latents) = video frames 0..4n = pose tokens 0..2n
    (2 video frames per token); its real Mimi conditioning covers codes 0..2n+1, so N = 2n + 2 tokens (capped at the actor's 150).
    Person package = videort's engine.person_dir rule (demo-data/prompts, then dmd-rt/actors/pkg)."""
    if not person or pf <= 0:
        return None
    from safetensors.numpy import load_file
    for root in (VIDEO_PEOPLE, "/home/param/datasets/pose-demo/actors"):
        f = f"{root}/{person}/gt/pose_tokens.safetensors"
        if os.path.isfile(f):
            c = np.asarray(load_file(f)["codes"], np.int64)
            assert c.ndim == 2 and c.shape[0] == 32 and c.min() >= 0 and c.max() < 2048, c.shape
            n = int(round(pf / 0.16))
            N = min(c.shape[1], 2 * n + 2)
            log(f"pose prefill: {person} {N} real pose frames ({N / 12.5:.2f} s) from {f}")
            return c[:, :N].T.copy()
    log(f"pose prefill: no gt/pose_tokens for person {person!r}: video-only prefill")
    return None


def voice_person(voice):
    """10-25 user: "pick the actor whose voice I'm cloning as the starting character": {STATE}/VOICE_PERSON.json (videort's map, re-read per
    call) = {preset id: {"person": videort person id | null (no video of that actor), "max_prefill_s": s}} -> that entry or None."""
    try:
        m = json.load(open(f"{STATE}/VOICE_PERSON.json"))
    except (OSError, ValueError):
        return None
    v = m.get(str(voice or "")) if isinstance(m, dict) else None
    return v if isinstance(v, dict) else None


def voice_person_all():
    try:
        m = json.load(open(f"{STATE}/VOICE_PERSON.json"))
        return {k: v for k, v in m.items() if isinstance(v, dict)} if isinstance(m, dict) else {}
    except (OSError, ValueError):
        return {}


def state_line(name, env, default=""):
    """first non-comment line of {STATE}/<name> (re-read per call: changeable without a restart), else env, else default."""
    try:
        for ln in open(f"{STATE}/{name}"):
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                return ln
    except OSError:
        pass
    return os.environ.get(env, default)


def swap_ready(d):
    """aloe-posedup-live: a SWAP_TO.txt candidate is offered only once its download is VERIFIED: tools/hfdl.py (snapshot_download) writes
    the export's own COMPLETE file before the other files are done / checked, so COMPLETE alone is not enough. Ready = DL_VERIFY.json with
    no bad file (hfdl.py), or PDSAMPLES_STAGE.json (stage_pd.py: renamed into place after its sha check), or a SWAP_OK file (manual)."""
    try:
        if os.path.isfile(f"{d}/DL_VERIFY.json"):
            return not json.load(open(f"{d}/DL_VERIFY.json")).get("bad")
    except (OSError, ValueError):
        return False
    return (os.path.isfile(f"{d}/PDSAMPLES_STAGE.json") or os.path.isfile(f"{d}/SWAP_OK")
            or os.path.isfile(f"{d}/.metadata"))                   # a consolidated DCP export (the trainer writes COMPLETE last)


def ck_step(d):
    """the export's training step (meta.json / train_cfg.json 'step', else the dir name's trailing number), -1 if unknown."""
    for f in ("meta.json", "train_cfg.json"):
        try:
            return int(json.load(open(f"{d}/{f}"))["step"])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    b = os.path.basename(str(d).rstrip("/")).split("-")[-1]
    return int(b) if b.isdigit() else -1


def feed_url():
    u = state_line("FEED_URL.txt", "PPMLIVE_FEED_URL", FD.DEFAULT_URL)
    return "" if u.lower() in ("off", "none", "0") else u
ADAPT_HINT = ("While B is SPEAKING (it emitted a word piece in its last N frames) or SILENT, use that state's repetition penalties instead of "
              "the static ones; off = the static penalties.")
ADAPT_LABELS = {"adaptive_rep_on": "Adaptive repetition penalty", "speak_window_frames": "Speaking window (frames)",
                "audio_rep_speaking": "Audio rep-pen while speaking", "audio_rep_silent": "Audio rep-pen while silent",
                "text_rep_speaking": "Text rep-pen while speaking", "text_rep_silent": "Text rep-pen while silent",
                "pose_rep_cb0_speaking": "Pose cb0 rep-pen while speaking", "pose_rep_cb0_silent": "Pose cb0 rep-pen while silent",
                "pose_rep_light_speaking": "Pose light rep-pen while speaking", "pose_rep_light_silent": "Pose light rep-pen while silent"}
CFG_NAMES = ("cfg_on", "cfg_s", "cfg_cb0", "cfg_uncond_src")
FACT_KEYS = ("factorised_text", "cat_temp", "pad_bias") + LP.PADAD_KEYS   # 10-17 factorised text sampling (+ 10-18 PAD bias, + 10-24 adaptive PAD bias): own rows at the top of 'Text sampling'
FACT_BOOL = ("factorised_text", "adaptive_pad_on")
FACT_LABELS = {"factorised_text": "Factorised text sampling", "cat_temp": "Category temperature", "pad_bias": "PAD bias (negative = speak more)",
               "adaptive_pad_on": "Adaptive PAD bias", "pad_silent_s": "Silence threshold (s)", "pad_bias_silent": "PAD bias while silent",
               "pad_bias_other": "PAD bias otherwise"}
FACT_STEP = {"cat_temp": 0.05, "pad_bias": 0.1, "pad_silent_s": 0.1, "pad_bias_silent": 0.1, "pad_bias_other": 0.1}
FACT_GROUP_HINT = ("B's text is sampled in two steps (default): first PAD / NEW_WORD / WORD from the model's probabilities (category temperature, PAD bias), "
                   "then, only if WORD, which word piece. Every other text knob here acts on that word choice only; PAD / NEW_WORD timing is never touched by them. "
                   "Adaptive PAD bias: a PAD bias that depends on how long B has been silent (replaces the static PAD bias while on).")
WORD_HINT_KEYS = ("text_temp", "text_topk", "text_topp", "min_p", "min_p_rel", "freq_pen", "text_rep")   # shared-schema rows: this engine's words-only hint
PAGE_SIDE = ("aec", "ns", "agc", "mic_aec", "mic_ns", "mic_agc")
# ONE alias table (= s4live server.py): canonical shared-schema name (v4) -> internal name. Legacy names are accepted inbound too.
ALIAS = {"minp": "min_p", "minp_mode": "min_p_rel", "audio_rep": "audio_rep_pen", "max_len_s": "max_s", "a_gain": "a_gain_db",
         "a_lufs_on": "a_lufs_on", "lim_on": "limit_on", "lim_ceiling": "limit_ceil", "lim_attack_ms": "limit_attack_ms", "lim_release_ms": "limit_release_ms"}
LEGACY = {"min_p": "min_p", "min_p_mode": "min_p_rel", "text_minp": "min_p", "audio_rep_pen": "audio_rep_pen", "max_s": "max_s",
          "a_gain_db": "a_gain_db", "a_lufs_match": "a_lufs_on", "limit_ceil": "limit_ceil", "limit_attack_ms": "limit_attack_ms",
          "limit_release_ms": "limit_release_ms", "cfg_audio": "cfg_cb0", "cfg_strength": "cfg_s", "freq_pen_win": "freq_win",
          "audio_temp": "_audio_temp", "audio_topk": "_audio_topk"}
INV = {v: k for k, v in ALIAS.items()}
CFG_LAT = {"ms": None}                                                 # measured per-frame latency with CFG on (filled by the smoke test)
POSE_CFG_LAT = {"p50": None, "p95": None, "src": None, "lag_growth": None}   # 10-25: per-frame ms with pose CFG on (smoke, then live sessions)
RT_BUDGET_MS = 80.0


def _known_internal(k):
    return k in LP.PARAMS or k in HOST_PARAMS or k in START_PARAMS


def canon_in(d):
    """canonical (or legacy) names -> internal names (engine / host / start). Unknown / unsupported names are returned separately."""
    out, unk = {}, {}
    for k, v in d.items():
        if k in LP.GRAMMAR_GONE:
            continue                                                    # 10-28 USER ORDER: no lane grammar, ever: legacy key dropped silently
        if k in ("type", "client", "preset", "persona", "starting_context", "clip", "persona_only", "params", "t", "_rejected", "person", "model"):
            continue
        ik = ALIAS.get(k) or LEGACY.get(k) or k
        if ik == "min_p_rel":
            v = 1 if str(v).lower().startswith("rel") or v in (1, True) else 0
        elif ik in ("a_lufs_on", "limit_on"):
            v = 1 if v in (True, 1, "1", "true", "on") else 0
        elif ik in ("cfg_on", "cfg_cb0", "adaptive_rep_on", "factorised_text", "adaptive_pad_on") + POSE_CFG_BOOL:
            v = 1 if v in (True, 1, "1", "true", "on") else 0
        if k == "seed":
            out["seed"] = 0 if v in (None, "", "random") else v
        elif k in PAGE_SIDE:
            continue
        elif ik in ("_audio_temp", "_audio_topk"):                    # v1 names -> both audio stages
            for t in (("cb0_temp", "light_temp") if ik == "_audio_temp" else ("cb0_topk", "light_topk")):
                out.setdefault(t, v)
        elif k == "cfg_uncond_src":
            if str(v) in LP.UNCOND_SRC:
                out["cfg_uncond_src"] = str(v)
            else:
                unk[k] = f"uncond source must be one of {list(LP.UNCOND_SRC)}"
        elif _known_internal(ik):
            out[ik] = v
        else:
            unk[k] = "not supported by this demo"
    return out, unk


def canon_out(eng_params, hp, st=None):
    """internal values -> {canonical name: value} (what 'in_effect' reports)."""
    d = {}
    for k, v in eng_params.items():
        if k == "min_p_rel":
            d["minp_mode"] = "relative" if v else "absolute"
        elif k in ("cfg_on", "cfg_cb0", "adaptive_rep_on", "factorised_text", "adaptive_pad_on") + POSE_CFG_BOOL:
            d[k] = bool(v)
        else:
            d[INV.get(k, k)] = v
    for k, v in hp.items():
        d[INV.get(k, k)] = bool(v) if k in ("a_lufs_on", "limit_on") else v
    for k, v in (st or {}).items():
        d[INV.get(k, k)] = v
    return d


def _rej(bad):
    out = {}
    for b in bad:
        k = b.split(":")[0].split("=")[0].replace("unknown ", "").strip()
        out[INV.get(k, k)] = b
    return out


def _cfg_note():
    ms = CFG_LAT["ms"]
    if ms is None:
        return "the uncond stream runs every frame in the batch-2 engine (no extra pass)"
    return (f"the uncond stream runs every frame (batch-2 engine): measured {ms:.1f} ms per frame with CFG on"
            + (" (above the 80 ms real-time budget!)" if ms > 80 else " (real time)"))


def pose_cfg_note():
    """10-25: the measured per-frame compute with pose CFG on (2-GPU backend) vs the 80 ms real-time budget (shown next to the toggle)."""
    p50, p95 = POSE_CFG_LAT["p50"], POSE_CFG_LAT["p95"]
    if p50 is None:
        return "Per-frame cost with pose CFG on: not measured yet."
    over = p95 > RT_BUDGET_MS or (POSE_CFG_LAT.get("lag_growth") or 0) > 2.0
    return (f"pose CFG: {p50:.0f} ms/frame (p50; p95 {p95:.0f} ms, {POSE_CFG_LAT['src']}) "
            + (f"> {RT_BUDGET_MS:.0f} ms real-time budget: replies will fall behind!" if over else f"within the {RT_BUDGET_MS:.0f} ms real-time budget."))


def ready_schema(eng_params=None, hp=None, st=None):
    """the shared schema's rows (canonical names) with THIS engine's support / defaults / ranges / live flags, + its extra knobs."""
    try:
        shared = json.load(open(SHARED_SCHEMA))["params"]
    except Exception:  # noqa: BLE001
        shared = []
    cur = canon_out(LP.DEFAULTS, HOST_DEFAULTS, dict({k: v[3] for k, v in START_PARAMS.items()}, cfg_uncond_src=LP.UNCOND_SRC[0]))
    rng = {k: [lo, hi] for k, (t, lo, hi) in LP.PARAMS.items()}
    rng.update({k: [v[1], v[2]] for k, v in list(HOST_PARAMS.items()) + list(START_PARAMS.items())})
    rows, covered = [], set()
    for p in shared:
        n = p["name"]
        r = dict(p); r.pop("support", None)
        ik = ALIAS.get(n, n)
        if n in PAGE_SIDE:
            r.update(supported=True, model2_status="page-side (browser getUserMedia constraint); use headphones")
        elif n in ("persona", "voice_clip"):
            r.update(supported=True, live=False)
        elif n == "cfg_cb0":
            r.update(supported=True, live=True, default=bool(cur.get(n)),
                     model2_status="live: with cfg_on, B's cb0 logits = uncond + s (cond - uncond) before the audio rep-pen; one cb0 draw, cb1-31 from the cond decoder; " + _cfg_note())
            covered.add(ik)
        elif n == "cfg_uncond_src":
            r.update(supported=True, live=False, default=LP.UNCOND_SRC[0], choices=list(LP.UNCOND_SRC),
                     hint="What the uncond stream hears through the same Voxtral front + adapter: quiet_frames = s4live's quiet A frames (held-out "
                          "training rows, training level); digital_zero = encoded digital silence. Applies on Start.",
                     model2_status="on Start")
            covered.add("cfg_uncond_src")
        elif n in ("cfg_on", "cfg_s"):
            r.update(supported=True, live=True, default=cur.get(n), range=(rng.get(n) if n == "cfg_s" else None), model2_status="live; " + _cfg_note())
            if n == "cfg_on":
                r["hint"] = (r.get("hint") or "") + " " + _cfg_note() + "."
            covered.add(ik)
        elif _known_internal(ik):
            if n == "lim_ceiling":                                      # 10-17: the limiter's parent tick box, right before its 3 settings
                rows.append({"name": "lim_on", "group": r.get("group") or "B output", "type": "bool", "default": bool(HOST_DEFAULTS["limit_on"]),
                             "range": None, "step": None, "live": True, "label": "Limiter on", "supported": True, "model2_only": True,
                             "hint": "Soft peak limiter on B's audio before int16 (ceiling / attack / release below). Off (default): bypassed, B's "
                                     "audio goes straight to int16 with plain clipping only if a sample is out of range."})
                covered.add("limit_on")
            r.update(supported=True, default=cur.get(n, r.get("default")))
            if ik in WORD_HINT_KEYS:                                    # 10-17: factorised sampling: these act on the word choice only
                r["hint"] = LP.HINTS[ik][:1].upper() + LP.HINTS[ik][1:]
            if ik in rng and n != "minp_mode" and n != "a_lufs_on":
                r["range"] = rng[ik]
            r["live"] = ik not in START_PARAMS
            if n == "a_lufs_on":
                r.update(default=bool(HOST_DEFAULTS["a_lufs_on"]), live=True,
                         hint="Live -23 LUFS loudness match of your mic (the model trained on -23 LUFS audio). Off by default: your mic reaches the "
                              "model exactly as captured (with A gain 0 dB).")
            if n == "a_gain":
                r["hint"] = "Gain on your mic audio before the model hears it (dB), used only when the loudness match is off. 0 dB (default) = exactly as captured."
            if n == "seed":
                r.update(default=None, live=False)
            if n in ("max_len_s",):
                r["range"] = rng["max_s"]
            covered.add(ik)
        else:
            r.update(supported=False, model2_status="not in this model")
        rows.append(r)
    adapt, fact = [], []
    for k, (t, lo, hi) in LP.PARAMS.items():
        if k in covered or k == "min_p_rel":
            continue
        r = {"name": INV.get(k, k), "group": EXTRA_GROUPS.get(k, "Text sampling"), "type": "int" if t is int else "float",
             "default": LP.DEFAULTS.get(k), "range": [lo, hi], "step": 1 if t is int else 0.01, "live": True,
             "hint": LP.HINTS.get(k, ""), "supported": True, "model2_only": True}
        if k in LP.ADAPT_KEYS:                                       # 10-17: own group, right after the audio rep-pen rows
            r.update(label=ADAPT_LABELS[k], group_hint=ADAPT_HINT, model2_status="live")
            if k == "adaptive_rep_on":
                r.update(type="bool", default=bool(LP.DEFAULTS[k]), range=None, step=None)
            adapt.append(r)
            continue
        if k in LP.POSE_KEYS:                                        # aloe-posedup-live: own group 'Pose sampling'
            r.update(label=POSE_LABELS[k], model2_status="live", step=1 if t is int else 0.05)
        if k in LP.POSE_CFG_KEYS:                                    # 10-25 pose CFG: own group 'Pose CFG' (after 'Pose sampling')
            r.update(label=POSE_CFG_LABELS[k], group_hint=POSE_CFG_GROUP_HINT, model2_status="live", step=0.1 if t is float else 1)
            if k in POSE_CFG_BOOL:
                r.update(type="bool", default=bool(LP.DEFAULTS[k]), range=None, step=None)
            if k == "pose_cfg_on":
                r["hint"] = LP.HINTS[k] + " " + pose_cfg_note()
                r["latency_note"] = pose_cfg_note()
                r["latency"] = dict(POSE_CFG_LAT, budget_ms=RT_BUDGET_MS)
        if k in FACT_KEYS:                                           # 10-17: first rows of 'Text sampling'
            r.update(label=FACT_LABELS[k], group="Text sampling", group_hint=FACT_GROUP_HINT, model2_status="live")
            if k in FACT_BOOL:
                r.update(type="bool", default=bool(LP.DEFAULTS[k]), range=None, step=None)
            else:
                r["step"] = FACT_STEP[k]
            fact.append(r)
            continue
        rows.append(r)
    at = next((i + 1 for i, r in enumerate(rows) if r.get("name") == "audio_rep_win"), len(rows))
    rows[at:at] = adapt
    at = next((i for i, r in enumerate(rows) if r.get("group") == "Text sampling"), 0)
    rows[at:at] = fact
    for k, v in list(HOST_PARAMS.items()) + list(START_PARAMS.items()):
        if k in covered:
            continue
        rows.append({"name": INV.get(k, k), "group": EXTRA_GROUPS.get(k, "A input" if k.startswith("a_") else "Session"), "type": v[0].__name__,
                     "default": v[3], "range": [v[1], v[2]], "step": 0.5 if v[0] is float else 1, "live": k in HOST_PARAMS, "hint": v[4],
                     "supported": True, "model2_only": True})
        if k == "prefill_s":                                         # 10-25: one video latent per step, in the avatar's own group
            rows[-1].update(step=0.16, group="Avatar video", label="Prefill (s)")
        if k == "prefill_on":                                           # 10-27 user: the Prefill toggle, next to its amount
            rows[-1].update(group="Avatar video", label="Prefill")
    return rows


def param_schema():
    """GET /api/params_schema (= s4live): every knob with group, type, range, default, live / start, hint."""
    rows = []
    for k, (t, lo, hi) in LP.PARAMS.items():
        rows.append({"name": k, "type": t.__name__, "min": lo, "max": hi, "default": LP.DEFAULTS.get(k), "live": True,
                     "group": LP.GROUPS.get(k, "other"), "hint": LP.HINTS.get(k, "")})
    for k, (t, lo, hi, dflt, h) in HOST_PARAMS.items():
        rows.append({"name": k, "type": t.__name__, "min": lo, "max": hi, "default": dflt, "live": True, "group": "A input", "hint": h})
    for k, (t, lo, hi, dflt, h) in START_PARAMS.items():
        rows.append({"name": k, "type": t.__name__, "min": lo, "max": hi, "default": dflt, "live": False, "group": "session", "hint": h})
    rows.append({"name": "cfg_uncond_src", "type": "enum", "choices": list(LP.UNCOND_SRC), "default": LP.UNCOND_SRC[0], "live": False,
                 "group": "audio CFG", "hint": "what the CFG uncond stream hears (on Start)"})
    return rows


def _val(spec, d):
    out, bad = {}, []
    for k, v in d.items():
        if k not in spec:
            continue
        t, lo, hi = spec[k][:3]
        try:
            x = t(float(v))
        except (TypeError, ValueError):
            bad.append(f"{k}: not a number"); continue
        if not lo <= x <= hi:
            bad.append(f"{k}={x} outside [{lo}, {hi}] (clamped)"); x = min(max(x, lo), hi)
        out[k] = x
    return out, bad


def validate_host(d):
    return _val(HOST_PARAMS, d)


def validate_start(cfg):
    p = dict(cfg.get("params") or {})
    for k in ("seed", "max_s"):
        if k in cfg and k not in p:
            p[k] = cfg[k]
    st, bad = _val(START_PARAMS, p)
    return {k: st.get(k, START_PARAMS[k][3]) for k in START_PARAMS}, bad


def decode_upload(data):
    """any common audio file -> float32 mono 24 kHz (soundfile, else the bundled ffmpeg) (= s4live)."""
    import io
    import subprocess
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    try:
        w, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        w = w.mean(1)
        if sr != 24000:
            g = gcd(int(sr), 24000)
            w = resample_poly(w, 24000 // g, int(sr) // g).astype(np.float32)
        return w, "soundfile"
    except Exception:  # noqa: BLE001
        import tempfile
        import imageio_ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".upload") as tf:
            tf.write(data); tf.flush()
            r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", tf.name, "-ac", "1", "-ar", "24000",
                                "-f", "f32le", "pipe:1"], capture_output=True, timeout=60)
        if r.returncode != 0 or not r.stdout:
            raise ValueError("could not decode audio: " + r.stderr.decode(errors="replace")[:200])
        return np.frombuffer(r.stdout, np.float32).copy(), "ffmpeg"


class Model:
    """the loaded checkpoint + the engine (graphs captured once) + presets."""

    def __init__(self, a, ck, presets):
        t0 = time.time()
        devs = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())][:2]
        assert len(devs) == 2, "needs 2 GPUs"
        adapter = a.adapter if (a.adapter and ck == a.ck_init) else None
        self.m, ldev, cfgs, self.info = LP.build_model(ck, devs, adapter=adapter)
        assert self.info["pose"] and self.info["decp_bb"], ("the pose-duplex demo needs a checkpoint with pose parts + the D33 link", self.info)
        self.eng = LP.LivePPM(self.m, ldev, cfgs, devs, max_s=600.0)   # 10-25: static KV / graph buffers for 600 s (+~1.3 / 1.1 GiB)
        assert self.eng.pose
        log(f"engine: pose CFG rows {self.eng.pcfg} (trunk rows {self.eng.NR})")
        self.eng.capture()
        self.ck, self.step = ck, int(self.info["step"])
        self.tok = LP.IP.tok()
        self.presets = presets
        self.load_s = round(time.time() - t0, 1)
        log(f"MODEL ready: {ck} step {self.step} in {self.load_s}s (s = {self.info['s']:.6g}, {self.info['s_src']})")

    @torch.no_grad()
    def mimi_encode(self, w):
        with torch.cuda.device(self.eng.d0):
            x = torch.as_tensor(np.asarray(w, np.float32), device=self.eng.d0)[None, None]
            return self.eng.mimi.encode(x, num_quantizers=32).audio_codes[0].cpu().numpy().astype(np.int64)

    def free(self):
        self.eng.graphs.clear()
        for k in ("eng", "m"):
            setattr(self, k, None)
        gc.collect()
        torch.cuda.empty_cache()


def upload_codes(Mo, data, clip_min=38, clip_max=125):
    """upload -> mono 24 kHz -> first <= 10 s (training clip_max 125 frames) -> -23 LUFS (whole-clip BS.1770-4 gain, peak 0.995 = the Mimi
    recipe) -> Mimi codes [32, L] (3 s minimum = training clip_min 38 frames)."""
    w, how = decode_upload(data)
    L = min(len(w) // SPF, clip_max)
    if L < clip_min:
        raise ValueError(f"clip too short: {len(w) / 24000:.1f} s (needs >= {clip_min / 12.5:.1f} s)")
    w = w[:L * SPF]
    w, li = VS.lufs_normalize(w)
    codes = Mo.mimi_encode(w)[:, :L]
    assert codes.shape == (32, L), codes.shape
    return codes, w, {"seconds": L / 12.5, "decoder": how, "lufs_in": li.get("lufs_in"), "gain_db": li.get("gain_db"), "lufs_status": li.get("status")}


class Session:
    def __init__(self, srv, ws, loop, cfg):
        self.srv, self.ws, self.loop, self.cfg = srv, ws, loop, cfg
        self.inq = queue.Queue()
        self.outq = asyncio.Queue()
        self.stop = threading.Event()
        self.thread = None
        self.last_rx = time.time()
        self.end_reason = None
        self.ctl = queue.Queue()
        self.wsx = ws                                  # the socket the session talks to now (None while detached, waiting for a resume)
        self.rtoken = secrets.token_hex(8)
        self.gen = 0
        self.resumes = 0
        self.finished = False
        self.client_stop = False
        self.sid = None
        self.call_over = False                         # 10-26 busy fix: generation ended (the post-call save may still be running, in the background)
        self.save_thread = None

    def in_call(self):
        """10-26 main (busy-after-call fix): True only while the call is live = a connected client socket, generation not stopped / over.
        A detached session (socket closed, waiting up to RESUME_S for a resume) or an ended one (saving in the background) is NOT busy."""
        w = self.wsx
        return (not self.finished and not self.call_over and not self.stop.is_set()
                and w is not None and not w.closed)

    def limit(self, y, hb):
        """soft peak limiter on B's audio (= s4live)."""
        g0 = getattr(self, "_lg", 1.0)
        ceil = float(hb.get("limit_ceil", 0.98))
        pk = float(np.abs(y).max()) if len(y) else 0.0
        tgt = min(1.0, ceil / pk) if pk > 0 else 1.0
        n = len(y)
        if tgt < g0:
            na = max(1, min(n, int(hb.get("limit_attack_ms", 2.0) * 24)))
            ramp = np.concatenate([np.linspace(g0, tgt, na, dtype=np.float32), np.full(n - na, tgt, np.float32)])
            g1 = tgt
        else:
            a = float(np.exp(-n / max(1.0, hb.get("limit_release_ms", 50.0) * 24)))
            g1 = tgt + (g0 - tgt) * a
            ramp = np.linspace(g0, g1, n, dtype=np.float32)
        self._lg = g1
        return np.clip(y * ramp, -ceil, ceil).astype(np.float32)

    def send(self, obj):
        self.loop.call_soon_threadsafe(self.outq.put_nowait, obj)

    def run(self):
        try:
            with self.srv.lock:
                self._run()
        except BaseException as e:  # noqa: BLE001
            log("SESSION ERROR " + traceback.format_exc())
            self.send({"type": "error", "msg": repr(e)[:300]})
        finally:
            self.finished = True
            self.feed_end()
            self.send(None)

    # ---- FEED for videort (feed.py; never blocks: serialised here, sent by the feed thread, logged by the session's FeedLog thread)
    def feed_msg(self, kind, obj):
        txt = json.dumps(obj, separators=(",", ":"))
        self.srv.feed.put(kind, self.sid, txt)
        if self.flog is not None:
            self.flog.put(txt)

    def feed_end(self):
        if getattr(self, "feed_open", False):
            self.feed_open = False
            self.feed_msg("end", {"type": "end", "session": self.sid})
        if getattr(self, "flog", None) is not None:
            n = self.flog.close()
            self.flog = None
            log(f"feed {self.sid}: {n} messages in feed.jsonl; endpoint {self.srv.feed.session_stats(self.sid)}")

    def prompt(self):
        """-> (prefix, info) from the page's preset / persona text / clip choice."""
        srv, cfg = self.srv, self.cfg
        pres = srv.presets
        p = pres.get(cfg.get("preset")) or pres.items[0]
        mode = ck_meta(srv.M.ck)["prompt"]                            # 10-17: 'voice_clip_only' = the v2 training prompt (no text by default)
        dflt = "" if mode in ("voice_clip_only", "persona_prose") else p["persona"]   # 10-24 persona_prose: no persona text unless the user gives one
        txt = cfg.get("persona")
        txt = dflt if txt is None or not str(txt).strip() else str(txt)
        edited = txt.strip() != dflt.strip()
        cid = str(cfg.get("clip") or ("preset:" + p["id"]))
        clip, csrc = None, None
        if cid.startswith("upload:"):
            up = srv.uploads.get(cid[7:])
            assert up is not None, "uploaded clip not found (re-upload after a server restart)"
            clip, csrc = PR.upload_clip(up["codes"]), {"src": "upload", "lane": False, "seconds": up["seconds"]}
        elif cid.startswith("preset:"):
            cp = pres.get(cid[7:])
            assert cp is not None, ("unknown clip", cid)
            clip, csrc = cp["clip"], {"src": "preset", "lane": False, "speaker": cp["name"], "seconds": round(cp["clip"]["L"] / 12.5, 2),
                                      "src_key": cp["clip"]["src_key"], "clip_src": "rule"}
        # 10-27 user ("can we pick a prefill clip the conversation starts with"): the start message's clip_src "start" (DEFAULT) | "rule".
        # "start" = B's voice clip IS the first 10 s of the voice's conversation opening (shared prefill_clip.clip_for, whole package from one
        # root), and the voice-clip-window prefill then takes its first N s; "rule" (or a voice without an opening) = the preset clip above.
        want_cs = "start" if str(cfg.get("clip_src") or "rule").lower() == "start" else "rule"   # 10-27 18:00 PT main: DEFAULT = the training rule
        self.clip_src = "rule"
        vid0 = cid[7:] if cid.startswith("preset:") else ""
        if want_cs == "start" and PERSON_RE.match(vid0 or "") and PC.start_available(vid0, openings_root()):
            scodes, sci = PC.clip_for(vid0, "start", openings_root())
            clip = PR.upload_clip(scodes)
            clip.update(src="conversation_start", src_key=sci["key"])
            csrc = {"src": "conversation_start", "clip_src": "start", "lane": False, "speaker": (pres.get(vid0) or {}).get("name", vid0),
                    "seconds": round(scodes.shape[1] / 12.5, 2), "src_key": sci["key"], "sha256": sci["sha256"]}
            self.clip_src = "start"
        elif want_cs == "start" and csrc is not None:
            csrc["clip_note"] = "no conversation-start package for this voice: training-rule clip"
        if csrc is not None and "sha256" not in csrc and clip_codes_of(clip) is not None:
            csrc["sha256"] = PC.sha(clip_codes_of(clip))
        log(f"voice clip: {cid} clip_src={self.clip_src} (asked {want_cs}) {csrc.get('src_key') if csrc else None} sha {str((csrc or {}).get('sha256'))[:16]}")
        self.voice_clip = clip                                          # 10-27: the clip B gets = the prefill window's source (voice clip window)
        pre, info = PR.build_prefix(txt, clip, prose=(mode == "persona_prose"))
        info.update(preset=p["id"], persona_edited=edited, clip=csrc, prompt_mode=mode,
                    route=("persona prose (no labels) + voice clip" if mode == "persona_prose" and edited else
                           "custom persona" if edited else ("voice clip only, no text (= the v2 training prompt)" if mode in ("voice_clip_only", "persona_prose")
                                                            else "held-out preset persona")))
        return pre, info, txt

    @torch.no_grad()
    def _run(self):
        Mo = self.srv.M
        cfg = self.cfg
        eng = Mo.eng
        t0 = time.time()
        self.send({"type": "status", "msg": "building system prompt"})
        hp, hbad = validate_host(dict(HOST_DEFAULTS, **{k: v for k, v in (cfg.get("params") or {}).items() if k in HOST_PARAMS}))
        st, sbad = validate_start(cfg)
        self.hp, self.st_params = hp, st
        max_s = min(float(st["max_s"]), eng.max_s)
        pre, pinfo, ptxt = self.prompt()
        # aloe-posedup-live: the session id (= sessions/<id> = the feed's session) + the videort feed (endpoint re-read per session)
        self.sid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        self.od = f"{self.srv.a.sessions}/{self.sid}"
        self.srv.feed.set_url(feed_url())
        cid = str(cfg.get("clip") or "")
        self.voice = cid[7:] if cid.startswith("preset:") else (cid if cid.startswith("upload:") else pinfo.get("preset"))
        # 10-25: avatar person = a typed id, else the cloned voice's actor (VOICE_PERSON.json), else the default (uploads / voices without video)
        typed = str(cfg.get("person") or "").strip()
        vp = None if str(self.voice or "").startswith("upload:") else voice_person(str(self.voice or "").split(":audio")[0])
        pers = typed or (vp or {}).get("person") or state_line("PERSON.txt", "PPMLIVE_PERSON", "")
        self.person_src = "typed" if typed else ("voice" if (vp or {}).get("person") else "default")
        self.person = pers if PERSON_RE.match(pers or "") else None
        if pers and not self.person:
            log(f"person id {pers[:80]!r} rejected (allowed: {PERSON_RE.pattern})")
        pf = round(round(float(st.get("prefill_s") or 0.0) / 0.16) * 0.16, 2)
        if not int(st.get("prefill_on", 1)):                                # 10-27 user: the Prefill toggle off = no assistant prefill at all
            pf = 0.0
        self.prefill_note = None
        if int(st.get("prefill_src", 3)) == 3:                           # 10-27 user: ALWAYS the first N s of the SAME clip selected as the voice
            vid = str(self.voice or "")
            want_pose = bool(int(st.get("pose_prefill", 1))) and eng.pose
            same = bool(self.person) and self.person == vid             # pose / real video only when the avatar IS this voice clip's actor
            ccodes = clip_codes_of(getattr(self, "voice_clip", None))
            codes, lane, pose, info = PC.prefill_from_clip(None if vid.startswith("upload:") else vid, pf, openings_root(), want_pose and same, LP,
                                                           self.srv.M.tok, clip_codes=ccodes, want_words=bool(int(st.get("prefill_words", 1))),
                                                           clip_src=getattr(self, "clip_src", "rule"))
            if want_pose and not same and info.get("on"):
                info["notes"].append("no pose prefill: the avatar is not this voice clip's actor")
            want_audio = bool(int(st.get("prefill_audio", 1)))
            pre_ = {}
            if codes is not None and want_audio:
                pre_["codes"] = codes.T.copy()
            if lane is not None and (want_audio or bool(int(st.get("prefill_words", 1)))):
                pre_["lane"] = lane
            self.pose_pf, self.full_pf, self.pf_info = pose, (pre_ or None), info
            self.pf_info["prefill_on"] = bool(int(st.get("prefill_on", 1)))
            pf = float(info.get("prefill_s") or 0.0) if info.get("on") else 0.0
            st["prefill_s"] = self.prefill_s = pf
            self.pose_pf_n = 0 if self.pose_pf is None else int(len(self.pose_pf))
            vmax = (vp or {}).get("max_prefill_s")                       # the avatar clip window IS its real video (from t = 0): real-video prefill
            self.prefill_video_s = (min(pf, float(vmax)) if vmax is not None else pf) if (same and (info.get("src") == "video clip window" or
                                    (info.get("src") == "conversation start" and opening_has_video(self.person)))) else 0.0   # 10-27: real video only
            # where it IS this window (the avatar clip window, or an opening that is the avatar package's own t = 0)
            log(f"prefill (voice clip window): voice {vid} on={self.pf_info['prefill_on']} {info.get('src')} {info.get('clip_key')} N={info.get('frames')} "
                f"codes sha {str(info.get('sha256_codes'))[:16]} streams {info.get('streams')} pose {self.pose_pf_n} video {self.prefill_video_s} {info.get('notes')}")
        else:
            if self.person_src == "voice" and vp.get("max_prefill_s") is not None:
                pf = min(pf, float(vp["max_prefill_s"]))
            if pf > 0 and self.person_src == "default":                      # 10-25 main: never prefill another person's real video
                self.prefill_note = f"prefill {pf:.2f} s refused: the avatar is the default person, not this voice's actor"
                log(self.prefill_note); pf = 0.0
            st["prefill_s"] = self.prefill_s = pf
            self.pose_pf, self.full_pf, self.pf_info = prefill_inputs(self.person, pf, int(st.get("prefill_src", 0)),
                                                                      bool(int(st.get("pose_prefill", 1))) and eng.pose, bool(int(st.get("prefill_audio", 1))),
                                                                      bool(int(st.get("prefill_words", 1))), self.srv.M.tok)
            self.pose_pf_n = 0 if self.pose_pf is None else int(len(self.pose_pf))
            if self.pf_info.get("src") == "conversation start" and not opening_has_video(self.person):   # the avatar's real video is a later clip,
                self.prefill_video_s = 0.0                                                                # not the opening: no real-video prefill
            else:
                self.prefill_video_s = pf
        self.pf_info["video"] = "real" if (self.prefill_video_s > 0 and self.pf_info.get("frames")) else "generated"
        self.flog = FD.FeedLog(f"{self.od}/feed.jsonl", log)
        self.feed_open = False
        P = int(pre["P"])
        seed = int(st["seed"]) if st["seed"] > 0 else int.from_bytes(os.urandom(3), "little") + 1
        st["seed"] = seed
        p0 = {k: v for k, v in (cfg.get("params") or {}).items() if k in LP.PARAMS}
        usrc = str((cfg.get("params") or {}).get("cfg_uncond_src") or LP.UNCOND_SRC[0])
        st["cfg_uncond_src"] = usrc
        self.send({"type": "status", "msg": f"prefilling the system prompt ({P} positions)"})
        eng.start(pre, seed, p0, uncond_src=usrc, pose_prefill=self.pose_pf, prefill=self.full_pf)
        _, self.param_problems = eng.set_params(p0)
        st_msg = {"type": "start", "session": self.sid, "model": FEED_MODEL, "voice": self.voice, "fps": 12.5, "prefill_s": self.prefill_video_s,
                  "pose_prefill_frames": self.pose_pf_n, "prefill": self.pf_info}
        self.f0gs = self.pf_info.get("frame0_global_state") if self.pose_pf_n > 0 else None
        if self.f0gs:                                                     # 10-26 posefix: decode against the opening's frame 0
            st_msg["frame0_global_state"] = self.f0gs
        if self.person:
            st_msg["person"] = self.person
        self.feed_msg("start", st_msg)
        self.feed_open = True
        self.send({"type": "ready", "session": self.sid, "person": self.person, "person_src": self.person_src, "prefill_s": self.prefill_s, "pose_prefill_frames": self.pose_pf_n, "prefill": self.pf_info, "prefill_note": self.prefill_note, "voice": self.voice,
                   "feed": {"endpoint": self.srv.feed.url or "off", "model": FEED_MODEL},
                   "ck": Mo.ck, "step": Mo.step, "label": ck_meta(Mo.ck)["label"], "P": P, "seed": seed, "a_mimi": 0, "setup_s": round(time.time() - t0, 1),
                   "max_s": max_s, "host_params": hp, "start_params": st, "prompt_route": pinfo.get("route"), "prompt": pinfo,
                   "params": dict(eng.params), "param_problems": self.param_problems, "live_params": True,
                   "schema": ready_schema(), "in_effect": canon_out(eng.params, hp, st),
                   "rejected": dict(_rej(self.param_problems + sbad + hbad), **(cfg.get("_rejected") or {})),
                   "clip": pinfo.get("clip") or {}, "s": Mo.info["s"], "pose_cfg_rows": bool(eng.pcfg), "pose_cfg_latency": dict(POSE_CFG_LAT, note=pose_cfg_note()),
                   "resume": self.rtoken, "resume_s": RESUME_S, "alt_ws": alt_urls()})
        log(f"session ready: P {P}, seed {seed}, setup {time.time() - t0:.1f}s, prompt {pinfo}")
        log("session params in effect (" + LP.PROFILE + " defaults): " + json.dumps(canon_out(eng.params, hp, st), sort_keys=True))
        importlib.reload(AL); AL.LIVE = True
        loud = AL.StreamLevel(g0_db=st["a_g0_db"])
        hb = hp
        raw = np.zeros(int((max_s + 5) * 24000) + 10 * SPF, np.float32)
        n_raw = n_full = 0
        f = 0
        T = P
        Tmax = P + int(max_s * 12.5)
        rec = {"a": [], "b": [], "ms": [], "tok": [], "pp": [], "pn": [], "lag": [], "spk": [], "arp": [], "trp": [], "pon": [], "sil": [], "pb": [],
               "pose": [], "mimi": [], "pcf": [], "pcg": []}
        tok = Mo.tok
        while not self.stop.is_set() and T < Tmax:
            need = (f + 1) * SPF + LOOK
            while n_raw < need and not self.stop.is_set():
                try:
                    x = self.inq.get(timeout=0.5)
                except queue.Empty:
                    if time.time() - self.last_rx > HEARTBEAT_S:
                        self.end_reason = "timeout"
                        self.send({"type": "timeout", "msg": f"no audio or ping for {HEARTBEAT_S:.0f} s"})
                        self.stop.set()
                    continue
                if x is None:
                    self.stop.set(); break
                x = x[:len(raw) - n_raw]
                raw[n_raw:n_raw + len(x)] = x; n_raw += len(x)
            if self.stop.is_set():
                break
            t_ready = time.perf_counter()
            while n_raw - n_full >= SPF:
                loud.target, loud.slew = float(hb["a_target_lufs"]), float(hb["a_slew_db_s"])
                y, _ = loud.push(raw[n_full:n_full + SPF])
                if not hb["a_lufs_on"]:
                    y = (raw[n_full:n_full + SPF] * np.float32(10 ** (hb["a_gain_db"] / 20))).astype(np.float32)
                eng.write_a(n_full, y)
                rec["a"].append(y)
                n_full += SPF
            if n_full < need:
                gdb = loud.g_db if hb["a_lufs_on"] else hb["a_gain_db"]
                eng.write_a(n_full, raw[n_full:need] * np.float32(10 ** (gdb / 20)))
            while not self.ctl.empty():
                upd, unk = self.ctl.get_nowait()
                hu = {k: upd.pop(k) for k in list(upd) if k in HOST_PARAMS}
                clean, bad = eng.set_params({k: v for k, v in upd.items() if k not in START_PARAMS and k != "cfg_uncond_src"})
                if hu:
                    hc, hbad2 = validate_host(hu)
                    hb = dict(hb, **hc); self.hp = hb
                    clean.update(hc); bad += hbad2
                bad += [f"{k}: applies at Start only" for k in upd if k in START_PARAMS or k == "cfg_uncond_src"]
                rej = dict(_rej(bad), **unk)
                self.send({"type": "settings", "f": f, "in_effect": canon_out(eng.params, hb, st),
                           "changed": [INV.get(k, k) for k in clean], "rejected": rej})
                log(f"params at frame {f}: {clean} {bad if bad else ''}")
                self.param_log = getattr(self, "param_log", []) + [{"frame": f, "applied": clean, "problems": bad}]
            pcf = eng.pose_cfg_live()                                   # 10-25: pose CFG in effect this frame
            dec = eng.step(T)
            tokB, p_pad, p_nw, y = eng.result(T, dec)
            ms = (time.perf_counter() - t_ready) * 1e3
            t_out = time.time()
            cf = f - LP.CD                                              # code frame of this step's B Mimi / pose codes (= the audio sent now)
            pose = eng.last_pose if cf >= 0 else None
            lag = (n_raw - need) / SPF
            piece, nw = None, False
            if tokB < LP.TE:
                piece = tok.convert_ids_to_tokens(tokB)
                nw = piece.startswith("Ġ")
                piece = tok.decode([tokB])
            self.send({"type": "f", "f": f, "tok": piece, "ws": nw, "ctl": ("PAD" if tokB == LP.PAD else "NW" if tokB == LP.NW else None if tokB < LP.TE else str(tokB)),
                       "pp": round(p_pad, 4), "pn": round(p_nw, 4), "ms": round(ms, 1), "lag": round(lag, 2),
                       "ps": round(eng.last_ptok[0], 4), "pd": round(eng.last_ptok[1], 4),
                       "s": (float(eng.params.get("cfg_s", 1.0)) if eng.params.get("cfg_on") else 1.0),
                       "sa": (float(eng.params.get("cfg_s", 1.0)) if eng.cb0_cfg_live() else 1.0),
                       "cd": round(eng.last_cfgd[0], 4), "tv": round(eng.last_cfgd[1], 4), "tv0": (round(eng.last_cfgd[3], 4) if eng.cb0_cfg_live() else None),
                       "spk": int(eng.last_adapt[0] > 0.5), "aon": int(bool(eng.params.get("adaptive_rep_on"))),
                       "arp": round(eng.last_adapt[1], 4), "trp": round(eng.last_adapt[2], 4),
                       "pon": int(eng.last_padad[0] > 0.5), "sil": round(eng.last_padad[1], 2), "pb": round(eng.last_padad[2], 4),
                       "cf": cf if cf >= 0 else None, "pose": pose,
                       "pcf": int(pcf), "pcs": (float(eng.params.get("pose_cfg_s", 1.0)) if pcf else 1.0),
                       "pcg": int(eng.last_pcg)})                       # 10-26: pose CFG actually applied this frame (speaking gate incl.)
            if y is not None:
                if hb.get("limit_on"):
                    y = self.limit(y, hb)
                else:                                                   # 10-17: limiter off (default) = bypassed; int16 clip below only
                    self._lg = 1.0
                pcm = np.clip(y * 32767, -32768, 32767).astype("<i2").tobytes()
                self.send(pcm)
                rec["b"].append(y)
                if cf >= 0:                                             # FEED: one message per code frame, every frame (listening too)
                    self.feed_msg("f", {"type": "f", "session": self.sid, "f": cf, "t": round(t_out, 3), "pose": pose,
                                        "mimi": eng.last_codes, "pcm_b64": base64.b64encode(pcm).decode("ascii")})
                    rec["pose"].append(pose); rec["mimi"].append(eng.last_codes)
            rec["ms"].append(ms); rec["tok"].append(tokB); rec["pp"].append(p_pad); rec["pn"].append(p_nw); rec["lag"].append(lag)
            rec["spk"].append(eng.last_adapt[0]); rec["arp"].append(eng.last_adapt[1]); rec["trp"].append(eng.last_adapt[2])
            rec["pon"].append(eng.last_padad[0]); rec["sil"].append(eng.last_padad[1]); rec["pb"].append(eng.last_padad[2])
            rec["pcf"].append(int(pcf))
            rec["pcg"].append(int(eng.last_pcg))
            f += 1
            T += 1
        ms = np.asarray(rec["ms"])
        stats = {}
        if len(ms):
            stats = {"frames": int(len(ms)), "p50": round(float(np.percentile(ms, 50)), 2), "p95": round(float(np.percentile(ms, 95)), 2),
                     "p99": round(float(np.percentile(ms, 99)), 2), "max": round(float(ms.max()), 2), "mean": round(float(ms.mean()), 2),
                     "rtf": round(float(ms.mean()) / 80.0, 3)}
            pc = np.asarray(rec["pcf"], bool)
            for name, sel in (("pose_cfg_on", pc), ("pose_cfg_off", ~pc)):
                if sel.sum() >= 20:
                    x = ms[sel][3:]
                    lg = np.asarray(rec["lag"])[sel]
                    stats[name] = {"frames": int(sel.sum()), "p50": round(float(np.percentile(x, 50)), 2), "p95": round(float(np.percentile(x, 95)), 2),
                                   "mean": round(float(x.mean()), 2), "rtf": round(float(x.mean()) / 80.0, 3),
                                   "lag_growth": round(float(lg[-1] - lg[0]), 2)}
            stats["cfg_guided_frames"] = int(sum(rec["pcg"]))           # 10-26: = the playground workers' summary fields
            stats["cfg_frames_total"] = max(int(len(ms)) - LP.CD, 0)
            if pc.sum() >= 100:                                         # 10-25: refresh the measured pose-CFG cost shown next to the toggle
                q = stats["pose_cfg_on"]
                POSE_CFG_LAT.update(p50=q["p50"], p95=q["p95"], lag_growth=q["lag_growth"], src=f"session {self.sid}, {q['frames']} frames")
        words = " ".join(w["text"] for w in LP.IP.lane_words(np.asarray(rec["tok"]))) if rec["tok"] else ""
        self._finish(eng, Mo, rec, ms, stats, words, seed, pinfo, ptxt, P, f)

    def _finish(self, eng, Mo, rec, ms, stats, words, seed, pinfo, ptxt, P, f):
        """10-26 main (busy-after-call fix): the end of _run (still on the session thread, still holding srv.lock). Everything that reads the
        GPU / the shared engine is snapshotted HERE (eng.vrec is reused by the next session); the file writes (CPU / disk only) run in a
        background thread, so the server is free (busy=false, lock released) for the next call at once. Same files, same contents, same order
        (session.npz, then session.json last)."""
        self.feed_end()
        sid, od = self.sid, self.od
        a_vec = eng.vrec[:f].float().cpu().numpy().astype(np.float16)
        meta = {"ck": Mo.ck, "step": Mo.step, "label": ck_meta(Mo.ck)["label"], "s": Mo.info["s"], "params": dict(eng.params)}
        self.call_over = True
        self.send({"type": "done", "latency": stats, "session": sid})
        self.save_thread = threading.Thread(target=self._save, name=f"save-{sid}", daemon=False,
                                            args=(od, sid, rec, ms, a_vec, meta, seed, pinfo, ptxt, stats, words, P, f))
        self.save_thread.start()

    def _save(self, od, sid, rec, ms, a_vec, meta, seed, pinfo, ptxt, stats, words, P, f):
        """10-26 busy fix: the post-call save, off the session thread (no GPU, no srv.lock). Each file is written under a temp name and renamed,
        session.json last (the playground importer starts on session.json)."""
        t0 = time.time()
        try:
            os.makedirs(od, exist_ok=True)
            np.savez_compressed(f"{od}/session.tmp.npz", a=np.concatenate(rec["a"]) if rec["a"] else np.zeros(0), b=np.concatenate(rec["b"]) if rec["b"] else np.zeros(0),
                                ms=ms, tok=np.asarray(rec["tok"]), a_vec=a_vec,
                                pp=np.asarray(rec["pp"]), pn=np.asarray(rec["pn"]), lag=np.asarray(rec["lag"]),
                                spk=np.asarray(rec["spk"], np.float32), arp=np.asarray(rec["arp"], np.float32), trp=np.asarray(rec["trp"], np.float32),
                                pon=np.asarray(rec["pon"], np.float32), sil=np.asarray(rec["sil"], np.float32), pb=np.asarray(rec["pb"], np.float32),
                                pose=np.asarray(rec["pose"], np.int16).reshape(-1, 32), mimi=np.asarray(rec["mimi"], np.int16).reshape(-1, 32),
                                pcf=np.asarray(rec["pcf"], np.int8), pcg=np.asarray(rec["pcg"], np.int8))
            os.replace(f"{od}/session.tmp.npz", f"{od}/session.npz")
            with open(f"{od}/session.json.tmp", "w") as fh:
                json.dump({"ck": meta["ck"], "step": meta["step"], "label": meta["label"], "s": meta["s"], "seed": seed, "pose_prefill_frames": self.pose_pf_n, "prefill": self.pf_info,
                           "frame0_global_state": getattr(self, "f0gs", None), "params": meta["params"], "param_log": getattr(self, "param_log", []),
                           "cfg": {k: v for k, v in self.cfg.items() if k != "type"}, "prompt": pinfo, "persona": ptxt, "latency_ms": stats, "B_words": words,
                           "P": P, "end_reason": self.end_reason, "session": sid, "person": self.person, "voice": self.voice,
                           "pose_frames": len(rec["pose"]), "pose_missing_frames": int(sum(1 for q in rec["pose"] if q is None or min(q) < 0)),
                           "cfg_guided_frames": int(sum(rec["pcg"])), "cfg_frames_total": max(f - LP.CD, 0),
                           "feed": self.srv.feed.session_stats(sid)}, fh, indent=1, default=str)
            os.replace(f"{od}/session.json.tmp", f"{od}/session.json")
            log(f"session {sid} done: {stats} | {words[:200]} (files saved in the background in {time.time() - t0:.1f} s)")
        except Exception:  # noqa: BLE001
            log(f"SESSION SAVE ERROR {sid}: " + traceback.format_exc())


class Server:
    def __init__(self, a):
        self.a = a
        self.token = open(a.token_file).read().strip()
        self.lock = threading.Lock()                   # held by a running session and by a model (re)load
        self.active = None
        self.M = None
        self.swap_to = None
        self.state = "loading"
        self.uploads = {}
        self.presets = None
        self.prefix_selftest = None
        self.feed = FD.Feed(feed_url(), log, qmax=512)      # aloe-posedup-live: the videort feed client (own thread; never blocks). 10-26: 512 (was 64)
        # so the ~2 s connect/handshake at a call's start is buffered, not dropped (call b991 lost 20 frames before the first connect)

    def load(self, ck):
        self.state = "loading " + "/".join(ck.rstrip("/").split("/")[-2:])
        if self.M is not None:
            self.M.free(); self.M = None
        codes, first, extra = ck_meta(ck)["preset_codes"], ck_meta(ck)["default_preset"], ck_meta(ck)["extra_presets"]   # 10-24/25: follow the checkpoint
        if self.presets is None or (self.presets.codes, self.presets.first, self.presets.extra) != (codes, first, extra):
            from packaged_presets import PackagedPresets
            self.presets = PackagedPresets(os.environ["PPMLIVE_PACKAGED_OPENINGS"], codes=codes, first=first, extra=extra)
            add_video_people(self.presets)
            log(f"PRESETS codes={codes}: {[(p['id'], p['name'], p['clip']['src_key']) for p in self.presets.items]}")
            self.prefix_selftest = self.presets.selftest()
            log(f"PREFIX SELFTEST {'PASS' if self.prefix_selftest['PASS'] else 'FAIL'} {json.dumps(self.prefix_selftest)}")
            assert self.prefix_selftest["PASS"], "prompt builder != the trainer's podcast_prefix_vox"
            self.prefix_selftest_v2 = self.presets.selftest()
            log(f"PACKAGED RAW PREFIX SELFTEST {'PASS' if self.prefix_selftest_v2['PASS'] else 'FAIL'} {json.dumps(self.prefix_selftest_v2)}")
            assert self.prefix_selftest_v2["PASS"], "packaged raw prefix integrity failed"
        self.M = Model(self.a, ck, self.presets)
        self.smoke()
        self.state = "ready"
        json.dump({"ck": ck, "step": self.M.step, "label": ck_meta(ck)["label"], "prompt_mode": ck_meta(ck)["prompt"], "last_step": ck_last_step(self.M),
                   "loaded": time.strftime("%F %T %Z"), "info": self.M.info, "load_s": self.M.load_s,
                   "smoke": self.smoke_res}, open(f"{STATE}/LOADED.json", "w"), indent=1, default=str)

    @torch.no_grad()
    def smoke(self):
        """internal Start smoke (preset 0, its clip, a few frames of silence through the graphed step) before the model is offered."""
        eng = self.M.eng
        p0 = self.presets.items[0]
        pm = ck_meta(self.M.ck)["prompt"]
        pre, info = PR.build_prefix("" if pm == "voice_clip_only" else p0["prose"] if pm == "persona_prose" else p0["persona"], p0["clip"],
                                    prose=(pm == "persona_prose"))
        eng.start(pre, 1234)
        ms = []
        for f in range(12):
            eng.write_a(f * SPF, np.zeros(SPF + LOOK, np.float32))
            t1 = time.perf_counter()
            dec = eng.step(eng.P + f)
            tok, pp, pn, y = eng.result(eng.P + f, dec)
            ms.append((time.perf_counter() - t1) * 1e3)
            assert np.isfinite(pp) and np.isfinite(pn) and (y is None or np.isfinite(y).all())
        eng.start(pre, 1234, {"cfg_on": 1, "cfg_s": 2.0, "cfg_cb0": 1, "audio_rep_pen": 1.1, "text_rep": 1.1, "freq_pen": 0.3, "min_p": 0.05,
                              "adaptive_rep_on": 1, "audio_rep_speaking": 1.1, "audio_rep_silent": 1.3, "text_rep_speaking": 1.1, "text_rep_silent": 1.2,
                              "pad_bias": -1.0, "adaptive_pad_on": 1, "pad_silent_s": 1.0, "pad_bias_silent": -2.0, "pad_bias_other": 0.5})
        msc = []
        for f in range(40):
            eng.write_a(f * SPF, np.zeros(SPF + LOOK, np.float32))
            t1 = time.perf_counter()
            dec = eng.step(eng.P + f)
            tok, pp, pn, y = eng.result(eng.P + f, dec)
            msc.append((time.perf_counter() - t1) * 1e3)
            assert np.isfinite(pp) and np.isfinite(pn) and (y is None or np.isfinite(y).all())
        CFG_LAT["ms"] = float(np.percentile(msc[3:], 95))
        self.smoke_res = {"frames": 12, "ms_p50": round(float(np.median(ms[2:])), 1), "ms": [round(x, 1) for x in ms], "P": eng.P,
                          "cfg_on_all_knobs_ms_p50": round(float(np.median(msc[3:])), 1), "cfg_on_ms_p95": round(CFG_LAT["ms"], 1),
                          "all_knobs_incl_adaptive_rep": True, "all_knobs_incl_adaptive_pad": True}
        if eng.pcfg:                                                  # 10-25 POSE CFG pass: off for 10 frames, then on (scope all, s 3) mid-session
            eng.start(pre, 1234)
            msp, pose_off = [], []
            for f in range(70):
                if f == 10:
                    eng.set_params({"pose_cfg_on": 1, "pose_cfg_s": 3.0, "pose_cfg_cb0": 1, "pose_cfg_light": 1})
                eng.write_a(f * SPF, np.zeros(SPF + LOOK, np.float32))
                t1 = time.perf_counter()
                dec = eng.step(eng.P + f)
                tok, pp, pn, y = eng.result(eng.P + f, dec)
                (msp if f >= 10 else pose_off).append((time.perf_counter() - t1) * 1e3)
                assert np.isfinite(pp) and np.isfinite(pn) and (y is None or np.isfinite(y).all())
                assert f < LP.CD or min(eng.last_pose) >= 0, ("no pose codes with pose CFG on", f, eng.last_pose)
            x = np.asarray(msp[5:])
            POSE_CFG_LAT.update(p50=round(float(np.percentile(x, 50)), 1), p95=round(float(np.percentile(x, 95)), 1), lag_growth=None,
                                src=f"start-up smoke, {len(x)} frames, silent mic")
            self.smoke_res.update(pose_cfg_on_ms_p50=POSE_CFG_LAT["p50"], pose_cfg_on_ms_p95=POSE_CFG_LAT["p95"], pose_cfg_switched_on_at_frame=10)
        log(f"SMOKE PASS {self.smoke_res}")

    def latest_swap(self):
        """the newest COMPLETE candidate: env PPMLIVE_SWAP_TO + the lines of SWAP_TO.txt (re-read every poll: the fallback watcher appends the
        node-local step-366 copy there if the consolidated /data export never appears)."""
        best = None
        extra = []
        try:
            extra = [ln.strip() for ln in open(f"{STATE}/SWAP_TO.txt") if ln.strip() and not ln.startswith("#")]
        except OSError:
            pass
        for d in SWAP_TO + extra:
            if os.path.isfile(f"{d}/COMPLETE") and (d in SWAP_TO or swap_ready(d)):
                k = (ck_step(d), os.path.getmtime(f"{d}/COMPLETE"))  # aloe-posedup-live: the HIGHEST training step wins (then the newest)
                if best is None or k > best[0]:
                    best = (k, d)
        return best[1] if best else None

    def watcher(self):
        while True:
            time.sleep(20)
            try:
                ck = self.latest_swap()
                if ck and self.M is not None and os.path.abspath(ck) != os.path.abspath(self.M.ck) and ck != getattr(self, "_failed", None):
                    if self.swap_to != ck:
                        log(f"HOT SWAP pending: {ck}")
                    self.swap_to = ck
                if self.swap_to and self.active is None:
                    with self.lock:
                        ck, self.swap_to = self.swap_to, None
                        old = self.M.ck if self.M is not None else None
                        try:
                            self.load(ck)
                            log(f"HOT SWAP done: {old} -> {ck}")
                        except Exception:  # noqa: BLE001
                            log("HOT SWAP FAILED, reloading the previous checkpoint: " + traceback.format_exc())
                            self._failed = ck
                            self.M = None
                            gc.collect(); torch.cuda.empty_cache()
                            self.load(old)
            except Exception:  # noqa: BLE001
                log("watcher: " + traceback.format_exc())

    def ok(self, req):
        return hmac.compare_digest(req.query.get("token", ""), self.token)

    def busy(self):
        """10-26 busy fix: busy = a LIVE call (connected socket, generating). A detached (resume-wait) or ended (saving) session is not busy."""
        s = self.active
        return s is not None and s.in_call()

    async def index(self, req):
        if not self.ok(req):
            return web.Response(status=403, text="forbidden")
        return web.FileResponse(f"{HERE}/index.html", headers={"Cache-Control": "no-store"})

    async def changelog(self, req):
        if not self.ok(req):
            return web.Response(status=403, text="forbidden")
        try:
            with open(f"{HERE}/CHANGELOG.json") as f:
                d = json.load(f)
        except (OSError, ValueError):
            d = {"running": {}, "entries": []}
        d["live"] = dict(LOADED, ck=self.M and self.M.ck)
        return web.json_response(d, headers={"Cache-Control": "no-store"})

    async def presets_api(self, req):
        if not self.ok(req):
            return web.Response(status=403, text="forbidden")
        M, pr = self.M, self.presets
        return web.json_response({"state": self.state, "ck": M and M.ck, "step": M and M.step, "busy": self.busy(),
                                  "presets": pr.public(prose=ck_meta(M and M.ck)["prompt"] == "persona_prose") if pr else [], "clips": clips_labelled(pr) if pr else [],
                                  "default_clip": pr.default_clip() if pr else {}, "default_preset": pr.items[0]["id"] if pr else None,
                                  "swap_pending": self.swap_to, "start_name": "", "upload_ok": True, "last_step": ck_last_step(M),
                                  "label": M and ck_meta(M.ck)["label"], "prompt_default": ck_meta(M and M.ck)["prompt"],
                                  "s": M and M.info["s"], "schema": ready_schema(), "defaults_profile": LP.PROFILE,
                                  "avatar_url": state_line("AVATAR_URL.txt", "PPMLIVE_AVATAR_URL", ""),
                                  "person_default": state_line("PERSON.txt", "PPMLIVE_PERSON", ""), "feed_endpoint": feed_url() or "off",
                                  "voice_person": voice_person_all(),
                                  "pose": bool(M and M.info.get("pose"))})

    async def clip(self, req):
        if not self.ok(req):
            return web.Response(status=403, text="forbidden")
        if self.M is None:
            return web.json_response({"error": "model loading"}, status=503)
        data = await req.read()
        if not data or len(data) > 25 << 20:
            return web.json_response({"error": "empty or > 25 MB"}, status=400)
        try:
            codes, w, info = await asyncio.get_running_loop().run_in_executor(None, upload_codes, self.M, data)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)[:300]}, status=400)
        import secrets
        cid = secrets.token_hex(6)
        self.uploads[cid] = {"codes": codes, **info}
        od = f"{self.a.sessions}/uploads"
        os.makedirs(od, exist_ok=True)
        np.savez_compressed(f"{od}/{cid}.npz", codes=codes, wav=w.astype(np.float16))
        return web.json_response({"id": "upload:" + cid, **info, "label": f"your upload, {info['seconds']:.1f} s (audio-only clip)"})

    def _lost(self, s, gen):
        if s.wsx is None and s.gen == gen and not s.finished:
            log(f"session {s.sid} lost: no resume within {RESUME_S:.0f} s")
            s.end_reason = "lost"
            s.stop.set(); s.inq.put(None)

    async def ws(self, req):
        if not self.ok(req):
            return web.Response(status=403, text="forbidden")
        ws = web.WebSocketResponse(max_msg_size=8 << 20, heartbeat=20)
        await ws.prepare(req)
        loop = asyncio.get_running_loop()
        sess = None                    # the session this socket feeds (its input counts only while sess.wsx is this socket)

        async def pump(s):             # one per session, outlives its sockets: sends to s.wsx; while detached, holds the queue until a resume
            while True:
                item = await s.outq.get()
                if item is None:
                    break
                while True:
                    w = s.wsx
                    if w is None or w.closed:
                        if s.finished and s.wsx is None:
                            break
                        await asyncio.sleep(0.05)
                        continue
                    try:
                        if isinstance(item, bytes):
                            await w.send_bytes(item)
                        else:
                            await w.send_str(json.dumps(item))
                        break
                    except Exception:  # noqa: BLE001
                        if s.wsx is w:
                            await asyncio.sleep(0.05)
                if s.finished and s.wsx is None and s.outq.empty():
                    break
            if self.active is s:
                self.active = None

        try:
            async for msg in ws:
                live = sess is not None and sess.wsx is ws
                if live:
                    sess.last_rx = time.time()
                if msg.type == WSMsgType.BINARY:
                    if live:
                        sess.inq.put(np.frombuffer(msg.data, "<i2").astype(np.float32) / 32768.0)
                elif msg.type == WSMsgType.TEXT:
                    d = json.loads(msg.data)
                    if d.get("type") == "ping":
                        await ws.send_str(json.dumps({"type": "pong", "t": d.get("t")}))
                    elif d.get("type") == "start":
                        if self.state != "ready" or self.M is None:
                            await ws.send_str(json.dumps({"type": "error", "msg": f"model {self.state}; try again in about a minute"}))
                            continue
                        old = self.active
                        # 10-26 busy fix: an ENDED session (stopped / generation over; its files save in the background) blocks nobody; a DETACHED
                        # one (socket closed, resume window) is replaced by a real user's new call (browser / playground) but never by an
                        # automated client; a LIVE call is taken over only by a 'browser' start (the proxy rewrites real users' steals to it).
                        ended = old is not None and (old.finished or old.call_over or old.stop.is_set())
                        detached = old is not None and not ended and (old.wsx is None or old.wsx.closed)
                        cl = d.get("client")
                        if old is not None and not ended and not (cl == "browser" or (detached and cl in REAL_CLIENTS)):
                            await ws.send_str(json.dumps({"type": "error", "msg": "busy: a session is active (automated clients never take over)"}))
                            await ws.close(code=1013, message=b"busy")
                            break
                        if old is not None:                                      # TAKEOVER: the newest real Start wins (ended: just wait for its thread)
                            log(f"session taken over by a new {cl} session" if not ended else f"previous session {old.sid} ended; starting the next one")
                            if not ended:                                        # an ended call keeps its own end_reason (its save reads it)
                                old.end_reason = "taken_over"
                                old.send({"type": "taken_over", "msg": "a newer session took over"})
                            old.stop.set(); old.inq.put(None)
                            if old.thread is not None:
                                await loop.run_in_executor(None, old.thread.join, 60)
                            if self.active is old:
                                self.active = None
                        pin, unk = canon_in(dict(d, **(d.get("params") or {})))
                        d = dict(d, params=pin, _rejected=unk)
                        sess = Session(self, ws, loop, d)
                        self.active = sess
                        sess.pump = asyncio.ensure_future(pump(sess))
                        sess.thread = threading.Thread(target=sess.run, daemon=True)
                        sess.thread.start()
                    elif d.get("type") == "resume":                            # 10-25: continue the active session on this socket
                        s = self.active
                        if (s is None or s.finished or s.call_over or s.stop.is_set() or not s.sid or s.sid != d.get("session")
                                or not hmac.compare_digest(str(d.get("rtoken", "")), s.rtoken)):
                            log(f"resume refused for {d.get('session')}: " + ("no active session" if s is None else "session over" if (s.finished or s.stop.is_set()) else "id/token mismatch"))
                            await ws.send_str(json.dumps({"type": "error", "resume_failed": True, "msg": "could not resume: the session is over"}))
                            await ws.close(code=4404, message=b"no such session")
                            break
                        old_ws, gap = s.wsx, time.time() - s.last_rx
                        s.wsx = ws; s.gen += 1; s.resumes += 1; s.last_rx = time.time(); sess = s
                        log(f"session {s.sid} resumed (#{s.resumes}, {d.get('why', '')[:80]}) after {gap:.1f} s without input")
                        await ws.send_str(json.dumps({"type": "resumed", "session": s.sid, "n": s.resumes, "gap_s": round(gap, 1)}))
                        if old_ws is not None and old_ws is not ws and not old_ws.closed:
                            asyncio.ensure_future(old_ws.close(code=4001, message=b"resumed on another connection"))
                    elif d.get("type") in ("set", "params") and live:
                        pin, unk = canon_in(dict(d, **(d.get("params") or {})))
                        sess.ctl.put((pin, unk))
                    elif d.get("type") == "stop" and live:
                        sess.client_stop = True
                        sess.inq.put(None)
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        finally:
            if sess is not None and sess.wsx is ws:
                if sess.client_stop or sess.finished or sess.stop.is_set() or sess.call_over:
                    sess.stop.set(); sess.inq.put(None)
                    try:
                        await asyncio.wait_for(asyncio.shield(sess.pump), 30)
                    except Exception:  # noqa: BLE001
                        pass
                    if self.active is sess:
                        self.active = None
                else:                                                            # socket lost mid-call: keep the session for a resume
                    sess.wsx = None
                    log(f"session {sess.sid} detached (socket closed); waiting up to {RESUME_S:.0f} s for a resume")
                    loop.call_later(RESUME_S, self._lost, sess, sess.gen)
        return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument("--adapter", default="", help="VOX_ADAPTER dir: used only when --ck is the PPM init (test path: init + adapter + s init)")
    ap.add_argument("--port", type=int, default=8785)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--sessions", default=f"{STATE}/sessions")
    a = ap.parse_args()
    a.ck_init = a.ck if a.adapter else None
    os.makedirs(a.sessions, exist_ok=True)
    srv = Server(a)
    ck = srv.latest_swap() or a.ck                                    # a COMPLETE swap target at start-up wins over --ck
    try:
        srv.load(ck)
    except Exception:  # noqa: BLE001
        if ck == a.ck:
            raise
        log(f"load of {ck} failed, falling back to --ck {a.ck}: " + traceback.format_exc())
        srv._failed = ck
        srv.M = None; gc.collect(); torch.cuda.empty_cache()
        srv.load(a.ck)
    threading.Thread(target=srv.watcher, daemon=True).start()
    app = web.Application(client_max_size=26 << 20)
    app.router.add_get("/", srv.index)
    app.router.add_get("/api/presets", srv.presets_api)
    app.router.add_get("/api/changelog", srv.changelog)
    app.router.add_get("/api/params_schema", lambda r: web.json_response(param_schema()) if srv.ok(r) else web.Response(status=403, text="forbidden"))
    app.router.add_get("/ws", srv.ws)
    app.router.add_post("/api/clip", srv.clip)
    app.router.add_get("/healthz", lambda r: web.json_response({"state": srv.state, "busy": srv.busy(), "ck": srv.M and srv.M.ck}))
    host = os.environ.get("PPMLIVE_BIND", "127.0.0.1")              # 10-25: 0.0.0.0 for a backend another node's front proxy routes to (token-checked)
    log(f"serving on {host}:{a.port}")
    web.run_app(app, host=host, port=a.port, access_log=None, print=None)


if __name__ == "__main__":
    main()
