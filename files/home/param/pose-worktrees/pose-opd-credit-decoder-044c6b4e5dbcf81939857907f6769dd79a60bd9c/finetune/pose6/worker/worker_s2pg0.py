#!/usr/bin/env python3
"""MONOPLEX PLAYGROUND worker (10-25): one hot monoplex engine (aloe-pose-mono-8020 export-357 through ppm/aloe-pose-monoplex-infer's
fast_ppm FastPPM, exactly as posefix/sim/mono.py runs it) serving the genweb run queue. Contract (see server.py): runs/<id>/READY ->
claimed by os.rename(READY, CLAIMED.<worker>) -> status.json updates -> outputs b.wav, words.json, tokens.npz, feed.jsonl, summary.json;
the feed is handed to videort's render service as a playground job (render/jobs/<id>.feed.jsonl -> render/out/<id>.mp4 + _sxs.mp4 + .json).

request.json = {run_id, seed, clip_s (generated s after the prefill), prefill: {person, s, mode "pose"}, prompt: {voice: person id |
"upload:<abs wav>" | null, persona: text}, params: {<FP.PARAMS name>: value}}. Prefill = the actor's real pose tokens forced for pose frames
g < N (N = 2 x round(s / 0.16) + 2, = mono.py / the live server), words + audio sampled from frame 0; the feed's start carries person +
prefill_s so the render prefills the actor's real video for s. The first worker writes OPTIONS.json + SCHEMA.json (every engine knob).
usage: worker.py <worker id>   (CUDA_VISIBLE_DEVICES = 2 GPUs; env as posefix/sim/mono_run.sh)"""
# POSE6 VARIANT (ppm/pose6/worker, 10-27; NOT deployed): the deployed genweb worker.py byte for byte except the lines marked "pose6":
#  - the engine is pose6/infer (= pose-s2p engine + G0COND; absolute path), which serves a checkpoint carrying s2p cb0 + G0COND
#    (tower_p.s_emb + inp.g0); the worker refuses any other checkpoint (check_mods);
#  - every run passes g0f = the frame-0 features of the frame its pose is decoded / rendered against (g0src.g0_from_opening; g0_source
#    below), or the request's explicit "g0f" override (8 finite floats); a missing source fails THAT run loudly (never f = 0);
#  - this file lives in pose6/worker, so the playground dir is the absolute genweb path (+ genweb on sys.path for stick.py, as the deployed
#    script dir); GENWEB_CK is required; GENWEB_WEIGHTS defaults to "s2pg0" (claims only runs with mono_weights "s2pg0", never base / rosie).
import base64
import json
import os
import sys
import time
import traceback

MI = "/data/home/amartyavarma/ppm/pose6/infer"                         # pose6: the pose6 engine (s2p cb0 + G0COND) + g0src.py
GW = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb"            # pose6: the playground dir (the deployed worker's own dir)
sys.path.insert(0, GW)                                                  # pose6: = the deployed script dir on sys.path (import stick)
sys.path.insert(0, MI)
sys.path.insert(0, f"{MI}/ppm")
os.environ.setdefault("PPM_CD", "2")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import infer as MIF  # noqa: E402
import fast_ppm as FP  # noqa: E402
import ppm_io as IO  # noqa: E402
import g0src as G0S  # noqa: E402   (pose6: pose6/infer/g0src.py, A2)
from safetensors.numpy import load_file  # noqa: E402

G0_KINDS = ("openings", "openings-sam", "clip")                          # pose6: the kinds this worker passes (briefs/CONTRACT.md)
assert all(k in G0S.KINDS for k in G0_KINDS), ("pose6: g0src.KINDS lacks a kind this worker passes", G0S.KINDS, G0_KINDS)
G = GW                                                                  # pose6 (deployed: os.path.dirname(os.path.abspath(__file__)))
LIVE = "/data/home/amartyavarma/ppm/aloe-posedup-live"
RUNS, RJOBS, ROUT = f"{G}/runs", f"{G}/render/jobs", f"{G}/render/out"
RJOBS_V = f"{G}/render/jobs_vansh"   # 10-26 user: Vansh-weights runs render from their own queue (videort's Vansh renderers)
RJOBS_V50 = f"{G}/render/jobs_vansh50"   # 10-26 user: Vansh at 20 steps is the default (jobs_vansh); 50 steps on request (vansh_steps 50)
CK = os.environ.get("GENWEB_CK", "")                                   # pose6: no default (export-357 has neither mod); required by Worker
# 10-27 user ("put up rosie"): a worker serves ONE set of monoplex weights; runs pick theirs with request "mono_weights" (base | rosie)
WEIGHTS = os.environ.get("GENWEB_WEIGHTS", "s2pg0")                     # pose6: own weights id by default (never claims base / rosie runs)
WEIGHTS_LABEL = {"base": "export-357", "rosie": "Rosie FT (aloe-pose-rosie export-18)"}
WEIGHTS_LABEL[WEIGHTS] = "s2p cb0 + G0COND (" + (os.path.relpath(CK.rstrip("/"), "/data/home/amartyavarma/ppm") if CK else "GENWEB_CK unset") + ")"   # pose6
PEOPLE = ["/data/home/amartyavarma/video/demo-data/prompts", "/data/home/amartyavarma/video/dmd-rt/actors/pkg",
          "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/people_extra"]   # 10-27: speakers with real data but no avatar (Rosie)
CD, OFF, PAD, DEPTH, SPF = FP.CD, FP.OFF, FP.PAD, FP.DEPTH, 1920
MAX_CLIP_S, MAX_PF_S = 120.0, 11.68
# defaults = posefix's MONO_PARAMS: the fantastic set as far as this engine has the knobs; pose T1 / top-k 30 (cb0 + light); CFG off
DEF = dict(FP.DEFAULTS, **{k: v for k, v in MIF.TONIGHT.items() if k in FP.PARAMS})
DEF.update(text_temp=1.0, text_topk=50, text_topp=1.0, min_p=0.05, min_p_rel=0, freq_pen=0.25, freq_win=50, cb0_temp=1.0, cb0_topk=30,
           light_temp=1.0, light_topk=30, pose_temp=1.0, pose_topk=30, pose_light_temp=1.0, pose_light_topk=151, pose_cfg_on=0)   # 10-25: pose light k151 (user's call setting, posefix's fix)
LABEL = {"text_temp": "text temperature", "text_topk": "text top-k", "text_topp": "text top-p", "min_p": "word min-p", "min_p_rel": "min-p relative",
         "freq_pen": "frequency penalty", "freq_win": "frequency window", "text_rep": "text repetition penalty", "text_rep_win": "text rep window",
         "cb0_temp": "speech cb0 temperature", "cb0_topk": "speech cb0 top-k", "light_temp": "speech cb1-31 temperature", "light_topk": "speech cb1-31 top-k",
         "audio_rep_pen": "speech repetition penalty", "audio_rep_win": "speech rep window", "pose_temp": "pose cb0 temperature", "pose_topk": "pose cb0 top-k",
         "pose_light_temp": "pose cb1-31 temperature", "pose_light_topk": "pose cb1-31 top-k", "pose_rep_cb0": "pose cb0 rep-pen", "pose_rep_light": "pose light rep-pen",
         "pose_rep_cb0_speaking": "pose cb0 rep-pen speaking", "pose_rep_cb0_silent": "pose cb0 rep-pen silent",
         "pose_rep_light_speaking": "pose light rep-pen speaking", "pose_rep_light_silent": "pose light rep-pen silent",   # 10-26 user: cb0 / light split (posefix)
         "pose_rep_win": "pose rep window (0 = all)", "pose_cfg_on": "pose CFG on", "pose_cfg_s": "pose CFG strength", "pose_cfg_cb0": "pose CFG on cb0",
         "pose_cfg_light": "pose CFG on cb1-31", "pose_cfg_speaking_only": "pose CFG only while speaking"}
GROUP = {"text": "Text sampling", "text grammar": "Text grammar", "audio": "Speech sampling", "pose": "Pose sampling", "pose CFG": "Pose CFG",
         "adaptive repetition penalty": "Adaptive repetition penalty"}


def log(*a):
    print(f"[genweb-worker {WID} {time.strftime('%H:%M:%S')}]", *a, flush=True)


def jwrite(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def person_file(person, rel):
    for r in PEOPLE:
        if os.path.isfile(f"{r}/{person}/{rel}"):
            return f"{r}/{person}/{rel}"
    raise FileNotFoundError(f"person {person}: no {rel}")


def opening_frame0_gs(person):
    """10-26 posefix (for main): an opening packaged from a recording WINDOW (meta sources.pose_tokens.frame0.fields) was tokenized against
    that window's frame 0, not the avatar package's -> [root_y, root_x, width] in pose.global_state's units | None (the opening IS the package).
    Same as LIVE/server.py opening_frame0_gs; the feed 'start' carries it as frame0_global_state and the avatar engine decodes against it."""
    try:
        f = json.load(open(f"{G}/openings/{person}/meta.json"))["sources"]["pose_tokens"]["frame0"]["fields"]
        return [float(f["frame0_root"][0]), float(f["frame0_root"][1]), float(f["frame0_width"])]
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def package_dir(person):
    """pose6: the person's avatar package dir (the one holding gt/pose_tokens.safetensors: demo-data/prompts/<p>, people_extra/<p> = Rosie)."""
    return os.path.dirname(os.path.dirname(person_file(person, "gt/pose_tokens.safetensors")))


def g0_source(person, src, po, proot, noav, f0gs):
    """pose6 G0COND: the run's frame-0 source = the frame 0 its pose is decoded / rendered against, exactly the deployed f0gs / sgs choice
    in run(): pose only -> the pose-only source (openings-sam / openings dir, else the actor clip); an avatar run whose start prefill carries
    the opening's own frame 0 (f0gs) -> that opening; a no-avatar person (Rosie) with src start -> the opening when its meta has frame-0
    fields (= the stick figure's sgs); everything else (clip prefill, no pose prefill, openings cut at the package's own frame 0) -> the
    avatar package. -> (dir, kind) for g0src.g0_from_opening."""
    if po:
        return (proot, os.path.basename(os.path.dirname(proot))) if proot else (package_dir(person), "clip")
    if f0gs is not None or (noav and src == "start" and opening_frame0_gs(person) is not None):
        return f"{G}/openings/{person}", "openings"
    return package_dir(person), "clip"


def run_g0f(req, person, src, po, proot, noav, f0gs):
    """pose6: -> (g0f np.float32[8], source dict). request "g0f" (optional, 8 finite floats) overrides the source; otherwise
    g0src.g0_from_opening(dir, kind) of g0_source(); any failure raises a short, clear error for this run (never f = 0)."""
    if req.get("g0f") is not None:
        try:
            f = np.asarray([float(v) for v in req["g0f"]], np.float32)
        except (TypeError, ValueError) as e:
            raise ValueError(f"G0COND: request g0f must be 8 numbers: {e}") from None
        if f.shape != (8,) or not np.isfinite(f).all():
            raise ValueError(f"G0COND: request g0f must be 8 finite numbers, got {req['g0f']!r}"[:300])
        return f, {"kind": "request override", "dir": None}
    d, kind = g0_source(person, src, po, proot, noav, f0gs)
    assert kind in G0_KINDS, kind
    try:
        f = np.asarray(G0S.g0_from_opening(d, kind), np.float32)
    except Exception as e:  # noqa: BLE001   (re-raised with the source named: the page shows it as the run's error)
        raise ValueError(f"G0COND: no frame-0 features for {kind} {os.path.basename(d)}: {type(e).__name__}: {e}"[:380]) from e
    if f.shape != (8,) or not np.isfinite(f).all():
        raise ValueError(f"G0COND: g0src returned a bad f for {kind} {os.path.basename(d)}: {f.tolist()}"[:380])
    return f, {"kind": kind, "dir": d}


def g0_warm(pp):
    """pose6: warm g0src's caches at worker start (its first call scans the v2 items, ~1.4 s), over every source a run can pick for the
    playground's persons. A source that fails here is only logged: the run that picks it still fails loudly in run_g0f."""
    ok, bad = 0, []
    for p in pp:
        srcs = [(f"{G}/{r}/{p['id']}", r) for r in ("openings", "openings-sam") if os.path.isfile(f"{G}/{r}/{p['id']}/meta.json")]
        try:
            srcs.append((package_dir(p["id"]), "clip"))
        except FileNotFoundError as e:
            bad.append(f"{p['id']} clip: {e}")
        for d, kind in srcs:
            try:
                G0S.g0_from_opening(d, kind)
                ok += 1
            except Exception as e:  # noqa: BLE001
                bad.append(f"{p['id']} {kind}: {type(e).__name__}: {str(e)[:120]}")
    return ok, bad


def check_mods(m, binfo):
    """pose6: this worker serves ONLY a checkpoint that carries BOTH mods as trained weights: s2p cb0 (tower_p.s_emb) and G0COND (inp.g0).
    A zero-filled / missing term (the engine's flag-on-without-key path) is refused."""
    zi = sorted(k for k, v in binfo.items() if "zero_init" in str(k) and v)
    assert binfo.get("s2p") == "cb0", ("pose6 worker: the checkpoint must carry tower_p.s_emb in mode cb0", binfo.get("s2p"), CK)
    assert bool(binfo.get("g0cond", True)) and hasattr(m.inp, "g0"), ("pose6 worker: the checkpoint must carry inp.g0 (G0COND)", CK)
    assert not zi, ("pose6 worker: new params zero-filled, not loaded from the checkpoint", zi, CK)


def persons():
    out = []
    for r in PEOPLE[:1]:
        for d in sorted(os.listdir(r)):
            if os.path.isfile(f"{r}/{d}/gt/pose_tokens.safetensors") and os.path.isfile(f"{r}/{d}/gt/mimi_local.safetensors"):
                out.append({"id": d, "label": f"Person {d[:2]}", "max_prefill_s": MAX_PF_S})
    ex = PEOPLE[2]                                                    # 10-27 user: Rosie (her real data from the stores; NO_AVATAR -> stick figure)
    for d in sorted(os.listdir(ex)) if os.path.isdir(ex) else []:
        if os.path.isfile(f"{ex}/{d}/gt/pose_tokens.safetensors") and os.path.isfile(f"{ex}/{d}/gt/mimi_local.safetensors"):
            try:
                lab = json.load(open(f"{ex}/{d}/meta.json")).get("label") or d
            except (OSError, ValueError):
                lab = d
            out.append({"id": d, "label": lab, "max_prefill_s": MAX_PF_S, "no_avatar": os.path.isfile(f"{ex}/{d}/NO_AVATAR")})
    return out


def no_avatar(person):
    return os.path.isfile(f"{PEOPLE[2]}/{person}/NO_AVATAR")


def schema_rows(pp):
    """every engine knob (FP.PARAMS / DEFAULTS / HINTS) + the run controls, in the live demo's schema row format."""
    ids = [p["id"] for p in pp]
    rows = [
        {"name": "seed", "label": "seed (0 = random)", "group": "Run", "type": "int", "default": 0, "range": [0, 2 ** 31 - 1], "step": 1,
         "hint": "sampling seed; with several samples, seeds seed .. seed+N-1 (0 = a random seed per sample)"},
        {"name": "n_samples", "label": "samples per Generate", "group": "Run", "type": "int", "default": 1, "range": [1, 16], "step": 1,
         "hint": "independent samples (one run each, generated in parallel on the free workers)"},
        {"name": "clip_s", "label": "generated seconds (after the prefill)", "group": "Run", "type": "float", "default": 20.0, "range": [2.0, MAX_CLIP_S],
         "step": 1.0, "hint": "how long the model generates after the prefill"},
        {"name": "pose_only", "label": "pose only (real speech)", "group": "Run", "type": "bool", "default": False, "range": None, "step": None,
         "hint": "10-27: B's speech and timed words are the actor's REAL ones for the whole clip (forced every frame); only B's pose is generated, "
                 "shown as a stick figure (no avatar video). Source: the actor's conversation-start opening (SAM-isolated codes when "
                 "genweb/openings-sam has the actor, e.g. Person 06 = the duplex clip), else the actor clip; capped to the real speech on disk (12 s)"},
        {"name": "prefill_person", "label": "prefill actor", "group": "Prefill", "type": "enum", "default": "01_ab63d745", "choices": ids,
         "choice_labels": {p["id"]: p["label"] for p in pp}, "hint": "the avatar person; their real pose (and real video, in the render) for the prefill seconds"},
        {"name": "prefill_s", "label": "prefill seconds", "group": "Prefill", "type": "float", "default": 1.92, "range": [0.0, MAX_PF_S], "step": 0.16,
         "hint": "first S s: B's pose = the actor's real pose tokens (the render shows their real video for S); words + audio are sampled from frame 0. 0 = no prefill"},
        {"name": "prefill_src", "label": "prefill source", "group": "Prefill", "type": "enum", "default": "start", "choices": ["start", "clip"],
         "choice_labels": {"start": "conversation start (the recording's first seconds)", "clip": "actor clip (the avatar package's 12 s chunk)"},
         "hint": "where the prefill seconds come from; 'conversation start' needs the actor's opening on disk (else the run falls back to the clip and says so)"},
        {"name": "prefill_pose", "label": "prefill: pose", "group": "Prefill", "type": "bool", "default": True, "range": None, "step": None,
         "hint": "force B's pose to the actor's real pose tokens for the prefill frames"},
        {"name": "prefill_audio", "label": "prefill: audio", "group": "Prefill", "type": "bool", "default": True, "range": None, "step": None,
         "hint": "force B's speech codes to the actor's real Mimi codes (gt/mimi_local, -23 LUFS) for the prefill frames"},
        {"name": "prefill_words", "label": "prefill: words", "group": "Prefill", "type": "bool", "default": True, "range": None, "step": None,
         "hint": "force B's word lane to the actor's timed words (only Persons 10 and 19 have timed words; others get a PAD lane = off-distribution)"},
        {"name": "voice", "label": "voice clip", "group": "Prompt", "type": "enum", "default": "same", "choices": ["same", "none"] + ids,
         "choice_labels": dict({"same": "the prefill actor's own voice", "none": "no voice clip"}, **{p["id"]: p["label"] + "'s voice" for p in pp}),
         "hint": "B's voice clip in the system prompt (first 10 s of the person's own audio). Upload your own below"},
        {"name": "persona", "label": "persona text", "group": "Prompt", "type": "text", "default": "", "hint": "optional persona text in the system prompt"},
    ]
    for k, (t, lo, hi) in FP.PARAMS.items():
        r = {"name": k, "label": LABEL.get(k, k.replace("_", " ")), "group": GROUP.get(FP.GROUPS.get(k, "text"), "Other"),
             "type": "bool" if (t is int and lo == 0 and hi == 1) else ("int" if t is int else "float"), "default": DEF.get(k), "range": [lo, hi],
             "step": 1 if t is int else (0.01 if hi <= 3 else 0.05), "hint": str(FP.HINTS.get(k, "")) if hasattr(FP, "HINTS") else ""}
        if r["type"] == "bool":
            r.update(default=bool(DEF.get(k)), range=None, step=None)
        rows.append(r)
    return rows


def lufs23(w):
    try:
        import pyloudnorm as pyln
        L = pyln.Meter(24000).integrated_loudness(w.astype(np.float64))
        g = (-23.0 - L) if np.isfinite(L) else 0.0
    except Exception:  # noqa: BLE001   (no pyloudnorm: RMS-based fallback ~ -23 LUFS for speech)
        r = float(np.sqrt(np.mean(w ** 2)) + 1e-9)
        g = -20.0 - 20 * np.log10(r)
    y = w * np.float32(10 ** (g / 20))
    pk = float(np.abs(y).max()) if len(y) else 0
    return (y * (0.995 / pk) if pk > 0.995 else y).astype(np.float32)


def load_wav24(path):
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    w = w.mean(1)
    if sr != 24000:
        g = gcd(int(sr), 24000)
        w = resample_poly(w, 24000 // g, int(sr) // g).astype(np.float32)
    return w


class Worker:
    def __init__(self):
        devs = [torch.device("cuda:0"), torch.device("cuda:1")]
        torch.cuda.set_device(devs[-1])
        self.pp = persons()
        assert CK, "pose6 worker: set GENWEB_CK to an s2p cb0 + G0COND checkpoint (export / w- path)"   # pose6
        with torch.inference_mode():
            m, ldev, cfgs, binfo = FP.build_model(CK, devs)
            assert binfo["pose"] and binfo["decp_bb"], binfo
            check_mods(m, binfo)                                        # pose6: both mods, loaded from the checkpoint
            self.step = binfo.get("step")
            self.mimi = IO.Mimi(devs[-1])
            self.max_s = MAX_CLIP_S + MAX_PF_S + 2
            nfr = int(round(self.max_s * 12.5)) + 8
            sil = self.mimi.encode(np.zeros((2048 + nfr) * SPF, np.float32))[:, :2048 + nfr]
            self.eng = FP.FastPPM(m, ldev, cfgs, devs, n_inst=1, cfg_rows=True, max_s=self.max_s, mimi_stream=False, silence=sil)
            self.eng.capture()
        g0ok, g0bad = g0_warm(self.pp)                                  # pose6: g0src caches warm before the first claim
        log(f"g0src warm: {g0ok} sources ok, {len(g0bad)} failing (their runs will fail loudly){': ' + '; '.join(g0bad[:8]) if g0bad else ''}")
        log(f"model step {self.step} loaded + captured ({CK})")

    def feed_client(self):
        """10-25 (user: renders must stream): videort's live avatar feed client (the live demo's feed.py: own thread, bounded queue, never
        blocks); each run streams start / f / end to it while it generates, the page shows videort's live player for the run."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("genweb_feed", f"{LIVE}/feed.py")
        FD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(FD)
        try:                                                            # = the live server's feed_url(): FEED_URL.txt, else feed.py's default
            url = open(f"{G}/FEED_URL.txt" if os.path.exists(f"{G}/FEED_URL.txt") else f"{LIVE}/FEED_URL.txt").read().split()[0]   # 10-26: the playground's own avatar front (videort front P)
        except (OSError, IndexError):
            url = FD.DEFAULT_URL
        url = "" if url.lower() in ("off", "none", "0") else url
        self.feed = FD.Feed(url, log, qmax=4096)
        log(f"live feed -> {url or 'off'}")

    def stream(self, rid, P, f, sent, final=False):
        """frames sent .. f - CD (conversation frames) -> the live feed: pose + mimi (drive the avatar) + this frame's pcm (decoded with 2 s
        of left context, so the stream carries B's voice). -> new 'sent'."""
        K = f - CD + 1
        if K <= sent:
            return sent
        S = self.eng.readback(0, P + f + 1)
        c = S["dep_b"][P + CD:P + CD + K] - OFF
        p = S["pdep"][P + CD:P + CD + K] - OFF
        if c.min() < 0 or p.min() < 0:
            return sent
        a0 = max(0, sent - 25)
        w = np.asarray(self.mimi.decode(c[a0:K].T.astype(np.int64)), np.float32).reshape(-1)[(sent - a0) * SPF:]
        for k in range(sent, K):
            seg = w[(k - sent) * SPF:(k - sent + 1) * SPF]
            seg = seg if len(seg) == SPF else np.zeros(SPF, np.float32)
            pcm = np.clip(seg * 32767, -32768, 32767).astype("<i2").tobytes()
            self.feed.put("f", rid, json.dumps({"type": "f", "session": rid, "f": k, "t": round(k * 0.08, 3), "pose": [int(v) for v in p[k]],
                                                "mimi": [int(v) for v in c[k]], "pcm_b64": base64.b64encode(pcm).decode("ascii")}))
        return K

    def options(self):
        if WEIGHTS != "base":                                           # 10-27: only base workers write the shared schema / options
            return
        rows = schema_rows(self.pp)
        jwrite(f"{G}/SCHEMA.json", rows)
        o = {"models": ["monoplex"], "model_label": f"pose monoplex ({WEIGHTS_LABEL[WEIGHTS]}, step {self.step})",   # pose6: this worker's weights
             "persons": self.pp, "defaults": {r["name"]: r["default"] for r in rows}, "max_clip_s": MAX_CLIP_S,
             "max_prefill_s": MAX_PF_S, "written_by": WID, "t": time.time()}
        try:                                                            # 10-25: keep the duplex worker's keys (worker_dup.py writes them)
            old = json.load(open(f"{G}/OPTIONS.json"))
            if isinstance(old.get("duplex"), dict):
                o.update({k: v for k, v in old.items() if k.startswith("duplex")})
                o["models"] = ["monoplex", "duplex"]
        except (OSError, ValueError, AttributeError):
            pass
        jwrite(f"{G}/OPTIONS.json", o)

    def gt_render(self, only=None):
        """one GT render job per person (their real 12 s pose + audio through the avatar engine), once. 10-26 user ("i dont need to see gt
        video", default off): queued on demand only, for the run's person when its request asks for GT (gt: true), never at startup."""
        os.makedirs(RJOBS, exist_ok=True)
        for p in self.pp:
            name = f"gt-{p['id']}"
            if only is not None and p["id"] not in only:
                continue
            if os.path.exists(f"{RJOBS}/{name}.feed.jsonl") or os.path.exists(f"{ROUT}/{name}.mp4"):
                continue
            try:
                pose = np.asarray(load_file(person_file(p["id"], "gt/pose_tokens.safetensors"))["codes"], np.int64)
                codes = np.asarray(load_file(person_file(p["id"], "gt/mimi_local.safetensors"))["codes"], np.int64)
                n = min(pose.shape[1], codes.shape[1])
                with torch.inference_mode():
                    w = np.asarray(self.mimi.decode(codes[:, :n]), np.float32).reshape(-1)
                self.write_feed(f"{RJOBS}/{name}.feed.jsonl", name, p["id"], 0.0, pose[:, :n].T, codes[:, :n].T, w, n)
            except Exception as e:  # noqa: BLE001
                log(f"gt render job {name} failed: {e}")

    @staticmethod
    def write_feed(path, name, person, pf, pose, mimi, w, n, f0gs=None):
        tmp = path + ".tmp"
        with open(tmp, "w") as fo:
            fo.write(json.dumps({"type": "start", "session": name, "model": "genweb-mono", "voice": person, "fps": 12.5, "person": person,
                                 "prefill_s": float(pf), **({"frame0_global_state": f0gs} if f0gs else {})}) + "\n")
            for k in range(n):
                seg = w[k * SPF:(k + 1) * SPF] if len(w) >= (k + 1) * SPF else np.zeros(SPF, np.float32)
                pcm = np.clip(seg * 32767, -32768, 32767).astype("<i2").tobytes()
                fo.write(json.dumps({"type": "f", "session": name, "f": k, "t": round(k * 0.08, 3), "pose": [int(v) for v in pose[k]],
                                     "mimi": [int(v) for v in mimi[k]], "pcm_b64": base64.b64encode(pcm).decode("ascii")}) + "\n")
            fo.write(json.dumps({"type": "end", "session": name}) + "\n")
        os.replace(tmp, path)

    def run(self, rd, req):
        rid = req["run_id"]
        st = {"state": "running", "progress": 0.0, "msg": "building the prompt", "worker": WID, "t_start": time.time()}
        jwrite(f"{rd}/status.json", st)
        pr = dict(req.get("params") or {})
        po = bool(pr.pop("pose_only", False))                          # 10-27 user: pose only (real speech, stick figure); not an engine knob
        for old, new in (("pose_rep", ("pose_rep_cb0", "pose_rep_light")), ("pose_rep_speaking", ("pose_rep_cb0_speaking", "pose_rep_light_speaking")),
                         ("pose_rep_silent", ("pose_rep_cb0_silent", "pose_rep_light_silent"))):   # 10-26: pre-split pose rep-pen keys set BOTH new keys
            if old in pr and old not in FP.PARAMS and all(n in FP.PARAMS for n in new):            # (new keys win; = fast_ppm.validate's alias)
                v = pr.pop(old)
                for n in new:
                    pr.setdefault(n, v)
        bad = [k for k in pr if k not in FP.PARAMS]
        if bad:
            raise ValueError(f"unknown engine knobs: {bad}")
        params = dict(DEF, **pr)
        for k, (t, lo, hi) in FP.PARAMS.items():
            params[k] = t(min(max(t(params[k]), lo), hi))
        seed = int(req.get("seed") or 0) or int.from_bytes(os.urandom(3), "little") + 1
        clip_s = min(max(float(req.get("clip_s") or 20.0), 1.0), MAX_CLIP_S)
        pf = req.get("prefill") or {}
        person = pf.get("person") or "01_ab63d745"
        noav = no_avatar(person)                                       # 10-27: a person with no avatar (Rosie): stick figure, no GT / live avatar
        if req.get("gt") is True and not noav:                        # 10-26: GT render only when the user shows GT
            self.gt_render(only={person})
        pfs = min(max(float(pf.get("s") or 0.0), 0.0), MAX_PF_S)
        prm = req.get("prompt") or {}
        voice, persona = prm.get("voice"), (prm.get("persona") or "").strip() or None
        if voice in (None, "", "none"):
            clip, vsrc = None, "none"
        elif str(voice).startswith("upload:"):
            w = lufs23(load_wav24(str(voice)[7:]))
            L = min(len(w) // SPF, 125)
            if L < 38:
                raise ValueError(f"uploaded clip too short: {len(w) / 24000:.1f} s (needs >= 3 s)")
            with torch.inference_mode():
                clip = np.asarray(self.mimi.encode(w[:L * SPF]), np.int64)[:, :L]
            vsrc = f"upload ({L / 12.5:.1f} s, -23 LUFS)"
        else:
            vp = person if voice == "same" else str(voice)
            clip = np.asarray(load_file(person_file(vp, "gt/mimi_local.safetensors"))["codes"], np.int64)[:, :125]
            vsrc = f"{vp} own voice (10 s)"
        sp = MIF.system_block(clip, persona)
        # 10-25 FULL PREFILL (user via main / posefix): for the first N frames (N = 2 x round(s / 0.16) + 2, = the real-video prefill)
        # force any of B's three streams to the actor's own: pose (gt/pose_tokens), audio (gt/mimi_local = lufs23-local) and the word lane
        # (infer.lane_from_words over the actor's timed words; lane frames 0..N+1, codes / pose frames 0..N-1). Persons without timed words:
        # the lane is PAD on 0..N+1 when audio is forced (off-distribution, said in the run summary).
        use_pose, use_audio, use_words = bool(pf.get("pose", True)), bool(pf.get("audio", True)), bool(pf.get("words", True))
        pp = pa = lane_f = None
        words_used = False
        N = 0
        notes = []                                                      # (set before the source pick below)
        src = pf.get("src") or "start"
        odir = f"{G}/openings/{person}"
        if src == "start" and not (os.path.isfile(f"{odir}/pose_tokens.safetensors") and os.path.isfile(f"{odir}/mimi_local.safetensors")):
            notes.append("no conversation-start data on disk for this actor: prefill from the actor clip")
            src = "clip"
        vpf = pfs                                                      # the avatar's REAL-video prefill: only when that video is the same moment
        if src == "start":
            try:
                ov = (json.load(open(f"{G}/openings/SUMMARY.json")).get(person) or {}).get("video")
            except (OSError, ValueError):
                ov = None
            if not ov:
                vpf = 0.0
                notes.append("no opening video on disk for this actor: the avatar's first seconds are rendered from the forced pose (no real-video prefill)")

        def gt(rel):
            return f"{odir}/{rel}" if src == "start" else person_file(person, f"gt/{rel}")
        if pfs > 0 and (use_pose or use_audio or use_words):
            c = np.asarray(load_file(gt("pose_tokens.safetensors"))["codes"], np.int64)
            N = min(c.shape[1], 2 * int(round(pfs / 0.16)) + 2)
            if use_pose:
                pp = c[:, :N].T.copy()
            if use_audio:
                a_ = np.asarray(load_file(gt("mimi_local.safetensors"))["codes"], np.int64)
                pa = a_[:, :N].T.copy()
            wf = f"{odir}/words.json" if src == "start" else f"{G}/words/{person}/words.json"
            words_used = False
            if use_words and os.path.isfile(wf):                       # built over the whole span, then cut (posefix: a word starting just after
                lane_f = MIF.lane_from_words(MIF.read_words(wf), max(N + CD, 152) + 64)[:N + CD]   # the window keeps its NEW_WORD on the last frame)
                words_used = True
            elif use_words or use_audio:
                lane_f = np.full(N + CD, PAD, np.int64)
                notes.append("no timed words for this actor: the prefill lane is PAD (off-distribution)" if use_words else "prefill lane PAD (words off)")
        f0gs = opening_frame0_gs(person) if (src == "start" and pp is not None and N > 0) else None   # 10-26 posefix: the opening's frame 0
        n = N + int(round(clip_s * 12.5))
        NA = N                                                          # frames whose audio / lane are forced (prefill only, unless pose only)
        sgs = None
        if po:                                                          # 10-27 POSE ONLY: real audio + timed words every frame, pose generated
            notes.clear()                                               # (the avatar-prefill notes above do not apply: no avatar video)
            proot = next((r for r in (f"{G}/openings-sam/{person}", f"{G}/openings/{person}")
                          if all(os.path.isfile(f"{r}/{x}") for x in ("pose_tokens.safetensors", "mimi_local.safetensors", "meta.json"))), None)
            pget = (lambda rel: f"{proot}/{rel}") if proot else (lambda rel: person_file(person, f"gt/{rel}"))
            pt = load_file(pget("pose_tokens.safetensors"))
            cpo, apo = np.asarray(pt["codes"], np.int64), np.asarray(load_file(pget("mimi_local.safetensors"))["codes"], np.int64)
            T = min(cpo.shape[1], apo.shape[1])
            N = min(T - 1, 2 * int(round(pfs / 0.16)) + 2) if pfs > 0 else 0
            pp = cpo[:, :N].T.copy() if (N > 0 and use_pose) else None
            n = min(N + int(round(clip_s * 12.5)), T)
            NA = n
            pa = apo[:, :n].T.copy()
            wf = f"{proot}/words.json" if proot else f"{G}/words/{person}/words.json"
            if os.path.isfile(wf):
                lane_f = MIF.lane_from_words(MIF.read_words(wf), max(n + CD, 152) + 64)[:n + CD]
                words_used = True
            else:
                lane_f = np.full(n + CD, PAD, np.int64)
                words_used = False
                notes.append("pose only: no timed words for this actor's real speech, so the word lane is PAD (off-distribution)")
            if proot:
                f0m = json.load(open(f"{proot}/meta.json"))["sources"]["pose_tokens"]["frame0"]
                if "fields" in f0m:
                    fl = f0m["fields"]
                    sgs = [float(fl["frame0_root"][0]), float(fl["frame0_root"][1]), float(fl["frame0_width"])]
                else:                                                   # 10-27: openings cut at the package's own frame 0 (13/14/15...) name
                    sgs = [float(v) for v in np.asarray(load_file(f0m["path"])["global_state"], np.float32).reshape(-1)[:3]]   # its llm_inputs
            else:
                sgs = [float(v) for v in np.asarray(pt["global_state"], np.float32).reshape(-1)[:3]]
            src = ("pose only: " + (("SAM-isolated opening " if "openings-sam" in proot else "conversation-start opening ") + os.path.basename(proot)
                                    if proot else "actor clip"))
            if n < N + int(round(clip_s * 12.5)):
                notes.append(f"pose only: the actor's real speech on disk is {T / 12.5:.1f} s, so the clip is {n / 12.5:.1f} s")
            vpf, f0gs = 0.0, None
        g0f, g0s = run_g0f(req, person, src, po, proot if po else None, noav, f0gs)   # pose6: G0COND f (a missing source fails this run)
        lane = np.full(n + CD, PAD, np.int64)
        x = MIF.assemble(sp, lane, np.zeros((DEPTH, 0), np.int64), np.zeros((DEPTH, 0), np.int64), n, 0)
        P = int(x["P"])
        lane_ids = None
        if lane_f is not None:                                         # the forced lane in the engine's own ids (assemble's placement)
            lw = np.full(n + CD, PAD, np.int64)
            lw[:NA + CD] = lane_f[:NA + CD]
            lane_ids = np.asarray(MIF.assemble(sp, lw, np.zeros((DEPTH, 0), np.int64), np.zeros((DEPTH, 0), np.int64), n, 0)["ids"][1], np.int64)
        t0 = time.time()
        live = (getattr(self, "feed", None) is not None and self.feed.enabled() and req.get("stream") is True
                and not req.get("no_live") and not po and not noav)                     # 10-26 user: never livestream unless the request asks for it
        sent = 0
        with torch.inference_mode():
            self.eng.start([dict(x, a_mode="empty", g0f=g0f)], seed, params, f0=0)   # pose6: g0f per instance (engine: SLOT = P - 1)
            if live:
                self.feed.put("start", rid, json.dumps({"type": "start", "session": rid, "model": "genweb-mono", "voice": person, "fps": 12.5,
                                                         "person": person, "prefill_s": vpf, "prefill_src": src, "record": f"{rd}/avatar.mp4",
                                                         **({"weights": "vansh"} if req.get("video_weights") == "vansh" else {}),
                                                         **({"frame0_global_state": f0gs} if f0gs else {})}))
            for f in range(n + CD):
                g = f - CD
                frc = {}
                if pp is not None and 0 <= g < N:
                    frc["pose"] = (pp[g] + OFF)[None]
                if pa is not None and 0 <= g < NA:
                    frc["codes"] = (pa[g] + OFF)[None]
                if lane_ids is not None and f < NA + CD:
                    frc["lane"] = lane_ids[P + f:P + f + 1]
                self.eng.step(f, force=frc or None, cold_pad=lane_ids is None)
                if live and f >= CD + 4 and (f - CD) % 5 == 4:
                    sent = self.stream(rid, P, f, sent)
                if f % 25 == 0:
                    if os.path.exists(f"{rd}/CANCEL"):
                        raise RuntimeError("cancelled")
                    el = time.time() - t0
                    jwrite(f"{rd}/status.json", dict(st, progress=round(f / (n + CD), 3), frames_done=f, frames_total=n + CD,
                                                     ms_per_frame=round(1e3 * el / max(1, f), 1), msg=f"generating {f}/{n + CD} frames"))
            if live:
                sent = self.stream(rid, P, n + CD - 1, sent)
                self.feed.put("end", rid, json.dumps({"type": "end", "session": rid}))
            S = self.eng.readback(0, P + n + CD)
            wall = time.time() - t0
            lane_b, dep_b, pdep = S["lane_b"], S["dep_b"], S["pdep"]
            c1, p1 = dep_b[P + CD:P + CD + n] - OFF, pdep[P + CD:P + CD + n] - OFF
            assert c1.min() >= 0 and p1.min() >= 0, "codes missing"
            if pp is not None:
                assert (p1[:N] == pp).all(), "pose prefill != actor tokens"
            if pa is not None:
                assert (c1[:NA] == pa[:NA]).all(), "forced audio != actor codes"
            w = np.asarray(self.mimi.decode(c1.T.astype(np.int64)), np.float32).reshape(-1)
        import soundfile as sf
        sf.write(f"{rd}/b.wav.part.wav", np.clip(w, -1, 1), 24000, subtype="PCM_16")
        os.replace(f"{rd}/b.wav.part.wav", f"{rd}/b.wav")
        words = IO.lane_words(lane_b[:P + n], P, P, P)
        ts = [float(q["t"]) for q in words]                              # lane_words: {text, frame, t (s from conversation frame 0)}
        wl = [{"t0": ts[i], "t1": round(min(ts[i + 1] if i + 1 < len(ts) else ts[i] + 0.4, ts[i] + 1.5), 2), "w": q["text"], "ch": "B",
               "prefill": ts[i] < N / 12.5} for i, q in enumerate(words)]
        jwrite(f"{rd}/words.json", wl)
        np.savez_compressed(f"{rd}/tokens.npz", pose=p1.astype(np.int16), mimi=c1.astype(np.int16), lane=lane_b[P:P + n])
        if (po or noav) and sgs is None:                              # 10-27 no-avatar person without pose only: decode frame of the prefill source
            try:
                fl = json.load(open(f"{G}/openings/{person}/meta.json"))["sources"]["pose_tokens"]["frame0"]["fields"] if src == "start" else None
                sgs = [float(fl["frame0_root"][0]), float(fl["frame0_root"][1]), float(fl["frame0_width"])] if fl else None
            except (OSError, ValueError, KeyError, TypeError):
                sgs = None
            if sgs is None:
                sgs = [float(v) for v in np.asarray(load_file(person_file(person, "gt/pose_tokens.safetensors"))["global_state"], np.float32).reshape(-1)[:3]]
        if po or noav:                                                  # 10-27 pose only / no-avatar person: a stick figure instead of the avatar render
            import stick
            stick.stick_mp4(p1, N, sgs, f"{rd}/b.wav", f"{rd}/stick.mp4")
        else:
            self.write_feed(f"{rd}/feed.jsonl", rid, person, vpf, p1, c1, w, n, f0gs=f0gs)
            jd = (RJOBS_V if req.get("vansh_steps") == 20 else RJOBS_V50) if req.get("video_weights") == "vansh" else RJOBS   # no field = 50 (old)
            os.makedirs(jd, exist_ok=True)
            with open(f"{rd}/feed.jsonl", "rb") as fi, open(f"{jd}/{rid}.feed.jsonl.tmp", "wb") as fo:
                fo.write(fi.read())
            os.replace(f"{jd}/{rid}.feed.jsonl.tmp", f"{jd}/{rid}.feed.jsonl")
        gen = p1[N:, 0].astype(int)
        chg = float((gen[1:] != gen[:-1]).mean()) if len(gen) > 1 else None
        text = " ".join(q["w"] for q in wl)
        summ = {"run_id": rid, "model": f"monoplex {WEIGHTS_LABEL.get(WEIGHTS, WEIGHTS)} (step {self.step})", "mono_weights": WEIGHTS, "person": person, "seed": seed, "P": P, "frames": n,
                "pose_prefill_frames": N if pp is not None else 0, "prefill_frames": N, "prefill_s": pfs,
                "prefill_streams": [k for k, v in (("pose", pp is not None), ("audio", pa is not None), ("words", words_used)) if v],
                "prefill_notes": notes, "prefill_source": src, "pose_only": po, "forced_audio_frames": NA if pa is not None else 0, "frame0_global_state": f0gs, "clip_s": clip_s, "voice": vsrc, "persona": persona, "params": params,
                "g0f": [float(v) for v in g0f], "g0_source": g0s,                                        # pose6: the G0COND f this run used
                "wall_s": round(wall, 1), "ms_per_frame": round(1e3 * wall / (n + CD), 1), "gen_cb0_change": chg,
                "cfg_guided_frames": int(self.eng.pcfg_guided[0].item()) if getattr(self.eng, "pcfg_guided", None) is not None else None,
                "cfg_frames_total": n,
                "gen_cb0_distinct": int(len(np.unique(gen))) if len(gen) else 0, "n_words": len(wl), "words": text[:600]}
        jwrite(f"{rd}/summary.json", summ)
        head = (f"{len(wl)} words · pose cb0 changes on {100 * chg:.0f}% of generated frames · {summ['ms_per_frame']} ms/frame · seed {seed}"
                if chg is not None else f"{len(wl)} words · seed {seed}")
        jwrite(f"{rd}/status.json", dict(st, state="done", progress=1.0, msg=("pose only: stick figure ready (real speech, generated pose)" if po else ("stick figure ready (no avatar of this person)" if noav else "generated; avatar rendering")), headline=head, t_end=time.time(),
                                         ms_per_frame=summ["ms_per_frame"], frames_done=n + CD, frames_total=n + CD, seed=seed))
        log(f"run {rid}: {head}")

    def loop(self):
        while True:
            if os.path.exists(f"{G}/flex/drain/{WID}"):                    # 10-26: DRAIN (a unit switch is pending): claim nothing new; the
                time.sleep(1.0)                                         # flex controller stops this worker only once its last run is done
                continue
            try:
                cands = sorted(r for r in os.listdir(RUNS) if os.path.exists(f"{RUNS}/{r}/READY"))
                def _hid(r):                                            # 10-26 user: agent test runs (request "hidden") go last
                    try:
                        return bool(json.load(open(f"{RUNS}/{r}/request.json")).get("hidden"))
                    except (OSError, ValueError):
                        return False
                cands.sort(key=_hid)                                    # stable: oldest first within each group
            except OSError:
                cands = []
            got = False
            for r in cands:
                rd = f"{RUNS}/{r}"
                try:                                                    # 10-25: "model": "duplex" runs belong to worker_dup.py
                    rq_ = json.load(open(f"{rd}/request.json"))
                    if (rq_.get("model") or "monoplex") == "duplex":
                        continue
                    if (rq_.get("mono_weights") or "base") != WEIGHTS:  # 10-27: another worker's weights (base | rosie)
                        continue
                except (OSError, ValueError):
                    pass                                                # unreadable request: claim + report the error as before
                try:
                    os.rename(f"{rd}/READY", f"{rd}/CLAIMED.{WID}")
                except OSError:
                    continue                                            # another worker took it
                got = True
                try:
                    req = json.load(open(f"{rd}/request.json"))
                    self.run(rd, req)
                except Exception as e:  # noqa: BLE001
                    log(f"run {r} failed: {traceback.format_exc()[-1500:]}")
                    jwrite(f"{rd}/status.json", {"state": "error", "error": f"{type(e).__name__}: {str(e)[:400]}", "worker": WID, "t_end": time.time()})
                break
            if not got:
                time.sleep(0.5)


if __name__ == "__main__":
    WID = sys.argv[1] if len(sys.argv) > 1 else f"w{os.getpid()}"
    os.makedirs(RUNS, exist_ok=True)
    wk = Worker()
    wk.feed_client()
    wk.options()                                                        # 10-26: every worker start rewrites SCHEMA/OPTIONS (one mono worker now; engine knobs change)
    wk.loop()
