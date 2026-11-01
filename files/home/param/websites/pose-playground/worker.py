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
import base64
import gc
import fcntl
import glob
import hashlib
import json
import os
CANOPY_ROOT = os.environ.get("CANOPY_ROOT") or os.path.expanduser("~")   # cluster home holding ppm/, venvs/, models/ (see README: paths)
HOME_DIR = os.path.expanduser("~")
import re
import sys
import threading
import time
import traceback
from collections import OrderedDict

MI = os.environ.get("GENWEB_MONO_ENGINE", CANOPY_ROOT + "/ppm/aloe-pose-monoplex-infer")
sys.path.insert(0, MI)
sys.path.insert(0, f"{MI}/ppm")
os.environ.setdefault("PPM_CD", "2")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import infer as MIF  # noqa: E402
import fast_ppm as FP  # noqa: E402
import ppm_io as IO  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

G = os.path.dirname(os.path.abspath(__file__))
LIVE = os.environ.get("GENWEB_LIVE", CANOPY_ROOT + "/ppm/aloe-posedup-live")
RUNS, RJOBS, ROUT = f"{G}/runs", f"{G}/render/jobs", f"{G}/render/out"
RJOBS_V = f"{G}/render/jobs_vansh"   # 10-26 user: Vansh-weights runs render from their own queue (videort's Vansh renderers)
RJOBS_V50 = f"{G}/render/jobs_vansh50"   # 10-26 user: Vansh at 20 steps is the default (jobs_vansh); 50 steps on request (vansh_steps 50)
CK = os.environ.get("GENWEB_CK", CANOPY_ROOT + "/ppm/aloe-pose-mono-8020/exports/export-357")
# 10-27 user ("put up rosie"): a worker serves ONE set of monoplex weights; runs pick theirs with request "mono_weights" (base | rosie)
WEIGHTS = os.environ.get("GENWEB_WEIGHTS", "base")
_UW = f"{G}/UNIT_WEIGHTS/{os.environ.get('SLURM_JOB_ID', '')}.txt"    # 10-28: a unit may set its worker's weights id (time-share one pair)
if os.environ.get("SLURM_JOB_ID") and os.path.isfile(_UW):
    WEIGHTS = open(_UW).read().strip() or WEIGHTS
WEIGHTS_LABEL = {"base": "export-357", "rosie": "Rosie FT (aloe-pose-rosie export-18)"}
# 10-27 main (user: "put the 32b gen model in the playground instead of the monoplex", picked (B)): a BASE worker serves the checkpoint named
# in MONO_CK.json {ck, label, short, prompt} (stage-v2pure/swap.sh writes it; restore = swap.sh 357 moves it away). Absent -> export-357 with
# its own prompt, byte-for-byte as before. Same engine + sampling knobs either way.
BASE_LABEL, BASE_SHORT, BASE_PROMPT = "aloe-pose-mono-8020 export-357", "", "mono8020"
_ov = {}
if WEIGHTS == "base" and os.path.isfile(f"{G}/MONO_CK.json"):
    _ov = json.load(open(f"{G}/MONO_CK.json"))
    CK, BASE_LABEL, BASE_SHORT, BASE_PROMPT = _ov["ck"], _ov["label"], _ov.get("short", ""), _ov.get("prompt", "mono8020")
    WEIGHTS_LABEL["base"] = BASE_LABEL
# 10-28 main (user: "set up backend server for both when they are ready"): more monoplex models, one weights id each, configured in
# genweb/weights/<id>.json {ck, label, short, prompt, openings, voice}: openings = the prefill-data root under genweb (openings | openings-sam),
# voice = own (the actor package's gt/mimi_local, first 10 s: today's monoplex) | opening_clip (<openings>/<voice>/voice_clip.safetensors)
OPEN_DIR, W_VOICE, W_SHORT, W_PERSONS = "openings", "own", "", None
_WCFG = {}
if WEIGHTS not in ("base", "rosie") and os.path.isfile(f"{G}/weights/{WEIGHTS}.json"):
    _wc = json.load(open(f"{G}/weights/{WEIGHTS}.json"))
    _WCFG = _wc
    CK, BASE_PROMPT, W_SHORT = _wc["ck"], _wc.get("prompt", "mono8020"), _wc.get("short", "")
    WEIGHTS_LABEL[WEIGHTS] = _wc["label"]
    OPEN_DIR, W_VOICE, W_PERSONS = _wc.get("openings", "openings"), _wc.get("voice", "own"), _wc.get("persons")
SAM_ONLY = OPEN_DIR != "openings"                                       # 10-28 audit: SAM-code weights never fall back to raw codes
PEOPLE = [os.environ.get("GENWEB_PEOPLE", CANOPY_ROOT + "/video/demo-data/prompts"),
          os.environ.get("GENWEB_ACTORS", CANOPY_ROOT + "/video/dmd-rt/actors/pkg"),
          os.path.dirname(os.path.abspath(__file__)) + "/people_extra"]   # 10-27: speakers with real data but no avatar (Rosie)
CD, OFF, PAD, DEPTH, SPF = FP.CD, FP.OFF, FP.PAD, FP.DEPTH, 1920
MAX_CLIP_S, MAX_PF_S = 120.0, 11.68
MAX_BATCH = 16
# defaults = posefix's MONO_PARAMS: the fantastic set as far as this engine has the knobs; pose T1 / top-k 30 (cb0 + light); CFG off
DEF = dict(FP.DEFAULTS, **{k: v for k, v in MIF.TONIGHT.items() if k in FP.PARAMS})
DEF.update(text_temp=1.0, text_topk=50, text_topp=1.0, min_p=0.05, min_p_rel=0, freq_pen=0.25, freq_win=50, cb0_temp=1.0, cb0_topk=30,
           light_temp=1.0, light_topk=30, pose_temp=1.0, pose_topk=30, pose_light_temp=1.0, pose_light_topk=151, pose_cfg_on=0)   # 10-25: pose light k151 (user's call setting, posefix's fix)
DEF.update({k: v for k, v in _ov.get("params", {}).items() if k in FP.PARAMS})
LABEL = {"text_temp": "text temperature", "text_topk": "text top-k", "text_topp": "text top-p", "min_p": "word min-p", "min_p_rel": "min-p relative",
         "freq_pen": "frequency penalty", "freq_win": "frequency window", "text_rep": "text repetition penalty", "text_rep_win": "text rep window",
         "cb0_temp": "speech cb0 temperature", "cb0_topk": "speech cb0 top-k", "light_temp": "speech cb1-31 temperature", "light_topk": "speech cb1-31 top-k",
         "audio_rep_pen": "speech repetition penalty", "audio_rep_win": "speech rep window", "pose_temp": "pose cb0 temperature", "pose_topk": "pose cb0 top-k",
         "pose_light_temp": "pose cb1-31 temperature", "pose_light_topk": "pose cb1-31 top-k", "pose_rep_cb0": "pose cb0 rep-pen", "pose_rep_light": "pose light rep-pen",
         "pose_rep_cb0_speaking": "pose cb0 rep-pen speaking", "pose_rep_cb0_silent": "pose cb0 rep-pen silent",
         "pose_rep_light_speaking": "pose light rep-pen speaking", "pose_rep_light_silent": "pose light rep-pen silent",   # 10-26 user: cb0 / light split (posefix)
         "pose_rep_win": "pose rep window (0 = all)", "pose_cfg_on": "pose CFG on", "pose_cfg_s": "pose CFG strength", "pose_cfg_cb0": "pose CFG on cb0",
         "pose_cfg_light": "pose CFG on cb1-31", "pose_cfg_speaking_only": "pose CFG only while speaking"}
GROUP = {"text": "Text sampling", "audio": "Speech sampling", "pose": "Pose sampling", "pose CFG": "Pose CFG",
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


def opening_frame0_gs(person, root=None, pdir=None):
    """10-26 posefix (for main): an opening packaged from a recording WINDOW (meta sources.pose_tokens.frame0.fields) was tokenized against
    that window's frame 0, not the avatar package's -> [root_y, root_x, width] in pose.global_state's units | None (the opening IS the package).
    Same as LIVE/server.py opening_frame0_gs; the feed 'start' carries it as frame0_global_state and the avatar engine decodes against it."""
    try:
        f = json.load(open(f"{pdir or f'{G}/{root or OPEN_DIR}/{person}'}/meta.json"))["sources"]["pose_tokens"]["frame0"]["fields"]
        return [float(f["frame0_root"][0]), float(f["frame0_root"][1]), float(f["frame0_width"])]
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return None


def persons():
    out = []
    for r in PEOPLE[:1]:
        for d in sorted(os.listdir(r)):
            if os.path.isfile(f"{r}/{d}/gt/pose_tokens.safetensors") and os.path.isfile(f"{r}/{d}/gt/mimi_local.safetensors"):
                try:
                    label = json.load(open(f"{r}/{d}/meta.json")).get("label")
                except (OSError, ValueError, AttributeError):
                    label = None
                out.append({"id": d, "label": label or (f"Person {d[:2]}" if d[:2].isdigit() else d), "max_prefill_s": MAX_PF_S})
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
         "hint": "sampling seed (0 = random). Multiple samples share this batch seed and use distinct engine rows; replay the same batch to reproduce them."},
        {"name": "n_samples", "label": "samples per Generate", "group": "Run", "type": "int", "default": 1, "range": [1, 16], "step": 1,
         "hint": "samples generated together in one CUDA-graph batch on the resident model"},
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
        {"name": "voice_clip", "label": "voice clip prompt", "group": "Prompt", "type": "bool", "default": False, "range": None, "step": None,
         "hint": "10-27 user: off (default) = NO voice clip in the system prompt, built exactly like a training row whose clip draw was false; on = the voice clip below, as before"},
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
    for row in rows:
        if row["name"] in _ov.get("run_defaults", {}):
            row["default"] = _ov["run_defaults"][row["name"]]
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
    def start_health(self):
        self.active_run = None

        def heartbeat():
            while True:
                jwrite(f"{G}/mono_ready.json", {"worker": WID, "model": "monoplex", "weights": WEIGHTS,
                                                "checkpoint": CK, "step": self.step, "pid": os.getpid(),
                                                "ready": True, "active_run": self.active_run, "time": time.time(),
                                                "batch_capacity": MAX_BATCH, "engine": "Amu FastPPM CUDA graphs",
                                                "graph_cache": [list(k) for k in list(self.engines)]})
                time.sleep(5)

        threading.Thread(target=heartbeat, name="worker-health", daemon=True).start()

    def __init__(self):
        devs = [torch.device("cuda:0"), torch.device("cuda:1")]
        torch.cuda.set_device(devs[-1])
        self.pp = persons()
        with torch.inference_mode():
            if os.path.isfile(f"{G}/FASTLOAD") or os.path.isfile(f"{G}/FASTLOAD.{WID}"):   # 10-28: opt-in parallel loader (sha-manifested, same bytes)
                os.environ["GENWEB_FASTLOAD"] = "1"
                if os.path.isfile(f"{G}/FASTLOAD_VERIFY"):
                    os.environ["GENWEB_FASTLOAD_VERIFY"] = "1"
            m, ldev, cfgs, binfo = FP.build_model(CK, devs)
            assert binfo["pose"] and binfo["decp_bb"], binfo
            self.step = binfo.get("step")
            self.mimi = IO.Mimi(devs[-1])
            self.max_s = MAX_CLIP_S + MAX_PF_S + 2
            nfr = int(round(self.max_s * 12.5)) + 8
            sil = self.mimi.encode(np.zeros((2048 + nfr) * SPF, np.float32))[:, :2048 + nfr]
            self.model_args = (m, ldev, cfgs, devs)
            self.silence = sil
            self.engines = OrderedDict()
            self.eng = None
            # The unchanged engine and Amu's persistent NI-row server pattern:
            # weights stay resident; each static graph shape is captured once.
            self.get_engine(1, self.max_s, 2048)
            self.get_engine(4, 32.0, 512)
            self.get_engine(16, 32.0, 512)
        log(f"model step {self.step} loaded + captured ({CK})")

    def get_engine(self, rows, seconds, prefix):
        key = (int(rows), float(seconds), int(prefix))
        if key in self.engines:
            self.engines.move_to_end(key)
            self.eng = self.engines[key]
            return self.eng, True
        m, ldev, cfgs, devs = self.model_args
        c32, c4, _ = cfgs
        length = int(prefix) + int(round(seconds * FP.FPS)) + 8 + CD + 8
        # Bound graph-cache residency using the new shape's actual KV layout,
        # plus room for capture workspaces. Eviction never reloads model weights.
        need = {}
        for d in devs:
            bb = sum(x == d for x in ldev) * 2 * rows * c32.num_key_value_heads * c32.head_dim
            tw = ((2 * rows * len(m.tower_p.layers) if d == devs[0] else 0)
                  + (rows * len(m.tower_s.layers) if d == devs[-1] else 0)) * c4.num_key_value_heads * c4.head_dim
            need[d] = 4 * length * (bb + tw) + (6 << 30)
        while self.engines:
            for d in devs:
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()
            if all(torch.cuda.mem_get_info(d)[0] >= need[d] for d in devs):
                break
            old, evicted = self.engines.popitem(last=False)
            if self.eng is evicted:
                self.eng = None
            del evicted
            gc.collect()
            log(f"released cached graph shape {old}; model weights remain resident")
        for d in devs:
            with torch.cuda.device(d):
                torch.cuda.empty_cache()
            if torch.cuda.mem_get_info(d)[0] < need[d] - (4 << 30):
                raise ValueError("This batch and prompt length exceed available GPU memory; shorten the clip/prompt or request fewer samples")
        log(f"capturing graph shape rows={rows}, seconds={seconds}, prefix={prefix}")
        eng = FP.FastPPM(m, ldev, cfgs, devs, n_inst=rows, cfg_rows=True, max_s=seconds,
                         p_max=prefix, mimi_stream=False, silence=self.silence)
        eng.capture()
        self.engines[key] = self.eng = eng
        return eng, False

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
        self._FD, self.rfeeds = FD, {}                                  # 10-27 overlap render: Feeds to videort's eager-decode instance R

    def rec_pool_claim(self, rid):
        """10-28 main: the record instances are ONE shared pool (genweb/RECORD_POOL.txt, one ws URL per line): a run claims any FREE instance
        (lock genweb/rpool/<host_port>.lock holding the run id; free = no lock, or its run's mp4 exists, or older than 30 min) -> url | None"""
        try:
            urls = [x.split()[0] for x in open(f"{G}/RECORD_POOL.txt").read().splitlines() if x.strip() and not x.startswith("#")]
        except (OSError, IndexError):
            return None
        os.makedirs(f"{G}/rpool", exist_ok=True)
        for u in urls:
            lk = f"{G}/rpool/" + re.sub(r"[^A-Za-z0-9]+", "_", u.split("://")[-1]).strip("_") + ".lock"
            for _ in range(2):
                try:
                    fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, rid.encode()); os.close(fd)
                    return u, lk
                except FileExistsError:
                    try:
                        old = open(lk).read().strip()
                        stale = os.path.exists(f"{ROUT}/{old}.mp4") or time.time() - os.path.getmtime(lk) > 1800
                    except OSError:
                        continue                                        # vanished meanwhile: try again
                    if not stale:
                        break
                    tmp = f"{lk}.{rid}.stale"
                    try:
                        os.replace(lk, tmp)
                        if open(tmp).read().strip() != old:             # someone re-claimed in between: put it back, move on
                            os.replace(tmp, lk); break
                        os.remove(tmp)
                    except OSError:
                        break
        return None

    def rec_pool_release(self, rid, lk):
        try:
            if lk and open(lk).read().strip() == rid:
                os.remove(lk)
        except OSError:
            pass

    def rec_feed(self, rid=None):
        """10-27 (main GO, gated on videort's paced = all-at-once proofs): OVERLAP RENDER. genweb/RECORD_FEED_URL.txt (read per run; absent /
        off = today's queue path) -> a Feed to instance R, which renders each latent as its codes arrive and writes render/out/<rid>.mp4"""
        self.rec_lock = None
        if rid and os.path.isfile(f"{G}/RECORD_POOL.txt"):             # 10-28 main: shared pool, any free instance; none free -> queue lanes
            got = self.rec_pool_claim(rid)
            if not got:
                log(f"{rid}: no free record instance -> queue render")
                return None
            u, self.rec_lock = got
            if u not in self.rfeeds:
                self.rfeeds[u] = self._FD.Feed(u, log, qmax=8192, **({"hold": True} if "hold" in self._FD.Feed.__init__.__code__.co_varnames else {}))
                log(f"record feed (overlap render) -> {u}")
            log(f"{rid}: record instance {u}")
            return self.rfeeds[u]
        try:                                                            # 10-28: per-weights record instance first (models render in parallel)
            f = next((x for x in (f"{G}/RECORD_FEED_URL.{WID}.txt", f"{G}/RECORD_FEED_URL.{WEIGHTS}.txt") if os.path.isfile(x)), f"{G}/RECORD_FEED_URL.txt")   # worker > weights > default
            u = open(f).read().split()[0]
        except (OSError, IndexError):
            return None
        if not u or u.lower() in ("off", "none", "0"):
            return None
        if u not in self.rfeeds:
            self.rfeeds[u] = self._FD.Feed(u, log, qmax=8192, **({"hold": True} if "hold" in self._FD.Feed.__init__.__code__.co_varnames else {}))
            log(f"record feed (overlap render) -> {u}")
        return self.rfeeds[u]

    def rec_fallback(self, rid, rd, jd, n, vpf, person, t_end, lk=None):
        """R's mp4 must appear soon after the end (R may lag a few s, or hold while a live call pauses it); else queue the job as today"""
        mp4, rs = f"{ROUT}/{rid}.mp4", f"{ROUT}/{rid}.json"
        lim, part = max(30.0, 0.5 * n * 0.08), f"{ROUT}/{rid}.part.mp4"

        def growing():                                                  # R still writing this run's mp4 (it may lag the gen / share its GPU)
            try:
                return time.time() - os.path.getmtime(part) < 15
            except OSError:
                return False
        queued = lambda: os.path.exists(f"{ROUT}/{rid}.queued") or os.path.exists(f"{ROUT}/{rid}.mp4.queued")   # R serialises streams (oldest first)
        while time.time() - t_end < lim or os.path.exists(f"{G}/render/PAUSE_LIVE_S4") or growing() or queued():
            if os.path.exists(mp4):
                jwrite(rs, {"chunks": None, "seconds": round(n * 0.08, 2), "prefill_s": float(vpf), "person": person, "state": "done",
                            "render_s": None, "after_gen_s": round(time.time() - t_end, 2), "engine": "overlap (R, eager decode)", "at": time.time()})
                log(f"{rid}: overlap mp4 {round(time.time() - t_end, 2)} s after the gen")
                self.rec_pool_release(rid, lk)
                return
            if time.time() - t_end > 1800:
                break
            time.sleep(0.25)
        time.sleep(1.0)                                                 # 10-28 fix: R renames part -> mp4 at close; re-check before queuing a duplicate
        if os.path.exists(mp4):
            jwrite(rs, {"chunks": None, "seconds": round(n * 0.08, 2), "prefill_s": float(vpf), "person": person, "state": "done",
                        "render_s": None, "after_gen_s": round(time.time() - t_end, 2), "engine": "overlap (R, eager decode)", "at": time.time()})
            log(f"{rid}: overlap mp4 {round(time.time() - t_end, 2)} s after the gen (close race)")
            self.rec_pool_release(rid, lk)
            return
        log(f"{rid}: no overlap mp4 after {round(time.time() - t_end, 1)} s -> queued as today")
        self.rec_pool_release(rid, lk)
        jwrite(rs, {"state": "queued", "note": "overlap render missing; queued", "at": time.time()})
        with open(f"{rd}/feed.jsonl", "rb") as fi, open(f"{jd}/{rid}.feed.jsonl.tmp", "wb") as fo:
            fo.write(fi.read())
        os.replace(f"{jd}/{rid}.feed.jsonl.tmp", f"{jd}/{rid}.feed.jsonl")

    def stream(self, rid, P, f, sent, final=False, dests=None):
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
            ln = json.dumps({"type": "f", "session": rid, "f": k, "t": round(k * 0.08, 3), "pose": [int(v) for v in p[k]],
                             "mimi": [int(v) for v in c[k]], "pcm_b64": base64.b64encode(pcm).decode("ascii")})
            for d in (dests if dests is not None else [self.feed]):      # 10-27: the live front and / or the overlap render instance R
                d.put("f", rid, ln)
        return K

    def options(self):
        if WEIGHTS != "base":                                           # 10-27: only base workers write the shared schema / options
            if WEIGHTS != "rosie":                                      # 10-28: other monoplex weights announce themselves (the page's picker)
                os.makedirs(f"{G}/weights_live", exist_ok=True)
                jwrite(f"{G}/weights_live/{WEIGHTS}.json", {"id": WEIGHTS, "label": WEIGHTS_LABEL.get(WEIGHTS, WEIGHTS), "short": W_SHORT,
                                                            "worker": WID, "step": self.step, "t": time.time(),
                                                            **({"persons": W_PERSONS} if W_PERSONS else {}),
                                                            **{k: v for k, v in _WCFG.items() if k in ("note", "orig", "person_notes")}})   # 10-28: page extras from the config
            return
        rows = schema_rows(self.pp)
        jwrite(f"{G}/SCHEMA.json", rows)
        o = {"models": ["monoplex"], "model_label": f"pose monoplex ({BASE_LABEL}, step {self.step})", "model_short": BASE_SHORT,
             "persons": self.pp, "defaults": {r["name"]: r["default"] for r in rows}, "max_clip_s": MAX_CLIP_S,
             "max_prefill_s": MAX_PF_S, "written_by": WID, "t": time.time(),
             "monoplex_batch": {"max_samples": MAX_BATCH, "execution": "CUDA-graph instance rows",
                                "seed_mode": "batch_seed_and_row", "engine_source": "HF355 code/infer/ppm/fast_ppm.py"}}
        os.makedirs(f"{G}/state", exist_ok=True)
        with open(f"{G}/state/options.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:                                                        # keep the duplex worker's keys while serializing both writers
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

    def run_parts(self, rd, req):
        rid = req["run_id"]
        st = {"state": "running", "progress": 0.0, "msg": "building the prompt", "worker": WID, "t_start": time.time()}
        jwrite(f"{rd}/status.json", st)
        pr = dict(req.get("params") or {})
        po = bool(pr.pop("pose_only", False))                          # 10-27 user: pose only (real speech, stick figure); not an engine knob
        vcl = pr.pop("voice_clip", None)                               # 10-27 user: voice clip prompt toggle (page: prompt.voice_clip); not an engine knob
        vcl = (req.get("prompt") or {}).get("voice_clip", req.get("voice_clip", vcl))
        vc_on = True if vcl is None else vcl in (True, 1, "1", "true")   # absent (requests from before the toggle) = on, exactly as before
        for old, new in (("pose_rep", ("pose_rep_cb0", "pose_rep_light")), ("pose_rep_speaking", ("pose_rep_cb0_speaking", "pose_rep_light_speaking")),
                         ("pose_rep_silent", ("pose_rep_cb0_silent", "pose_rep_light_silent"))):   # 10-26: pre-split pose rep-pen keys set BOTH new keys
            if old in pr and old not in FP.PARAMS and all(n in FP.PARAMS for n in new):            # (new keys win; = fast_ppm.validate's alias)
                v = pr.pop(old)
                for n in new:
                    pr.setdefault(n, v)
        for k in FP.GRAMMAR_GONE:                                      # 10-28 USER ORDER: no lane grammar, ever: legacy keys (old links /
            pr.pop(k, None)                                            # saved settings / presets) dropped silently, never an error
        bad = [k for k in pr if k not in FP.PARAMS]
        if bad:
            raise ValueError(f"unknown engine knobs: {bad}")
        params = dict(DEF, **pr)
        for k, (t, lo, hi) in FP.PARAMS.items():
            params[k] = t(min(max(t(params[k]), lo), hi))
        seed = int(req.get("seed") or 0) or int.from_bytes(os.urandom(3), "little") + 1
        clip_s = min(max(float(req.get("clip_s") or 20.0), 1.0), MAX_CLIP_S)
        pf = req.get("prefill") or {}
        person = pf.get("person") or _ov.get("run_defaults", {}).get("prefill_person", "01_ab63d745")
        noav = no_avatar(person)                                       # 10-27: a person with no avatar (Rosie): stick figure, no GT / live avatar
        if req.get("gt") is True and not noav:                        # 10-26: GT render only when the user shows GT
            self.gt_render(only={person})
        pfs = min(max(float(pf.get("s") or 0.0), 0.0), MAX_PF_S)
        prm = req.get("prompt") or {}
        voice, persona = prm.get("voice"), (prm.get("persona") or "").strip() or None
        od, wv = OPEN_DIR, W_VOICE                                      # 10-28 main: HIDDEN isolation runs may swap the input set (2x2: model x opening)
        pcodes = req.get("prefill_codes") if req.get("prefill_codes") in ("raw", "sam") else None   # 10-27 picker (user): the prefill's code type
        if pcodes:                                                      # raw = openings / the actor clip, sam = openings-sam (+ its voice clip); a data
            od, wv = ("openings", "own") if pcodes == "raw" else ("openings-sam", "opening_clip")   # source only; absent = the model's own (as before)
            log(f"{rid}: prefill codes {pcodes}: openings={od} voice={wv}")
        dbg = req.get("debug_inputs") if req.get("hidden") is True else None
        if isinstance(dbg, dict) and dbg.get("openings") in ("openings", "openings-sam"):
            od, wv = dbg["openings"], dbg.get("voice") or ("own" if dbg["openings"] == "openings" else "opening_clip")
            log(f"{rid}: hidden debug inputs: openings={od} voice={wv}")
        if not vc_on:                                                   # 10-27 user: voice clip prompt off -> no clip (MIF.system_block(None, persona) =
            clip, vsrc = None, "no voice clip (voice clip prompt off)"   # the trainers' no-clip row: empty block, or persona only)
        elif voice in (None, "", "none"):
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
            rc = f"{G}/v2pure-clips/{vp}.safetensors"
            if BASE_PROMPT == "v2pure" and WEIGHTS == "base" and pcodes != "sam" and os.path.isfile(rc):   # 10-27 main ("match what it was trained on"): the v2pure
                clip = np.asarray(load_file(rc)["codes"], np.int64)                   # training-rule clip (later same-speaker clip, 2nd half of the
                vsrc = f"{vp} v2pure rule clip ({clip.shape[1] / 12.5:.1f} s)"          # recording, 3-10 s, >= 50% speech; genweb/v2pure-clips)
            elif wv == "opening_clip" and not os.path.isfile(f"{G}/{od}/{vp}/voice_clip.safetensors"):
                raise ValueError(f"no {od} voice clip for {vp}: pick an actor with SAM data or upload a clip")
            elif wv == "opening_clip":                             # 10-28 per-weights clip (SAM: openings-sam voice_clip = the SAM store's codes)
                clip = np.asarray(load_file(f"{G}/{od}/{vp}/voice_clip.safetensors")["codes"], np.int64)
                vsrc = f"{vp} {od} voice clip ({clip.shape[1] / 12.5:.1f} s)"
            else:
                clip = np.asarray(load_file(person_file(vp, "gt/mimi_local.safetensors"))["codes"], np.int64)[:, :125]
                vsrc = f"{vp} own voice (10 s)"
        if BASE_PROMPT == "v2pure" and WEIGHTS == "base" and persona:       # 10-27: v2pure never trained with a persona block -> dropped
            log(f"{rid}: persona text dropped (v2pure was never trained with a persona)"); persona = None
        sp = MIF.system_block(clip, persona)
        sysb = {"P": int(sp["P"]), "sha1": hashlib.sha1(b"".join(np.ascontiguousarray(sp[k]).tobytes() for k in ("ids", "depth", "present"))).hexdigest()[:16]}   # 10-27: read-only digest of the system block
        # 10-25 FULL PREFILL (user via main / posefix): for the first N frames (N = 2 x round(s / 0.16) + 2, = the real-video prefill)
        # force any of B's three streams to the actor's own: pose (gt/pose_tokens), audio (gt/mimi_local = lufs23-local) and the word lane
        # (infer.lane_from_words over the actor's timed words; lane frames 0..N+1, codes / pose frames 0..N-1). Persons without timed words:
        # the lane is PAD on 0..N+1 when audio is forced (off-distribution, said in the run summary).
        use_pose, use_audio, use_words = bool(pf.get("pose", True)), bool(pf.get("audio", True)), bool(pf.get("words", True))
        pp = pa = lane_f = None
        words_used = False
        N = 0
        notes = []                                                      # (set before the source pick below)
        pf_fb = False                                                   # 10-27 picker: conversation start asked for, actor clip used
        src = pf.get("src") or "start"
        odir = f"{G}/{od}/{person}"                                    # 10-28: the weights' own prefill-data root (SAM models: openings-sam)
        if src == "start_orig":                                         # 10-28 main: a re-cut opening's ORIGINAL window stays selectable ("<actor> (orig)")
            if os.path.isfile(f"{odir}/orig/pose_tokens.safetensors"):
                odir = f"{odir}/orig"
                notes.append("prefill from the original opening window (before the 10-28 re-cut)")
            src = "start"
        if od != "openings":                                                    # 10-28 audit: SAM weights = SAM codes only (training stores); no raw fallback
            if not all(os.path.isfile(f"{odir}/{x}") for x in ("pose_tokens.safetensors", "mimi_local.safetensors", "meta.json")):
                raise ValueError(f"{person} has no SAM data on disk: this model only runs on actors "
                                 + ", ".join(sorted(os.path.basename(d) for d in glob.glob(f"{G}/{od}/[0-9][0-9]_*"))))
            if src != "start":
                notes.append("SAM model: prefill from the SAM-isolated opening (the actor clip has raw codes only)")
                src = "start"
        if src == "start" and not (os.path.isfile(f"{odir}/pose_tokens.safetensors") and os.path.isfile(f"{odir}/mimi_local.safetensors")):
            notes.append("no conversation-start data on disk for this actor: prefill from the actor clip")
            src = "clip"
            pf_fb = True
        vpf = pfs                                                      # the avatar's REAL-video prefill: only when that video is the same moment
        if src == "start":
            try:
                ov = (json.load(open(f"{G}/{od}/SUMMARY.json")).get(person) or {}).get("video")   # 10-28: the prefill data's own root
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
        f0gs = opening_frame0_gs(person, od, pdir=odir) if (src == "start" and pp is not None and N > 0) else None   # 10-26 posefix: the opening's frame 0
        n = N + int(round(clip_s * 12.5))
        NA = N                                                          # frames whose audio / lane are forced (prefill only, unless pose only)
        sgs = None
        if po:                                                          # 10-27 POSE ONLY: real audio + timed words every frame, pose generated
            notes.clear()                                               # (the avatar-prefill notes above do not apply: no avatar video)
            proot = next((r for r in (f"{G}/{od}/{person}",)       # 10-28 audit: the weights' own code type only (raw models: raw)
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
        # 10-27 picker: the page's status line, from what this run actually used (the page computes the same words before Generate)
        _fs = lambda s_: f"{s_:.2f}".rstrip("0").rstrip(".") + " s"
        _lab = lambda p_: next((x_["label"] for x_ in self.pp if x_["id"] == p_), p_)
        if po:
            pfl = "pose only (" + ("SAM opening" if od != "openings" else ("conversation start" if proot else "actor clip")) + ")"
        elif N == 0:
            pfl = "none"
        elif src == "start":
            pfl = "conversation start" + (", original window" if odir.endswith("/orig") else "") + f" ({_fs(pfs)})"
        else:
            pfl = "actor clip (no conversation start for this actor)" if pf_fb else f"actor clip ({_fs(pfs)})"
        vl = ("off" if not vc_on else "none" if voice in (None, "", "none") else "on (upload)" if str(voice).startswith("upload:")
              else f"on ({_lab(person if voice == 'same' else str(voice))})")
        pline = f"{_lab(person)} · prefill: {pfl} · codes: {'SAM-isolated Mimi' if od != 'openings' else 'raw Mimi'} · voice clip: {vl}"
        lane = np.full(n + CD, PAD, np.int64)
        x = MIF.assemble(sp, lane, np.zeros((DEPTH, 0), np.int64), np.zeros((DEPTH, 0), np.int64), n, 0)
        P = int(x["P"])
        lane_ids = None
        if lane_f is not None:                                         # the forced lane in the engine's own ids (assemble's placement)
            lw = np.full(n + CD, PAD, np.int64)
            lw[:NA + CD] = lane_f[:NA + CD]
            lane_ids = np.asarray(MIF.assemble(sp, lw, np.zeros((DEPTH, 0), np.int64), np.zeros((DEPTH, 0), np.int64), n, 0)["ids"][1], np.int64)
        # Preparation and file publication retain the original playground path.
        # The queue adapter between them runs all samples through Amu's NI-row
        # FastPPM.start/step/readback API, without changing any inference kernels.
        result = yield {"rd": rd, "rid": rid, "req": req, "st": st, "params": params, "seed": seed,
                        "x": x, "P": P, "n": n, "N": N, "NA": NA, "pp": pp, "pa": pa, "lane_ids": lane_ids}
        S, wall, execution = result["S"], result["wall"], result["execution"]
        requested_seed, seed = int(req.get("seed") or 0), execution["batch_seed"]
        rfeed = None
        lane_b, dep_b, pdep = S["lane_b"], S["dep_b"], S["pdep"]
        c1, p1 = dep_b[P + CD:P + CD + n] - OFF, pdep[P + CD:P + CD + n] - OFF
        assert c1.min() >= 0 and c1.max() < FP.CBS and p1.min() >= 0 and p1.max() < FP.CBS, "invalid or missing codes"
        if pp is not None:
            assert (p1[:N] == pp).all(), "pose prefill != actor tokens"
        if pa is not None:
            assert (c1[:NA] == pa[:NA]).all(), "forced audio != actor codes"
        with torch.inference_mode():
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
                fl = json.load(open(f"{odir}/meta.json"))["sources"]["pose_tokens"]["frame0"]["fields"] if src == "start" else None
                sgs = [float(fl["frame0_root"][0]), float(fl["frame0_root"][1]), float(fl["frame0_width"])] if fl else None
            except (OSError, ValueError, KeyError, TypeError):
                sgs = None
            if sgs is None:
                sgs = [float(v) for v in np.asarray(load_file(person_file(person, "gt/pose_tokens.safetensors"))["global_state"], np.float32).reshape(-1)[:3]]
        if po or noav:                                                  # 10-27 pose only / no-avatar person: a stick figure instead of the avatar render
            import stick
            stick.stick_mp4(p1, N, sgs, f"{rd}/b.wav", f"{rd}/stick.mp4")
        else:
            if not rfeed:
                self.write_feed(f"{rd}/feed.jsonl", rid, person, vpf, p1, c1, w, n, f0gs=f0gs)
            jd = (RJOBS_V if req.get("vansh_steps") == 20 else RJOBS_V50) if req.get("video_weights") == "vansh" else RJOBS   # no field = 50 (old)
            os.makedirs(jd, exist_ok=True)
            if rfeed:                                                   # 10-27 overlap: queue only if R's mp4 does not show up
                threading.Thread(target=self.rec_fallback, args=(rid, rd, jd, n, vpf, person, t_end, getattr(self, "rec_lock", None)), daemon=True).start()
            else:
                with open(f"{rd}/feed.jsonl", "rb") as fi, open(f"{jd}/{rid}.feed.jsonl.tmp", "wb") as fo:
                    fo.write(fi.read())
                os.replace(f"{jd}/{rid}.feed.jsonl.tmp", f"{jd}/{rid}.feed.jsonl")
        gen = p1[N:, 0].astype(int)
        chg = float((gen[1:] != gen[:-1]).mean()) if len(gen) > 1 else None
        text = " ".join(q["w"] for q in wl)
        summ = {"run_id": rid, "model": f"monoplex {WEIGHTS_LABEL.get(WEIGHTS, WEIGHTS)} (step {self.step})", "mono_weights": WEIGHTS, "person": person, "seed": seed, "P": P, "frames": n,
                "pose_prefill_frames": N if pp is not None else 0, "prefill_frames": N, "prefill_s": pfs,
                "prefill_streams": [k for k, v in (("pose", pp is not None), ("audio", pa is not None), ("words", words_used)) if v],
                "prefill_notes": notes, "prefill_source": src, "pose_only": po, "forced_audio_frames": NA if pa is not None else 0, "frame0_global_state": f0gs, "clip_s": clip_s, "voice": vsrc, "voice_clip": vc_on,
                "prefill_codes": "sam" if od != "openings" else "raw", "prefill_codes_choice": pcodes or "model-default", "prefill_line": pline, "sys_block": sysb, "persona": persona, "params": params,
                "wall_s": round(wall, 1), "ms_per_frame": round(1e3 * wall / (n + CD), 1), "gen_cb0_change": chg,
                "cfg_guided_frames": result["cfg_guided_frames"], "execution": execution, "requested_seed": requested_seed,
                "cfg_frames_total": n,
                "gen_cb0_distinct": int(len(np.unique(gen))) if len(gen) else 0, "n_words": len(wl), "words": text[:600]}
        jwrite(f"{rd}/summary.json", summ)
        head = (f"{len(wl)} words · pose cb0 changes on {100 * chg:.0f}% of generated frames · {summ['ms_per_frame']} ms/frame · seed {seed}"
                if chg is not None else f"{len(wl)} words · seed {seed}")
        jwrite(f"{rd}/status.json", dict(st, state="done", progress=1.0, msg=("pose only: stick figure ready (real speech, generated pose)" if po else ("stick figure ready (no avatar of this person)" if noav else "generated; avatar rendering")), headline=head, t_end=time.time(),
                                         ms_per_frame=summ["ms_per_frame"], frames_done=n + CD, frames_total=n + CD, seed=seed, execution=execution))
        log(f"run {rid}: {head}")

    @staticmethod
    def batch_signature(req):
        return json.dumps({k: req.get(k) for k in ("model", "mono_weights", "params", "clip_s", "prefill", "prompt",
                                                  "prefill_codes", "debug_inputs", "voice_clip", "hidden")}, sort_keys=True)

    @staticmethod
    def fail_run(rd, exc):
        try:
            state = json.load(open(f"{rd}/status.json"))
        except (OSError, ValueError):
            state = {}
        jwrite(f"{rd}/status.json", dict(state, state="error", error=f"{type(exc).__name__}: {str(exc)[:400]}",
                                        worker=WID, t_end=time.time()))

    def collect_batch(self, rd, req):
        """Wait briefly for the rest of one Generate submission, including older
        tabs that POST samples serially. Partial submissions remain runnable."""
        group = [(rd, req)]
        batch = req.get("batch") or {}
        bid = batch.get("id")
        expected = min(MAX_BATCH, max(1, int(batch.get("n") or 1)))
        if not bid or expected == 1:
            return group
        signature = self.batch_signature(req)
        seen = {rd}
        deadline = time.monotonic() + 3.0
        while len(group) < expected:
            for rid in sorted(os.listdir(RUNS)):
                other = f"{RUNS}/{rid}"
                if other in seen or not os.path.isfile(f"{other}/READY"):
                    continue
                try:
                    rq = json.load(open(f"{other}/request.json"))
                    if (rq.get("batch") or {}).get("id") != bid or self.batch_signature(rq) != signature:
                        continue
                    os.rename(f"{other}/READY", f"{other}/CLAIMED.{WID}")
                except (OSError, ValueError):
                    continue
                seen.add(other)
                group.append((other, rq))
                if len(group) >= expected:
                    break
            if len(group) >= expected or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        return sorted(group, key=lambda x: int((x[1].get("batch") or {}).get("k") or 0))

    def run_batch(self, jobs):
        """Amu's established persistent batch runner: one start, one graph replay
        per frame for every NI row, then readback each requested sample. The
        packaged HF355 inference engine remains byte-for-byte unchanged."""
        prepared = []
        for rd, req in jobs:
            try:
                if os.path.exists(f"{rd}/CANCEL"):
                    raise RuntimeError("cancelled before generation")
                if req.get("stream") is True:
                    raise ValueError("batch worker uses avatar replay; use Duplex Live mic for streaming")
                parts = self.run_parts(rd, req)
                ctx = next(parts)
                prepared.append((parts, ctx))
            except Exception as exc:
                self.fail_run(rd, exc)
        if not prepared:
            return
        contexts = [x[1] for x in prepared]
        first = contexts[0]
        count = len(contexts)
        capacity = 1 if count == 1 else 4 if count <= 4 else 16
        n, P, params = first["n"], first["P"], first["params"]
        def same_forcing(c):
            return (c["N"] == first["N"] and c["NA"] == first["NA"] and
                    all((c[k] is None and first[k] is None) or
                        (c[k] is not None and first[k] is not None and np.array_equal(c[k], first[k]))
                        for k in ("pp", "pa", "lane_ids")))
        if not all(c["n"] == n and c["P"] == P and c["params"] == params and same_forcing(c) for c in contexts):
            for parts, c in prepared:
                self.fail_run(c["rd"], ValueError("incompatible batch"))
                parts.close()
            return
        seconds = next((s for s in (32.0, 64.0, self.max_s) if n / FP.FPS <= s), self.max_s)
        prefix = max(512, ((P + 511) // 512) * 512)
        if count == 1:
            seconds, prefix = self.max_s, max(2048, prefix)
        batch = first["req"].get("batch") or {}
        # Same semantics as Amu's ppm_worker.py / samples_fast.py: independent
        # stochastic rows in one seeded batch, not seed+k singleton streams.
        raw_seed = int(first["req"].get("seed") or 0)
        seed = (raw_seed - int(batch.get("k") or 0)) if raw_seed and batch else first["seed"]
        if seed <= 0:
            seed = first["seed"]
        cancelled = set()
        try:
            with torch.inference_mode():
                eng, hit = self.get_engine(capacity, seconds, prefix)
                instances = [dict(c["x"], a_mode="empty") for c in contexts]
                instances += [instances[-1]] * (capacity - count)
                executions = []
                for i, c in enumerate(contexts):
                    execution = {"kind": "batched_cuda_graph", "sample_count": count, "engine_rows": capacity,
                                 "batch_id": batch.get("id"), "batch_seed": seed,
                                 "sample_index": int((c["req"].get("batch") or {}).get("k") or 0),
                                 "engine_row": i, "graph_cache_hit": hit, "engine_shape": [capacity, seconds, prefix],
                                 "seed_mode": "batch_seed_and_row", "engine_source": "HF355 code/infer/ppm/fast_ppm.py"}
                    executions.append(execution)
                    c["st"].update(seed=seed, execution=execution, msg=f"prefilling {count} samples together")
                    jwrite(f"{c['rd']}/status.json", c["st"])
                t0 = time.time()
                eng.start(instances, seed, params, f0=0)
                log(f"batch {batch.get('id', first['rid'])}: {count} samples in {capacity} CUDA-graph rows, seed {seed}, cached={hit}")
                for f in range(n + CD):
                    g, force = f - CD, {}
                    if first["pp"] is not None and 0 <= g < first["N"]:
                        force["pose"] = np.repeat((first["pp"][g] + OFF)[None], capacity, axis=0)
                    if first["pa"] is not None and 0 <= g < first["NA"]:
                        force["codes"] = np.repeat((first["pa"][g] + OFF)[None], capacity, axis=0)
                    if first["lane_ids"] is not None and f < first["NA"] + CD:
                        force["lane"] = np.full(capacity, first["lane_ids"][P + f], np.int64)
                    eng.step(f, force=force or None, cold_pad=first["lane_ids"] is None)
                    if f % 25 == 0 or f == n + CD - 1:
                        elapsed = time.time() - t0
                        for i, c in enumerate(contexts):
                            if os.path.exists(f"{c['rd']}/CANCEL"):
                                if i not in cancelled:
                                    self.fail_run(c["rd"], RuntimeError("cancelled"))
                                cancelled.add(i)
                            if i not in cancelled:
                                jwrite(f"{c['rd']}/status.json", dict(c["st"], progress=round((f + 1) / (n + CD), 3),
                                       frames_done=f + 1, frames_total=n + CD, batch_size=count, engine_rows=capacity,
                                       ms_per_frame=round(1e3 * elapsed / (f + 1), 1),
                                       msg=f"generating {count} samples together: {f + 1}/{n + CD} frames"))
                        if len(cancelled) == count:
                            break
                outputs = [(eng.readback(i, c["P"] + n + CD), int(eng.pcfg_guided[i].item()))
                           if i not in cancelled else (None, None) for i, c in enumerate(contexts)]
                wall = time.time() - t0
            for i, (parts, c) in enumerate(prepared):
                if i in cancelled or os.path.exists(f"{c['rd']}/CANCEL"):
                    self.fail_run(c["rd"], RuntimeError("cancelled"))
                    parts.close()
                    continue
                execution = executions[i]
                try:
                    parts.send({"S": outputs[i][0], "cfg_guided_frames": outputs[i][1], "wall": wall, "execution": execution})
                except StopIteration:
                    pass
                except Exception as exc:
                    log(f"publishing {c['rid']} failed: {traceback.format_exc()[-1500:]}")
                    self.fail_run(c["rd"], exc)
        except Exception as exc:
            log(f"batch failed: {traceback.format_exc()[-2000:]}")
            for parts, c in prepared:
                self.fail_run(c["rd"], exc)
                parts.close()

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
                self.active_run = r
                try:
                    req = json.load(open(f"{rd}/request.json"))
                    jobs = self.collect_batch(rd, req)
                    self.active_run = [x[1]["run_id"] for x in jobs]
                    self.run_batch(jobs)
                except Exception as e:  # noqa: BLE001
                    log(f"run {r} failed: {traceback.format_exc()[-1500:]}")
                    jwrite(f"{rd}/status.json", {"state": "error", "error": f"{type(e).__name__}: {str(e)[:400]}", "worker": WID, "t_end": time.time()})
                finally:
                    self.active_run = None
                break
            if not got:
                time.sleep(0.5)


if __name__ == "__main__":
    WID = sys.argv[1] if len(sys.argv) > 1 else f"w{os.getpid()}"
    os.makedirs(RUNS, exist_ok=True)
    wk = Worker()
    wk.feed_client()
    wk.options()                                                        # 10-26: every worker start rewrites SCHEMA/OPTIONS (one mono worker now; engine knobs change)
    wk.start_health()
    wk.loop()
