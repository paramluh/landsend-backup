"""PPM pose server: shared bits (stdlib only, importable from every venv): paths, clip registry, request normalisation, disk state, claims.
= sam-pose-server/code/srv_common.py's design (resumable on-disk queue keyed by request id) for the PPM 32B monoplex pose FT.

Disk layout (a restart re-scans requests/ and finishes whatever is not DONE):
  requests/<rid>/request.json         normalised request (written once by the front)
  requests/<rid>/gen.claim/           claim of the PPM worker running it (owner.json; mtime = heartbeat, stale after STALE_S)
  requests/<rid>/s<k>/ codes.npz gen.wav words.json meta.json   (sample k = 0 .. n_samples-1)
  requests/<rid>/GEN_DONE | GEN_FAIL  progress.json (generation step X / Y)
  requests/<rid>/s<k>/stick.mp4 STICK_DONE        (CPU subprocess started by the PPM worker)
  requests/<rid>/s<k>.video.claim/ -> s<k>/vdata/ video.mp4 video-flash.mp4 video_status.json progress.json VIDEO_DONE | VIDEO_FAIL
"""
import hashlib
import json
import os
import socket
import time

ROOT = "/data/home/amartyavarma/ppm/ppm-pose-server"
REQ = f"{ROOT}/requests"
WORKERS = f"{ROOT}/workers"
CTRL = f"{ROOT}/control"
STALE_S = 45
FPS_TOK = 12.5
PF_ROOT = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples"
DEFAULT_CKPT = "/data/home/amartyavarma/ppm/aloe-v2pure-16610/export-200"
FANTASTIC = "/data/home/amartyavarma/ppm/FANTASTIC_SET.json"

# ---- clip registry: the 20 true held-out clips (v2pure cache 'eval' recordings) + the 10 clips of the old SAM eval page (NOT held out) ----
_OLD = [("S-05", "10_27286bdd"), ("S-13", "06_74807fe0"), ("S-20", "19_47473a46"), ("E-03", "03_b437950e"), ("E-05", "05_e25c6859"),
        ("E-09", "09_d9ef3c94"), ("E-12", "12_72cb45d3"), ("E-13", "13_956c09af"), ("E-15", "15_d1e1322c"), ("E-18", "18_1ffa9083")]


def clips():
    out = []
    ids = open(f"{PF_ROOT}/control/ids.txt").read().strip().split(",")
    for i, idn in enumerate(ids):
        out.append({"id": idn, "group": "held-out (never trained on)", "label": f"held-out {i + 1:02d} · {idn}", "prompt_dir": f"{PF_ROOT}/prompts/{idn}",
                    "row": f"{PF_ROOT}/rows/{idn}", "gt_video": f"{PF_ROOT}/video/GT-{idn}/video-flash.mp4", "gt_stick": f"{PF_ROOT}/runs/GT-{idn}/stick.mp4",
                    "held_out": True})
    S = "/data/home/amartyavarma/video/studio-posemono/sam8b-eval-cfg"
    for row, idn in _OLD:
        out.append({"id": idn, "group": "old SAM eval page (in training data)", "label": f"SAM page {row} · {idn}",
                    "prompt_dir": f"/data/home/amartyavarma/video/demo-data/prompts/{idn}", "row": f"{PF_ROOT}/s20/rows/{idn}",
                    "gt_video": f"{S}/video/GT-{idn}/video-flash.mp4", "gt_stick": f"{S}/runs/{row}/stick-gt.mp4", "held_out": False})
    return out


def clip(cid):
    return next((c for c in clips() if c["id"] == cid), None)


SCR_EXPORTS = "/scratch/amartyavarma/ppm-pose-server/stage"           # slinky-15 node-local copies (never stage models on /data)
EXPORTS = [
    {"label": "PPM pose FT from 16,610 (aloe-v2pure-16610 export-200)", "path": "/data/home/amartyavarma/ppm/aloe-v2pure-16610/export-200"},
    {"label": "PPM pose FT from 13,533 (aloe-v2pure-13533 export-200)", "path": f"{SCR_EXPORTS}/13533"},
    {"label": "PPM pose FT from 11,488 (aloe-v2pure-ft export-200)", "path": "/data/home/amartyavarma/ppm/aloe-v2pure-ft/export-200"},
    {"label": "PPM pose FT from 7,932 (aloe-v2pure-7932 export-200)", "path": f"{SCR_EXPORTS}/7932"},
]
PRESETS = [
    {"id": "default", "label": "defaults (fantastic set, pose top-k 30, CFG off)", "set": {}},
    {"id": "k250", "label": "pose light top-k 250", "set": {"pose_light_topk": 250}},
    {"id": "k250cfg", "label": "top-k 250 + pose CFG 3 + audio CFG 3",
     "set": {"pose_light_topk": 250, "pose_cfg": 3.0, "pose_cfg_scope": "all", "cfg_uncond": "silence_pad", "audio_cfg": 3.0}},
    {"id": "user1020", "label": "user 10-20 set",           # the user's 10-20 set: plain text, pose T .952 / top-k 250, light T .966 / top-k 75,
     "set": {"factorised_text": 0, "cat_temp": 1.0, "text_temp": 1.0, "text_topk": 0, "text_topp": 1.0, "min_p": 0.0, "freq_pen": 0.0,   # pose CFG
             "pad_bias": 0.0, "text_rep": 1.0, "cb0_temp": 1.0, "cb0_topk": 30, "cb0_topp": 1.0, "light_temp": 1.0, "light_topk": 30,          # 3.139 on
             "light_topp": 1.0, "pose_temp": 0.952, "pose_topk": 250, "pose_topp": 1.0, "pose_light_temp": 0.966, "pose_light_topk": 75,      # all 32,
             "pose_light_topp": 1.0, "pose_rep": 1.0, "pose_light_rep": 1.022, "pose_rep_win": 50, "pose_cfg": 3.139, "pose_cfg_scope": "all",  # silent
             "cfg_uncond": "silence_pad", "audio_cfg": 0.0}},                                                                              # speech
]


def ckpt_ok(path):
    """a COMPLETE consolidated DCP export (.metadata) or a COMPLETE ppm-local-bf16-shards-v1 dir (meta.json + rank*.pt + rep.pt)."""
    if not path or not os.path.isfile(f"{path}/COMPLETE"):
        return False
    return os.path.isfile(f"{path}/.metadata") or (os.path.isfile(f"{path}/meta.json") and os.path.isfile(f"{path}/rep.pt"))


def exports():
    return [dict(e, available=ckpt_ok(e["path"])) for e in EXPORTS]


def personas():
    try:
        return json.load(open(f"{CTRL}/personas.json"))
    except Exception:  # noqa: BLE001
        return []


def _fs():
    return json.load(open(FANTASTIC))["params"]


def defaults():
    f = _fs()
    d = {"prompt": "clip", "clip": clips()[0]["id"], "prefill_s": 2.0, "gen_s": 18.0, "seed": 0, "n_samples": 1, "ckpt": None,
         "voice_clip": "rule", "persona": ""}
    for k in ("factorised_text", "cat_temp", "text_temp", "text_topk", "text_topp", "min_p", "freq_pen", "freq_win", "pad_bias", "text_rep",
              "text_rep_win", "cb0_temp", "cb0_topk", "light_temp", "light_topk"):
        d[k] = f[k]
    d.update(cb0_topp=1.0, light_topp=1.0, pose_temp=1.0, pose_topk=30, pose_topp=1.0, pose_light_temp=1.0, pose_light_topk=30, pose_light_topp=1.0,
             pose_rep=1.0, pose_light_rep=1.0, pose_rep_win=0,
             pose_cfg=0.0, pose_cfg_scope="all", cfg_uncond="silence_pad", audio_cfg=0.0,
             stick=True, video=True, video_stab_steps=40, video_res_steps=10, video_seed=2026)   # 10-21: 9-sinks + reference images (demo-v3 settings)
    return d


HELP = {
    "prompt": "clip = an eval clip's real speech + pose as prefill; upload = your wav (+ optional pose npz) as prefill",
    "clip": "prompt clip (its real speech + pose feed the prefill)",
    "voice_clip": "B's voice clip in the system prompt: rule = the clip's training-rule clip (a later chunk of the same speaker); none; upload = your voice wav (3-10 s)",
    "persona": "persona text in the system prompt (empty = none, the default); pick a held-out persona to edit",
    "prefill_s": "seconds of the clip's real pose + speech fed as prefill (0.08 .. 12)", "gen_s": "seconds generated after the prefill (0.08 .. 48)",
    "seed": "sampling seed", "n_samples": "independent samples of this prompt, generated in parallel in one batch (1 .. 4)",
    "ckpt": "export dir to serve (empty = the server default)",
    "factorised_text": "1 = factorised text draw (PAD / NEW_WORD / WORD, then the word); 0 = joint", "cat_temp": "category temperature (factorised)",
    "text_temp": "text temperature", "text_topk": "text top-k (0 = off)", "text_topp": "text top-p (1 = off)", "min_p": "word min-p (absolute)",
    "freq_pen": "frequency penalty strength", "freq_win": "frequency penalty window (word pieces)", "pad_bias": "added to B's PAD logit (< 0 = talk more)",
    "text_rep": "text CTRL rep-pen (1 = off)", "text_rep_win": "text rep-pen window",
    "cb0_temp": "speech cb0 temperature", "cb0_topk": "speech cb0 top-k", "cb0_topp": "speech cb0 top-p",
    "light_temp": "speech cb1-31 temperature", "light_topk": "speech cb1-31 top-k", "light_topp": "speech cb1-31 top-p",
    "pose_temp": "pose cb0 temperature", "pose_topk": "pose cb0 top-k", "pose_topp": "pose cb0 top-p",
    "pose_light_temp": "pose cb1-31 temperature", "pose_light_topk": "pose cb1-31 top-k", "pose_light_topp": "pose cb1-31 top-p",
    "pose_rep": "CTRL rep-pen on pose cb0 (1 = off)", "pose_light_rep": "CTRL rep-pen on pose cb1-31 (1 = off)",
    "pose_rep_win": "pose rep-pen window in frames (0 = every pose code generated so far)",
    "pose_cfg": "pose CFG scale w (0 = off): l = l_uncond + w (l_cond - l_uncond)", "pose_cfg_scope": "all = 32 pose codebooks, cb0 = cb0 only, light = cb1-31 only",
    "cfg_uncond": "uncond stream (shared by pose and audio CFG): silence_pad = every Mimi input -> digital-silence codes AND B's word lane -> PAD in the "
                  "conversation; silence = Mimi -> silence codes only (lane kept). System prompt text and pose history unchanged",
    "audio_cfg": "audio CFG scale on speech cb0 only (live-engine cfg_cb0 rule; 0 = off)",
    "stick": "render the stick figure", "video": "render the video WITH the 9 reference images (9-sinks video model; 10-21 rule: always refs)", "video_stab_steps": "stabiliser steps (multiple of 4; default 40 = demo-v3)",
    "video_res_steps": "resolver steps (default 10 = demo-v3)", "video_seed": "video noise seed",
}
FLOATS = ("prefill_s", "gen_s", "cat_temp", "text_temp", "text_topp", "min_p", "freq_pen", "pad_bias", "text_rep", "cb0_temp", "cb0_topp", "light_temp",
          "light_topp", "pose_temp", "pose_topp", "pose_light_temp", "pose_light_topp", "pose_cfg", "audio_cfg", "pose_rep",
          "pose_light_rep")
INTS = ("seed", "n_samples", "factorised_text", "text_topk", "pose_rep_win", "freq_win", "text_rep_win", "cb0_topk", "light_topk", "pose_topk", "pose_light_topk",
        "video_stab_steps", "video_res_steps", "video_seed")
BOOLS = ("stick", "video")
UPLOAD_KEYS = ("_upload_sha", "_upload_pose_sha", "_voice_sha")


def normalize(body):
    D = defaults()
    unknown = set(body) - set(D) - {"id", "gt_force"} - set(UPLOAD_KEYS)
    if unknown:
        raise ValueError(f"unknown fields: {sorted(unknown)}")
    r = dict(D)
    r.update({k: v for k, v in body.items() if k in D and v is not None and v != ""})
    for k in FLOATS:
        r[k] = float(r[k])
    for k in INTS:
        r[k] = int(float(r[k]))
    for k in BOOLS:
        r[k] = r[k] in (True, 1, "1", "true", "on", "yes")
    r["persona"] = str(r.get("persona") or "")
    if r["prompt"] not in ("clip", "upload") or r["voice_clip"] not in ("rule", "none", "upload"):
        raise ValueError("prompt clip|upload; voice_clip rule|none|upload")
    if r["prompt"] == "clip" and clip(r["clip"]) is None:
        raise ValueError(f"unknown clip {r['clip']}")
    if r["prompt"] == "upload":
        if not body.get("_upload_sha"):
            raise ValueError("prompt 'upload' needs an uploaded wav")
        r["clip"] = None
        if r["voice_clip"] == "rule":
            r["voice_clip"] = "none"
    if r["voice_clip"] == "upload" and not body.get("_voice_sha"):
        raise ValueError("voice_clip 'upload' needs an uploaded voice wav")
    for k in UPLOAD_KEYS:
        if body.get(k):
            r[k.lstrip("_")] = body[k]
    for k in ("pose_cfg", "audio_cfg"):
        if r[k] and not 1.0 <= r[k] <= 6.0:
            raise ValueError(f"{k}: 0 (off) or a scale in [1, 6]")
    if not (0.08 <= r["prefill_s"] <= 12.0) or not (0.08 <= r["gen_s"] <= 48.0) or r["prefill_s"] + r["gen_s"] > 60.0:
        raise ValueError("prefill_s in [0.08, 12], gen_s in [0.08, 48]")
    if not 1 <= r["n_samples"] <= 4:
        raise ValueError("n_samples in 1..4")
    if r["pose_cfg_scope"] not in ("all", "cb0", "light") or r["cfg_uncond"] not in ("silence_pad", "silence"):
        raise ValueError("pose_cfg_scope all|cb0|light; cfg_uncond silence_pad|silence")
    if r["video_stab_steps"] % 4 or r["video_stab_steps"] < 4 or r["video_res_steps"] < 1:
        raise ValueError("video_stab_steps: a positive multiple of 4; video_res_steps >= 1")
    if not r["ckpt"]:
        r["ckpt"] = default_ckpt()
    if not ckpt_ok(r["ckpt"]):
        raise ValueError(f"ckpt {r['ckpt']}: not a COMPLETE PPM export (DCP or node-local shards)")
    if body.get("gt_force") in (True, 1, "1", "true", "on", "yes"):   # (10-23) GT teacher forcing of B's lane + Mimi (prompt_build GT FORCING);
        if r["prompt"] != "clip" or r["voice_clip"] != "rule" or r["persona"]:   # opt-in only: absent from every other request (same rids)
            raise ValueError("gt_force: clip prompt, voice_clip rule, no persona")
        r["gt_force"] = True
    canon = json.dumps({k: r[k] for k in sorted(r)}, sort_keys=True)
    rid = body.get("id") or hashlib.sha1(canon.encode()).hexdigest()[:12]
    if not all(c.isalnum() or c in "-_" for c in rid) or len(rid) > 64:
        raise ValueError("id: [A-Za-z0-9_-]{1,64}")
    r["frames_prefill"] = max(1, round(r["prefill_s"] * FPS_TOK))
    r["frames_gen"] = max(1, round(r["gen_s"] * FPS_TOK))
    return rid, r


def default_ckpt():
    try:
        return json.load(open(f"{CTRL}/ckpt.json"))["default"]
    except Exception:  # noqa: BLE001
        return DEFAULT_CKPT


def rdir(rid):
    return f"{REQ}/{rid}"


def write_json(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def has(rid, name):
    return os.path.exists(f"{rdir(rid)}/{name}")


def rj(path):
    try:
        return json.load(open(path))
    except Exception:  # noqa: BLE001
        return None


def state(rid):
    d = rdir(rid)
    req = rj(f"{d}/request.json")
    if req is None:
        return None
    n = int(req.get("n_samples", 1))
    s = {"id": rid, "submitted": req.get("_submitted"), "request": {k: v for k, v in req.items() if not k.startswith("_")}}
    samples = []
    for k in range(n):
        sd = f"{d}/s{k}"
        samples.append({"k": k, "gen": os.path.exists(f"{sd}/meta.json"), "stick": os.path.exists(f"{sd}/STICK_DONE"),
                        "video": os.path.exists(f"{sd}/VIDEO_DONE"), "video_fail": (open(f"{sd}/VIDEO_FAIL").read().strip() if os.path.exists(f"{sd}/VIDEO_FAIL") else None),
                        "video_progress": rj(f"{sd}/progress.json"), "meta": rj(f"{sd}/meta.json"), "video_status": rj(f"{sd}/video_status.json")})
    s["samples"] = samples
    s["gen_progress"] = rj(f"{d}/progress.json")
    if has(rid, "GEN_FAIL"):
        s["state"], s["error"] = "failed", open(f"{d}/GEN_FAIL").read().strip()
    elif not has(rid, "GEN_DONE"):
        s["state"] = "generating" if claim_live(d, "gen") else "queued"
    elif req.get("video") and not all(x["video"] or x["video_fail"] for x in samples):
        s["state"] = "video"
    elif req.get("stick") and not all(x["stick"] for x in samples):
        s["state"] = "stick"
    else:
        s["state"] = "done"
    s["t_done"] = rj(f"{d}/t_done.json")
    return s


def list_requests():
    out = []
    for rid in os.listdir(REQ):
        p = f"{REQ}/{rid}/request.json"
        if os.path.exists(p):
            out.append((os.path.getmtime(p), rid))
    return [rid for _, rid in sorted(out)]


# ---- claims: mkdir <dir>/<name>.claim (atomic), owner.json inside, heartbeat = os.utime on the claim dir ----
def claim_live(d, name):
    try:
        return time.time() - os.stat(f"{d}/{name}.claim").st_mtime < STALE_S
    except FileNotFoundError:
        return False


def try_claim(d, name, tag):
    c = f"{d}/{name}.claim"
    try:
        os.mkdir(c)
    except FileExistsError:
        try:
            if time.time() - os.stat(c).st_mtime < STALE_S:
                return False
            os.rename(c, f"{c}.stale.{tag}.{int(time.time())}")
        except (FileNotFoundError, OSError):
            return False
        try:
            os.mkdir(c)
        except FileExistsError:
            return False
    me = {"worker": tag, "host": socket.gethostname(), "pid": os.getpid(), "t": time.time()}
    try:   # 10-22: stale-claim race - another worker may rename our just-made claim dir away (it saw the old stale one) -> we lost; never crash
        write_json(f"{c}/owner.json", me)
        with open(f"{c}/owner.json") as f:
            got = json.load(f)
    except (FileNotFoundError, OSError, ValueError):
        return False
    return got.get("worker") == tag and got.get("pid") == me["pid"]


def heartbeat(paths):
    for c in list(paths):
        try:
            os.utime(c)
        except FileNotFoundError:
            pass


def release(d, name):
    c = f"{d}/{name}.claim"
    try:
        for f in os.listdir(c):
            os.remove(f"{c}/{f}")
        os.rmdir(c)
    except FileNotFoundError:
        pass


def worker_beat(tag, info):
    os.makedirs(WORKERS, exist_ok=True)
    write_json(f"{WORKERS}/{tag}.json", {**info, "tag": tag, "host": socket.gethostname(), "pid": os.getpid(), "t": time.time()})
