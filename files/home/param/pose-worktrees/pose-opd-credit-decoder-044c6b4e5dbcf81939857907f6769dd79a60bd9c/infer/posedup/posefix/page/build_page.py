"""posefix samples page: the user's pose MONOPLEX (aloe-pose-mono-8020 export-357) vs the pose-DUPLEX (Aloe pose-duplex FT export-498)
at five pose-prefill lengths. = the pose-duplex samples page (ppm/aloe-ft-32-samsep/poseduplex/samples/page/build_page.py, artifact
JfjmHg9GUw9MooWA33aQbC; descends from the PPM 2-side page samples/2side-local-10386-v2): same data model (conv -> ck -> versions -> 2 tracks
with mono MP3 + waveform peaks + word lane, cards, video per speaker), data loading adapted to the posefix simulators:
  generated runs  posefix/out/{mono,dupM}/<input>_<pf>.{npz,json}  (npz: pose [F, 32] from code frame 0, tok = B lane per conversation
                  frame, b = B audio 24 kHz aligned with pose frame 0; json: pose_prefill_frames, seed, ...). A run counts once its .json exists.
  inputs          1086 = replay of the user's live call 20261025-142453-1086 (A = the user's mic, session.npz 'a'; the monoplex has no
                  partner input); c{i}s0 = held-out Aloe conv i (convs/c{i}.npz: B = side 0, partner = stem_1, partner words = laneB_1).
  GT ("Real")     c{i}s0: that conversation's real B (stem_0 audio, laneB_0 words, clips/gt/c{i}s0.mp4); 1086: Person 01's own real 12 s
                  (her gt/mimi_local codes decoded with Mimi on CPU, untimed transcript text.txt, clips/gt/p01.mp4), no partner.
  avatar clips    posefix/clips/{mono,dupM,gt}/<name>[ _sxs].mp4, B side only; missing clips = no video cell (re-run as they land).
                  > 700 kbps or > 15 MB -> H.264 ~600 kbps, no audio track (the page plays the MP3s); else copied.
  scores          posefix/score/score.py --person 01_ab63d745 --filters none --start-token <pose_prefill_frames> on the run npz (generated
                  tokens only), cached in scores/<model>_<name>.json keyed on the npz mtime; group headline = mean over the group's runs of
                  cb0 repeat % (100 x (1 - cb0_change)), out_of_frame %, conf_all. Real reference = score/out/gt_aloe.json [none] rows.
Idempotent and incremental (media re-encoded only when the source is newer). Writes index.html's data.json, files.json, a/, scores/, cache/.
usage (compute node only, CPU):
  srun -w slinky-4 -x slinky-9 -n1 -c 16 --mem=48G -t 60 /data/home/amartyavarma/venvs/qwen-s2/bin/python <this dir>/build_page.py"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

assert os.environ.get("SLURM_JOB_ID"), "run on a compute node via srun (never the login node)"
OUT = os.environ.get("PAGE_OUT") or os.path.dirname(os.path.abspath(__file__))     # PAGE_OUT / PAGE_RUN_DIRS: test overrides only
for d in ("a", "scores", "cache"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
PFX = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix"
RUNS, CLIPS = f"{PFX}/out", f"{PFX}/clips"
RUN_DIRS = {"mono": f"{RUNS}/mono", "dupM": f"{RUNS}/dupM"} | json.loads(os.environ.get("PAGE_RUN_DIRS", "{}"))
SESS = "/data/home/amartyavarma/ppm/aloe-posedup-live/sessions/20261025-142453-1086"
CONVS = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs"
P01 = "/data/home/amartyavarma/video/demo-data/prompts/01_ab63d745"
FFD = "/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static"
SCORER, SPY = f"{PFX}/score/score.py", "/data/home/amartyavarma/video/venv-distilled/bin/python"
QWEN32 = "/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer/ppm/configs/Qwen3-32B"
MIMI_DIR = "/data/home/amartyavarma/models/mimi"
PERSON = "01_ab63d745"
TE, FPS, SPF, SR, NB = 151643, 12.5, 1920, 24000, 2400
INPUTS = ["1086", "c0s0", "c1s0", "c2s0", "c3s0"]
GROUPS = [(k, f"Pose prefill {lab}") for k, lab in (("pf0", "0 s"), ("pf2", "2 s"), ("pf5", "5 s"), ("pf10", "10 s"), ("pf1168", "11.68 s"))]
ARMS = [("lk151", "pose light top-k 151 (the user's own call setting)"), ("lk0", "pose light top-k off"),
        ("rep11", "pose rep-pen 1.1 cb0 (window 64)"), ("cfg3", "pose CFG 3")]   # 10-25 page size cap: t12 / cfg3lk151 / k0 / k0lk0 / rep13 /
# rep12all dropped from the page (posefix/STATUS.md keeps their numbers)
BASE = "pf2"                       # sampling groups: 2 s prefill, run <input>_pf2_<arm>; header diff = run params vs <input>_pf2 params
GROUPS += [(f"{BASE}_{a}", f"2 s prefill · {lab}") for a, lab in ARMS]
MODELS = [("mono", "Monoplex (export-357)"), ("dupM", "Duplex (export-498)")]
VIDEO_PREF = ("", "_sxs")          # avatar clip first; the avatar | skeleton side-by-side only when the plain one is missing
VKBPS = 400
SPAN_S = 30.0                      # every cell shows (and the scores measure) at most the first 30 s after the pose prefill
GT_CLIP = {"1086": ("gt", "01_ab63d745_gt")} | {f"c{i}s0": ("gtc", f"c{i}s0") for i in range(4)}   # Person 01's own GT / conv GT renders
PARAMS_LINE = ("Pose: temp 1, top-k 30 (cb0 + cb1-31), CFG off, no rep-pen. Text/audio: the fantastic set (duplex: factorised, top-k 50, "
               "min-p 0.05, freq pen 0.25/50, PAD bias -0.3; audio T1 k30/k30); monoplex engine has no factorised text / PAD bias: joint "
               "lane top-k 50, min-p 0.05, freq pen 0.25/50. Prompt: Person 01's voice clip, no persona. Pose prefill = Person 01's own real "
               "pose tokens for the first S s (video prefill = the same S s of her real video). No align / orient / smoothing.")
MAXTOK = int(SPAN_S * FPS)
SCORE_KEYS = ("cb0_change", "out_of_frame", "conf_all", "conf_body", "conf_hand", "speed_p50", "jumps_per_s", "centre_actor_p50",
              "width_actor_p50", "cb0_distinct_per30s", "seconds")


def log(*a):
    print(f"[page {time.strftime('%H:%M:%S')}]", *a, flush=True)


def fresh(dst, *srcs):
    return os.path.isfile(dst) and all(os.path.getmtime(dst) >= os.path.getmtime(s) for s in srcs if s and os.path.isfile(s))


def peaks(y):
    n = max(len(y) // NB * NB, NB)
    y2 = np.zeros(n, np.float32); y2[:min(n, len(y))] = y[:n]
    pk = np.abs(y2).reshape(NB, -1).max(1)
    return [round(float(v), 3) for v in pk / max(1e-6, float(pk.max()))]


def mp3(y, fn, *srcs):
    """the existing builder's MP3 (libsndfile, ~32 kb/s mono 24 kHz); re-encoded only when a source is newer."""
    if not fresh(f"{OUT}/{fn}", *srcs):
        sf.write(f"{OUT}/{fn}.tmp", np.clip(y, -1, 1), SR, format="MP3", subtype="MPEG_LAYER_III")
        os.replace(f"{OUT}/{fn}.tmp", f"{OUT}/{fn}")
    return fn


def silence(nfr):
    return np.zeros(int(nfr) * SPF, np.float32)


_TOK = [None]


def lane_words(lane, prompt_end=None):
    """= infer_ppm / ppm_io lane_words(lane, 0, 0): word pieces grouped into words (a new word = a piece starting with 'G' or after a
    non-word frame); lane index f -> t = f / 12.5 s (B's audio / pose frame 0 = lane frame 0). -> [[t, text, prompt]]"""
    if _TOK[0] is None:
        from transformers import AutoTokenizer
        _TOK[0] = AutoTokenizer.from_pretrained(QWEN32)
    tk = _TOK[0]
    words, cur = [], None
    for f, t in enumerate(np.asarray(lane).tolist()):
        t = int(t)
        if 0 <= t < TE:
            if cur is None or tk.convert_ids_to_tokens(t).startswith("Ġ"):
                cur = [f, [t]]; words.append(cur)
            else:
                cur[1].append(t)
        else:
            cur = None
    return [[round(f / FPS, 2), tk.decode(p).strip(), False] for f, p in words if tk.decode(p).strip()]


# ------------------------------------------------------------------------------------------------ video
def probe(src):
    r = subprocess.run([f"{FFD}/ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", src], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def video_job(src, fn, tmax):
    """copy, or (longer than tmax / > 700 kbps / > 15 MB) H.264 ~600 kbps trimmed to the first tmax s, no audio track."""
    dst = f"{OUT}/{fn}"
    if fresh(dst, src):
        return fn
    dur, size = probe(src), os.path.getsize(src)
    kbps = size * 8 / 1e3 / dur if dur else 1e9
    tmp = f"{dst}.tmp.mp4"
    if size > 15e6 or kbps > VKBPS * 7 / 6 or (dur or 1e9) > tmax + 0.1:
        cmd = [f"{FFD}/ffmpeg", "-y", "-v", "error", "-i", src, "-t", f"{tmax:.2f}", "-an", "-c:v", "libx264", "-preset", "medium", "-b:v", f"{VKBPS}k",
               "-maxrate", f"{int(VKBPS * 4 / 3)}k", "-bufsize", f"{2 * VKBPS}k", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-threads", "4", tmp]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            log("ffmpeg FAILED", src, r.stderr[-500:])
            return None
        log(f"video {fn}: {dur} s {kbps:.0f} kbps {size / 1e6:.1f} MB -> {tmax:.1f} s {os.path.getsize(tmp) / 1e6:.1f} MB")
    else:
        shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return fn


def clip_src(kind, name):
    for suf in VIDEO_PREF:
        p = f"{CLIPS}/{kind}/{name}{suf}.mp4"
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            return p, suf
    return None, None


# ------------------------------------------------------------------------------------------------ scores
def score_runs(runs):
    """runs [(model, name, npz, start_token)] -> {(model, name): whole metrics}; stale caches (npz mtime changed) re-scored, one scorer
    process per (model, start token), 4 at a time."""
    res, stale = {}, {}
    for model, name, npz, st in runs:
        cp = f"{OUT}/scores/{model}_{name}.json"
        if os.path.isfile(cp):
            c = json.load(open(cp))
            if c.get("npz_mtime") == os.path.getmtime(npz) and c.get("start_token") == st and c.get("max_tokens") == MAXTOK:
                res[(model, name)] = c["whole"]; continue
        stale.setdefault((model, st), []).append((name, npz))

    def run(key):
        model, st = key
        items = stale[key]
        tmp = f"{OUT}/scores/.tmp-{model}-{st}-{os.getpid()}.json"
        cmd = [SPY, SCORER, *[p for _, p in items], "--person", PERSON, "--filters", "none", "--start-token", str(st),
               "--max-tokens", str(MAXTOK), "--out", tmp]
        env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode:
            log("scorer FAILED", key, r.stderr[-800:]); return
        d = json.load(open(tmp)); os.remove(tmp)
        d = d if isinstance(d, list) else [d]
        assert len(d) == len(items), (len(d), len(items))
        for (name, npz), x in zip(items, d):
            w = {k: x["whole"].get(k) for k in SCORE_KEYS}
            w["measured_tokens"] = x["measured_tokens"]
            json.dump({"npz": npz, "npz_mtime": os.path.getmtime(npz), "start_token": st, "max_tokens": MAXTOK, "whole": w}, open(f"{OUT}/scores/{model}_{name}.json", "w"))
            res[(model, name)] = w
        log(f"scored {model} start {st}: {len(items)} runs")

    if stale:
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(run, list(stale)))
    return res


def summ(ws):
    ws = [w for w in ws if w and w.get("cb0_change") is not None]
    if not ws:
        return None
    return {"n": len(ws), "repeat": round(100 * (1 - float(np.mean([w["cb0_change"] for w in ws]))), 1),
            "oof": round(100 * float(np.mean([w["out_of_frame"] for w in ws])), 1), "conf": round(float(np.mean([w["conf_all"] for w in ws])), 3)}


def param_diff(base, new):
    """'key base -> new' over the sampling params that differ (base = the same model's plain 2 s prefill run)."""
    if base is None:
        return "base run (pf2) not present yet"
    ks = [k for k in sorted(set(base) | set(new)) if base.get(k) != new.get(k)]
    return ", ".join(f"{k} {base.get(k, '-')} \u2192 {new.get(k, '-')}" for k in ks) or "no param change vs base"


def one(w):
    return summ([w]) if w else None


# ------------------------------------------------------------------------------------------------ inputs / GT
def mimi_decode(codes):
    import torch
    from transformers import MimiModel
    torch.set_num_threads(8)
    m = MimiModel.from_pretrained(MIMI_DIR).eval()
    with torch.no_grad():
        c = torch.as_tensor(np.asarray(codes, np.int64))[None]
        return m.decode(c).audio_values[0, 0].float().numpy()[:c.shape[-1] * SPF]


def p01_audio():
    """Person 01's real 12 s (gt/mimi_local decoded) and the 10 s voice clip the prompt uses (gt/mimi_local[:, :125]), cached."""
    src = f"{P01}/gt/mimi_local.safetensors"
    out = {}
    for tag, n in (("gt", None), ("clip", 125)):
        cp = f"{OUT}/cache/p01_{tag}.wav"
        if not fresh(cp, src):
            from safetensors.numpy import load_file
            c = np.asarray(load_file(src)["codes"], np.int64)
            y = mimi_decode(c if n is None else c[:, :n])
            sf.write(cp, y, SR, subtype="PCM_16")
        out[tag], _ = sf.read(cp, dtype="float32")
    return out


INP = {}


def input_info(inp):
    """partner audio (A), GT B audio / words / video source, descriptions; loaded once per input."""
    if inp in INP:
        return INP[inp]
    if inp == "1086":
        a = np.load(f"{SESS}/session.npz")["a"].astype(np.float32)
        p = p01_audio()
        tw = open(f"{P01}/text.txt").read().split()
        d = {"A": a, "A_src": f"{SESS}/session.npz", "A_words": [], "A_role": "A · user's mic (live call 1086)",
             "gt_B": p["gt"], "gt_B_src": f"{P01}/gt/mimi_local.safetensors", "gt_B_words": [[None, w, False] for w in tw],
             "gt_A_role": "A · none (solo recording)", "gt_B_role": "B · Person 01 (her real 12 s)", "gt_clip": "p01", "gt_A_silent": True,
             "title": "1086 · live call replay",
             "desc": "Input: replay of the user's live call 20261025-142453-1086 (Person 01). The duplex hears the user's recorded mic on "
                     "channel A; the monoplex has no partner input (channel A empty). Real = Person 01's own real 12 s (no partner; "
                     "transcript untimed).",
             "clip": p["clip"]}
    else:
        i = int(inp[1])
        z = np.load(f"{CONVS}/c{i}.npz")
        n = 1500 * SPF
        d = {"A": z["stem_1"].astype(np.float32)[:n], "A_src": f"{CONVS}/c{i}.npz", "A_words": lane_words(z["laneB_1"]),
             "A_role": f"A · partner (real, Aloe c{i})", "gt_B": z["stem_0"].astype(np.float32)[:n], "gt_B_src": f"{CONVS}/c{i}.npz",
             "gt_B_words": lane_words(z["laneB_0"]), "gt_A_role": f"A · partner (real, Aloe c{i})", "gt_B_role": f"B · real speaker (Aloe c{i})",
             "gt_clip": inp, "gt_A_silent": False, "title": f"c{i} · held-out Aloe",
             "desc": f"Input: held-out Aloe conversation c{i} (B = side 0). The duplex hears the real partner (stem 1) on channel A; the "
                     f"monoplex has no partner input (channel A empty). Real = this conversation's real B (pose, audio, words).",
             "clip": None}
    INP[inp] = d
    return d


# ------------------------------------------------------------------------------------------------ build
def main():
    t0 = time.time()
    files = {"data.json": "data.json"}
    vjobs = {}
    # runs present (json = done marker)
    runs = {}
    for model, _ in MODELS:
        for inp in INPUTS:
            for g, _ in GROUPS:
                nm = f"{inp}_{g}"
                js, npz = f"{RUN_DIRS[model]}/{nm}.json", f"{RUN_DIRS[model]}/{nm}.npz"
                if os.path.isfile(js) and os.path.isfile(npz):
                    runs[(model, nm)] = (json.load(open(js)), npz)
    log(f"{len(runs)} runs present:", sorted(f"{m}/{n}" for m, n in runs))
    scores = score_runs([(m, n, npz, int(J.get("pose_prefill_frames", 0))) for (m, n), (J, npz) in runs.items()])

    def add_video(kind, name, tmax):
        src, suf = clip_src(kind, name)
        if not src:
            return None, None
        fn = f"a/{kind}-{name}{suf}-{int(round(tmax * FPS))}.mp4"
        vjobs[fn] = (src, tmax)
        return fn, suf

    def silent_track(nfr):
        fn = f"a/silence-{nfr}.mp3"
        mp3(silence(nfr), fn); files[fn] = fn
        return fn, peaks(silence(nfr))

    convs, groups = [], []
    for gi, (g, glab) in enumerate(GROUPS):
        head, diffs = {}, {}
        for model, _ in MODELS:
            head[model] = summ([scores.get((model, f"{inp}_{g}")) for inp in INPUTS if (model, f"{inp}_{g}") in runs])
            if g != BASE and g.startswith(BASE + "_"):
                for inp in INPUTS:
                    if (model, f"{inp}_{g}") in runs:
                        base = runs.get((model, f"{inp}_{BASE}")) or next((runs[(model, f"{i}_{BASE}")] for i in INPUTS
                                                                           if (model, f"{i}_{BASE}") in runs), None)
                        diffs[model] = param_diff(base[0].get("params", {}) if base else None, runs[(model, f"{inp}_{g}")][0].get("params", {}))
                        break
        dl = list(dict.fromkeys(diffs.values()))
        diff = (dl[0] if len(dl) == 1 else " | ".join(f"{dict(MODELS)[m].split(' ')[0]}: {d}" for m, d in diffs.items())) if dl else None
        groups.append({"key": g, "label": glab, "head": head, "diff": diff})
        for inp in INPUTS:
            nm = f"{inp}_{g}"
            have = [(m, lab) for m, lab in MODELS if (m, nm) in runs]
            if not have:
                continue
            I = input_info(inp)
            vers = []
            npfw = max(int(runs[(m, nm)][0].get("pose_prefill_frames", 0)) for m, _ in have)
            tmax = npfw / FPS + SPAN_S                                       # window shown for every cell of this clip
            ns = int(round(tmax * SR))
            nfw = int(round(tmax * FPS))
            tw = lambda ws: [w for w in ws if w[0] is None or w[0] < tmax]
            A_in = I["A"][:ns]
            fa_in = f"a/in-{inp}-A-{nfw}.mp3"
            # Real (GT)
            gB = I["gt_B"][:ns]
            nfr = int(np.ceil(len(gB) / SPF))
            fb = f"a/gt-{inp}-B-{nfw}.mp3"; mp3(gB, fb, I["gt_B_src"]); files[fb] = fb
            if I["gt_A_silent"]:
                fa, pa = silent_track(nfr)
            else:
                fa = fa_in; mp3(A_in, fa, I["A_src"]); files[fa] = fa; pa = peaks(A_in)
            v = {"label": "Real (GT)", "kind": "gt", "np_s": None, "dur": len(gB) / SR, "seed": None, "score": None,
                 "tracks": [{"ch": "A", "role": I["gt_A_role"], "src": fa, "peaks": pa, "words": [] if I["gt_A_silent"] else tw(I["A_words"])},
                            {"ch": "B", "role": I["gt_B_role"], "src": fb, "peaks": peaks(gB), "words": tw(I["gt_B_words"])}]}
            vf, suf = add_video(*GT_CLIP[inp], min(tmax, len(gB) / SR))
            if vf:
                v["video"] = {"B": {"src": vf, "cap": ("real pose (Person 01's own 12 s)" if inp == "1086" else "real pose")
                                    + (" · avatar | skeleton" if suf else "")}}
            vers.append(v)
            seeds = set()
            for model, lab in have:
                J, npz = runs[(model, nm)]
                z = np.load(npz)
                b, tok = z["b"].astype(np.float32)[:ns], z["tok"][:nfw]
                npf = int(J.get("pose_prefill_frames", 0))
                seeds.add(J.get("seed"))
                fb = f"a/{model}-{nm}-B-{nfw}.mp3"; mp3(b, fb, npz); files[fb] = fb
                if model == "mono":
                    fa, pa = silent_track(int(np.ceil(len(b) / SPF)))
                    ta = {"ch": "A", "role": "A · empty (monoplex hears nothing)", "src": fa, "peaks": pa, "words": []}
                else:
                    fa = fa_in; mp3(A_in, fa, I["A_src"]); files[fa] = fa
                    ta = {"ch": "A", "role": I["A_role"], "src": fa, "peaks": peaks(A_in), "words": tw(I["A_words"])}
                v = {"label": lab, "kind": model, "np_s": npf / FPS, "np_frames": npf, "dur": len(b) / SR, "seed": J.get("seed"),
                     "score": one(scores.get((model, nm))),
                     "tracks": [ta, {"ch": "B", "role": "B · Person 01 (generated)", "src": fb, "peaks": peaks(b), "words": lane_words(tok)}]}
                vf, suf = add_video(model, nm, len(b) / SR)
                if vf:
                    v["video"] = {"B": {"src": vf, "cap": (f"real pose for the first {npf / FPS:.2f} s, generated after" if npf else "generated pose")
                                        + (" · avatar | skeleton" if suf else "")}}
                vers.append(v)
            cards = {"A": {"role": "A · partner", "text": None, "clip": None}, "B": {"role": "B · Person 01", "text": None, "clip": None}}
            if I.get("clip") is None:
                I["clip"] = p01_audio()["clip"]
            fc = "a/p01-clip.mp3"; mp3(I["clip"], fc, f"{P01}/gt/mimi_local.safetensors"); files[fc] = fc
            cards["B"]["clip"] = {"src": fc, "seconds": len(I["clip"]) / SR}
            npf0 = max(v.get("np_frames", 0) for v in vers)
            convs.append({"name": nm, "group": gi, "title": I["title"], "input": inp, "desc": I["desc"], "arm": glab if diff else None, "diff": diff,
                          "seed": sorted(s for s in seeds if s is not None), "np_frames": npf0,
                          "ck": {"posefix": {"step": "", "sampling": {}, "versions": vers, "cards": cards, "score": None}},
                          "mode": "continue", "np_s": npf0 / FPS, "order": [{"kind": "clip", "ch": "B"}]})
    # videos (parallel)
    if vjobs:
        with ThreadPoolExecutor(4) as ex:
            got = dict(zip(vjobs, ex.map(lambda kv: video_job(kv[1][0], kv[0], kv[1][1]), vjobs.items())))
        for fn, ok in got.items():
            if ok:
                files[fn] = fn
        bad = {fn for fn, ok in got.items() if not ok}
        for c in convs:
            for v in c["ck"]["posefix"]["versions"]:
                if v.get("video") and v["video"]["B"]["src"] in bad:
                    del v["video"]
    # video_ok per version; default_conv = first 2 s prefill clip whose GT, Monoplex and Duplex videos all exist (else the first clip of
    # any group with all three, else the first clip with any video, else 0)
    for c in convs:
        for v in c["ck"]["posefix"]["versions"]:
            v["video_ok"] = bool(v.get("video") and v["video"]["B"]["src"] in files and os.path.isfile(f"{OUT}/{v['video']['B']['src']}"))

    def full(c):
        vs = c["ck"]["posefix"]["versions"]
        return {v["kind"] for v in vs if v["video_ok"]} >= {"gt", "mono", "dupM"}
    gpf2 = next(i for i, (k, _) in enumerate(GROUPS) if k == BASE)
    default_conv = next((i for i, c in enumerate(convs) if c["group"] == gpf2 and full(c)), None)
    if default_conv is None:
        default_conv = next((i for i, c in enumerate(convs) if full(c)), None)
    if default_conv is None:
        default_conv = next((i for i, c in enumerate(convs) if any(v["video_ok"] for v in c["ck"]["posefix"]["versions"])), 0)
    # real reference (score/out: gt_aloe [none] rows, Person 01's own 12 s from gt_actors)
    ref = {}
    try:
        ga = [x["whole"] for x in json.load(open(f"{PFX}/score/out/gt_aloe.json")) if x["filters"] == "none"]
        ref["aloe"] = summ(ga)
        gp = [x["whole"] for x in json.load(open(f"{PFX}/score/out/gt_actors.json")) if x["filters"] == "none" and x["name"] == PERSON]
        ref["p01"] = summ(gp)
    except (OSError, KeyError, ValueError) as e:
        log("reference scores unavailable:", e)
    n_total = len(MODELS) * len(INPUTS) * len(GROUPS)
    n_vid = sum(1 for c in convs for v in c["ck"]["posefix"]["versions"] if v.get("video"))
    meta = {"ckpts": [{"tag": "posefix", "step": ""}], "params_line": PARAMS_LINE, "groups": groups, "ref": ref,
            "n_runs": len(runs), "n_total": n_total, "n_video": n_vid, "built": time.strftime("%Y-%m-%d %H:%M %Z")}
    json.dump({"meta": meta, "default_conv": default_conv, "convs": convs}, open(f"{OUT}/data.json.tmp", "w"), separators=(",", ":"))
    os.replace(f"{OUT}/data.json.tmp", f"{OUT}/data.json")
    json.dump(files, open(f"{OUT}/files.json", "w"))
    tot = sum(os.path.getsize(f"{OUT}/{f}") for f in files)
    big = max(files, key=lambda f: os.path.getsize(f"{OUT}/{f}"))
    log(f"{len(convs)} clips in {sum(1 for g in groups)} groups, {len(runs)}/{n_total} runs, {n_vid} video cells; {len(files)} files "
        f"{tot / 1e6:.1f} MB (largest {big} {os.path.getsize(f'{OUT}/{big}') / 1e6:.1f} MB); {time.time() - t0:.0f} s")
    for gr in groups:
        log(f"group {gr['label']}: " + " | ".join(f"{m} {gr['head'][m]}" for m, _ in MODELS) + (f"  [{gr['diff']}]" if gr.get("diff") else ""))
    log("ref", ref)
    log(f"default_conv {default_conv} = {convs[default_conv]['name'] if convs else None}")


if __name__ == "__main__":
    main()
