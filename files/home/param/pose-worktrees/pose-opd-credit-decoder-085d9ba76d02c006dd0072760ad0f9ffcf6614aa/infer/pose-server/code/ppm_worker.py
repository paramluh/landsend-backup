"""PPM pose server: persistent PPM 32B pose-FT worker (2 GPUs, qwen-s2 venv). Loads the export ONCE (fast_ppm_srv engine: CUDA graphs, bf16,
NI = 4 instance rows + 4 CFG uncond rows, captured once; every sampling / CFG knob is a live device tensor, so nothing recompiles per request),
then claims queued requests from requests/ (oldest first) whose ckpt is the one it holds. Per request: the clip's prompt = the training-path
row (system prompt with B's voice clip by the training rule, no persona; the clip's real Mimi + pose for the prefill frames; the store row's
word lane for the prefill frames), generation of words + Mimi + pose for gen_s by backbone + towers + decoders, n_samples instance rows in
one batch. Writes s<k>/{codes.npz, gen.wav, words.json, meta.json}, progress.json, GEN_DONE; then starts the stick figures (CPU subprocess).
A request for another ckpt: the worker exits with code 3 after writing control/next-<tag>.txt (its loop restarts it on that export).
   python ppm_worker.py --tag TAG      (CUDA_VISIBLE_DEVICES = 2 GPUs; env PPM_CK = export dir)"""
import argparse
import importlib
import json
import os
import pickle
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import srv_common as C  # noqa: E402
import fast_ppm_srv as FP  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

I = FP.IP
CD, DEPTH, CBS, FPS, OFF, PAD = FP.CD, FP.DEPTH, FP.CBS, FP.FPS, FP.OFF, FP.PAD
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
args = ap.parse_args()
TAG = args.tag
CK = os.environ.get("PPM_CK") or C.default_ckpt()
NI, MAX_S = 4, 62.0
HELD = set()
CHILD = {}
STAT = {"kind": "ppm", "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "busy": None, "done": 0, "ckpt": CK}


def log(m):
    print(f"[ppm {TAG}] {time.strftime('%H:%M:%S')} {m}", flush=True)


def beat():
    while True:
        for c, ps in list(CHILD.items()):
            if all(p.poll() is not None for p in ps):
                CHILD.pop(c, None)
        C.heartbeat(HELD)
        try:
            C.worker_beat(TAG, STAT)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)


assert os.environ.get("GEMM_PRECISION", "bf16") == "bf16", "bf16 only"
t0 = time.time()
threading.Thread(target=beat, daemon=True).start()
devs = [torch.device("cuda:0"), torch.device("cuda:1")]
torch.cuda.set_device(devs[-1])
with torch.inference_mode():
    M, LDEV, CFGS, INFO = FP.build_model(CK, devs, last_w=float(os.environ.get("PPM_LAST_W", "0.75")))
    assert INFO["pose"] and INFO["decp_bb"], INFO
    MIMI = I.Mimi(devs[-1])
    ENG = FP.FastPPM(M, LDEV, CFGS, devs, n_inst=NI, cfg_rows=True, max_s=MAX_S, mimi_stream=False)
    ENG.capture()
STAT.update(step=INFO["step"], load_s=round(time.time() - t0, 1))
log(f"loaded {CK} (step {INFO['step']}) + captured in {time.time() - t0:.0f} s; " +
    ", ".join(f"cuda:{d.index} {torch.cuda.memory_allocated(d) / 2 ** 30:.1f} GiB" for d in devs))


def engine_params(req):
    p = {k: req[k] for k in ("factorised_text", "cat_temp", "text_temp", "text_topk", "text_topp", "min_p", "freq_pen", "freq_win", "pad_bias",
                             "text_rep", "text_rep_win", "cb0_temp", "cb0_topk", "cb0_topp", "light_temp", "light_topk", "light_topp", "pose_temp",
                             "pose_topk", "pose_topp", "pose_light_temp", "pose_light_topk", "pose_light_topp", "pose_rep", "pose_light_rep",
                             "pose_rep_win")}
    p.update(min_p_rel=0, allow_eob=0, free_text=0, word_greedy=0, audio_rep_pen=1.0, audio_rep_win=64, adaptive_rep_on=0, speak_window_frames=25,
             audio_rep_speaking=1.0, audio_rep_silent=1.0, text_rep_speaking=1.0, text_rep_silent=1.0, pose_rep_speaking=1.0, pose_rep_silent=1.0)
    w = float(req["pose_cfg"])
    p.update(pose_cfg_on=int(w > 0), pose_cfg_s=min(max(w, 1.0), 6.0) if w > 0 else 3.0, pose_cfg_cb0=int(req["pose_cfg_scope"] in ("all", "cb0")),
             pose_cfg_light=int(req["pose_cfg_scope"] in ("all", "light")), uncond_pad_lane=int(req["cfg_uncond"] == "silence_pad"))
    a = float(req["audio_cfg"])
    p.update(speech_cfg_on=int(a > 0), speech_cfg_s=min(max(a, 1.0), 6.0) if a > 0 else 3.0, speech_cfg_cb0=1)
    clean, bad = FP.validate(p)
    assert not bad, bad
    return clean


def progress(d, **p):
    C.write_json(f"{d}/progress.json", {**p, "t": time.time()})


@torch.no_grad()
def run(rid):
    import prompt_build
    PB = importlib.reload(prompt_build)                                  # prompt code re-read per request (no model reload)
    d = C.rdir(rid)
    req = json.load(open(f"{d}/request.json"))
    t_start = time.time()
    inst, plan = PB.build(req, d, MIMI, C.clip(req["clip"]) if req.get("prompt", "clip") == "clip" else None)
    info, force = plan["info"], plan["force"]
    prm = engine_params(req)
    nS = int(req["n_samples"])
    P, PF, n, F0 = inst["P"], info["PF"], info["n"], info["f0"]
    nsteps = n + CD - F0
    progress(d, stage="prefill", step=0, total=nsteps)
    with torch.inference_mode():
        ENG.start([inst] * NI, int(req["seed"]), prm, f0=F0)
        assert all(ENG.params[k] == v for k, v in prm.items()), "engine params != requested"
        t1 = time.time()
        for i, f in enumerate(range(F0, F0 + nsteps)):
            fo = None
            if force is not None and f < PF + CD:                         # pose-less upload: its audio (+ PAD lane) teacher-forced
                fo = {"lane": np.full(NI, force["lane"][f], np.int64)}
                if f >= CD:
                    fo["codes"] = np.tile(force["codes"][f - CD][None], (NI, 1))
            ENG.step(f, cold_pad=(F0 == 0), force=fo)
            if i % 10 == 9 or i == nsteps - 1:
                torch.cuda.synchronize(devs[-1])
                el = time.time() - t1
                progress(d, stage="generating", step=i + 1, total=nsteps, eta_s=round(el / (i + 1) * (nsteps - i - 1), 1))
        S = [ENG.readback(k, P + n + CD) for k in range(nS)]
    gen_s = time.time() - t1
    fz = info.get("frame0_pose")
    for k in range(nS):
        sd = f"{d}/s{k}"
        os.makedirs(sd, exist_ok=True)
        lane_b, dep_b, pdep = S[k]["lane_b"], S[k]["dep_b"], S[k]["pdep"]
        c1 = (dep_b[P + CD:P + CD + n] - OFF).T
        p1 = (pdep[P + CD:P + CD + n] - OFF).T
        assert c1.shape == (DEPTH, n) and c1.min() >= 0 and c1.max() < CBS and p1.min() >= 0 and p1.max() < CBS, (c1.shape, p1.shape)
        assert np.array_equal(lane_b[:P + F0], inst["ids"][1][:P + F0]), "prefill lane != the prompt's"
        if force is not None:
            assert np.array_equal(c1[:, :PF].T + OFF, force["codes"]), "forced upload audio not in the output"
        w1 = MIMI.decode(c1)
        I.write_wav(f"{sd}/gen.wav", w1)
        words = I.lane_words(lane_b[:P + n], P, P, P + PF + CD)
        json.dump({"s1": words, "prompt_s": PF / FPS, "prompt_lane_frames": PF + CD}, open(f"{sd}/words.json", "w"))
        np.savez_compressed(f"{sd}/codes.npz", audio=c1.astype(np.int16), pose=p1.astype(np.int16), lane=lane_b[P:P + n],
                            global_state=info["global_state"], frame0_pose=fz if fz is not None else np.zeros((133, 3), np.float32),
                            prefill_frames=PF)
        gen = [w for w in words if not w["prompt"]]
        C.write_json(f"{sd}/meta.json", {
            "id": rid, "sample": k, "ckpt": CK, "step": INFO["step"], "worker": TAG, "prompt": req.get("prompt", "clip"), "clip": req.get("clip"),
            "held_out": info.get("held_out"), "row_key": info.get("row_key"), "voice_clip": info.get("voice_clip"),
            "voice_clip_frames": info.get("voice_clip_frames"), "persona_tokens": info.get("persona_tokens"),
            "prefix_equals_training_row": info.get("prefix_equals_training_row", False), "P": P, "prefill_frames": PF, "frames": n,
            "seconds": n / FPS, "pose_from_frame0": bool(info.get("pose_from_frame0")), "gen_words": len(gen),
            "gen_text": " ".join(w["text"] for w in gen), "sink": info.get("sink") if fz is not None else None,
            "engine_params": prm, "audio": I.audio_stats(w1[PF * 1920:]), "pose_gen_cb0_unique": int(len(np.unique(p1[0, PF:]))),
            "timing": {"setup_s": round(t1 - t_start, 2), "gen_s": round(gen_s, 2), "ms_per_frame": round(1000 * gen_s / max(nsteps, 1), 1),
                       "total_s": round(time.time() - t_start, 2)}, "dtype": "bf16"})
    progress(d, stage="done", step=nsteps, total=nsteps, eta_s=0)
    open(f"{d}/GEN_DONE", "w").write("ok\n")
    return {"n": n, "samples": nS, "gen_s": round(gen_s, 1), "ms_per_frame": round(1000 * gen_s / max(nsteps, 1), 1),
            "total_s": round(time.time() - t_start, 1)}


def start_sticks(rid):
    d = C.rdir(rid)
    req = json.load(open(f"{d}/request.json"))
    if not req.get("stick"):
        return
    ps = []
    env = dict(os.environ, PPM_CD="2", CUDA_VISIBLE_DEVICES="")
    for k in range(int(req["n_samples"])):
        sd = f"{d}/s{k}"
        if os.path.exists(f"{sd}/STICK_DONE"):
            continue
        lf = open(f"{sd}/stick.log", "w")
        ps.append(subprocess.Popen(["nice", "-n", "10", "/data/home/amartyavarma/venvs/qwen-s2/bin/python", "-u", f"{HERE}/srv_stick.py", sd],
                                   env=env, stdout=lf, stderr=subprocess.STDOUT))
    if ps:
        CHILD[rid] = ps


# warm-up: one short session on the first clip (CUDA graphs replayed, Mimi decode, prompt builder import)
tw = time.time()
import prompt_build  # noqa: E402
_r = dict(C.normalize({"gen_s": 2.0, "id": f"_warm-{TAG}"})[1])
_inst, _plan = prompt_build.build(_r, "/nonexistent", MIMI, C.clip(_r["clip"]))
with torch.inference_mode():
    ENG.start([_inst] * NI, 0, engine_params(_r), f0=_plan["info"]["f0"])
    for f in range(_plan["info"]["f0"], _plan["info"]["f0"] + _plan["info"]["NG"]):
        ENG.step(f, cold_pad=False)
    _S = ENG.readback(0, _inst["P"] + _plan["info"]["n"] + CD)
    MIMI.decode((_S["dep_b"][_inst["P"] + CD:_inst["P"] + CD + _plan["info"]["n"]] - OFF).T)
log(f"warm-up {time.time() - tw:.1f} s; ready {time.time() - t0:.0f} s after start")
STAT["ready"] = time.time()

_ENG_T = time.time()


def maybe_rebuild_engine():
    """control/rebuild-engine newer than the current engine -> re-import fast_ppm_srv and rebuild + recapture the FastPPM engine on the
    RESIDENT weights (no checkpoint reload; ~30-40 s)."""
    global ENG, FP, _ENG_T
    f = f"{C.CTRL}/rebuild-engine"
    if not os.path.exists(f) or os.path.getmtime(f) <= _ENG_T:
        return
    t = time.time()
    log("engine rebuild requested: re-importing fast_ppm_srv, rebuilding + recapturing on the resident weights")
    import gc
    ENG = None
    gc.collect()
    for dv in devs:
        with torch.cuda.device(dv):
            torch.cuda.empty_cache()
    FP = importlib.reload(FP)
    with torch.inference_mode():
        ENG = FP.FastPPM(M, LDEV, CFGS, devs, n_inst=NI, cfg_rows=True, max_s=MAX_S, mimi_stream=False)
        ENG.capture()
    _ENG_T = time.time()
    log(f"engine rebuilt in {time.time() - t:.0f} s (weights untouched)")


while True:
    maybe_rebuild_engine()
    took = False
    for rid in C.list_requests():
        d = C.rdir(rid)
        if C.has(rid, "GEN_DONE") or C.has(rid, "GEN_FAIL"):
            continue
        req = C.rj(f"{d}/request.json")
        if req is None:
            continue
        if req.get("ckpt") != CK:
            others = [w for w in os.listdir(C.WORKERS) if w.endswith(".json")]
            live = [C.rj(f"{C.WORKERS}/{w}") for w in others]
            if any(w and w.get("kind") == "ppm" and w.get("ckpt") == req["ckpt"] and time.time() - w.get("t", 0) < C.STALE_S for w in live):
                continue                      # another worker holds that export
            if STAT["busy"] is None and not any(C.rj(f"{C.rdir(r)}/request.json") and C.rj(f"{C.rdir(r)}/request.json").get("ckpt") == CK
                                                and not C.has(r, "GEN_DONE") and not C.has(r, "GEN_FAIL") for r in C.list_requests()):
                open(f"{C.CTRL}/next-{TAG}.txt", "w").write(req["ckpt"] + "\n")
                log(f"hot-swap requested by {rid}: {CK} -> {req['ckpt']}; exiting for the reload")
                sys.exit(3)
            continue
        if not C.try_claim(d, "gen", TAG):
            continue
        c = f"{d}/gen.claim"
        HELD.add(c)
        STAT["busy"] = rid
        log(f"claim {rid}")
        try:
            r = run(rid)
            STAT["done"] += 1
            log(f"gen done {rid}: {r}")
            start_sticks(rid)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            open(f"{d}/GEN_FAIL", "w").write(f"{TAG}: {type(e).__name__}: {e}\n")
            log(f"FAILED {rid}: {type(e).__name__}: {e}")
        HELD.discard(c)
        C.release(d, "gen")
        STAT["busy"] = None
        took = True
        break
    if not took:                       # resumability: GEN_DONE without sticks (worker died mid-stick) -> redo them
        for rid in C.list_requests():
            if C.has(rid, "GEN_DONE") and rid not in CHILD:
                req = C.rj(f"{C.rdir(rid)}/request.json") or {}
                if req.get("stick") and any(not os.path.exists(f"{C.rdir(rid)}/s{k}/STICK_DONE") and
                                            not os.path.exists(f"{C.rdir(rid)}/s{k}/STICK_FAIL") for k in range(int(req.get("n_samples", 1)))):
                    if time.time() - os.path.getmtime(f"{C.rdir(rid)}/GEN_DONE") > 60:
                        start_sticks(rid)
        time.sleep(0.3)
