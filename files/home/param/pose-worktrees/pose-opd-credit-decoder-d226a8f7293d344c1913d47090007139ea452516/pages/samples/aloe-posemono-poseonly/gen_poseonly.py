#!/usr/bin/env python3
"""aloe-posemono-poseonly (10-19): POSE-ONLY samples from the PPM 32B monoplex pose FT of job 24272 (aloe-pose-mono-8020 export-357).
= aloe-v2pure-200/work-k30/gen_k30.py's 'text forced' path (fastinfer/fast_ppm.FastPPM, CUDA graphs, bf16) with B's Mimi audio ALSO
teacher-forced, so only the pose stream is generated. Per row (rows from build_rows.py: the trainer's micro of a held-out track, B's system
block = its voice clip [+ persona text], channel A empty, cut to the window's first n frames), one engine row + its pose-CFG uncond row:
  PREFILL  positions [0, P + F0), F0 = 25 + CD = 27, teacher-forced: system block, B's real Mimi + pose of frames 0..24 (2.0 s), real lane.
  STEPS    f = F0 .. n - 1 + CD (positions P + f):
    text : B's lane forced to the row's real lane via the engine's in-graph text_force (= gen_k30 mode 'force'), asserted == real.
    audio: B's Mimi codes forced to the row's real codes at every position (engine step(force={'codes'}): the forced frame replaces the
           draw after the speech graph, before the trunk / next speech-tower step read it; the uncond rows still see digital-silence codes
           at those positions), asserted == real at every position.
    pose : SAMPLED (pose tower cb0 + fused 0.6B pose decoder cb1-31, decp_bb link).
SAMPLING = the v2pure k30 page's 'CFG 3' sessions (gen_k30.py PARAMS, variant cfg3), every knob explicit: pose temp 1.0, pose cb0 top-k 30,
  pose cb1-31 top-k 30, pose CFG 3 on cb0 + cb1-31 with the silent-speech uncond (every Mimi input -> Mimi codes of digital silence; same
  text lane, system block and pose history); no repetition / frequency penalty anywhere; bf16. (Text / audio knobs are inert: forced.)
Outputs per row: <out>/runs-poseonly/<row>/ s1.wav (= the real audio, decoded from the forced codes) gt_s1.wav clip_s1.wav words.json
  codes.npz pose_s1.mp4 gt_pose_s1.mp4 (skeleton renders, samples_fast.render) meta.json DONE
   python gen_poseonly.py --ckpt <export-357> --rows-dir <rows> --out <work root> --rows R-000,R-001,... --seed S --devs 0,1"""
import argparse
import json
import os
import pickle
import sys
import threading
import time
import types

sys.path.insert(0, "/data/home/amartyavarma/ppm/fastinfer")
import fast_ppm as FP  # noqa: E402
import samples_fast as SF  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

I = FP.IP
CD, DEPTH, CBS, FPS, OFF, PAD = FP.CD, FP.DEPTH, FP.CBS, FP.FPS, FP.OFF, FP.PAD
log = FP.log
NPF = 25
F0 = NPF + CD
STEP = 357
PARAMS = {"text_temp": 1.0, "text_topk": 0, "text_topp": 1.0, "min_p": 0.05, "min_p_rel": 0, "freq_pen": 0.0, "text_rep": 1.0,
          "allow_eob": 0, "free_text": 0, "word_greedy": 0,
          "cb0_temp": 1.0, "cb0_topk": 30, "light_temp": 1.0, "light_topk": 30, "audio_rep_pen": 1.0,
          "pose_temp": 1.0, "pose_topk": 30, "pose_light_temp": 1.0, "pose_light_topk": 30,
          "pose_cfg_on": 1, "pose_cfg_s": 3.0, "pose_cfg_cb0": 1, "pose_cfg_light": 1,
          "pose_rep": 1.0, "pose_rep_win": 0, "adaptive_rep_on": 0, "audio_rep_speaking": 1.0, "audio_rep_silent": 1.0,
          "text_rep_speaking": 1.0, "text_rep_silent": 1.0, "pose_rep_speaking": 1.0, "pose_rep_silent": 1.0}
K30 = "/data/home/amartyavarma/ppm/samples/aloe-v2pure-200/runs-force-k30-cfg3/B-000/meta.json"


def post(a, mimi, row, m, S, rinfo, seed, t_start, gen_s, spstep, einfo):
    out = f"{a.out}/runs-poseonly/{row}"
    assert not os.path.exists(f"{out}/DONE"), ("refusing to overwrite a finished run", out)
    os.makedirs(out, exist_ok=True)
    P, n = int(m["P"]), int(rinfo["n"])
    lane_b, dep_b, pdep = S["lane_b"], S["dep_b"], S["pdep"]
    real = np.asarray(m["ids"][1])
    assert (S["lane_a"][:P + n + CD] == PAD).all() and (S["dep_a"][:P + n + CD] < 0).all(), "channel A was written"
    # UNIT CHECKS: text and audio forced == real at every position of the window (prefill + forced steps)
    eq = lane_b[:P + n + CD] == real[:P + n + CD]
    assert eq.all(), ("FORCED LANE != REAL LANE", row, int((~eq).sum()))
    rd = np.asarray(m["depth"][1])
    eqa = dep_b[:P + n + CD] == rd[:P + n + CD]
    assert eqa.all(), ("FORCED AUDIO != REAL AUDIO", row, int((~eqa).sum()))
    c1 = (dep_b[P + CD:P + CD + n] - OFF).T
    p1 = (pdep[P + CD:P + CD + n] - OFF).T
    gp1 = (m["pdepth"][P + CD:P + CD + n] - OFF).T
    for c in (c1, p1, gp1):
        assert c.shape == (DEPTH, n) and c.min() >= 0 and c.max() < CBS, (c.shape, c.min(), c.max())
    assert (p1[:, :NPF] == gp1[:, :NPF]).all(), "2 s pose prefill != real"
    chk = {"lane_equals_real_positions": int(eq.size), "audio_equals_real_positions": int(eqa.size),
           "forced_steps": [P + F0, P + n + CD], "pose_generated_frames": [NPF, n],
           "pose_frames_differing_from_real_after_prefill": int((p1[:, NPF:] != gp1[:, NPF:]).any(0).sum())}
    log(f"[check] {row}: lane == real at {eq.size} positions, audio == real at {eqa.size} positions OK")
    w1 = mimi.decode(c1)
    I.write_wav(f"{out}/s1.wav", w1)
    I.write_wav(f"{out}/gt_s1.wav", w1)                                    # forced audio: the real recording's codes
    cpos = np.flatnonzero(rd[:P, 0] >= 0)
    assert len(cpos) and [int(cpos[0]), int(cpos[-1]) + 1] == rinfo["clip"]["pos"], (cpos[:3], rinfo["clip"])
    wc = mimi.decode((rd[cpos] - OFF).T)
    I.write_wav(f"{out}/clip_s1.wav", wc)
    lw = lambda lane: I.lane_words(lane[:P + n], P, P, P + F0)  # noqa: E731
    words = lw(real)
    json.dump({"s1": words, "gt": {"s1": words}, "prompt_s": NPF / FPS, "prompt_lane_frames": F0, "lane_s1": lane_b[P:P + n].tolist()},
              open(f"{out}/words.json", "w"))
    extra = {"pose_s1": p1.astype(np.int16), "gt_pose_s1": gp1.astype(np.int16)}
    np.savez_compressed(f"{out}/codes.npz", s1=c1.astype(np.int16), gt_s1=c1.astype(np.int16), lane_s1=lane_b[P:P + n], **extra)
    meta = {"row": row, "mode": "poseonly", "row_key": m["row_key"], "item": m["item"], "P": P, "n_frames": n, "dur_s": round(n / FPS, 2),
            "prompt": rinfo["prompt"], "prefill": f"positions [0, P + {F0}) teacher-forced: system block, real Mimi + pose of frames 0..{NPF - 1} "
            f"(2.0 s), real lane frames 0..{F0 - 1}", "generated": f"pose frames {NPF}..{n - 1} ({(n - NPF) / FPS:.2f} s) to the window's end",
            "text": "teacher-forced to the real lane (real words + timing) for the whole window",
            "audio": "teacher-forced to the real Mimi codes for the whole window", "force_check": chk, "chan_A": "empty", "pose": 1,
            "words": len(words), "text_all": " ".join(w["text"] for w in words), "audio_stats": I.audio_stats(w1),
            "pose_gen_cb0_unique": int(len(np.unique(p1[0, NPF:]))), "speed_s_per_step": round(spstep, 4), "gen_wall_s": round(gen_s, 1),
            "seed": seed, "column": "text + audio forced · CFG 3 · pose top-k 30", "variant": "cfg3", "sampling": dict(PARAMS),
            "pose_cfg_uncond": "every Mimi input (system-block clip, prefill, forced) -> Mimi codes of digital silence; same text lane, system "
            "block and pose history", "rep_pen": None, "dtype": "bf16", "ckpt": a.ckpt, "step": STEP, "engine": einfo,
            "made": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    SF.render(types.SimpleNamespace(rows_dir=a.rows_dir), out, row, extra, NPF, ("s1",))
    meta["wall_s"] = round(time.time() - t_start, 1)
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    open(f"{out}/DONE", "w").write("ok\n")
    log(f"[poseonly] {row} done: pose cb0 uniq {meta['pose_gen_cb0_unique']}, {chk['pose_frames_differing_from_real_after_prefill']} / "
        f"{n - NPF} generated frames differ from real -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--devs", default="0,1")
    a = ap.parse_args()
    ref = json.load(open(K30))["sampling"]
    assert ref == PARAMS, ("sampling != the v2pure k30 page's CFG 3 sessions", {k: (ref.get(k), v) for k, v in PARAMS.items() if ref.get(k) != v})
    names = [r for r in a.rows.split(",") if r]
    for r in names:
        assert not os.path.exists(f"{a.out}/runs-poseonly/{r}/DONE"), ("refusing to overwrite", r)
    assert os.environ.get("GEMM_PRECISION", "bf16") == "bf16", "bf16 only"
    T00 = time.time()
    devs = [torch.device(f"cuda:{int(x)}") for x in a.devs.split(",")]
    torch.cuda.set_device(devs[-1])
    rows = {r["row"]: r for r in json.load(open(f"{a.rows_dir}/rows.json"))}
    assert all(r in rows for r in names), names
    NI = len(names)
    ns = {rows[r]["n"] for r in names}
    assert len(ns) == 1, ("one window length per session", ns)
    n = ns.pop()
    max_s = n / FPS + 2
    with torch.inference_mode():
        m, ldev, cfgs, info = FP.build_model(a.ckpt, devs)
        assert info["pose"] and info["decp_bb"], info
        assert info["step"] == STEP, info["step"]
        mimi = I.Mimi(devs[-1])
        p_max = 2048
        nfr = int(round(max_s * FPS)) + 8
        sil = mimi.encode(np.zeros((p_max + nfr) * 1920, np.float32))[:, :p_max + nfr]
        eng = FP.FastPPM(m, ldev, cfgs, devs, n_inst=NI, cfg_rows=True, max_s=max_s, mimi_stream=False, silence=sil)
        eng.capture()
        clean, bad = FP.validate(PARAMS)
        assert not bad and clean == PARAMS, (bad, clean)
        einfo = f"fastinfer/fast_ppm.py (CUDA graphs, bf16, {NI} rows + {NI} pose-CFG uncond rows, in-graph text_force + per-step audio force)"
        insts, mm = [], []
        nsteps = n + CD - F0
        CF = np.zeros((NI, nsteps, DEPTH), np.int64)
        for k, r in enumerate(names):
            x = pickle.load(open(f"{a.rows_dir}/{r}.pkl", "rb"))
            P = int(x["P"])
            assert P == rows[r]["P"] and x["ids"].shape[1] == P + n + CD and x["row_key"] == rows[r]["row_key"], (r, x["ids"].shape)
            assert (x["ids"][0] == PAD).all() and (x["depth"][0] < 0).all() and (x["present"][0] == 0).all(), "channel A must be empty"
            assert (x["pdepth"][P + CD:P + CD + n] >= 0).all() and (x["pdepth"][:P + CD] < 0).all()
            cf = np.asarray(x["depth"][1][P + F0:P + n + CD], np.int64)
            assert cf.shape == (nsteps, DEPTH) and (cf >= 0).all(), (r, cf.shape)
            CF[k] = cf
            insts.append(dict(x, a_mode="empty", text_force=np.asarray(x["ids"][1], np.int64)))
            mm.append(x)
        log(f"[poseonly] ready in {time.time() - T00:.0f}s: rows {names}, window {n} frames, seed {a.seed}, params {PARAMS}")
        t_start = time.time()
        eng.start(insts, a.seed, PARAMS, f0=F0)
        assert all(eng.params[k] == v for k, v in PARAMS.items()), "engine params != requested"
        t1 = time.time()
        for i, f in enumerate(range(F0, F0 + nsteps)):
            eng.step(f, cold_pad=False, force={"codes": CF[:, i]})
        S = [eng.readback(k, int(insts[k]["P"]) + n + CD) for k in range(NI)]
        gen_s = time.time() - t1
        spstep = gen_s / nsteps
        log(f"[poseonly] prefill {t1 - t_start:.1f}s, {nsteps} steps x {NI} rows (+ {NI} uncond) in {gen_s:.1f}s ({1e3 * spstep:.1f} ms / step)")
        err = []

        def one(k):
            r = names[k]
            try:
                with torch.inference_mode():
                    post(a, mimi, r, mm[k], S[k], rows[r], a.seed, t_start, gen_s, spstep, einfo)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                err.append(f"{r} {e!r}")
        ths = [threading.Thread(target=one, args=(k,)) for k in range(NI)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
    log(f"[poseonly] all done in {time.time() - T00:.0f}s{'; ERRORS ' + str(err) if err else ''}")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
