#!/usr/bin/env python3
"""aloe-v2pure-200 SAMPLES with the fast pose engine (fastinfer/fast_ppm.FastPPM: CUDA graphs, bf16, batched rows) on the PPM 32B monoplex
pose fine-tune of job 23051 (export-200). Per job (row, mode), one engine row (+ its pose-CFG uncond row):
  PREFILL  the row's micro (build_rows.py = the trainer's Stores.conv: B's empty system block, channel A empty) teacher-forced over positions
           [0, P + F0), F0 = 25 + CD = 27: B's real Mimi + pose codes of the clip's first 2.0 s (frames 0..24; codes sit CD = 2 positions after
           their lane frame) and B's real lane up to frame 26.
  GENERATE frames from 2.0 s to the clip's real end (steps f = F0 .. n - 1 + CD; codes of frames 25 .. n - 1):
    mode gen   : B's text lane, Mimi and pose all sampled.
    mode force : B's text lane TEACHER-FORCED to the row's real lane (real words + real timing) via the engine's text_force (the forced token
                 replaces the draw inside the text graph, before lane / towers / trunk read it); Mimi + pose sampled. The forced lane is
                 asserted equal to the real lane at every position of the row (unit check, logged + meta.json).
SAMPLING (= the 8B v2pure README defaults; every knob explicit): temperature 1.0 on every stream and codebook; audio cb0 + cb1-31 top-k 30, no
  audio CFG; pose cb0 + cb1-31 top-k 250; pose CFG weight 3 on cb0 + cb1-31 (uncond rows = the same input with every Mimi input replaced by
  the Mimi codes of digital silence: the engine's silent-speech uncond); text (mode gen only) factorised: PAD / NEW_WORD / WORD at temperature
  1, then the word at temperature 1 with min-p 0.05 (absolute) over word pieces only (the engine's t_word_min_p keeps the WORD mass: the
  identical distribution, one draw), no top-k / top-p; no repetition / frequency penalty anywhere; bf16.
A worker's jobs share ONE session (run to its longest clip; each row cut at its own end). Outputs per job (= samples_fast.post_mono's files):
  <out>/runs-<mode>/B-iii/: s1.wav gt_s1.wav words.json codes.npz pose_s1.mp4 gt_pose_s1.mp4 meta.json DONE
   python gen_v2pure.py --ckpt <dir> --rows-dir <rows> --out <work root> --jobs B-000:gen,B-000:force,... [--seed S] [--devs 0,1]
          [--pose-topk 250 --pose-topk-light 250 --suffix '']   (10-18 k140 variant: --pose-topk 30 --pose-topk-light 140 --suffix=-k140)"""
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
import samples_fast as SF  # noqa: E402   (render: the samples pages' skeleton videos)
import numpy as np  # noqa: E402
import torch  # noqa: E402

I = FP.IP
CD, DEPTH, CBS, FPS, OFF, PAD = FP.CD, FP.DEPTH, FP.CBS, FP.FPS, FP.OFF, FP.PAD
log = FP.log
NPF = 25                      # prefill frames of real audio + pose (2.0 s)
F0 = NPF + CD                 # first generated position offset (lane frame 27)
PARAMS = {"text_temp": 1.0, "text_topk": 0, "text_topp": 1.0, "min_p": 0.05, "min_p_rel": 0, "freq_pen": 0.0, "text_rep": 1.0,
          "allow_eob": 0, "free_text": 0, "word_greedy": 0,
          "cb0_temp": 1.0, "cb0_topk": 30, "light_temp": 1.0, "light_topk": 30, "audio_rep_pen": 1.0,
          "pose_temp": 1.0, "pose_topk": 250, "pose_light_temp": 1.0, "pose_light_topk": 250,
          "pose_cfg_on": 1, "pose_cfg_s": 3.0, "pose_cfg_cb0": 1, "pose_cfg_light": 1,
          "pose_rep": 1.0, "pose_rep_win": 0, "adaptive_rep_on": 0, "audio_rep_speaking": 1.0, "audio_rep_silent": 1.0,
          "text_rep_speaking": 1.0, "text_rep_silent": 1.0, "pose_rep_speaking": 1.0, "pose_rep_silent": 1.0}


def post(a, mimi, row, mode, m, S, rinfo, seed, t_start, gen_s, spstep, einfo):
    out = f"{a.out}/runs-{mode}{a.suffix}/{row}"
    os.makedirs(out, exist_ok=True)
    P, n = int(m["P"]), int(rinfo["n"])
    lane_b, dep_b, pdep = S["lane_b"], S["dep_b"], S["pdep"]
    real = np.asarray(m["ids"][1])
    assert (S["lane_a"][:P + n + CD] == PAD).all() and (S["dep_a"][:P + n + CD] < 0).all(), "channel A was written"
    c1 = (dep_b[P + CD:P + CD + n] - OFF).T
    g1 = (m["depth"][1, P + CD:P + CD + n] - OFF).T
    p1 = (pdep[P + CD:P + CD + n] - OFF).T
    gp1 = (m["pdepth"][P + CD:P + CD + n] - OFF).T
    for c in (c1, g1, p1, gp1):
        assert c.shape == (DEPTH, n) and c.min() >= 0 and c.max() < CBS, (c.shape, c.min(), c.max())
    assert (c1[:, :NPF] == g1[:, :NPF]).all() and (p1[:, :NPF] == gp1[:, :NPF]).all(), "2 s prefill codes != real"
    assert np.array_equal(lane_b[:P + F0], real[:P + F0]), "prefill lane != real"
    chk = None
    if mode == "force":                                                                    # UNIT CHECK: forced lane == the real lane exactly
        eq = lane_b[P + F0:P + n + CD] == real[P + F0:P + n + CD]
        assert eq.all(), ("FORCED LANE != REAL LANE", row, int((~eq).sum()))
        assert np.array_equal(lane_b[:P + n + CD], real[:P + n + CD])
        chk = {"forced_positions": [P + F0, P + n + CD], "n_forced": int(eq.size), "n_equal": int(eq.sum()), "lane_equals_real": True,
               "real_word_pieces": int((real[P + F0:P + n + CD] < FP.TE).sum())}
        log(f"[check] {row} force: lane == real lane at all {eq.size} forced positions (+ {F0} prefill) OK")
    w1, gw1 = mimi.decode(c1), mimi.decode(g1)
    I.write_wav(f"{out}/s1.wav", w1)
    I.write_wav(f"{out}/gt_s1.wav", gw1)
    lw = lambda lane: I.lane_words(lane[:P + n], P, P, P + F0)  # noqa: E731
    words, gtw = lw(lane_b), lw(real)
    if mode == "force":
        assert [(w["text"], w["frame"]) for w in words] == [(w["text"], w["frame"]) for w in gtw], "forced words != real words"
    json.dump({"s1": words, "gt": {"s1": gtw}, "prompt_s": NPF / FPS, "prompt_lane_frames": F0, "lane_s1": lane_b[P:P + n].tolist()},
              open(f"{out}/words.json", "w"))
    extra = {"pose_s1": p1.astype(np.int16), "gt_pose_s1": gp1.astype(np.int16)}
    np.savez_compressed(f"{out}/codes.npz", s1=c1.astype(np.int16), gt_s1=g1.astype(np.int16), lane_s1=lane_b[P:P + n], **extra)
    gen = [w for w in words if not w["prompt"]]
    meta = {"row": row, "mode": mode, "row_key": m["row_key"], "item": m["item"], "P": P, "n_frames": n, "dur_s": round(n / FPS, 2),
            "prefill": f"positions [0, P + {F0}) teacher-forced: real Mimi + pose of frames 0..{NPF - 1} (2.0 s), real lane frames 0..{F0 - 1}",
            "generated": f"frames {NPF}..{n - 1} ({(n - NPF) / FPS:.2f} s) to the clip's end",
            "text": "teacher-forced to the real lane (real words + timing)" if mode == "force" else "sampled (factorised, word min-p 0.05)",
            "text_force_check": chk, "chan_A": "empty (PAD / Mimi -1 / present 0)", "pose": 1,
            "audio": {"s1_gen": I.audio_stats(w1[NPF * 1920:]), "gt_s1": I.audio_stats(gw1)},
            "gen_words": len(gen), "gen_text": " ".join(w["text"] for w in gen),
            "gt_cont_text": " ".join(w["text"] for w in gtw if not w["prompt"]),
            "pose_gen_cb0_unique": int(len(np.unique(p1[0, NPF:]))), "speed_s_per_step": round(spstep, 4), "gen_wall_s": round(gen_s, 1),
            "seed": seed, "sampling": dict(PARAMS), "pose_cfg_uncond": "every Mimi input (prefill + generated) -> Mimi codes of digital silence; "
            "same text lane, system block and pose history", "rep_pen": None, "dtype": "bf16", "ckpt": a.ckpt, "engine": einfo,
            "made": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    SF.render(types.SimpleNamespace(rows_dir=a.rows_dir), out, row, extra, NPF, ("s1",))
    meta["wall_s"] = round(time.time() - t_start, 1)
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    open(f"{out}/DONE", "w").write("ok\n")
    log(f"[{mode}] {row} done: {meta['gen_words']} gen words, pose cb0 uniq {meta['pose_gen_cb0_unique']} -> {out}\n  {meta['gen_text'][:240]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", required=True, help="B-000:gen,B-000:force,...")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--devs", default="0,1")
    ap.add_argument("--pose-topk", type=int, default=250, help="pose cb0 top-k (8B README default 250)")
    ap.add_argument("--pose-topk-light", type=int, default=250, help="pose cb1-31 top-k (8B README default 250)")
    ap.add_argument("--suffix", default="", help="runs dir suffix: <out>/runs-<mode><suffix>/ (e.g. -k140)")
    a = ap.parse_args()
    PARAMS.update(pose_topk=int(a.pose_topk), pose_light_topk=int(a.pose_topk_light))
    assert os.environ.get("GEMM_PRECISION", "bf16") == "bf16", "bf16 only"
    T00 = time.time()
    devs = [torch.device(f"cuda:{int(x)}") for x in a.devs.split(",")]
    torch.cuda.set_device(devs[-1])
    rows = {r["row"]: r for r in json.load(open(f"{a.rows_dir}/rows.json"))}
    jobs = [tuple(j.split(":")) for j in a.jobs.split(",") if j]
    assert all(r in rows and md in ("gen", "force") for r, md in jobs), jobs
    NI = len(jobs)
    nmax = max(rows[r]["n"] for r, _ in jobs)
    max_s = nmax / FPS + 2
    with torch.inference_mode():
        m, ldev, cfgs, info = FP.build_model(a.ckpt, devs)
        assert info["pose"] and info["decp_bb"], info
        assert info["step"] == 200, info["step"]
        mimi = I.Mimi(devs[-1])
        p_max = 2048
        nfr = int(round(max_s * FPS)) + 8
        sil = mimi.encode(np.zeros((p_max + nfr) * 1920, np.float32))[:, :p_max + nfr]
        eng = FP.FastPPM(m, ldev, cfgs, devs, n_inst=NI, cfg_rows=True, max_s=max_s, mimi_stream=False, silence=sil)
        eng.capture()
        clean, bad = FP.validate(PARAMS)
        assert not bad, bad
        einfo = f"fastinfer/fast_ppm.py (CUDA graphs, bf16, {NI} rows + {NI} pose-CFG uncond rows in one session, text_force per row)"
        insts, mm = [], []
        for r, md in jobs:
            x = pickle.load(open(f"{a.rows_dir}/{r}.pkl", "rb"))
            P, n = int(x["P"]), int(rows[r]["n"])
            assert x["ids"].shape[1] == P + n + CD and x["item"] == rows[r]["item"], (r, x["ids"].shape)
            assert (x["ids"][0] == PAD).all() and (x["depth"][0] < 0).all() and (x["present"][0] == 0).all(), "channel A must be empty"
            assert (x["pdepth"][P + CD:P + CD + n] >= 0).all() and (x["pdepth"][:P + CD] < 0).all()
            it = dict(x, a_mode="empty")
            if md == "force":
                it["text_force"] = np.asarray(x["ids"][1], np.int64)
            insts.append(it)
            mm.append(x)
        log(f"[v2pure] ready in {time.time() - T00:.0f}s: jobs {jobs}, longest clip {nmax} frames, params {PARAMS}")
        t_start = time.time()
        eng.start(insts, a.seed, PARAMS, f0=F0)
        t1 = time.time()
        nsteps = nmax + CD - F0
        for f in range(F0, F0 + nsteps):
            eng.step(f, cold_pad=False)
        S = [eng.readback(k, int(insts[k]["P"]) + rows[r]["n"] + CD) for k, (r, _) in enumerate(jobs)]
        gen_s = time.time() - t1
        spstep = gen_s / nsteps
        log(f"[v2pure] prefill {t1 - t_start:.1f}s, generation {nsteps} frames x {NI} rows (+ {NI} uncond) in {gen_s:.1f}s "
            f"({1e3 * spstep:.1f} ms / frame)")
        err = []

        def one(k):
            r, md = jobs[k]
            try:
                with torch.inference_mode():
                    post(a, mimi, r, md, mm[k], S[k], rows[r], a.seed, t_start, gen_s, spstep, einfo)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                err.append(f"{r}:{md} {e!r}")
        ths = [threading.Thread(target=one, args=(k,)) for k in range(NI)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
    log(f"[v2pure] all done in {time.time() - T00:.0f}s{'; ERRORS ' + str(err) if err else ''}")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
