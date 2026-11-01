#!/usr/bin/env python3
"""aloe-v2pure LONG samples (10-18): 60 s held-out windows (rows from build_rows_long.py), 2 s real prefill, then ~58 s with ALL streams
generated using EXACTLY the sampling of ppm_aloe_v2pure_samples' column 'all streams · CFG3 top-k 140': the resolved dict is read from
aloe-v2pure-200/runs-gen-k140/<row>/meta.json (asserted identical over its 10 rows, gen and force) and used unchanged (pose CFG 3 with the
silent-speech uncond, pose cb0 top-k 30 / cb1-31 top-k 140, audio top-k 30, temp 1 everywhere, factorised text with word min-p 0.05).
Output: <out>/runs-gen-k140/<row>/. Base: work-k30/gen_k30fp.py (docstring follows).
(10-18 fp05 variant = gen_k30.py + text frequency penalty (--freq-pen, window freq_win 20 = the engine default, now explicit), run-dir
suffix (--suffix, e.g. -fp05), --post-modes (write only these modes; the other rows still run so every session keeps the k30 batch layout
and RNG stream), and --groups SEED=JOBS;SEED=JOBS: several seed groups in one process on one model load (engine rebuilt only when the
row count changes; max_s = the longest clip over all groups).)
aloe-v2pure-200 k30 SAMPLES (10-18, copy of work/gen_v2pure.py; only pose + audio sampling changed): pose cb0 top-k 30 temp 1, pose cb1-31
top-k 30 temp 1, audio cb0 top-k 30 temp 1, audio cb1-31 top-k 30 temp 1; TWO sessions per process on the same engine and the same seed:
pose CFG off (runs-<mode>-k30-nocfg) and pose CFG 3 on cb0 + cb1-31 with the silent-speech uncond (runs-<mode>-k30-cfg3).
Original docstring follows.
aloe-v2pure-200 SAMPLES with the fast pose engine (fastinfer/fast_ppm.FastPPM: CUDA graphs, bf16, batched rows) on the PPM 32B monoplex
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
K140 = "/data/home/amartyavarma/ppm/samples/aloe-v2pure-200"
_ref = [json.load(open(f"{K140}/runs-{md}-k140/B-{i:03d}/meta.json"))["sampling"] for md in ("gen", "force") for i in range(10)]
assert all(r == _ref[0] for r in _ref), "k140 sampling differs between its rows"
PARAMS = dict(_ref[0])
assert PARAMS["pose_topk"] == 30 and PARAMS["pose_light_topk"] == 140 and PARAMS["pose_cfg_on"] == 1 and PARAMS["pose_cfg_s"] == 3.0
assert PARAMS["cb0_topk"] == 30 and PARAMS["light_topk"] == 30 and PARAMS["min_p"] == 0.05


def post(a, mimi, row, mode, m, S, rinfo, seed, t_start, gen_s, spstep, einfo, prm, var):
    out = f"{a.out}/runs-gen-k140/{row}"
    assert not os.path.exists(f"{out}/DONE"), ("refusing to overwrite a finished run", out)
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
            "seed": seed, "column": f"all streams · CFG3 top-k 140 ({n / FPS:.0f} s window)", "sampling_source": f"{K140}/runs-gen-k140/<row>/meta.json",
            "engine_max_s": round(a.max_s, 2),
            "variant": var, "sampling": dict(prm), "pose_cfg_uncond": "every Mimi input (prefill + generated) -> Mimi codes of digital silence; "
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
    ap.add_argument("--groups", required=True, help="SEED=B-000:gen,B-000:force;SEED=...  (one session pair per group, in order)")
    ap.add_argument("--devs", default="0,1")
    ap.add_argument("--variants", default="k140")
    ap.add_argument("--post-modes", default="gen", help="modes whose outputs are written")
    a = ap.parse_args()
    VARS = {"k140": dict(PARAMS)}
    variants = [v for v in a.variants.split(",") if v]
    pmodes = set(a.post_modes.split(","))
    assert variants and all(v in VARS for v in variants) and pmodes <= {"gen", "force"}, (variants, pmodes)
    groups = []
    for g in a.groups.split(";"):
        sd, js = g.split("=")
        groups.append((int(sd), [tuple(j.split(":")) for j in js.split(",") if j]))
    for v in variants:
        for _, jobs in groups:
            for r, md in jobs:
                if md in pmodes:
                    assert md == "gen" and not os.path.exists(f"{a.out}/runs-gen-k140/{r}/DONE"), ("refusing to overwrite", v, r, md)
    assert os.environ.get("GEMM_PRECISION", "bf16") == "bf16", "bf16 only"
    T00 = time.time()
    devs = [torch.device(f"cuda:{int(x)}") for x in a.devs.split(",")]
    torch.cuda.set_device(devs[-1])
    rows = {r["row"]: r for r in json.load(open(f"{a.rows_dir}/rows.json"))}
    for _, jobs in groups:
        assert all(r in rows and md in ("gen", "force") for r, md in jobs), jobs
    nall = max(rows[r]["n"] for _, jobs in groups for r, _ in jobs)
    a.max_s = max_s = nall / FPS + 2
    err = []
    with torch.inference_mode():
        m, ldev, cfgs, info = FP.build_model(a.ckpt, devs)
        assert info["pose"] and info["decp_bb"], info
        assert info["step"] == 200, info["step"]
        mimi = I.Mimi(devs[-1])
        p_max = 2048
        nfr = int(round(max_s * FPS)) + 8
        sil = mimi.encode(np.zeros((p_max + nfr) * 1920, np.float32))[:, :p_max + nfr]
        for v in variants:
            clean, bad = FP.validate(VARS[v])
            assert not bad and clean == VARS[v], (v, bad, clean)
        eng, eNI = None, None
        for seed, jobs in groups:
            NI = len(jobs)
            nmax = max(rows[r]["n"] for r, _ in jobs)
            if eNI != NI:
                if eng is not None:
                    del eng
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                eng = FP.FastPPM(m, ldev, cfgs, devs, n_inst=NI, cfg_rows=True, max_s=max_s, mimi_stream=False, silence=sil)
                eng.capture()
                eNI = NI
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
            log(f"[long] group seed {seed} ready at {time.time() - T00:.0f}s: jobs {jobs}, longest clip {nmax} frames, sessions {variants}, "
                f"write {sorted(pmodes)}")
            for var in variants:
                prm = VARS[var]
                log(f"[long] session {var}: seed {seed}, params {prm}")
                t_start = time.time()
                eng.start(insts, seed, prm, f0=F0)
                assert all(eng.params[k] == v for k, v in prm.items()), ("engine params != requested", var)
                t1 = time.time()
                nsteps = nmax + CD - F0
                for f in range(F0, F0 + nsteps):
                    eng.step(f, cold_pad=False)
                S = [eng.readback(k, int(insts[k]["P"]) + rows[r]["n"] + CD) for k, (r, _) in enumerate(jobs)]
                gen_s = time.time() - t1
                spstep = gen_s / nsteps
                log(f"[long] {var}: prefill {t1 - t_start:.1f}s, generation {nsteps} frames x {NI} rows (+ {NI} uncond) in {gen_s:.1f}s "
                    f"({1e3 * spstep:.1f} ms / frame)")

                def one(k, S=S, insts=insts, mm=mm, jobs=jobs, seed=seed, t_start=t_start, gen_s=gen_s, spstep=spstep, prm=prm, var=var,
                        einfo=einfo):
                    r, md = jobs[k]
                    try:
                        with torch.inference_mode():
                            post(a, mimi, r, md, mm[k], S[k], rows[r], seed, t_start, gen_s, spstep, einfo, prm, var)
                    except Exception as e:  # noqa: BLE001
                        import traceback
                        traceback.print_exc()
                        err.append(f"{r}:{md}:{var} {e!r}")
                ths = [threading.Thread(target=one, args=(k,)) for k in range(NI) if jobs[k][1] in pmodes]
                for th in ths:
                    th.start()
                for th in ths:
                    th.join()
    log(f"[long] all done in {time.time() - T00:.0f}s{'; ERRORS ' + str(err) if err else ''}")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
