#!/usr/bin/env python3
"""(NO-CFG variant, 10-19 user: no CFG on pose; else identical) STUDIO final eval (10-19, user: prefill from the 4 picked studio prompt packages): gen_allgen.py on rows-studio (build_rows_studio.py:
our v2pure store row of the package's recording / speaker / window, B's voice clip, no persona; frames 0..24 Mimi + pose = the package's
llm_inputs prefill_audio / prefill_pose; lane = the store row's words aligned to the package window). Prefill [0, P + 27), then words, audio
and pose all sampled to 60 s: temp 1 every stream, no min-p, audio top-k 30, pose top-k 30, pose CFG 3 (silent-speech uncond), no rep-pen, bf16.
Outputs <out>/runs-studio-nocfg/<row>/: s1.wav (package audio 0-2 s, generated after) words.json codes.npz (s1, pose_s1, lane_s1) meta.json DONE
   python gen_studio.py --ckpt <export-357> --rows-dir <rows-studio> --out <work root> --rows S-05,... --seed S --devs 0,1"""
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
PARAMS = {"text_temp": 1.0, "text_topk": 0, "text_topp": 1.0, "min_p": 0.0, "min_p_rel": 0, "freq_pen": 0.0, "text_rep": 1.0,
          "allow_eob": 0, "free_text": 0, "word_greedy": 0,
          "cb0_temp": 1.0, "cb0_topk": 30, "light_temp": 1.0, "light_topk": 30, "audio_rep_pen": 1.0,
          "pose_temp": 1.0, "pose_topk": 30, "pose_light_temp": 1.0, "pose_light_topk": 30,
          "pose_cfg_on": 0, "pose_cfg_s": 3.0, "pose_cfg_cb0": 1, "pose_cfg_light": 1,
          "pose_rep": 1.0, "pose_rep_win": 0, "adaptive_rep_on": 0, "audio_rep_speaking": 1.0, "audio_rep_silent": 1.0,
          "text_rep_speaking": 1.0, "text_rep_silent": 1.0, "pose_rep_speaking": 1.0, "pose_rep_silent": 1.0}
K30 = "/data/home/amartyavarma/ppm/samples/aloe-v2pure-200/runs-force-k30-cfg3/B-000/meta.json"


def post(a, mimi, row, m, S, rinfo, seed, t_start, gen_s, spstep, einfo):
    out = f"{a.out}/runs-studio-nocfg/{row}"
    assert not os.path.exists(f"{out}/DONE"), ("refusing to overwrite a finished run", out)
    os.makedirs(out, exist_ok=True)
    P, n = int(m["P"]), int(rinfo["n"])
    lane_b, dep_b, pdep = S["lane_b"], S["dep_b"], S["pdep"]
    assert (S["lane_a"][:P + n + CD] == PAD).all() and (S["dep_a"][:P + n + CD] < 0).all(), "channel A was written"
    c1 = (dep_b[P + CD:P + CD + n] - OFF).T
    p1 = (pdep[P + CD:P + CD + n] - OFF).T
    for c in (c1, p1):
        assert c.shape == (DEPTH, n) and c.min() >= 0 and c.max() < CBS, (c.shape, c.min(), c.max())
    pa = (np.asarray(m["depth"][1])[P + CD:P + CD + NPF] - OFF).T
    pp = (np.asarray(m["pdepth"])[P + CD:P + CD + NPF] - OFF).T
    assert (c1[:, :NPF] == pa).all() and (p1[:, :NPF] == pp).all(), "2 s prefill codes != the package's"
    assert np.array_equal(lane_b[:P + F0], np.asarray(m["ids"][1])[:P + F0]), "prefill lane != the row's"
    w1 = mimi.decode(c1)
    I.write_wav(f"{out}/s1.wav", w1)
    rd = np.asarray(m["depth"][1])
    cpos = np.flatnonzero(rd[:P, 0] >= 0)
    I.write_wav(f"{out}/clip_s1.wav", mimi.decode((rd[cpos] - OFF).T))
    words = I.lane_words(lane_b[:P + n], P, P, P + F0)
    json.dump({"s1": words, "prompt_s": NPF / FPS, "prompt_lane_frames": F0, "lane_s1": lane_b[P:P + n].tolist()}, open(f"{out}/words.json", "w"))
    np.savez_compressed(f"{out}/codes.npz", s1=c1.astype(np.int16), pose_s1=p1.astype(np.int16), lane_s1=lane_b[P:P + n],
                        prefill_audio=pa.astype(np.int16), prefill_pose=pp.astype(np.int16))
    gen = [w for w in words if not w["prompt"]]
    meta = {"row": row, "mode": "studio-allgen", "card": rinfo["card"], "identity": rinfo["identity"], "store_row_key": rinfo["store_row_key"],
            "P": P, "n_frames": n, "prompt": rinfo["prompt"], "prefill": f"positions [0, P + {F0}): system block (voice clip only), the package's "
            f"prefill_audio + prefill_pose (frames 0..{NPF - 1}), the store row's lane frames 0..{F0 - 1}", "generated": "words, audio, pose to 60 s",
            "gen_words": len(gen), "gen_text": " ".join(w["text"] for w in gen), "prefill_words": [w["text"] for w in words if w["prompt"]],
            "audio": I.audio_stats(w1[NPF * 1920:]), "pose_gen_cb0_unique": int(len(np.unique(p1[0, NPF:]))), "speed_s_per_step": round(spstep, 4),
            "gen_wall_s": round(gen_s, 1), "seed": seed, "sampling": dict(PARAMS), "rep_pen": None, "min_p": None, "dtype": "bf16", "ckpt": a.ckpt,
            "step": STEP, "engine": einfo, "made": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    meta["wall_s"] = round(time.time() - t_start, 1)
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    open(f"{out}/DONE", "w").write("ok\n")
    log(f"[studio] {row} done: prefill words {meta['prefill_words']}, {meta['gen_words']} gen words\n  {meta['gen_text'][:200]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--devs", default="0,1")
    a = ap.parse_args()
    ref = dict(json.load(open(K30))["sampling"], min_p=0.0, pose_cfg_on=0)        # = the k30 CFG 3 sessions with min-p off (standing rule: no min-p)
    assert ref == PARAMS, ({k: (ref.get(k), v) for k, v in PARAMS.items() if ref.get(k) != v})
    names = [r for r in a.rows.split(",") if r]
    for r in names:
        assert not os.path.exists(f"{a.out}/runs-studio-nocfg/{r}/DONE"), ("refusing to overwrite", r)
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
        einfo = f"fastinfer/fast_ppm.py (CUDA graphs, bf16, {NI} rows + {NI} pose-CFG uncond rows, nothing forced after the prefill)"
        insts, mm = [], []
        nsteps = n + CD - F0
        for k, r in enumerate(names):
            x = pickle.load(open(f"{a.rows_dir}/{r}.pkl", "rb"))
            P = int(x["P"])
            assert P == rows[r]["P"] and x["ids"].shape[1] == P + n + CD and x["row_key"] == rows[r]["store_row_key"], (r, x["ids"].shape)
            assert (x["ids"][0] == PAD).all() and (x["depth"][0] < 0).all() and (x["present"][0] == 0).all(), "channel A must be empty"
            assert (x["pdepth"][P + CD:P + CD + NPF] >= 0).all() and (x["pdepth"][:P + CD] < 0).all() and (x["depth"][1][P + CD:P + CD + NPF] >= 0).all()
            insts.append(dict(x, a_mode="empty"))
            mm.append(x)
        log(f"[studio] ready in {time.time() - T00:.0f}s: rows {names}, window {n} frames, seed {a.seed}, params {PARAMS}")
        t_start = time.time()
        eng.start(insts, a.seed, PARAMS, f0=F0)
        assert all(eng.params[k] == v for k, v in PARAMS.items()), "engine params != requested"
        t1 = time.time()
        for i, f in enumerate(range(F0, F0 + nsteps)):
            eng.step(f, cold_pad=False)
        S = [eng.readback(k, int(insts[k]["P"]) + n + CD) for k in range(NI)]
        gen_s = time.time() - t1
        spstep = gen_s / nsteps
        log(f"[allgen] prefill {t1 - t_start:.1f}s, {nsteps} steps x {NI} rows (+ {NI} uncond) in {gen_s:.1f}s ({1e3 * spstep:.1f} ms / step)")
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
    log(f"[allgen] all done in {time.time() - T00:.0f}s{'; ERRORS ' + str(err) if err else ''}")
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
