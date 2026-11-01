"""si20k-vox CPU checks (compute node, no GPU, no checkpoints): run against a built store set (ALOE_SI20K_SRC / _STORE_ROOT / _STORES_JSON).
  1 STORE   sample rows: check_row strict; words / pieces == the generated lane (the trainer's lane rule rebuilds every piece + NEW_WORD
            before every word-initial piece after PAD); codes == the npz slice, PAD -1 past n; persona sidecar 'missing'; the B voice clip
            == the bank npy of manifest voice.B; the A ledger entry == the row window; keep-span cut honoured (no onset >= cut).
  2 A AUDIO ch0 slice length n x 1920; soundfile decode == ffmpeg (pan c0) decode bit for bit; speech activity aligns with the user words
            (frame RMS on user-word frames vs assistant-only frames, dB gap); c01 rows start at sample 1920 x 7500.
  3 MICROS  Stores.conv on sample items (train + eval cfg): T = P + n + CD, channel A empty in the conversation and the prefix, vin = (P, n),
            the B clip block == the bank codes, targets only on B at >= P, light ratio of the sample.
  4 PROMPTS the rule over the whole plan (finalize_plan's dry run) + realised flags on the sampled micros.
  5 VOXTRAL the frozen encoder (CPU, sdpa) on a 16 s A window of a real row -> [1, 4 n, 1280], finite.
  6 CONFIG  each trainer imported with its run env (+ PLAN values of this store set): resolved cfg vs the reference run's logged cfg.
usage: python -I cpu_checks.py --stores DIR --out JSON [--n 120] [--skip-vox] [--runs ppm,intl]"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

P = "/data/home/amartyavarma/ppm"
CODE = f"{P}/si20k-vox-ppm/code"
BUILD = "/data/home/amartyavarma/qwen-1ch-data/si20k-vox"


def env_file(path):
    out = {}
    for ln in open(path):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1); out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--skip-vox", action="store_true"); ap.add_argument("--runs", default="ppm,intl")
    a = ap.parse_args()
    t0 = time.time()
    os.environ.update(ALOE_SI20K_SRC=a.stores, ALOE_SI20K_STORE_ROOT=a.stores, ALOE_SI20K_STORES_JSON=f"{a.stores}/STORES.json", PPM_CD="2")
    for p in (CODE, f"{CODE}/lib", BUILD):
        sys.path.insert(0, p)
    import data_si20k_vox as A
    import si20k_common as SC
    import speech_q as SPQ
    D = A.D
    R = {"stores": a.stores, "t": time.strftime("%F %T")}
    sj = json.load(open(f"{a.stores}/STORES.json"))
    rng = np.random.default_rng(0)
    st = A.Stores()
    bank = SC.bank_paths()
    # ---------------------------------------------------------------- 1 STORE
    s1 = {"rows": 0, "check_row": 0, "lane_exact": 0, "nw_rule": 0, "codes_eq": 0, "pad_ok": 0, "persona_missing": 0, "clip_eq": 0, "ledger_ok": 0,
          "rows_cut": 0}
    samp = []
    for k in range(4):
        node = st.open(k)[0]
        N = len(node["length"])
        samp += [(k, int(r)) for r in rng.choice(N, size=min(a.n // 4, N), replace=False)]
    for k, r in samp:
        node, rows_, ps, vc, led = st.open(k)
        row = rows_[r]
        s1["rows"] += 1
        s1["check_row"] += not D.R3.check_row(row, strict=False)
        cid = row["episode_id"]
        mk = json.load(open(f"{SC.RUN}/queue/done/{cid}.json"))
        z = np.load(f"{SC.RUN}/{mk['codes']}")
        f0, n = int(row["frame_start"]), int(row["n_frames"])
        ok_l, ok_nw, ok_c, ok_p = True, True, True, True
        for side, C in (("user", "A"), ("assistant", "B")):
            lane = np.asarray(z[f"lane_{C}"], np.int64)[f0:f0 + n]
            wif = np.zeros(n, np.int64)
            fr = SPQ.lane(row[f"words_{side}"], row[f"piece_ids_{side}"], row[f"piece_offsets_{side}"], n, wif).copy()
            wf = lane < SC.CTRL_LO
            # a word whose onset is in a previous row is not this row's (its tail pieces at the row start are the only allowed difference)
            first = int(np.argmax(~wf)) if wf[0] else 0
            ok_l &= bool(np.array_equal(fr[first:][wf[first:]], lane[first:][wf[first:]]))
            fr2 = fr.copy(); SPQ.insert_new_word(fr2, wif)
            ok_nw &= bool(((fr2 == SC.NW_ID) <= ((np.roll(wif, -1) == 1) & (fr == SC.PAD_ID))).all())
            cc = np.asarray(row[f"codes_{side}"])
            ok_c &= bool(np.array_equal(cc[:, :n], np.asarray(z[f"codes_{C}"])[:, f0:f0 + n]))
            ok_p &= bool((cc[:, n:] == -1).all())
        s1["lane_exact"] += ok_l; s1["nw_rule"] += ok_nw; s1["codes_eq"] += ok_c; s1["pad_ok"] += ok_p
        pp = ps.get(r, row["row_key"])
        s1["persona_missing"] += pp["assistant"]["status"] == "missing" and pp["user"]["status"] == "missing"
        vid, cd, sha = st.clip(k, r, row["row_key"])
        ref = np.load(bank[mk["voice"]["B"]][0]).astype(np.int16)
        s1["clip_eq"] += vid == mk["voice"]["B"] and np.array_equal(np.asarray(cd), ref)
        e = led[row["row_key"]]
        s1["ledger_ok"] += e["frame0"] == f0 and e["n_frames"] == n and e["flac"].endswith(f"/out/flac/{cid}.flac") and e["flac_bytes"] == mk["flac_bytes"]
        s1["rows_cut"] += f0 + n < int(mk["frames"])
    R["1_store"] = s1
    print("[1 store]", s1, flush=True)
    # keep-span cut honoured (whole conversations): rows of a conversation end at its cut; no word onset at or past the cut in any row
    b = json.load(open(f"{a.stores}/BUILD.json"))
    spans = {}
    if b.get("keep_spans"):
        for ln in open(b["keep_spans"]):
            p_ = ln.rstrip("\n").split("\t")
            if p_[0] != "id" and len(p_) > 1 and p_[1]:
                spans[p_[0]] = float(p_[1])
    import pyarrow.parquet as pq
    cutchk = {"convs": 0, "cut_convs": 0, "end_ge_span": 0, "end_le_F": 0, "guard_frames_max": 0}
    for k in range(4):
        t = pq.read_table(f"{a.stores}/s20n{k}/rows.parquet", columns=["episode_id", "frame_end", "src_frames", "cut_frames"]).to_pydict()
        ends = {}
        for e_, fe, sf_, cf in zip(t["episode_id"], t["frame_end"], t["src_frames"], t["cut_frames"]):
            ends.setdefault(e_, [0, sf_, cf]); ends[e_][0] = max(ends[e_][0], fe)
        for e_, (fe, sf_, cf) in ends.items():
            cutchk["convs"] += 1
            if cf < sf_:
                cutchk["cut_convs"] += 1
                es = spans.get(e_)
                if es is not None:
                    c0 = int(np.ceil(es * 12.5 - 1e-6))
                    cutchk["end_ge_span"] += cf >= min(c0, sf_)
                    cutchk["guard_frames_max"] = max(cutchk["guard_frames_max"], cf - c0)
            cutchk["end_le_F"] += cf <= sf_ and (fe == cf or fe < cf)
    R["1_cut"] = cutchk
    print("[1 cut]", cutchk, flush=True)
    # ---------------------------------------------------------------- 2 A AUDIO
    s2 = {"rows": 0, "len_ok": 0, "ffmpeg_eq": 0, "ffmpeg_checked": 0, "gap_db": [], "c01_rows": 0}
    import audio_s4q as AU
    for i, (k, r) in enumerate(samp[:40]):
        node, rows_, ps, vc, led = st.open(k)
        row = rows_[r]; e = led[row["row_key"]]
        w = A.a_audio(e)
        n = int(row["n_frames"])
        s2["rows"] += 1; s2["len_ok"] += w.shape == (n * 1920,)
        s2["c01_rows"] += int(row["chunk_index"]) > 0
        if i < 4:
            rr = subprocess.run([AU.FFMPEG_EXE, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", e["flac"], "-af", "pan=mono|c0=c0", "-f", "f32le",
                                 "-ar", "24000", "-"], capture_output=True, check=True)
            x = np.frombuffer(rr.stdout, np.float32)[int(e["frame0"]) * 1920:(int(e["frame0"]) + n) * 1920]
            s2["ffmpeg_checked"] += 1; s2["ffmpeg_eq"] += x.shape == w.shape and x.tobytes() == w.tobytes()
        fr = w.reshape(n, 1920)
        db = 10 * np.log10(np.maximum((fr.astype(np.float64) ** 2).mean(1), 1e-12))
        uf = np.zeros(n, bool); bf = np.zeros(n, bool)
        for ws_, m_ in ((row["words_user"], uf), (row["words_assistant"], bf)):
            for w_ in ws_:
                m_[int(w_["start_frame"]):min(n, int(w_["start_frame"]) + 4)] = True
        only_b = bf & ~uf
        if uf.any() and only_b.any():
            s2["gap_db"].append(float(np.median(db[uf]) - np.median(db[only_b])))
    g = np.asarray(s2.pop("gap_db"))
    s2["user_word_vs_assistant_only_db_gap"] = {"min": float(g.min()), "median": float(np.median(g)), "max": float(g.max()), "n": len(g)}
    R["2_a_audio"] = s2
    print("[2 A audio]", s2, flush=True)
    # ---------------------------------------------------------------- 3 MICROS + 4 PROMPTS
    PL = A.make_plan(64)
    dg = hashlib.sha256(json.dumps(PL["items"]).encode()).hexdigest()[:16]
    s3 = {"items": 0, "T_ok": 0, "A_empty": 0, "prefix_A_empty": 0, "vin_ok": 0, "clip_eq": 0, "clips": 0, "targets_B_ok": 0, "voice": 0, "persona": 0,
          "light": 0, "cb0": 0, "P_hist": {}}
    cfgs = [("train", A.aloe_cfg(), PL["items"]), ("eval", A.eval_cfg(), PL["eval"])]
    for name, cfg, its in cfgs:
        for j in rng.choice(len(its), size=min(a.n // 2, len(its)), replace=False):
            it = its[int(j)]
            m = st.conv(it, cfg, int(j) % 64, audio=False)
            mic = A.as_micro(m)
            Pm, n = int(m["P"]), int(m["vin"][1])
            k, r, _ = it
            row = st.open(int(k))[1][int(r)]
            s3["items"] += 1
            s3["T_ok"] += len(m["pid"]) == Pm + n + D.CD and n == int(row["n_frames"])
            s3["A_empty"] += bool((m["ids"][0][Pm:] == D.PAD_ID).all() and (m["depth"][0] < 0).all() and not m["present"][0][Pm:].any())
            s3["prefix_A_empty"] += bool((m["ids"][0][:Pm] == D.PAD_ID).all() and not m["present"][0][:Pm].any())
            s3["vin_ok"] += tuple(int(x) for x in mic["vin"]) == (Pm, n)
            s3["targets_B_ok"] += bool((m["sel"] >= Pm).all() and (m["cb0"][:Pm + D.CD] < 0).all())
            s3["voice"] += int(m["voice_clip"]); s3["persona"] += int(m["persona_used"])
            if name == "train":
                s3["light"] += len(m["light"]); s3["cb0"] += int((m["cb0"] >= 0).sum())
            s3["P_hist"][str(Pm)] = s3["P_hist"].get(str(Pm), 0) + 1
            if int(m["voice_clip"]):
                vid, cd, _ = st.clip(int(k), int(r), row["row_key"])
                L = cd.shape[1]
                got = (np.asarray(m["depth"][1][4 + D.CD:4 + D.CD + L]) - np.arange(32)[None] * 2048).T
                s3["clips"] += 1; s3["clip_eq"] += bool(np.array_equal(got, np.asarray(cd, np.int64))) and Pm == 5 + L + 2 + D.CD
    s3["light_ratio_train_sample"] = s3.pop("light") / max(s3.pop("cb0"), 1)
    s3["P_hist"] = dict(sorted(s3["P_hist"].items(), key=lambda x: -x[1])[:8])
    R["3_micros"] = s3
    R["4_prompts"] = {"plan_digest": dg, "plan_digest_stores_json": sj["plan"]["digest"], "steps": PL["steps"], "n_train_rows": PL["n_train_rows"],
                      "n_eval_rows": PL["n_eval_rows"], "dry_run": sj["plan"]["prompt_fractions"], "realised": sj["plan"]["realised"],
                      "sample_voice_frac": s3["voice"] / max(s3["items"], 1), "sample_persona_frac": s3["persona"] / max(s3["items"], 1)}
    print("[3 micros]", s3, flush=True)
    print("[4 prompts]", R["4_prompts"], flush=True)
    # ---------------------------------------------------------------- 5 VOXTRAL (CPU)
    if not a.skip_vox:
        import torch
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
        import voxtral_front as VF
        k, r = samp[0]
        node, rows_, ps, vc, led = st.open(k)
        row = rows_[r]; e = dict(led[row["row_key"]])
        n = 200
        uw = [int(w["start_frame"]) for w in row["words_user"]]
        f0 = max(0, min(uw[0] if uw else 0, int(row["n_frames"]) - n))
        e.update(frame0=int(e["frame0"]) + f0, n_frames=n)
        w = torch.from_numpy(A.a_audio(e))
        t1 = time.time()
        fe = VF.VoxtralFront(device="cpu", attn_implementation="sdpa")
        lat = fe.encode(w, n)
        R["5_voxtral"] = {"row": row["row_key"], "window_frames": [f0, f0 + n], "latent_shape": list(lat.shape), "expect": [1, 4 * n, 1280],
                          "finite": bool(torch.isfinite(lat).all()), "rms": float(lat.float().pow(2).mean().sqrt()), "cpu_s": time.time() - t1}
        print("[5 voxtral]", R["5_voxtral"], flush=True)
    # ---------------------------------------------------------------- 6 CONFIG (each trainer imported in a fresh interpreter)
    R["6_cfg"] = {}
    for run in [x for x in a.runs.split(",") if x]:
        Q = f"{P}/si20k-vox-{run}"
        env = dict(os.environ)
        env.update(env_file(f"{Q}/si20k_vox_{run}.env"))
        env.update(QQA_STEPS=str(PL["steps"]), QQA_EXPORT_STEPS=",".join(str(x) for x in sj["plan"]["export_steps"]), QQA_EXPORT=f"{Q}/export-{PL['steps']}",
                   QQA_REF_PLAN_DIGEST=dg, ALOE_SI20K_SRC=a.stores, ALOE_SI20K_STORE_ROOT=a.stores, ALOE_SI20K_STORES_JSON=f"{a.stores}/STORES.json",
                   RANK="0", WORLD_SIZE="64", LOCAL_RANK="0", LOCAL_WORLD_SIZE="8", ALOE_LNODES="0;1;2;3;4;5;6;7", QQA_RUN=f"si20k-vox-{run}-jCPUCHECK",
                   QQA_LOG_ROOT=f"/tmp/si20k-cpucheck-{run}",
                   QQA_INIT={"ppm": f"{P}/ckpt-infer/ppm-v1/local-11488", "intl": f"{P}/next-arch/ckpt-weka/ppm-next/w-6242"}[run])
        code = (f"import sys; sys.path.insert(0, '{Q}/code'); sys.path.insert(0, '{Q}/code/lib'); import train_si20k_vox as T; "
                f"n, bad = T.cfg_diff_si20k(); print('CFGDIFF', n, bad)")
        rr = subprocess.run([sys.executable, "-I", "-c", code], env=env, capture_output=True, text=True, cwd="/tmp")
        txt = rr.stdout + rr.stderr
        R["6_cfg"][run] = {"rc": rr.returncode, "diff": [ln for ln in txt.splitlines() if ln.startswith("  ok ") or ln.startswith("  BAD ")],
                           "summary": [ln for ln in txt.splitlines() if "CFGDIFF" in ln or "[cfg-diff]" in ln][-2:], "tail": txt.splitlines()[-5:] if rr.returncode else []}
        print(f"[6 cfg {run}] rc {rr.returncode}", "\n".join(R["6_cfg"][run]["diff"]), R["6_cfg"][run]["summary"], R["6_cfg"][run]["tail"], flush=True)
    R["elapsed_s"] = time.time() - t0
    json.dump(R, open(a.out, "w"), indent=1, default=str)
    print("[done]", a.out, f"{R['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
