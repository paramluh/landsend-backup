#!/usr/bin/env python3
"""pdsamples DUO generation for the POSE-DUPLEX Aloe FT (job 30074 export) = ppm/ppmvox/samples/code/gen_duo.py (the PPMVOX Voxtral-A duo)
on the pose engine ppm/fastinfer/fast_ppm.py FastPPM (imported unchanged: the live Voxtral demo engine + channel-B pose; model code =
aloe-m1vox-ft/code model_ppm, pose maths = ppm/code). TWO FastPPM instances (NI = 1 each, Voxtral front, pose) on ONE weight copy (2 GPUs),
one per speaker: state k plays speaker k as channel B (lane + NEW_WORD + Mimi + POSE); its channel A = streaming Voxtral of the OTHER
state's audio: the partner's real raw-gain audio for the teacher-forced prefill frames, then the Mimi decode of the partner state's OWN
GENERATED codes (both sides generated; the real partner track is never used after the prefill; pose is never cross-fed, as in training).
  LINE DELAY: A(t) = partner(t - 5832 samples) = 243 ms everywhere (minimum causal: code delay 2 frames + 72-sample Voxtral lookahead).
PREFILL: the first 125 frames (10 s) of the real conversation teacher-forced on both states (B lane + B Mimi + B pose from the training micro;
  codes / pose at delay 2), then generated to the row's 1500 frames (120 s).
PROMPT: the exact poseduplex training micro's system block with B's voice clip (forced on), no persona (prep_convs.py).
SAMPLING: temperature 1 on every stream (text joint draw, no top-k / top-p / min-p), audio top-k 30 (cb0 + cb1-31), pose top-k 30 (cb0 +
  cb1-31), no repetition / frequency penalty, CFG off (pose CFG off, no uncond rows), bf16.
usage: python gen_pd.py --ck <export dir> --conv <convs/cN.npz> --out <runs/B-00N> --seed S"""
import argparse
import json
import math
import os
import sys
import time

FI = "/data/home/amartyavarma/ppm/fastinfer"
sys.path.insert(0, FI)
import fast_ppm as FP  # noqa: E402   (the pose engine, unchanged)
import numpy as np  # noqa: E402
import torch  # noqa: E402

IP, log = FP.IP, FP.log
SPF, LOOK, CD = 1920, 72, FP.CD
DELAY = 3 * SPF + LOOK
NPRE = 125
CBP = 2048                                                       # pose codebook size (global id = cb x 2048 + code)
PARAMS = dict(FP.DEFAULTS, text_temp=1.0, text_topk=0, text_topp=1.0, min_p=0.0, freq_pen=0.0, text_rep=1.0, cb0_temp=1.0, cb0_topk=30,
              light_temp=1.0, light_topk=30, audio_rep_pen=1.0, allow_eob=0, free_text=0, word_greedy=0, pose_temp=1.0, pose_topk=30,
              pose_light_temp=1.0, pose_light_topk=30, pose_cfg_on=0, pose_rep=1.0, adaptive_rep_on=0, audio_rep_speaking=1.0,
              audio_rep_silent=1.0, text_rep_speaking=1.0, text_rep_silent=1.0, pose_rep_speaking=1.0, pose_rep_silent=1.0)


def s_for(ck, meta):
    """s = the trainer's fp32 s at the export step (metrics train/vox_a_scale of the export's own run) or exp(fp32 theta in meta)."""
    if meta.get("vox_a_scale"):
        return float(meta["vox_a_scale"]), "meta vox_a_scale"
    c = meta.get("cfg") or {}
    mp = f"{c.get('log_root')}/{c.get('run_tag')}/metrics.jsonl" if c.get("log_root") and c.get("run_tag") else None
    if mp and os.path.isfile(mp):
        best = None
        for ln in open(mp):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if int(d.get("step", -1)) == int(meta["step"]) and "train/vox_a_scale" in d:
                best = float(d["train/vox_a_scale"])
        if best is not None:
            return best, f"trainer fp32 s at step {meta['step']} ({mp})"
    if meta.get("vgain_theta_fp32") is not None:
        return math.exp(float(meta["vgain_theta_fp32"])), "exp(meta vgain_theta_fp32)"
    return None, "exp(checkpoint theta)"


def mimi_wav(eng, codes):
    with torch.no_grad(), torch.cuda.device(eng.d0):
        return eng.mimi.decode(torch.as_tensor(np.asarray(codes, np.int64), device=eng.d0)[None]).audio_values[0, 0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", required=True)
    ap.add_argument("--conv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", default="20261025")
    ap.add_argument("--frames", type=int, default=0, help="0 = the row's full length (tests: fewer)")
    ap.add_argument("--params", default="", help="JSON sampling overrides of PARAMS (A/B arms, e.g. {\"pose_topk\": 500})")
    a = ap.parse_args()
    assert os.environ.get("GEMM_PRECISION", "bf16") == "bf16"
    t0 = time.time()
    meta_ck = json.load(open(f"{a.ck}/meta.json"))
    s_val, s_src = s_for(a.ck, meta_ck)
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    m, ldev, cfgs, info = FP.build_model(a.ck, devs, s_value=s_val)
    assert info["pose"] and info["vadapter"], ("pose-duplex export must carry pose parts and the Voxtral adapter", info)
    with torch.cuda.device(devs[0]):
        mm = FP.VS.MimiStream(devs[0]).mimi                       # one Mimi for both states' streaming decoders (+ clip previews)
    eng = [FP.FastPPM(m, ldev, cfgs, devs, n_inst=1, max_s=130.0, mimi=mm)]
    eng.append(FP.FastPPM(m, ldev, cfgs, devs, n_inst=1, max_s=130.0, mimi=mm, enc=eng[0].enc, silence=eng[0].sil))
    for e in eng:
        e.mimi = mm
    for e in eng:
        e.capture()
    log(f"model {a.ck} step {info['step']} s {info.get('s')} ({s_src}); D33 alpha {info['decp_bb_alpha']}; 2 states ready in {time.time() - t0:.0f}s")
    convs, outs, seeds = a.conv.split(","), a.out.split(","), [int(x) for x in str(a.seed).split(",")]
    assert len(convs) == len(outs), ("--conv / --out lists differ", convs, outs)
    for i, (cv, od) in enumerate(zip(convs, outs)):               # several conversations per warm engine (one after the other)
        run(eng, info, s_src, a.ck, cv, od, seeds[min(i, len(seeds) - 1)], a.frames, json.loads(a.params) if a.params else {})


def run(eng, info, s_src, ck, conv, out, seed, frames, over):
    a = argparse.Namespace(ck=ck, conv=conv, out=out, seed=seed)
    bad = sorted(set(over) - set(PARAMS))
    assert not bad, ("unknown sampling overrides", bad)
    prm = dict(PARAMS, **over)
    z = np.load(a.conv)
    cj = json.load(open(a.conv.replace(".npz", ".json")))
    NF = int(len(z["laneB_0"])) if not frames else min(int(frames), int(len(z["laneB_0"])))
    assert len(z["stem_0"]) >= NF * SPF + LOOK and len(z["stem_1"]) >= NF * SPF + LOOK
    insts = [{"P": int(z[f"pre_ids_{k}"].shape[1]), "ids": z[f"pre_ids_{k}"], "depth": z[f"pre_depth_{k}"], "present": z[f"pre_present_{k}"],
              "pdepth": z[f"pre_pdepth_{k}"], "a_mode": "voxtral"} for k in (0, 1)]
    stems = [z["stem_0"], z["stem_1"]]
    NFA = NPRE - CD
    part = [np.zeros((NF + 4) * SPF, np.float32) for _ in (0, 1)]
    for k in (0, 1):
        part[k][:NFA * SPF] = stems[k][:NFA * SPF]
    filled = [NFA * SPF, NFA * SPF]
    wr = [0, 0]
    for k in (0, 1):
        eng[k].start([insts[k]], a.seed + 10 * k, prm, f0=0)
    Ps = [eng[k].P[0] for k in (0, 1)]
    toks, pp, pn, ys = [[], []], [[], []], [[], []], [[], []]
    t1 = time.time()
    prog = f"{a.out}.progress"
    for f in range(NF):
        for k in (0, 1):
            hi = (f + 1) * SPF + LOOK
            src_hi = hi - DELAY
            assert src_hi <= filled[1 - k], ("partner audio not yet available", f, k, src_hi, filled[1 - k])
            lo = wr[k]
            x = np.zeros(hi - lo, np.float32)
            s0 = lo - DELAY
            if src_hi > 0:
                a0 = max(s0, 0)
                x[a0 - s0:] = part[1 - k][a0:src_hi]
            eng[k].write_a(lo, x)
            wr[k] = hi
            force = None
            if f < NPRE:
                force = {"lane": [int(z[f"laneB_{k}"][f])]}
                if f >= CD:
                    d, pd = z[f"depB_{k}"][f], z[f"pdepB_{k}"][f]
                    assert (d >= 0).all(), ("prefill B codes missing", k, f)
                    force["codes"] = [d]
                    force["pose"] = [pd]
            eng[k].step(f, force=force)
        for k in (0, 1):
            tk, p_pad, p_nw, y = eng[k].result(f, f >= CD)
            toks[k].append(tk[0]); pp[k].append(p_pad[0]); pn[k].append(p_nw[0])
            if y is not None:
                ys[k].append(y[0])
                g = f - CD
                if g >= NFA:
                    part[k][g * SPF:(g + 1) * SPF] = y[0]
                    filled[k] = (g + 1) * SPF
        if f % 250 == 0 or f == NF - 1:
            el = time.time() - t1
            open(prog, "w").write(json.dumps({"frame": f, "of": NF, "wall_s": round(el, 1), "ms_per_frame": round(1000 * el / (f + 1), 1)}) + "\n")
    gen_s = time.time() - t1
    log(f"{NF} frames x 2 states in {gen_s:.0f}s ({1000 * gen_s / NF:.1f} ms / frame)")
    os.makedirs(a.out, exist_ok=True)
    nG = NF - CD                                                  # audio / pose frames 0 .. NF - CD - 1
    nA = nG * SPF
    names, sides = ("s1", "s2"), ("assistant", "user")             # prep: direction 0 = speaker 1 = data 'assistant' is B
    words, cards, audio, posec = {"gt": {}}, {}, {}, {}
    for k in (0, 1):
        rb = eng[k].readback(0, Ps[k] + NF)
        pdep = rb["pdep"][Ps[k] + CD:Ps[k] + NF]                  # pose of frames 0 .. nG - 1 (global ids, -1 none)
        loc = np.where(pdep >= 0, pdep - (np.arange(32) * CBP)[None], -1).astype(np.int16).T
        np.save(f"{a.out}/pose_{names[k]}.npy", loc)
        np.save(f"{a.out}/gt_pose_{names[k]}.npy", np.asarray(z[f"gtpose_{k}"])[:, :nG].astype(np.int16))
        posec[names[k]] = {"frames": int(nG), "missing_frac_gen": round(float((loc[0, NPRE:] < 0).mean()), 4) if nG > NPRE else None}
        out = np.concatenate(ys[k]).astype(np.float32)[:nA]
        IP.write_wav(f"{a.out}/{names[k]}.wav", out)
        IP.write_wav(f"{a.out}/gt_{names[k]}.wav", stems[k][:nA])
        dep = z[f"pre_depth_{k}"][1]
        ok = (dep >= 0).all(1)
        cw = mimi_wav(eng[0], (dep[ok] - FP.OFF[None]).T) if ok.any() else np.zeros(0, np.float32)
        IP.write_wav(f"{a.out}/clip_{names[k]}.wav", cw)
        cards[names[k]] = {"text": "", "n_tokens": 0, "clip": {"wav": f"clip_{names[k]}.wav", "seconds": round(len(cw) / 24000, 2)}}
        words[names[k]] = IP.lane_words(np.asarray(toks[k]), 0, 0, prompt_end=NPRE)
        words["gt"][names[k]] = [{"t": float(w[0]), "text": w[1], "prompt": float(w[0]) < NPRE / 12.5} for w in cj.get(f"words_{sides[k]}", [])]
        audio[f"{names[k]}_gen"] = IP.audio_stats(out[NPRE * SPF:])
        audio[f"gt_{names[k]}"] = IP.audio_stats(stems[k][:nA])
    json.dump(words, open(f"{a.out}/words.json", "w"))
    np.savez_compressed(f"{a.out}/lanes.npz", lane_s1=np.asarray(toks[0]), lane_s2=np.asarray(toks[1]), p_pad_s1=np.asarray(pp[0], np.float32),
                        p_pad_s2=np.asarray(pp[1], np.float32), p_nw_s1=np.asarray(pn[0], np.float32), p_nw_s2=np.asarray(pn[1], np.float32))
    meta = {"row": os.path.basename(a.out), "model": "Aloe pose-duplex FT (v5 w-18100 init)", "ckpt": a.ck, "step": int(info["step"]),
            "s": info.get("s"), "s_src": s_src, "prov": {"row_key": cj["row_key"], "chan_B_data_side": sides[0]}, "dirs": cj.get("dirs"),
            "prompt_s": NPRE / 12.5, "gen_s": round((nG - NPRE) / 12.5, 2), "P": Ps, "cards": cards,
            "order": [{"kind": "clip", "spk": "s1"}, {"kind": "clip", "spk": "s2"}], "audio": audio, "pose": posec,
            "gen_words": {n: sum(1 for w in words[n] if not w["prompt"]) for n in names},
            "gen_text": {n: " ".join(w["text"] for w in words[n] if not w["prompt"]) for n in names},
            "seed": [a.seed, a.seed + 10], "engine_params": prm, "param_overrides": over, "line_delay_ms": round(1000 * DELAY / 24000, 1),
            "sampling": {"text_temp": 1.0, "text_topk": 0, "text_min_p": 0, "audio_temp": prm["cb0_temp"], "audio_topk": prm["cb0_topk"], "pose_temp": prm["pose_temp"], "pose_topk": prm["pose_topk"],
                         "pose_light_temp": prm["pose_light_temp"], "pose_light_topk": prm["pose_light_topk"],
                         "rep_pen": None, "cfg": None, "pose_cfg": None, "dtype": "bf16"},
            "engine": "fastinfer/fast_ppm.py FastPPM x 2 (NI 1, Voxtral front, pose; shared weights), duo", "gen_wall_s": round(gen_s, 1),
            "made": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1, default=str)
    open(f"{a.out}/DONE", "w").write("ok\n")
    log(f"[duo] {a.out}: s1 {audio['s1_gen']} | s2 {audio['s2_gen']} | pose {posec}\n  s1: {meta['gen_text']['s1'][:300]}\n  s2: {meta['gen_text']['s2'][:300]}")


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
