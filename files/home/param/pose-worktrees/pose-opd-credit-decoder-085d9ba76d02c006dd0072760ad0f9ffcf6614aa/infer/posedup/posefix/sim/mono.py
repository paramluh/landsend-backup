#!/usr/bin/env python3
"""posefix MONOPLEX arm (10-25, user: "monoplex generations from the exact same prefill"): the user's pose monoplex (aloe-pose-mono-8020
export-357) through its own inference engine (ppm/aloe-pose-monoplex-infer: ppm/fast_ppm.py FastPPM + infer.py's system_block / assemble),
with EXACTLY the duplex replays' prefill: system block = the actor's voice clip (gt/mimi_local [:, :125], no persona, = the live preset),
then from conversation frame 0 everything is sampled (lane cold PAD at frame 0, as the live engines) except B's POSE on pose frames
g < N, forced to the actor's real gt/pose_tokens (N = 2 x round(S / 0.16) + 2, = stage-next server.pose_prefill_codes). Channel A is
empty (monoplex). Output per run = replay.py's format: <name>.npz (pose [F, 32] local from code frame 0, mimi, tok, b audio),
<name>.feed.jsonl (videort feed), <name>.json.
usage: python mono.py <ck> <runs.json> <out dir>   (run = {name, seed, pose_prefill_s, frames, person, params})"""
import base64
import json
import os
import sys
import time

MI = "/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer"
sys.path.insert(0, MI)
sys.path.insert(0, f"{MI}/ppm")
os.environ.setdefault("PPM_CD", "2")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import infer as MIF  # noqa: E402   (system_block, assemble, TONIGHT)
import fast_ppm as FP  # noqa: E402
import ppm_io as IO  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

CD, OFF, PAD, DEPTH, SPF = FP.CD, FP.OFF, FP.PAD, FP.DEPTH, 1920
PEOPLE = ["/data/home/amartyavarma/video/demo-data/prompts", "/data/home/amartyavarma/video/dmd-rt/actors/pkg"]
# pose T1 / top-k 30 (cb0 + light), CFG off, no rep-pen; text / audio = the fantastic set as far as this engine has the knobs
# (no factorised text / PAD bias here: joint lane sampling with top-k 50, min-p 0.05 abs, freq pen 0.25 / 50)
MONO_PARAMS = dict(MIF.TONIGHT, text_temp=1.0, text_topk=50, text_topp=1.0, min_p=0.05, min_p_rel=0, freq_pen=0.25, freq_win=50,
                   cb0_temp=1.0, cb0_topk=30, light_temp=1.0, light_topk=30, pose_temp=1.0, pose_topk=30, pose_light_temp=1.0,
                   pose_light_topk=30, pose_cfg_on=0)


def log(*a):
    print(f"[mono {time.strftime('%H:%M:%S')}]", *a, flush=True)


def person_file(person, rel):
    for r in PEOPLE:
        if os.path.isfile(f"{r}/{person}/{rel}"):
            return f"{r}/{person}/{rel}"
    raise FileNotFoundError((person, rel))


def main():
    ck, runs, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
    os.makedirs(out, exist_ok=True)
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    torch.cuda.set_device(devs[-1])
    nmax = max(int(r.get("frames", 1500)) for r in runs)
    with torch.inference_mode():
        m, ldev, cfgs, binfo = FP.build_model(ck, devs)
        assert binfo["pose"] and binfo["decp_bb"], binfo
        mimi = IO.Mimi(devs[-1])
        max_s = nmax / 12.5 + 2
        p_max = 2048
        nfr = int(round(max_s * 12.5)) + 8
        sil = mimi.encode(np.zeros((p_max + nfr) * SPF, np.float32))[:, :p_max + nfr]
        eng = FP.FastPPM(m, ldev, cfgs, devs, n_inst=1, cfg_rows=True, max_s=max_s, mimi_stream=False, silence=sil)
        eng.capture()
        log(f"model step {binfo['step']} loaded + captured")
        for r in (runs[::-1] if os.environ.get("POSEFIX_REVERSE") else runs):
            name = r["name"]
            if os.path.isfile(f"{out}/{name}.json") and not os.environ.get("POSEFIX_REDO"):
                log(f"skip {name}")
                continue
            try:                                                       # claim: several instances share one run list
                os.close(os.open(f"{out}/{name}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                log(f"skip {name} (claimed)")
                continue
            person = r.get("person", "01_ab63d745")
            seed = int(r["seed"])
            n = int(r.get("frames", 1500))
            if r.get("conv_prefill"):                                  # CONVERSATION OPENING: B's voice clip from the conv's own system block
                zc = np.load(r["conv"])
                K = int(r.get("side", 0))
                d1 = zc[f"pre_depth_{K}"][1]
                clip = (d1[d1[:, 0] >= 0] - OFF).T.astype(np.int64)
                assert clip.shape[0] == DEPTH and clip.shape[1] >= 38, clip.shape
            else:
                clip = np.asarray(load_file(person_file(person, "gt/mimi_local.safetensors"))["codes"], np.int64)[:, :125]
            sp = MIF.system_block(clip, None)
            pf = float(r.get("pose_prefill_s", 0))
            pp = None
            if pf > 0:
                c = np.asarray(load_file(person_file(person, "gt/pose_tokens.safetensors"))["codes"], np.int64)
                pp = c[:, :min(c.shape[1], 2 * int(round(pf / 0.16)) + 2)].T.copy()
            N = 0 if pp is None else len(pp)
            lane = np.full(n + CD, PAD, np.int64)
            F0 = 0
            if r.get("conv_prefill"):                                  # B's real opening: lane frames 0..N+1, Mimi + pose 0..N-1
                N = min(len(zc[f"laneB_{K}"]) - CD, 2 * int(round(float(r["conv_prefill"]["s"]) / 0.16)) + 2)
                lane[:N + CD] = zc[f"laneB_{K}"][:N + CD]
                au = (zc[f"depB_{K}"][CD:CD + N] - OFF).T
                po = (zc[f"pdepB_{K}"][CD:CD + N] - OFF).T
                assert (au >= 0).all() and (po >= 0).all(), "opening has gaps"
                x = MIF.assemble(sp, lane, au, po, n, N)
                F0, pp = N + CD, po.T
            elif r.get("full_prefill"):                                # FULL PREFILL (pose + words + audio) = infer.py's package prefill path
                bz = np.load(r["full_prefill"]["bundle"])
                N = min(150, 2 * int(round(float(r["full_prefill"]["s"]) / 0.16)) + 2)
                lane[:N + CD] = bz["lane"][:N + CD]
                x = MIF.assemble(sp, lane, bz["audio"][:, :N], bz["pose"][:, :N], n, N)
                F0, pp = N + CD, bz["pose"][:, :N].T
            else:
                x = MIF.assemble(sp, lane, np.zeros((DEPTH, 0), np.int64), np.zeros((DEPTH, 0), np.int64), n, 0)
            P = int(x["P"])
            params = dict(MONO_PARAMS, **r.get("params", {}))
            t0 = time.time()
            eng.start([dict(x, a_mode="empty")], seed, params, f0=F0)
            for f in range(F0, n + CD):
                g = f - CD
                frc = {"pose": (pp[g] + OFF)[None]} if (pp is not None and 0 <= g < N and not F0) else None
                eng.step(f, force=frc, cold_pad=not F0)
            S = eng.readback(0, P + n + CD)
            wall = time.time() - t0
            lane_b, dep_b, pdep = S["lane_b"], S["dep_b"], S["pdep"]
            c1 = (dep_b[P + CD:P + CD + n] - OFF)
            p1 = (pdep[P + CD:P + CD + n] - OFF)
            assert c1.min() >= 0 and p1.min() >= 0, "codes missing"
            if pp is not None:
                assert (p1[:N] == pp).all(), "pose prefill != actor tokens"
            w = mimi.decode(c1.T.astype(np.int64))
            w = np.asarray(w, np.float32).reshape(-1)
            np.savez_compressed(f"{out}/{name}.npz", pose=p1.astype(np.int16), mimi=c1.astype(np.int16), tok=lane_b[P:P + n], b=w)
            with open(f"{out}/{name}.feed.jsonl", "w") as fo:
                fo.write(json.dumps({"type": "start", "session": name, "model": "posefix-mono", "voice": person, "fps": 12.5, "person": person,
                                     "prefill_s": float(r.get("video_prefill_s", pf))}) + "\n")
                for k in range(n):
                    seg = w[k * SPF:(k + 1) * SPF] if len(w) >= (k + 1) * SPF else np.zeros(SPF, np.float32)
                    pcm = np.clip(seg * 32767, -32768, 32767).astype("<i2").tobytes()
                    fo.write(json.dumps({"type": "f", "session": name, "f": k, "t": round(k * 0.08, 3), "pose": p1[k].tolist(),
                                         "mimi": c1[k].tolist(), "pcm_b64": base64.b64encode(pcm).decode("ascii")}) + "\n")
                fo.write(json.dumps({"type": "end", "session": name}) + "\n")
            gen = p1[N:, 0].astype(int)
            words = IO.lane_words(lane_b[:P + n], P, P, P)
            summ = {"name": name, "model": "monoplex export-357", "person": person, "seed": seed, "P": P, "frames": n, "pose_prefill_frames": N,
                    "pose_prefill_s": N / 12.5, "params": params, "wall_s": round(wall, 1), "ms_per_frame": round(1e3 * wall / (n + CD), 1),
                    "gen_cb0_change": float((gen[1:] != gen[:-1]).mean()) if len(gen) > 1 else None, "gen_cb0_distinct": int(len(np.unique(gen))),
                    "words": " ".join(x_["text"] for x_ in words)[:400]}
            json.dump(summ, open(f"{out}/{name}.json", "w"), indent=1)
            log(f"run {name}: " + json.dumps({k: summ[k] for k in ("frames", "pose_prefill_frames", "gen_cb0_change", "gen_cb0_distinct", "ms_per_frame")}))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
