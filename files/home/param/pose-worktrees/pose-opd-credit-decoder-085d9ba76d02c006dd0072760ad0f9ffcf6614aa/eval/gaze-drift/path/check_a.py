"""Goal A numeric checks: GT render path vs generated render path inputs. CPU only. Read-only on inputs."""
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

PR = "/data/home/amartyavarma/video/demo-data/prompts"
E = "/data/home/amartyavarma/video/studio-posemono/sam8b-eval-cfg"
S20 = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20"
RQ = "/data/home/amartyavarma/ppm/ppm-pose-server/requests"
ROWS = dict([("S-05", "10_27286bdd"), ("S-13", "06_74807fe0"), ("S-15", "10_27286bdd"), ("S-20", "19_47473a46"), ("E-03", "03_b437950e"),
             ("E-05", "05_e25c6859"), ("E-09", "09_d9ef3c94"), ("E-12", "12_72cb45d3"), ("E-13", "13_956c09af"), ("E-15", "15_d1e1322c"),
             ("E-18", "18_1ffa9083")])
out = {"rows": {}}
first = True
for tag in ("fs", "d1", "d2", "d3"):
    mp = json.load(open(f"{S20}/control/xs17641_{tag}.json"))
    for row, (rid, k) in mp.items():
        if row not in ROWS:
            continue
        idn = ROWS[row]
        sd = f"{RQ}/{rid}/s{k}"
        z = np.load(f"{sd}/codes.npz")
        if first:
            out["codes_npz_keys"] = {kk: [list(z[kk].shape), str(z[kk].dtype)] for kk in z.files}
            first = False
        li = load_file(f"{PR}/{idn}/llm_inputs.safetensors")
        gtp = load_file(f"{PR}/{idn}/gt/pose_tokens.safetensors")
        gk = list(gtp.keys())
        g = gtp["codes"].numpy().astype(np.int64)
        rec_extra = dict(pkg_tokens_frame0_vs_llm_frame0_maxabs=float((gtp["frame0_pose"] - li["frame0_pose"]).abs().max()),
                         gen_global_state=[float(x) for x in np.asarray(z["global_state"]).reshape(-1)],
                         pkg_global_state=[float(x) for x in li["global_state"].reshape(-1)],
                         llm_prefill_pose_eq_pkg_codes=bool(np.array_equal(li["prefill_pose"].numpy(), g[:, :25])),
                         llm_prefill_audio_eq_gen=bool(np.array_equal(li["prefill_audio"].numpy(), z["audio"][:, :25])))
        rz = np.load(f"{E}/runs/{row}/codes.npz")
        P = int(z["prefill_frames"])
        pose = z["pose"]
        rec = dict(rid=rid, k=k, clip=idn, prefill=P, pose_shape=list(pose.shape), audio_shape=list(z["audio"].shape),
                   gt_tokens_key=gk, gt_tokens_shape=list(g.shape), **rec_extra)
        f0a = z["frame0_pose"].astype(np.float64)
        f0b = li["frame0_pose"].double().numpy()
        rec["frame0_pose_shape"] = [list(f0a.shape), list(f0b.shape)]
        rec["frame0_pose_maxabs"] = float(np.abs(f0a.reshape(-1) - f0b.reshape(-1)).max()) if f0a.size == f0b.size else "size mismatch"
        rec["prefill_eq_pkg_gt_tokens"] = bool(np.array_equal(pose[:, :P], g[:, :P]))
        rec["prefill_eq_run_gt_pose"] = bool(np.array_equal(pose[:, :P], rz["gt_pose"][:, :P]))
        rec["pkg_gt_eq_run_gt_pose_full"] = bool(np.array_equal(g, rz["gt_pose"][:, :g.shape[1]]))
        rec["prefill_mismatch_count"] = int((pose[:, :P] != g[:, :P]).sum())
        # frames after prefill matching GT (should be ~0)
        T = min(pose.shape[1], g.shape[1])
        rec["post_prefill_token_match_frac"] = float((pose[:, P:T] == g[:, P:T]).mean())
        # audio
        ga = load_file(f"{PR}/{idn}/gt/mimi_local.safetensors")["codes"].numpy()
        rec["gt_mimi_shape"] = list(ga.shape)
        rec["audio_prefill_eq_gt_mimi_local"] = bool(np.array_equal(z["audio"][:, :P], ga[:, :P]))
        rec["audio_prefill_match_frac"] = float((z["audio"][:, :P] == ga[:, :P]).mean())
        rec["audio_post_match_frac"] = float((z["audio"][:, P:min(z['audio'].shape[1], ga.shape[1])] == ga[:, P:min(z['audio'].shape[1], ga.shape[1])]).mean())
        rec["run_gt_audio_eq_pkg_mimi_local"] = bool(np.array_equal(rz["gt_audio"], ga[:, :rz["gt_audio"].shape[1]]))
        # vdata latents: generated vs GT vdata
        gv = load_file(f"{sd}/vdata/pose_latents_electron_v3.safetensors")["latents"].float()
        tv = load_file(f"{E}/vdata/GT-{idn}/pose_latents_electron_v3.safetensors")["latents"].float()
        pk = load_file(f"{PR}/{idn}/gt/pose_latents_electron_v3.safetensors")["latents"].float()
        rec["gen_pose_latents_T"], rec["gt_pose_latents_T"] = int(gv.shape[0]), int(tv.shape[0])
        nl = 2 * P // 4 + 1   # latents fully inside the prefill (causal VAE: latent i covers pixel frames 4i-3..4i)
        per = []
        for i in range(min(gv.shape[0], tv.shape[0])):
            per.append(float((gv[i] - tv[i]).norm() / tv[i].norm()))
        rec["pose_latent_rel_diff_gen_vs_gtvdata_per_latent"] = [round(x, 4) for x in per]
        rec["prefill_latents_n"] = nl
        rec["prefill_latents_max_rel_diff"] = max(per[:nl])
        rec["post_latents_mean_rel_diff"] = float(np.mean(per[nl + 1:])) if len(per) > nl + 1 else None
        rec["gtvdata_vs_pkg_latents_max_rel"] = max(float((tv[i] - pk[i]).norm() / pk[i].norm()) for i in range(min(tv.shape[0], pk.shape[0])))
        # sink
        s_gen = load_file(f"{sd}/vdata/video_latents.safetensors")["latents"][0].float()
        s_gt = load_file(f"{E}/vdata/GT-{idn}/video_latents.safetensors")["latents"][0].float()
        s_pk = load_file(f"{PR}/{idn}/sink_latent.safetensors")["latents"][0].float()
        rec["sink_gen_eq_gt"] = bool(torch.equal(s_gen.reshape(-1), s_gt.reshape(-1)))
        rec["sink_gen_eq_pkg_sink"] = bool(torch.equal(s_gen.reshape(-1), s_pk.reshape(-1)))
        # status
        vs = json.load(open(f"{sd}/video_status.json"))
        gs = json.load(open(f"{S20}/vrefs/GT-{idn}/status.json"))
        rec["status_diff"] = {kk: [vs.get(kk), gs.get(kk)] for kk in sorted(set(vs) | set(gs))
                              if kk not in ("node", "gpu", "worker", "vdata", "package", "identity", "sampling_seconds", "total_seconds", "decode_seconds", "peak_allocated_gib") and vs.get(kk) != gs.get(kk)}
        meta = json.load(open(f"{sd}/meta.json"))
        rec["sink_path"] = meta.get("sink")
        rec["pose_from_frame0"] = meta.get("pose_from_frame0")
        req = json.load(open(f"{RQ}/{rid}/request.json"))
        rec["video_seed"] = req.get("video_seed")
        out["rows"][f"{tag}/{row}"] = rec
        print(tag, row, idn, "f0 maxabs", rec["frame0_pose_maxabs"], "prefill==GT", rec["prefill_eq_pkg_gt_tokens"], rec["prefill_eq_run_gt_pose"],
              "audio prefill==GT", rec["audio_prefill_eq_gt_mimi_local"], f"{rec['audio_prefill_match_frac']:.3f}",
              "lat prefill maxrel", round(rec["prefill_latents_max_rel_diff"], 4), "post", round(rec["post_latents_mean_rel_diff"], 3),
              "sink eq", rec["sink_gen_eq_gt"], "Tgen/Tgt", rec["gen_pose_latents_T"], rec["gt_pose_latents_T"], "statusdiff", rec["status_diff"], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
