"""PPM TRAINER (pose + podcast + text-duplex; model_ppm, data_ppm). torchrun / srun (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_*), FSDP2.

ONE OPTIMIZER STEP = gradient accumulation over PPM_GA_SEQ (default pose, speech, speech, text), EVERY rank the same task per micro
(the pose trainer's MIX_MODE=ga): each micro's loss / len(GA_SEQ); one clip + main AdamW step + LR-schedule step per optimizer step.
  pose    one pose row (channel B speaker): text_B CE + W_CB0 speech cb0 (4B speech tower) + W_PCB0 pose cb0 (4B pose tower)
          + speech decoder nanosteps + pose decoder nanosteps
  speech  one FULL 600 s podcast row + its system prompt (data_ppm): text_B CE + W_CB0 cb0_B + speech decoder nanosteps
  text    one <= 7,500-frame text-duplex pack: text_B CE only (pure CE, both tracks summed in the input; txtdup duplex_ce rendering)
NANOSTEPS (lightms_s3q = SAM's pose-light scheme, for BOTH 0.6B decoders): per micro, the decoder's frames are shuffled and split into K
  strided chunks; each chunk = fwd + bwd on the DETACHED tower output + clip + its own AdamW step (LR = sam_lr over decoder updates); the
  accumulated (size-weighted) input gradient x (W_CB x task weight / len(GA_SEQ)) then flows through tower + backbone in the micro's ONE
  backward. Decoders are not in the main optimizer.
CHECKPOINTS: DCP (model + main AdamW + both decoder AdamWs) + train_state.json, node-local PPM_SAVE_ROOT (/scratch); never /data.
D32 (10-12, PPM_POSE_HEAD=1; default off = old behaviour exactly): auxiliary direct pose cb0 head on the backbone (model_ppm.pose_head_ce),
  + W_POSE_HEAD x its CE on pose rows (same pairs / mask / weights as the pose tower's cb0 loss); own AdamW group 'pose_head' (LR
  PPM_LR_POSE_HEAD, clip PPM_CLIP_POSE_HEAD, linear warmup over PPM_POSE_HEAD_WARMUP steps from the head's FIRST step, then constant).
  Resume from a checkpoint without the head: head = zeros, its AdamW state fresh (load(): only these keys may be missing).
D33 (10-12, PPM_DECP_BB=1; default off = old behaviour exactly; TEMPORARY, user: "as a temporary measure to get the connection and then we
  detach from the backbone"): the backbone -> POSE decoder link model_ppm.decp_bb (Linear 5120 -> 1024, zero init): zb[T] = alpha x
  decp_bb(hn[T - 1]) added to the pose decoder's input slot 0. The nanosteps run on a DETACHED copy of zb (as of hv_p) and accumulate its
  input gradient (full-batch mean, |chunk| / n weights, as hv_p); run_micro adds (zb, acc_zb x w_task x W_POSE_CB / NM) to the backward
  roots (the hv_p rule), so the cb1..31 loss reaches decp_bb AND the backbone. Own main-AdamW group 'decp_bb' (LR PPM_LR_DECP_BB 2e-4,
  clip PPM_CLIP_DECP_BB 1.0, linear warmup over PPM_DECP_BB_WARMUP 100 steps from its FIRST step, then constant; as D32's head). alpha =
  PPM_DECP_BB_ALPHA (resume-changeable), optionally faded linearly to 0 over PPM_DECP_BB_FADE_STEPS steps from PPM_DECP_BB_FADE_FROM (-1 =
  off). Resume from a checkpoint without the link: decp_bb = zeros, AdamW state fresh (only its keys may be missing).
D35 (10-13, user OK; PPM_POSEHEAD_DETACH_FROM / PPM_POSEHEAD_DETACH_RAMP; default off (FROM -1) = D32 exactly): detach the D32 head from the
  backbone. The head reads model_ppm.posehead_in(hn, s) = hn.detach() + s x (hn - hn.detach()) (forward and the head's loss unchanged; its
  gradient into the backbone x s), s = 1 -> 0 linearly over RAMP steps from optimizer step FROM (posehead_bbgrad), then hn.detach() (full
  detach). The head keeps training (same LR / loss) as a probe; pose_head_cb0_ce still logged. One log line per step during the ramp.
D36 (10-14, user OK; PPM_POSEHEAD_OFF=1 with PPM_POSE_HEAD=1; default unset = D35 exactly, cfg unchanged): remove the D32 head from training.
  model.pose_head_on = False: no head forward / loss / backward (model_ppm), its two parameters in NO optimizer (frozen; asserted gradient-free
  every step), no pose_head_* eval / train stats, no gn_pose_head / lr_pose_head / pose_head_bbgrad_s. The module and its weights stay
  (saved in every checkpoint / export, unchanged) so it can be revived. Resume from a checkpoint whose head was training: weights restored,
  its saved AdamW state set aside (safety_ppm.load_local drop_opt: digest incl. the set-aside entries == saved), its LambdaLR entry removed
  (group names checked against the saved config). Revival (unset again later): weights as saved, AdamW fresh, LR warmup restarts from the
  revive step (as D32's ADDED path). Everything else (losses, LRs, schedules, data order, D31 / D33 / D34, eval sets) unchanged.
D37 (10-16, user request; PPM_POSE_HEAD=0 at a resume from a D36-off checkpoint): the unused head's MODULE removed (no weights / FSDP unit). Engineering
  only (it contributed nothing to forward / loss / gradients since D36). Resume: its saved weights set aside (safety_ppm.load_local drop_model:
  asserted absent from every optimizer section and rep.pt, digest incl. them == saved, so every other tensor + optimizer state is exact); a
  checkpoint whose head still TRAINS (not D36-off) is refused. Not revivable from later checkpoints (they no longer hold its weights).
8,1,1 SCHEDULE (code-811, 10-20 PT; user 21:00: "focus almost all the gains of pose into the tower and light pose and make sure the backbone
  doesn't forget the other tasks"; PPM_S811=1, every knob explicit in prod_811.env; README_811.md). Unset = the production trainer exactly.
  One optimizer step = the micros of PPM_GA_SEQ (multiplier syntax, e.g. pose*8,speech,text) with a per-micro backbone mode PPM_BB_MODE
  (full | frozen, e.g. full,frozen*7,full,full). full = the production micro (trunk forward + backward). frozen = the trunk runs LIVE,
  forward only under no_grad (no recompute, no trunk backward / reduce-scatter; model_ppm bb="frozen"), the towers get hn.detach();
  nothing is cached. EVERY micro trains its towers and its 0.6B light decoders (nanosteps unchanged: PPM_DEC_K updates per micro over
  100% of the micro's frames, asserted light_frac = pose_light_frac = 1.0; pose rows: speech tower + speech decoder too unless
  PPM_POSE_SPEECH_PARTS=full). Gradient coefficients per micro (s811.coefs): c_bb (backbone: trunk / spec / proj / audio groups), c_s
  (speech-tower side: tower_s + the speech decoder's input gradient), c_p (pose-tower side: tower_p, decp_bb, the pose decoder's input
  gradient). Default PPM_W_NORM=task: c = weight[task] / (micros of that task feeding the part) / Z, Z = PPM_BB_W_POSE + _SPEECH + _TEXT
  (tasks present); backbone weights PPM_BB_W_* (default 1 / 2 / 1), tower weights PPM_TW_W_POSE / _SPEECH (default 1 / 2) => with the
  defaults every part gets exactly PPM's per-task coefficients (PPM: each of pose, speech, speech, text / 4), the 8 pose micros' tower
  gradients averaged. A full micro's roots are c_bb x text CE + c_s x W_CB0 cb0 + c_p x W_POSE_CB0 pose cb0 (+ decoder input gradients x
  c_s / c_p), and the towers read hn through model_ppm.GradScale(c_bb / c_side) so the trunk receives exactly c_bb x the micro's gradient.
  Backbone gradients accumulate over the step's full micros, then ONE clip + AdamW step (main AdamW, same groups / LRs / clips / constant
  schedule as PPM). Tower-side groups (tower_s, tower_p, decp_bb): PPM_TOWER_CADENCE=step (default: accumulate over all micros, step with
  the backbone) | micro (clip + AdamW step after every micro that fed them; the backbone still steps once). Data (resume from a PPM
  checkpoint at step S0 = the switch; S0 and PPM's GA carried in every later checkpoint, cfg s811): text continues PPM's per-source pack
  order by global step; pose = the rest of PPM's epoch-1 order, then epoch 2, 3, .. = fresh node permutations (pose_s3q.plan, seed
  PPM_POSE_EP2_SEED + e - 2); speech = the rest of PPM's D2 stream, then epochs of the same rows (holdout excluded) permuted with
  PPM_SPEECH_EP2_SEED. The run never deletes the switch checkpoint dcp-S0 (PPM_PROTECT_SWITCH=1). Checks: PPM_811_CHECK_STEPS=N asserts
  for the first N steps that every frozen micro leaves the backbone gradients bit-identical, full micros change them, tower grads accumulate;
  PPM_811_LOG_ROWS=N logs the rows of the first N steps; PPM_811_EVAL_REF=<metrics.jsonl glob> compares the start eval with PPM's at S0.
G0COND (pose5, 10-27, user change #2: "condition every pose clip on its frame-0 root xy and shoulder width"): PPM_G0COND=1 -> model_ppm
  inp.g0 (Linear 8 -> 5120, no bias, ZERO init, OWN main-AdamW group 'g0cond' (main 10-27, audit S7) whose LR / clip value / warmup /
  schedule = those of 'audio', the pose input embedding inp.pose's group, model_ppm.JOIN; cfg lr.g0cond / clip.g0cond) adds g0(f) to the TRUNK input at every pose clip's SLOT position (data_ppm.g0_feat: f = r_y, r_x, log w, the 4 root-to-edge
  distances in shoulder widths, log(orig_h / orig_w); invalid frame 0 -> 0). Pose rows only: the pose datasets (train + eval) get
  dict(CFG['data'], g0cond=1) (CFG['data'] itself unchanged), the codependence micros carry the same features (pose_shuf: the partner
  clip's, pose_zero: 0 = codep_s3q's gfeat rule). cfg keys g0cond (+ g0cond_from_step) present only when set: unset / 0 = no module, no
  key, the cfg and maths of job 27653 exactly. Resume from a checkpoint without it (dcp-17850): ONLY inp.g0.weight may be missing (kept at
  its zero init, AdamW state fresh), everything else exact (digest); a checkpoint with it refuses a config without it. Exports record it.
LINK OFF (ported 10-20 23:10 PT from ppm/link-exp/code = job 26303 arm A; user: "can we get this training without the adapter straight
  away"): PPM_DECP_BB_OFF=1 (needs PPM_DECP_BB=1) removes the D33 backbone -> pose-decoder link: decp_bb stays in the module (weights loaded
  exactly, frozen, unchanged, in NO optimizer: its saved AdamW state set aside with the digest check, its LambdaLR entry removed), it is NOT
  computed in training (model.decp_bb_on False; asserted) and the pose decoder is called with zb=None (= the pre-D33 graph exactly), so the
  link contributes structurally 0; asserted gradient-free every step. Eval: the main pose light CE is the link-off path; probe
  decp_linkon_* re-adds the frozen link on the same frames. Exports record decp_bb_alpha 0 (+ decp_bb_off 1) so inference matches.
POSE-S2P (pose5, 10-27; port of ppm/pose-s2p's monoplex FT change, user: "speech now conditions pose also"): PPM_S2P=1 | full (all 32 speech
  codebooks) | cb0 -> model_ppm tower_p.s_emb (zero init): the pose tower's input at T += s_emb(channel B's Mimi codes at T) (same position,
  no shift; pose rows only: speech / text rows never run the pose tower). Its weight is a tower_p parameter (FSDP unit) in its OWN main-
  AdamW group 's2p' (main 10-27, audit S7) whose LR / clip value / warmup / schedule = tower_p's (model_ppm.JOIN; cfg lr.s2p / clip.s2p). cfg keys s2p (+ s2p_from_step) present only when set: unset / 0 = no module, no
  key, the cfg and maths of job 27653 exactly. Resume from a checkpoint without it (e.g. dcp-17850): ONLY tower_p.s_emb.weight may be missing
  (load(): kept at its zero init, AdamW state fresh: its first update uses Adam step 1 bias correction), every other tensor / optimizer state
  exact (digest); s2p_from_step = that step. Later checkpoints carry it (strict again); a checkpoint with s_emb refuses a config without it
  or with the other mode. Exports record s2p in train_cfg.json.
Knobs: env PPM_* (read once below; every value recorded in train_state.json / the log). Normalised word CE is the headline text metric."""
import datetime
import hashlib
import json
import math
import os
import pickle
import socket
import sys
import time
import zoneinfo

import numpy as np
import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_ppm as M  # noqa: E402
import data_ppm as D  # noqa: E402
import chat_ppm as CH  # noqa: E402
import safety_ppm as SF  # noqa: E402
import lightms_s3q as LMS  # noqa: E402
import sched_s3q as SCH  # noqa: E402
import codep_s3q as CDP  # noqa: E402   (10-12 codependence probe, frozen copy of the pose trainer's)
import s811 as S8  # noqa: E402   (8,1,1 schedule helpers)


def E(k, d):
    return os.environ.get(k, d)


CFG = {
    "run_tag": E("PPM_RUN", "ppm-dev"), "log_root": E("PPM_LOG_ROOT", "/data/home/amartyavarma/ppm/runs"),
    "save_root": E("PPM_SAVE_ROOT", "/scratch/amartyavarma/ppm/ckpt"),
    "layers": int(E("PPM_LAYERS", "64")), "tower_layers": int(E("PPM_TOWER_LAYERS", "36")), "dec_layers": int(E("PPM_DEC_LAYERS", "28")),
    "proj_r": int(E("PPM_PROJ_R", "2048")), "tower_vocab": int(E("PPM_TOWER_VOCAB", str(M.TOWER_VOCAB))),
    "ga_seq": S8.expand_seq(E("PPM_GA_SEQ", "pose,speech,speech,text")),            # 8,1,1: multiplier syntax (pose*8); plain list = as before
    "w_task": {"pose": float(E("PPM_W_TASK_POSE", "1")), "speech": float(E("PPM_W_TASK_SPEECH", "1")), "text": float(E("PPM_W_TASK_TEXT", "1"))},
    "w_text": float(E("PPM_W_TEXT_CE", "5")), "w_cat_word": float(E("PPM_W_CAT_WORD", "1")), "w_tim": float(E("PPM_W_TIM_CE", "1")),
    "w_nw": float(E("PPM_W_NW_CE", "1")), "w_eob": float(E("PPM_W_EOB_CE", "1")),
    "w_cb0": float(E("PPM_W_CB0", "1")), "w_cb": float(E("PPM_W_CB", "1")), "w_pcb0": float(E("PPM_W_POSE_CB0", "1")), "w_pcb": float(E("PPM_W_POSE_CB", "1")),
    "lr": {"trunk": float(E("PPM_LR_TRUNK", "2e-5")), "spec": float(E("PPM_LR_SPEC", "6e-5")), "proj": float(E("PPM_LR_PROJ", "3e-5")),
           "audio": float(E("PPM_LR_AUDIO", "2e-4")), "tower_s": float(E("PPM_LR_TOWER", "3e-4")), "tower_p": float(E("PPM_LR_TOWER", "3e-4"))},
    "clip": {"trunk": float(E("PPM_CLIP_TRUNK", "1.0")), "spec": float(E("PPM_CLIP_SPEC", "1.0")), "proj": float(E("PPM_CLIP_PROJ", "1.0")),
             "audio": float(E("PPM_CLIP_AUDIO", "1.0")), "tower_s": float(E("PPM_CLIP_TOWER", "1.0")), "tower_p": float(E("PPM_CLIP_TOWER", "1.0"))},
    "warmup": int(E("PPM_WARMUP", "100")), "horizon": int(E("PPM_HORIZON", "10000")), "min_lr_rate": float(E("PPM_MIN_LR_RATE", "0.1")),
    "adam_beta2": float(E("PPM_ADAM_BETA2", "0.95")),
    "sched": E("PPM_SCHED", "constant"),                    # D21: warmup then CONSTANT (default) | cosine (to min_lr_rate over horizon); NOT a resume
    "dec_sched": E("PPM_DEC_SCHED", "constant"),            #   key: switch to a decay later at a resume (LambdaLR keeps the step count)
    "s3_sched": "cosine", "warmup_groups": [g for g in M.GROUPS if g not in ("pose_head", "decp_bb", "g0cond", "s2p")],   # (D32 / D33 own warmups;
                                                                                                         #  g0cond / s2p: their JOIN group's)
    "dec_k": int(E("PPM_DEC_K", "4")), "dec_lr": float(E("PPM_DEC_LR", "4e-4")), "dec_floor": float(E("PPM_DEC_FLOOR", "1e-4")),
    "dec_mode": E("PPM_DEC_MODE", "replicate"), "dec_fast": E("PPM_DEC_FAST", "1") == "1", "dec_ac": E("PPM_DEC_AC", "0") == "1", "dec_max_frames": int(E("PPM_DEC_MAX_FRAMES", "7500")),
    "dec_micro_frames": int(E("PPM_DEC_MICRO_FRAMES", "300")),
    # SPEED knobs (T25, same maths: kernel / scheduling only; not resume keys). dec_even: each update's chunk is split into
    # ceil(m / dec_micro_frames) EQUAL micro-batches (no weight-0 padding frames computed); dec_dyn: one compiled decoder graph for every
    # micro-batch size; dec_pair: on pose rows the two decoders' updates are interleaved (speech update j, pose update j, ...) so each
    # decoder's gradient all-reduce runs while the other decoder computes (each decoder's own update order / data unchanged).
    "dec_even": E("PPM_DEC_EVEN", "1") == "1", "dec_dyn": E("PPM_DEC_DYN", "1") == "1", "dec_pair": E("PPM_DEC_PAIR", "1") == "1",
    "dec_mb_max": int(E("PPM_DEC_MB_MAX", "470")),
    "dec_compile_mode": E("PPM_DEC_COMPILE_MODE", "default"), "profile": E("PPM_PROFILE", "0") == "1",
    "dec_warmup_updates": int(E("PPM_DEC_WARMUP_UPDATES", "400")), "dec_clip": float(E("PPM_DEC_CLIP", "0.5")), "dec_seed": int(E("PPM_DEC_SEED", "20261011")),
    "max_steps": int(E("PPM_MAX_STEPS", "1")), "eval_every": int(E("PPM_EVAL_EVERY", "100")), "save_every": int(E("PPM_SAVE_EVERY", "0")),
    "eval_per_task": int(E("PPM_EVAL_PER_TASK", "1")), "eval_light_max": int(E("PPM_EVAL_LIGHT_MAX", "2048")),
    "pose_root": E("PPM_POSE_ROOT", "/scratch/amartyavarma/pose-s3-samdata/rows"), "speech_root": E("PPM_SPEECH_ROOT", "/scratch/amartyavarma/s3-rows"),
    "text_root": E("PPM_TEXT_ROOT", "/data/home/amartyavarma/data/gemma-txtdup-v1"),
    "pose_seed": int(E("PPM_POSE_SEED", "20261011")), "speech_seed": int(E("PPM_SPEECH_SEED", "20261011")),
    "speech_eval_frac": float(E("PPM_SPEECH_EVAL_FRAC", "0.0005")), "speech_eval_salt": E("PPM_SPEECH_EVAL_SALT", "ppm-v1"),
    "text_offset": int(E("PPM_TEXT_OFFSET", "0")), "chat_offset": int(E("PPM_CHAT_OFFSET", "0")),
    "text_mix": E("PPM_TEXT_MIX", "rand"),                  # rand: 50/50 per (seed, step, rank) Gemma duplex / corpus15m chat | alt | gemma | chat
    "chat_split": E("PPM_CHAT_SPLIT", "2,3"), "chat_seed": int(E("PPM_CHAT_SEED", "20261011")), "allow_foreign_data": int(E("PPM_ALLOW_FOREIGN_DATA", "0")),   # chat: klpre splits 2 then 3
    "node_swap": dict(x.split(":") for x in E("PPM_NODE_SWAP", "").split(",") if x),     # store node -> host carrying it (e.g. slinky-14:slinky-19)
    # D2 (10-12): ALL FULL podcast rows. extra_plan = plan_extra.py output (the 4 non-training slices split over the hosts); from step
    # extra_from_step on, each rank's remaining native rows + its extra rows are shuffled together (rows [0, from x nsm) = the old stream).
    "extra_plan": E("PPM_EXTRA_PLAN", ""), "extra_from_step": int(E("PPM_EXTRA_FROM_STEP", "-1")),
    # holdout salts: from extra_from_step on, training drops every row whose EPISODE is an eval episode under these salts (eval_frac as the
    # speech eval), e.g. s3eval-v1 = the stage-3 / Demeter 'ourrows' eval episodes (now a clean holdout)
    "train_exclude_salts": [x for x in E("PPM_TRAIN_EXCLUDE_SALTS", "").split(",") if x],
    # extra evals (fixed orientation, both lanes): speech_s3 = the 103 stage-3 eval rows Demeter 'ourrows' was scored on (pre-built micros,
    # tests/build_eval_s3.py); gemma_s3 = gemma-txtdup-v1 eval packs 0..N-1 (the Demeter text-duplex reference), cap = that store's cap
    "eval_s3_dir": E("PPM_EVAL_S3_DIR", ""), "eval_gemma_s3_root": E("PPM_EVAL_GEMMA_S3_ROOT", ""), "eval_gemma_s3_packs": int(E("PPM_EVAL_GEMMA_S3_PACKS", "152")),
    # 10-12 DEPENDENCE PROBES (eval only, forward passes, not resume keys; doc above evaluate()):
    #   codep  pose rows: the pose trainer's codependence/* probe (codep_s3q), same subset rule / variants / key names as q32-pose-s3sam311b
    #   xdep   speech rows: B given A on the first xdep_rows stage-3 eval rows (speech_s3) x lanes A, B, s3probe's oth-silent / oth-absent
    "codep": int(E("PPM_CODEP", "1")), "codep_frames": int(E("PPM_CODEP_FRAMES", "1024")), "codep_seed": int(E("PPM_CODEP_SEED", "20261009")),
    "codep_light_frac": float(E("PPM_CODEP_LIGHT_FRAC", "0.25")),
    "xdep": int(E("PPM_XDEP", "1")), "xdep_rows": int(E("PPM_XDEP_ROWS", "32")), "xdep_conds": [x for x in E("PPM_XDEP_CONDS", "silent,absent").split(",") if x],
    "abl_selftest": int(E("PPM_ABL_SELFTEST", "0")),
    "data": D.default_cfg(seed=int(E("PPM_DATA_SEED", "20261011")), sp_cap=int(E("PPM_SP_CAP", "2000")), persona_sides=E("PPM_PERSONA_SIDES", "both"),
                          light_frac=float(E("PPM_LIGHT_FRAC", "1.0")), pose_light_frac=float(E("PPM_POSE_LIGHT_FRAC", "1.0")),
                          txt_cap=int(E("PPM_TXT_CAP", "7500")), crop_conv=E("PPM_CROP_CONV", "1") == "1",
                          chat_side=E("PPM_CHAT_SIDE", "assistant"), chat_eob=E("PPM_CHAT_EOB", "1") == "1",
                          pack_attn=E("PPM_PACK_ATTN", "causal")),
    "cd": D.CD,                                               # code delay (lib/speech_q.CD <- env PPM_CD, default 2): Mimi AND pose
    "trace_step": int(E("PPM_TRACE_STEP", "-1")),            # torch.profiler over this optimizer step (ranks 0 and WORLD-1): kernel summary
    "dl_workers": int(E("PPM_DL_WORKERS", "3")), "keep_last": int(E("PPM_AC_KEEP_LAST", "0")),
    "resume_from": E("PPM_RESUME_FROM", ""),                 # '' (fresh) | auto (newest step COMPLETE on every node, else fresh) | <dcp dir>
    "run_base": E("PPM_RUN_BASE", "") or E("PPM_RUN", "ppm-dev"),   # checkpoints / control files are per run BASE (stable across resumes)
    "keep_ckpt": int(E("PPM_KEEP_CKPT", "2")), "final_save": int(E("PPM_FINAL_SAVE", "1")),
    "control_dir": E("PPM_CONTROL_DIR", ""), "export_every": int(E("PPM_EXPORT_EVERY", "0")),
    # D22 wall-clock cadence (minutes; 0 = off), decided on rank 0 at step boundaries and broadcast; step-count knobs above still work
    "eval_every_min": float(E("PPM_EVAL_EVERY_MIN", "0")), "save_every_min": float(E("PPM_SAVE_EVERY_MIN", "0")),
    "export_every_min": float(E("PPM_EXPORT_EVERY_MIN", "0")),
    "export_root": E("PPM_EXPORT_ROOT", "/data/home/amartyavarma/ppm/ckpt-weka"), "export_min_free_gb": float(E("PPM_EXPORT_MIN_FREE_GB", "300")),
    "export_parts": E("PPM_EXPORT_PARTS", "all"), "test_stop_at": int(E("PPM_TEST_STOP_AT", "0")), "test_savenow_at": int(E("PPM_TEST_SAVENOW_AT", "0")), "ckpt_roundtrip": int(E("PPM_CKPT_ROUNDTRIP", "0")), "smoke": int(E("PPM_SMOKE", "0")),
    "wandb": int(E("PPM_WANDB", "0")), "wandb_project": E("PPM_WANDB_PROJECT", "ppm"),
    "pose_head": int(E("PPM_POSE_HEAD", "0")),              # D32 (doc above); off: no module / loss / group, cfg below unchanged
    "decp_bb": int(E("PPM_DECP_BB", "0")),                  # D33 (doc above); off: no module / root / group, cfg below unchanged
}
if CFG["pose_head"]:
    CFG["w_pose_head"] = float(E("PPM_W_POSE_HEAD", "1.0"))
    CFG["lr"]["pose_head"] = float(E("PPM_LR_POSE_HEAD", "2e-4"))
    CFG["clip"]["pose_head"] = float(E("PPM_CLIP_POSE_HEAD", "1.0"))
    CFG["pose_head_warmup"] = int(E("PPM_POSE_HEAD_WARMUP", "100"))
    CFG["pose_head_from_step"] = 0                          # the head's first optimizer step (set by load() on a resume; saved in the cfg)
    CFG["pose_head_detach_from"] = int(E("PPM_POSEHEAD_DETACH_FROM", "-1"))   # D35: ramp the head's backbone grad 1 -> 0 from this step (-1 = off)
    CFG["pose_head_detach_ramp"] = int(E("PPM_POSEHEAD_DETACH_RAMP", "250"))  # D35: ramp length (steps); after it: full detach
    if E("PPM_POSEHEAD_OFF", "0") == "1":                   # D36 (doc above): keys present only when set (unset: cfg = D35 exactly)
        CFG["pose_head_off"] = 1
        CFG["pose_head_off_from_step"] = 0                  # the first step without the head (set by load() on a resume; saved in the cfg)
assert E("PPM_POSEHEAD_OFF", "0") in ("0", "1") and (CFG["pose_head"] or E("PPM_POSEHEAD_OFF", "0") == "0"), \
    ("PPM_POSEHEAD_OFF must be 0 / 1 and needs PPM_POSE_HEAD=1 (the head module stays, frozen)", E("PPM_POSEHEAD_OFF", "0"), CFG["pose_head"])
PH_OFF = bool(CFG.get("pose_head_off", 0))                  # D36: head present but removed from training
PH_ACTIVE = bool(CFG["pose_head"]) and not PH_OFF           # the head trains / is scored (D32 / D35 behaviour)
if CFG["decp_bb"]:
    CFG["decp_bb_alpha"] = float(E("PPM_DECP_BB_ALPHA", "1.0"))           # the link's scale (resume-changeable)
    CFG["decp_bb_fade_from"] = int(E("PPM_DECP_BB_FADE_FROM", "-1"))      # fade alpha -> 0 linearly over fade_steps from this step (-1 = off)
    CFG["decp_bb_fade_steps"] = int(E("PPM_DECP_BB_FADE_STEPS", "0"))
    CFG["lr"]["decp_bb"] = float(E("PPM_LR_DECP_BB", "2e-4"))
    CFG["clip"]["decp_bb"] = float(E("PPM_CLIP_DECP_BB", "1.0"))
    CFG["decp_bb_warmup"] = int(E("PPM_DECP_BB_WARMUP", "100"))
    CFG["decp_bb_from_step"] = 0                            # the link's first optimizer step (set by load() on a resume; saved in the cfg)
assert E("PPM_DECP_BB_OFF", "0") in ("0", "1") and (CFG["decp_bb"] or E("PPM_DECP_BB_OFF", "0") == "0"), "PPM_DECP_BB_OFF=1 needs PPM_DECP_BB=1"
if E("PPM_DECP_BB_OFF", "0") == "1":                        # LINK OFF (doc above): key present only when set (unset = PPM's cfg exactly)
    CFG["decp_bb_off"] = 1
BB_OFF = bool(CFG.get("decp_bb_off", 0))
S2P = M.s2p_mode(E("PPM_S2P", "0"))                          # POSE-S2P (doc above): None (off) | 'full' | 'cb0'
if S2P:                                                      # keys present only when on (off = the production cfg exactly)
    CFG["s2p"] = S2P
    CFG["s2p_from_step"] = 0                                # the step s_emb's AdamW state started (set by load() on a resume; saved in the cfg)
S2P_KEYS = ("tower_p.s_emb.weight",)
assert E("PPM_S811", "0") in ("0", "1"), E("PPM_S811", "0")
if E("PPM_S811", "0") == "1":                                # 8,1,1 (doc above): cfg keys present only when on (off = the production cfg exactly)
    CFG["s811"] = {"bb_mode": S8.expand_seq(E("PPM_BB_MODE", "")),
                   "bb_w": {"pose": float(E("PPM_BB_W_POSE", "1")), "speech": float(E("PPM_BB_W_SPEECH", "2")), "text": float(E("PPM_BB_W_TEXT", "1"))},
                   "tw_w": {"pose": float(E("PPM_TW_W_POSE", "1")), "speech": float(E("PPM_TW_W_SPEECH", "2"))},
                   "w_norm": E("PPM_W_NORM", "task"), "tower_cadence": E("PPM_TOWER_CADENCE", "step"),
                   "pose_speech_parts": E("PPM_POSE_SPEECH_PARTS", "all"),
                   "pose_ep_seed": int(E("PPM_POSE_EP2_SEED", "20261021")), "speech_ep_seed": int(E("PPM_SPEECH_EP2_SEED", "20261021")),
                   "protect_switch": int(E("PPM_PROTECT_SWITCH", "1")),
                   "from_step": None, "prev_ga_seq": None}       # the switch step + PPM's GA (set from the resume source; carried in every 811 checkpoint)
    CFG["s811_check_steps"] = int(E("PPM_811_CHECK_STEPS", "0"))
    CFG["s811_log_rows"] = int(E("PPM_811_LOG_ROWS", "0"))
    CFG["s811_eval_ref"] = E("PPM_811_EVAL_REF", "")
S811 = "s811" in CFG
assert E("PPM_G0COND", "0") in ("0", "1"), ("PPM_G0COND must be 0 | 1", E("PPM_G0COND", "0"))
G0 = E("PPM_G0COND", "0") == "1"                             # G0COND (doc above)
if G0:                                                       # keys present only when on (off = the production cfg exactly)
    CFG["g0cond"] = 1
    CFG["g0cond_from_step"] = 0                             # the step inp.g0's AdamW state started (set by load() on a resume; saved in the cfg)
for g_, on_ in (("g0cond", G0), ("s2p", bool(S2P))):      # OWN groups (audit S7): LR / clip value = the JOIN group's (keys only when on)
    if on_:
        CFG["lr"][g_], CFG["clip"][g_] = CFG["lr"][M.JOIN[g_]], CFG["clip"][M.JOIN[g_]]
G0_KEYS = ("inp.g0.weight",)
# 10-21 FRESH-DATA RESUME FLIPS (user, for the 8,1,1 resume from dcp-17012 only; overrides the no-swap rule there; data_ppm FLIP_BIT doc):
#   PPM_SPEECH_FLIP=1: speech TRAINING rows of 8,1,1 speech epoch >= 2 (the repeats; s811.speech_stream marks) are built with the sides
#     flipped (swap = 1 - side_of: the row's other speaker on the predicted channel B); the unseen rest of PPM's epoch 1 is unchanged.
#   PPM_TEXT_FLIP=1 (a SEPARATE knob, never coupled to the speech one) + PPM_TEXT_FLIP_FROM=<step>: text-duplex TRAINING packs of optimizer
#     steps >= FROM (0-based step index t, = the micro of step t + 1) flipped: Gemma txtdup swap = 1 - side_of('txtdup/<conv_id>'); corpus15m
#     chat = the USER on B (its turn ends carry the kept <|im_end|>, chat_ppm.eob_to_user). FROM is explicit so every later resume flips the
#     same packs. Evals (pose, speech, speech_A, speech_s3, xdep, gemma, chat, gemma_s3) are never flipped. Keys present only when set.
assert E("PPM_SPEECH_FLIP", "0") in ("0", "1") and E("PPM_TEXT_FLIP", "0") in ("0", "1"), (E("PPM_SPEECH_FLIP", "0"), E("PPM_TEXT_FLIP", "0"))
if E("PPM_SPEECH_FLIP", "0") == "1":
    CFG.setdefault("fresh", {})["speech_flip"] = 1
    # 10-22 (user: "just flip podcast from the get go as its marginal"): PPM_SPEECH_FLIP_FROM=<step index> flips EVERY speech training row of
    # steps >= FROM (not only the epoch >= 2 repeats). Key present only when set.
    if E("PPM_SPEECH_FLIP_FROM", ""):
        CFG["fresh"]["speech_flip_from"] = int(E("PPM_SPEECH_FLIP_FROM", ""))
if E("PPM_TEXT_FLIP", "0") == "1":
    assert int(E("PPM_TEXT_FLIP_FROM", "-1")) >= 0, "PPM_TEXT_FLIP=1 needs PPM_TEXT_FLIP_FROM=<first flipped optimizer step index>"
    CFG.setdefault("fresh", {}).update(text_flip=1, text_flip_from=int(E("PPM_TEXT_FLIP_FROM", "-1")))
    # 10-22 (user: "you switched the user assistant roles in chat text"): PPM_TEXT_FLIP_CHAT=0 keeps corpus15m chat packs UNFLIPPED (assistant on B)
    # while the Gemma txtdup flip stays; key present only when 0 (unset = the 27100 behaviour, chat flipped too)
    assert E("PPM_TEXT_FLIP_CHAT", "1") in ("0", "1"), E("PPM_TEXT_FLIP_CHAT", "1")
    if E("PPM_TEXT_FLIP_CHAT", "1") == "0":
        CFG["fresh"]["text_flip_chat"] = 0
SPEECH_FLIP = bool(CFG.get("fresh", {}).get("speech_flip", 0))
TEXT_FLIP_FROM = int(CFG["fresh"]["text_flip_from"]) if CFG.get("fresh", {}).get("text_flip", 0) else None
TEXT_FLIP_CHAT = bool(CFG.get("fresh", {}).get("text_flip_chat", 1))
# 10-22 (user: "the point is just about getting fresh data"; only exhausted data gets flipped): PPM_CHAT_FRESH_SPLITS=<comma list, e.g. 1> appends
# UNUSED corpus15m klpre splits AFTER chat_split's training packs (eval = chat_split's first split's heldout, unchanged), so a rank that runs past
# its chat_split packs reads never-seen packs instead of a 2nd pass. The per-rank prefix (world-strided) is unchanged. Key present only when set.
if E("PPM_CHAT_FRESH_SPLITS", ""):
    CFG.setdefault("fresh", {})["chat_extra_splits"] = E("PPM_CHAT_FRESH_SPLITS", "")
assert S811 or not (SPEECH_FLIP or TEXT_FLIP_FROM is not None), "the fresh-data flips are defined on the 8,1,1 streams (PPM_S811=1)"
# 10-21 FRESH POSE (user: "completely fresh pose data, the HF bucket new-pose-llm-data"; doc fresh/fresh_pose.py): PPM_FRESH_POSE_ROOT=<node-local
# plan dir> + PPM_FRESH_POSE_FROM=<first fresh optimizer step index S_f> -> the 8,1,1 pose TRAINING micros of steps t >= S_f come ONLY from the
# fresh rows (built on the fly from the host's node-local new-pose-llm-data parquet by its fixed PLAN; deduped, minus PPM's TikTok v2+v3 videos and
# the pose-eval videos). Rank local l takes host rows l, l + 8, l + 16, .. (one row per pose micro, never repeated): step t micro m reads the
# rank's fresh row (t - S_f) x npm + m. The run ends (normal final save) at the last step every rank can fill ("[fresh] pose exhausted").
# Pose EVAL rows stay the old store's (unchanged). Keys present only when set.
if E("PPM_FRESH_POSE_ROOT", ""):
    assert int(E("PPM_FRESH_POSE_FROM", "-1")) >= 0, "PPM_FRESH_POSE_ROOT needs PPM_FRESH_POSE_FROM=<first fresh optimizer step index>"
    CFG.setdefault("fresh", {}).update(pose_root=E("PPM_FRESH_POSE_ROOT", ""), pose_from=int(E("PPM_FRESH_POSE_FROM", "-1")))
FRESH_POSE = CFG.get("fresh", {}).get("pose_root") or None
assert S811 or not FRESH_POSE, "fresh pose is defined on the 8,1,1 streams (PPM_S811=1)"
# 10-22 FRESH POSE APPEND (user: "more distinct pose data soon, same format; tack it onto the end of the data schedule"; fresh/append/
# APPEND_READY.md): PPM_FRESH_POSE_ROOT2=<node-local plan dir>[,<dir>..] (fresh_pose.py plans with FRESH_TAG, built by fresh/append/
# ingest_more.sh; deduped against plan 1 + PPM's TikTok + eval videos) + PPM_FRESH_POSE_TAIL=1|0 (explicit) -> rank local l's fresh pose stream
# = its plan-1 rows l, l + 8, .. then plan 2's rows l, l + 8, .. then plan 3's .. (fresh_pose.append_stream; plan k + 1 rows tagged k << 40).
# TAIL=1: every rank first finishes ALL its own rows of the earlier plans (plan 1: the rows past FRESH_END1 = the end the slowest rank set,
# never used so far); TAIL=0: the stream is cut at the previous end before each appended plan (every rank starts it at the same step).
# Step t micro m still reads stream row (t - S_f) x npm + m, so every step before the previous end reads exactly the row it read before
# (resume-exact): each appended plan may only be added at a checkpoint step <= the end of the plans before it (asserted; a later checkpoint
# may append further plans to the list). The run ends where the combined stream ends ("[fresh] pose exhausted"). Each row at most once. The
# checkpoint's fresh block records host -> [plan digests] (pose_digests, re-checked on every later resume) and the step each appended plan
# was added at (pose_append_at). Keys present only when set: unset = the cfg, data and behaviour of job 27114 exactly.
assert E("PPM_FRESH_POSE_TAIL", "") in ("", "0", "1") and bool(E("PPM_FRESH_POSE_ROOT2", "")) == bool(E("PPM_FRESH_POSE_TAIL", "")), \
    ("PPM_FRESH_POSE_ROOT2 and PPM_FRESH_POSE_TAIL=1|0 go together", E("PPM_FRESH_POSE_ROOT2", ""), E("PPM_FRESH_POSE_TAIL", ""))
if E("PPM_FRESH_POSE_ROOT2", ""):
    assert FRESH_POSE, "PPM_FRESH_POSE_ROOT2 appends to the fresh pose plan: needs PPM_FRESH_POSE_ROOT"
    CFG["fresh"].update(pose_root2=[x for x in E("PPM_FRESH_POSE_ROOT2", "").split(",") if x], pose_tail=int(E("PPM_FRESH_POSE_TAIL", "")))
    assert len(set(CFG["fresh"]["pose_root2"] + [FRESH_POSE])) == len(CFG["fresh"]["pose_root2"]) + 1, ("fresh pose plan dirs repeat", CFG["fresh"])
FRESH_POSE2 = CFG.get("fresh", {}).get("pose_root2") or None
FRESH_APPEND_KEYS, FRESH_APPEND_LATE = ("pose_root2", "pose_tail"), ("pose_digests", "pose_append_at")
FRESH_SAVED_DIG = [None]                                       # the resume checkpoint's pose_digests (appended plans), asserted in main()


def fresh_append_guard(fo, fn, ck_step):
    """the fresh-block resume check when appended plans are involved (s811_init; pure, unit-tested in fresh/append/test_append_cpu.py).
    fo = the checkpoint's fresh block (None: a checkpoint without one), fn = this config's (late keys not filled yet). -> (pose_append_at:
    the step each appended plan was added at, the checkpoint's pose_digests or None). Without appended plans in the checkpoint (27114's
    saves): every other fresh key equal; all plans added at its step. With them: every key equal except the late ones and pose_root2, whose
    saved list must be a prefix of this one (plans appended further are added at this step); main() asserts each added step <= the end of
    the plans before it, and the saved digests == the prefix of the loaded ones."""
    strip = lambda d, ks: {k: v for k, v in (d or {}).items() if k not in ks}  # noqa: E731
    assert fn is not None and "pose_root2" in fn, ("resume: the checkpoint's fresh pose has appended plans; this config drops them", fo, fn)
    n = len(fn["pose_root2"])
    if fo is None:
        return [int(ck_step)] * n, None
    if "pose_root2" in fo:
        assert strip(fo, FRESH_APPEND_LATE + ("pose_root2",)) == strip(fn, FRESH_APPEND_LATE + ("pose_root2",)), \
            ("fresh-data resume: the fresh keys differ from the checkpoint's", fo, fn)
        m = len(fo["pose_root2"])
        assert 1 <= m <= n and list(fn["pose_root2"][:m]) == list(fo["pose_root2"]), ("resume: the checkpoint's appended plans must be a prefix "
                                                                                      "of PPM_FRESH_POSE_ROOT2", fo["pose_root2"], fn["pose_root2"])
        assert all(k in fo for k in FRESH_APPEND_LATE) and len(fo["pose_append_at"]) == m, ("checkpoint fresh block: appended plans without "
                                                                                          "digests / append steps", fo)
        return [int(x) for x in fo["pose_append_at"]] + [int(ck_step)] * (n - m), fo["pose_digests"]
    assert not set(FRESH_APPEND_LATE) & set(fo), fo
    assert fo == strip(fn, FRESH_APPEND_KEYS + FRESH_APPEND_LATE), ("fresh-data resume: the fresh keys (other than the appended plans) differ "
                                                                    "from the checkpoint's", fo, fn)
    return [int(ck_step)] * n, None


def _fresh_pose_module():
    """fresh/fresh_pose.py, imported without leaving its builder paths (qwen-1ch/stage2, samdata scripts) on sys.path (module-name shadowing);
    its speech_q is this trainer's (lib copy, identical lane / pieces code: only CD differs, unused there)."""
    sp_ = list(sys.path)
    sys.path.insert(0, os.path.join(HERE, "fresh"))
    try:
        import fresh_pose as FP_  # noqa: E402
    finally:
        sys.path[:] = sp_
    return FP_
RANK, WORLD, LOCAL = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
LW = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
HOST_PHYS = socket.gethostname().split(".")[0]
# 10-21 logical host slots (= ppm14b train_ilp.py; placement only, never maths; outside CFG): PPM_HOST_ALIAS=phys:logical[,..] -> this physical
# host plays the logical host's slot (its rank range via the sbatch's PPM_HOSTS order, its node-local stores' node check, its D2 extra plan
# entry, its name in the checkpoint's rank -> host map), its /scratch holding a verified copy of the logical host's data + checkpoint shards
# (8,1,1 fresh resume: slinky-7 -> slinky-10, whose GPU 7 / IB rail 7 hangs every NCCL op). Unset = the physical host (as before).
HOST_ALIAS = dict(x.split(":") for x in os.environ.get("PPM_HOST_ALIAS", "").split(",") if x)
assert len(set(HOST_ALIAS.values())) == len(HOST_ALIAS) and not (set(HOST_ALIAS) & set(HOST_ALIAS.values())), ("PPM_HOST_ALIAS", HOST_ALIAS)
HOST = HOST_ALIAS.get(HOST_PHYS, HOST_PHYS)
PT = zoneinfo.ZoneInfo("America/Los_Angeles")
LOGDIR = f"{CFG['log_root']}/{CFG['run_tag']}"
CFG["control_dir"] = CFG["control_dir"] or f"/data/home/amartyavarma/ppm/control/{CFG['run_base']}"
CKPT = f"{CFG['save_root']}/{CFG['run_base']}"
assert CFG["export_parts"] in ("all", "new"), CFG["export_parts"]
RESUME_KEYS = ("cd", "layers", "tower_layers", "dec_layers", "proj_r", "tower_vocab", "ga_seq", "pose_seed", "speech_seed", "chat_seed", "text_mix", "chat_split",
               "text_offset", "chat_offset", "pose_root", "speech_root", "text_root", "dec_k", "dec_seed", "data", "w_task", "warmup")
GiB = 2 ** 30
SEQ = tuple(CFG["ga_seq"])
NM = len(SEQ)
assert set(SEQ) <= {"pose", "speech", "text"} and NM > 0, SEQ
COEF, BB_GROUPS, TW_GROUPS = None, ("trunk", "spec", "proj", "audio", "g0cond"), ("tower_s", "tower_p", "decp_bb", "s2p")   # 8,1,1: backbone /
                                                                                                    # tower-side groups (g0cond: trunk input; s2p: pose tower)
if S811:
    _s8 = CFG["s811"]
    assert len(_s8["bb_mode"]) == NM, ("PPM_BB_MODE needs one mode (full | frozen) per GA micro", _s8["bb_mode"], SEQ)
    assert CFG["data"]["light_frac"] == 1.0 and CFG["data"]["pose_light_frac"] == 1.0 and CFG["dec_k"] >= 1, \
        ("8,1,1: the light decoders score 100% of every micro's frames (never subsampled)", CFG["data"]["light_frac"], CFG["data"]["pose_light_frac"])
    assert all(v == 1.0 for v in CFG["w_task"].values()), ("8,1,1: PPM_W_TASK_* must stay 1 (PPM_BB_W_* / PPM_TW_W_* weight the tasks)", CFG["w_task"])
    assert not CFG["pose_head"] and CFG["dec_mode"] == "replicate" and CFG["text_mix"] == "rand", "8,1,1 supports the ppm-v1 configuration (no D32 head, replicated decoders, rand text)"
    assert _s8["tower_cadence"] in ("step", "micro"), _s8["tower_cadence"]
    COEF = S8.coefs(SEQ, _s8["bb_mode"], _s8["bb_w"], _s8["tw_w"], _s8["w_norm"], _s8["pose_speech_parts"])
assert CFG["save_root"].startswith("/scratch/"), ("checkpoints go to node-local /scratch only", CFG["save_root"])


assert CFG["text_mix"] in ("rand", "alt", "gemma", "chat"), CFG["text_mix"]


def text_src(step, rank=None):
    """the text row's source at optimizer step `step` on `rank`: rand (default, user 10-11) = 50/50 per (seed, step, rank), deterministic
    (exact on resume; ranks of one micro may differ: both sources take the same text path) | alt (step parity) | gemma | chat."""
    rank = RANK if rank is None else rank
    if CFG["text_mix"] == "rand":
        return "gemma" if D.h64(f"ppm-text-src|{CFG['chat_seed']}|{int(step)}|{int(rank)}") & 1 == 0 else "chat"
    return CFG["text_mix"] if CFG["text_mix"] != "alt" else ("gemma" if int(step) % 2 == 0 else "chat")


def text_k(step):
    """index of `step` among THIS rank's steps of the same source (each source's packs consumed once, world-strided by rank)."""
    s = text_src(step)
    return sum(1 for u in range(int(step)) if text_src(u) == s)


def now_pt():
    return datetime.datetime.now(PT).strftime("%m-%d %H:%M:%S PT")


def log(*a):
    if RANK == 0:
        print(f"[{now_pt()}]", *a, flush=True)


def allsum(v, dev):
    t = torch.tensor(v, dtype=torch.float64, device=dev)
    dist.all_reduce(t)
    return t.tolist()


# ------------------------------------------------------------------------------------------------ micro -> device + weights
ARR = ("ids", "depth", "present", "pdepth", "pid", "sel", "y", "word", "pad", "eob", "nw", "cb0", "light", "pcb0", "plight",
       "ids_t", "depth_t", "present_t", "pdepth_t",                 # *_t: trunk-input overrides of the dependence probes (model_ppm.PPM.forward)
       "g0pos", "g0f")                                              # G0COND pose micros only (absent otherwise)


def prep(b, dev):
    t = {"kind": b["kind"], "src": b.get("src", b["kind"]), "segs": [int(x) for x in b["segs"]]}
    for k in ARR:
        if k in b:
            t[k] = torch.as_tensor(b[k]).to(dev, non_blocking=True)
    t["pid"] = t["pid"][None]
    S = float(b["n_text"])
    w = {k: t[k].float() for k in ("word", "pad", "eob", "nw")}
    t["wt"] = ((CFG["w_cat_word"] * w["word"] + CFG["w_tim"] * w["pad"] + CFG["w_eob"] * w["eob"] + CFG["w_nw"] * w["nw"]) / S).float()
    t["wn"] = (CFG["w_text"] * w["word"] / S).float()
    if "cb0" in t:
        m = (t["cb0"] >= 0).float()
        t["cb0_w"] = m / m.sum().clamp_min(1.0)
    if "pcb0" in t:
        m = (t["pcb0"] >= 0).float()
        t["pcb0_w"] = m / m.sum().clamp_min(1.0)
    return t


# ------------------------------------------------------------------------------------------------ stats
TK = ("word_ce", "word_norm", "n_word", "word_hit", "pad_ce", "n_pad", "nw_ce", "n_nw", "eob_ce", "n_eob")


def text_vec(t, st):
    w, p, n_, e = (t[k].float() for k in ("word", "pad", "nw", "eob"))
    return [((st[0] * w).sum()), ((st[3] * w).sum()), (w.sum()), ((st[1] * w).sum()), ((st[0] * p).sum()), (p.sum()),
            ((st[0] * n_).sum()), (n_.sum()), ((st[0] * e).sum()), (e.sum())]


def summarize(kind, v, pre, base=None):
    base = base or kind
    d = dict(zip(TK, v[:len(TK)]))
    m = lambda a, b: d[a] / max(d[b], 1.0)  # noqa: E731
    out = {f"{pre}_{kind}/word_ce_norm": m("word_norm", "n_word"), f"{pre}_{kind}/word_ce": m("word_ce", "n_word"), f"{pre}_{kind}/word_top1": m("word_hit", "n_word"),
           f"{pre}_{kind}/pad_ce": m("pad_ce", "n_pad"), f"{pre}_{kind}/nw_ce": m("nw_ce", "n_nw"),
           f"{pre}_{kind}/text_ce": (d["word_ce"] + d["pad_ce"] + d["nw_ce"] + d["eob_ce"]) / max(d["n_word"] + d["n_pad"] + d["n_nw"] + d["n_eob"], 1.0)}
    x = v[len(TK):]
    if base in ("speech", "pose"):
        out[f"{pre}_{kind}/cb0_ce"] = x[0] / max(x[1], 1.0); out[f"{pre}_{kind}/cb0_top1"] = x[2] / max(x[1], 1.0)
        out[f"{pre}_{kind}/light_ce"] = x[3] / max(x[4], 1.0)
    if base == "pose":
        out[f"{pre}_{kind}/pose_cb0_ce"] = x[5] / max(x[6], 1.0); out[f"{pre}_{kind}/pose_cb0_top1"] = x[7] / max(x[6], 1.0)
        out[f"{pre}_{kind}/pose_light_ce"] = x[8] / max(x[9], 1.0)
    if base in ("speech", "pose"):                                 # per-codebook light CE (each codebook's count = frames = light n / 31)
        o_ = 5 if base == "speech" else 10
        for j, c in enumerate(CB_S):
            out[f"{pre}_{kind}/light_ce_cb{c}"] = x[o_ + j] / max(x[4] / 31.0, 1.0)
    if base == "pose":
        for j, c in enumerate(CB_P):
            out[f"{pre}_{kind}/pose_light_ce_cb{c}"] = x[10 + len(CB_S) + j] / max(x[9] / 31.0, 1.0)
        o_ = 10 + len(CB_S) + len(CB_P)
        if PH_ACTIVE:                                              # D32 pose head: CE sum, n, top-1 hits (same positions as pose_cb0; D36 off: none)
            out[f"{pre}_{kind}/pose_head_cb0_ce"] = x[o_] / max(x[o_ + 1], 1.0); out[f"{pre}_{kind}/pose_head_cb0_top1"] = x[o_ + 2] / max(x[o_ + 1], 1.0)
            o_ += 3
        if CFG["decp_bb"] and pre == "eval":                       # D33 eval probes of the pose decoder's two input paths (same frames as
            for vi, vn in enumerate(DECP_VARS):                    #   pose_light_* = 'both'): toweronly = link term zeroed, bbonly = tower
                b_ = o_ + vi * (2 + len(CB_P))                     #   term zeroed (slot 0 = the link term alone); cb1-31 mean + cb 1/8/16/31
                if x[b_ + 1] > 0:                                  #   (micros without the probe, e.g. codep: count 0 -> no keys)
                    out[f"{pre}_{kind}/decp_{vn}_light_ce"] = x[b_] / x[b_ + 1]
                    for j, c in enumerate(CB_P):
                        out[f"{pre}_{kind}/decp_{vn}_light_ce_cb{c}"] = x[b_ + 2 + j] / (x[b_ + 1] / 31.0)
    return out


CB_S, CB_P = (8, 16, 31), (1, 8, 16, 31)                            # per-codebook light CE logged (speech decoder / pose decoder)
DECP_VARS = ("linkon",) if BB_OFF else ("toweronly", "bbonly")      # D33 eval probes (run_micro bbvar=True; stats appended in this order)
NV = {"text": len(TK), "speech": len(TK) + 5 + len(CB_S), "pose": len(TK) + 10 + len(CB_S) + len(CB_P) + (3 if PH_ACTIVE else 0)
      + (len(DECP_VARS) * (2 + len(CB_P)) if CFG["decp_bb"] else 0)}


# ------------------------------------------------------------------------------------------------ dependence probes (10-12, eval only)
# CODEPENDENCE (pose rows; = the pose trainer q32-pose-s3sam311b's codependence/* probe, codep_s3q, same key names): per rank a FIXED subset
#   of its pose eval rows' whole clips (codep_s3q.select_clips, <= PPM_CODEP_FRAMES positions), scored in 5 passes with identical layout /
#   targets / light frames, ONE forward each: real | pose_shuf (the trunk's pose codes from another subset clip, seeded derangement) | pose_zero
#   (no pose codes) | speech_shuf (the trunk's Mimi codes + word lane from the partner clip) | speech_sil (lane PAD, no Mimi codes). As in the
#   pose trainer the towers / decoders keep their REAL teacher-forced codes and labels; the lane input of the towers follows the variant.
#   dCE = CE(variant) - CE(real): pose -> speech (normalised word CE, speech cb0, speech light), speech -> pose (pose cb0, pose light); timing =
#   Pearson r of P(NEW_WORD) vs pose motion. PPM differences (documented): no frame-0 pose state input (PPM has none), and the subset clips are
#   packed PLAIN CAUSAL in one segment (PPM's pose-row rule, PPM_PACK_ATTN causal) where the pose trainer gave each clip its own segment.
SIL_CODES = "/data/home/amartyavarma/setup-logs/s3hestia/ourrows/silence_codes.json"     # = xbench s3probe / Demeter dm_xdep's silence codes
XD_KEYS = (("word_ce_norm", 1, 2), ("cb0_ce", 10, 11), ("light_ce", 13, 14))                   # speech stats vector: (metric, sum idx, count idx)


def codep_micro(c, v, dcfg):
    """codep_s3q.build output + variant name -> PPM pose micro: lane (trunk + towers' lane input) = the variant's, depth / pdepth = REAL
    (towers' teacher-forced inputs + decoder labels), depth_t / pdepth_t = the variant's trunk codes. NEW_WORD probes ride on sel (word False)."""
    b = c["base"]
    ids_b, dep, pdep, _ = c["var"][v]
    m = {"kind": "pose", "ids": np.stack([np.asarray(b["ids"][0]), ids_b]), "depth": np.asarray(b["depth"]), "pdepth": np.asarray(b["pdepth"]),
         "depth_t": np.stack([np.asarray(b["depth"][0]), dep]), "pdepth_t": pdep, "present": b["present"], "pid": b["pid"],
         "segs": [int(x) for x in b["segs"]], "sel": b["sel"], "y": b["y"], "word": b["word"], "pad": b["pad"], "eob": b["eob"], "nw": b["nw"],
         "n_text": float(c["info"]["ns"]), "cb0": b["cb0"], "light": b["light"], "pcb0": b["pcb0"], "plight": b["plight"]}
    assert (m["depth"][1] == np.asarray(b["depth_real"])).all() and (m["pdepth"] == np.asarray(b["pdepth_real"])).all()
    return D.packed(m, dcfg)


def codep_g0(cdp, rows):
    """G0COND: + g0pos / g0f on every codependence micro. The subset = codep_s3q.select_clips(rows, budget) (the clips build() packed, in
    order; their SLOT positions = the base micro's gpos); per variant the features follow codep_s3q's gfeat rule: real / speech_* = the
    clip's own, pose_shuf = the partner clip's (perm), pose_zero = 0."""
    sel = CDP.select_clips(rows, CFG["codep_frames"])
    assert [c.get("clip_id", "") for _, c in sel] == list(cdp["info"]["clips"]), "G0COND: codep subset != the built one"
    f = np.stack([D.g0_feat(c)[0] for _, c in sel]).astype(np.float32)
    bad = [(i, D.g0_problem(c)) for i, (_, c) in enumerate(sel) if D.g0_problem(c)]
    log(f"[g0cond] codependence subset: {len(sel)} clips, {len(bad)} without usable frame-0 fields (f = 0)" + (f": {bad[:8]}" if bad else ""))
    perm = np.asarray(cdp["info"]["perm"], np.int64)
    gpos = np.asarray(cdp["base"]["gpos"], np.int64)
    assert len(gpos) == len(f) == len(perm), (len(gpos), len(f), len(perm))
    for v, m in zip(CDP.VARIANTS, cdp["micros"]):
        m["g0pos"] = gpos
        m["g0f"] = f[perm] if v == "pose_shuf" else (np.zeros_like(f) if v == "pose_zero" else f)


def codep_vals(v):
    """PPM pose stats vector -> codep_s3q.reduce_variant's 10 sums: word norm (sum, n), cb0, light (frame-mean over 31 cb), pose cb0, pose light."""
    return [v[1], v[2], v[10], v[11], v[13] / 31.0, v[14] / 31.0, v[15], v[16], v[18] / 31.0, v[19] / 31.0]


def codep_timing(c, st):
    """codep_s3q.reduce_variant's timing sums (6 per motion kind) from the head stats [4, 2 ns] (row 2 = p(target); probes = the second half)."""
    ns, dev = c["info"]["ns"], st.device
    ok = torch.as_tensor(c["probe_ok"], device=dev).double()
    p = st[2, ns:2 * ns].double()
    out = []
    for mk in ("probe_motion_real", "probe_motion_shown"):
        m_ = torch.as_tensor(c[mk], device=dev).double()
        out += [ok.sum(), (p * ok).sum(), (m_ * ok).sum(), (p * p * ok).sum(), (m_ * m_ * ok).sum(), (p * m_ * ok).sum()]
    return torch.stack(out).tolist()


# XDEP (speech rows, B given A; = xbench s3probe's oth-silent / oth-absent, the probe the stage-3 runs and Demeter (dm_xdep.py) were scored
#   with): rows = the first PPM_XDEP_ROWS (32) of the 103 stage-3 eval rows = rows103.pkl[:32] = s3probe's eval_list(.., 32) = Demeter's xdep
#   rows; lanes B and A (PPM predicts only channel B; lane A = the row flipped) = both speakers on B = s3probe / Demeter channel B over swaps 0, 1.
#   The ablation acts on channel A over the CONVERSATION frames [P, T) of the TRUNK input only (towers see only channel B anyway):
#     silent  A's text lane = PAD and A's Mimi codes = the codec's silence codes (silence_codes.json 'seq', tiled from conversation frame 0)
#     absent  A contributes nothing (present_A = 0)
#   intact = the same rows' speech_s3 eval forwards (same micros). dCE = ablated - intact (> 0: B uses A); normalised word CE, cb0, light.
#   PPM differences (documented): PPM rows carry the system prompt (left intact); PPM has plain RoPE (no word positions to recompute, as Demeter).
def xdep_micro(m, cond, sil):
    """a speech_s3 micro + condition -> the micro with trunk-input overrides (ids_t, depth_t, present_t). 'none' = identity (self-test)."""
    P, T = int(m["P"]), int(m["ids"].shape[1])
    ids_t, dep_t, pres_t = np.array(m["ids"]), np.array(m["depth"]), np.array(m["present"], dtype=np.float32)
    if cond == "silent":
        ids_t[0, P:] = D.PAD_ID
        q = np.flatnonzero(dep_t[0, P:, 0] >= 0) + P
        f = q - (P + D.CD)
        assert len(q) and f.min() >= 0, ("channel-A audio before the conversation's first coded frame", P, q[:3])
        dep_t[0, q] = sil[:, f % sil.shape[1]].T + np.arange(D.DEPTH, dtype=np.int64) * D.CBS
        assert ((dep_t[0] >= 0) == (np.asarray(m["depth"][0]) >= 0)).all()
    elif cond == "absent":
        pres_t[0, P:] = 0.0
    else:
        assert cond == "none", cond
    return dict(m, ids_t=ids_t, depth_t=dep_t, present_t=pres_t)


# ------------------------------------------------------------------------------------------------ nanosteps
class DecOpt:
    def __init__(self, dec, name):
        self.ps = list(dec.parameters())
        self.rep = not hasattr(self.ps[0], "to_local")            # replicated decoder: grads = views of ONE flat fp32 buffer (one all-reduce)
        if self.rep:
            self.flat = torch.zeros(sum(p.numel() for p in self.ps), dtype=torch.float32, device=self.ps[0].device)
            o = 0
            for p in self.ps:
                p.grad = self.flat[o:o + p.numel()].view_as(p); o += p.numel()
        self.opt = torch.optim.AdamW([{"params": self.ps, "lr": CFG["dec_lr"]}], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True)
        self.micros, self.name = 0, name                          # decoder-micro counter (LR index = micros x K + j)

    def lr(self, s):
        hz = max(CFG["horizon"] * sum(1 for x in SEQ if x in (("speech", "pose") if self.name == "dec_s" else ("pose",))) * CFG["dec_k"], 1)
        if CFG["dec_sched"] == "constant":                         # D21: linear warmup over dec_warmup_updates, then the peak
            return CFG["dec_lr"] * min(1.0, (s + 1) / max(1, CFG["dec_warmup_updates"]))
        return LMS.sam_lr(s, CFG["dec_warmup_updates"], hz, CFG["dec_lr"], CFG["dec_floor"])


def _mb_plan(m):
    """micro-batches [(start, size)] of a decoder update's chunk of m frames. Reference (dec_even off): ceil(m / MB) micro-batches of the
    static size MB, the last one padded with weight-0 frames. dec_even: the same count ceil(m / MB) (or ceil(m / dec_mb_max) if set) of
    EQUAL sizes ceil(m / count) (static compile: rounded up to 32 frames, the remainder weight-0 padded). Padded frames carry weight 0
    (zero loss, exactly zero gradient), so the chunk gradient is the same sum either way (fp32 accumulation order only)."""
    MB = CFG["dec_micro_frames"]
    mm = max(m, 1)
    if not CFG["dec_even"]:
        return [(a0, MB) for a0 in range(0, -(-mm // MB) * MB, MB)]
    cap = CFG["dec_mb_max"] or MB
    nmb = -(-mm // cap)
    sz = -(-mm // nmb)
    if not CFG["dec_dyn"]:
        sz = -(-sz // 32) * 32
    plan = [(i * sz, sz) for i in range(nmb) if i * sz < mm]
    return [(a0, max(z, 16)) for a0, z in plan]                  # >= 16 frames (dynamic-shape graph never sees a size-0/1 batch)


def _dec_update_compute(dec, do, hd, depth, light, idx, m, st, zd=None):
    """forward + backward of one decoder update's chunk (idx [m] positions into light), gradient-accumulated over _mb_plan micro-batches
    into the decoder's grads (flat buffer) and hd.grad (D33: and zd.grad, the detached backbone-link input). = the reference loop body."""
    hd.grad = None
    if zd is not None:
        zd.grad = None
    if do.rep:
        do.flat.zero_()
    for a0, sz in _mb_plan(m):
        mv = max(0, min(sz, m - a0))
        fl = light.index_select(0, idx[a0:a0 + mv]) if mv else light[:1]
        if fl.numel() < sz:
            fl = torch.cat([fl, light[:1].expand(sz - fl.numel())])
        wf = torch.zeros(sz, dtype=torch.float32, device=hd.device)
        if mv:
            wf[:mv] = 1.0 / (31.0 * m)
        l_, s_, num_, cb_ = dec(hd, depth, fl, wf, n_valid=mv) if zd is None else dec(hd, depth, fl, wf, n_valid=mv, zb=zd)
        if mv:
            st["sum"] += s_; st["n"] += num_; st["cb"] += cb_.detach().float()    # per-codebook CE sums (logging only)
        l_.backward()


def _dec_update_finish(do, j, K, work=None):
    """the update's gradient mean over ranks (all-reduce; `work` = an async one already in flight), clip, AdamW step. -> lr."""
    if do.rep:
        if work is None:
            dist.all_reduce(do.flat)
        else:
            work.wait()
        do.flat.div_(WORLD)
    lr = do.lr(do.micros * K + j)
    for g in do.opt.param_groups:
        g["lr"] = lr
    torch.nn.utils.clip_grad_norm_(do.ps, CFG["dec_clip"])
    do.opt.step()
    if not do.rep:
        do.opt.zero_grad(set_to_none=True)
    return lr


def _nano_setup(do, hv, light, zb=None):
    hd = hv.detach().requires_grad_()
    n, K, dev = int(light.numel()), CFG["dec_k"], hv.device
    perm = torch.randperm(max(n, 1), generator=torch.Generator().manual_seed(CFG["dec_seed"] * 1000003 + do.micros * 1009 + RANK))
    sizes = [int(perm[j::K].numel()) if n else 0 for j in range(K)]
    perm_d = perm.pin_memory().to(dev, non_blocking=True)
    acc = torch.zeros(hd.shape, dtype=torch.float32, device=dev)
    st = {"sum": torch.zeros((), device=dev, dtype=torch.float32), "n": 0, "cb": torch.zeros(31, device=dev, dtype=torch.float32)}
    zd = zb.detach().requires_grad_() if zb is not None else None                   # D33: the detached backbone-link input (as hd)
    acc_z = torch.zeros(zd.shape, dtype=torch.float32, device=dev) if zb is not None else None
    return hd, n, perm_d, sizes, acc, st, zd, acc_z


def nanosteps(dec, do, hv, depth, light, zb=None):
    """K chunked decoder updates on the detached tower output -> (accumulated d loss / d hv (fp32, = full-batch mean), stats, LRs, D33:
    the same for the backbone-link input zb or None).
    = lightms_s3q.micro_steps (SAM: randperm, chunk j = perm[j::K], chunk loss = mean CE over its frames x 31, input grad accumulated with
    weight |chunk| / n, clip + optimizer step per chunk). Each chunk's forward/backward runs in MICRO-BATCHES (_mb_plan; gradient
    accumulation = the same chunk gradient; bounded activation memory), padded frames at weight 0 (zero loss / gradient); replicated
    decoders: ONE grad all-reduce per update. Compiled decoder, permutation on the GPU once, no host sync inside the loop."""
    if not do.rep:
        assert zb is None, "D33 (PPM_DECP_BB) runs on the replicated decoder path only (PPM_DEC_MODE=replicate)"
        return _nanosteps_fsdp(dec, do, hv, depth, light) + (None,)
    return nanosteps_pair([(dec, do, hv, depth, light, zb)])[0]


def nanosteps_pair(items):
    """nanosteps for one or more INDEPENDENT replicated decoders (pose rows: speech + pose decoder). Each decoder runs exactly its own
    sequence (update j: compute -> all-reduce -> clip + step -> update j + 1, same chunks, same LRs); with dec_pair the updates are
    interleaved across decoders and each all-reduce is asynchronous, overlapping the OTHER decoder's compute (no decoder ever computes
    with stale parameters: its next update waits for its own step). items: (dec, do, hv, depth, light[, zb]) (D33: zb = the backbone-link
    input of the pose decoder, detached + accumulated exactly as hv). -> [(acc, stats, lrs, acc_zb or None)] per item."""
    K = CFG["dec_k"]
    items = [tuple(it) + (None,) * (6 - len(it)) for it in items]
    S = [dict(zip(("hd", "n", "perm", "sizes", "acc", "st", "zd", "acc_z"), _nano_setup(do, hv, light, zb)), lrs=[], work=None)
         for (dec, do, hv, depth, light, zb) in items]
    seq = [(j, i) for j in range(K) for i in range(len(items))]
    asyn = CFG["dec_pair"] and len(items) > 1
    for j, i in seq:
        dec, do, hv, depth, light, _ = items[i]
        s_ = S[i]
        if s_["work"] is not None:                                  # this decoder's previous update: finish it before computing again
            s_["lrs"].append(_dec_update_finish(do, j - 1, K, s_["work"])); s_["work"] = None
        m = s_["sizes"][j]
        idx = s_["perm"][j::K] if m else s_["perm"][:0]
        _dec_update_compute(dec, do, s_["hd"], depth, light, idx, m, s_["st"], s_["zd"])
        if s_["hd"].grad is not None and m:
            s_["acc"] += s_["hd"].grad.float() * (m / s_["n"])
        if s_["zd"] is not None and s_["zd"].grad is not None and m:              # D33: the same |chunk| / n weight as hv
            s_["acc_z"] += s_["zd"].grad.float() * (m / s_["n"])
        if asyn:
            s_["work"] = dist.all_reduce(do.flat, async_op=True)
        else:
            s_["lrs"].append(_dec_update_finish(do, j, K))
    out = []
    for (dec, do, hv, depth, light, _), s_ in zip(items, S):
        if s_["work"] is not None:
            s_["lrs"].append(_dec_update_finish(do, K - 1, K, s_["work"])); s_["work"] = None
        s_["hd"].grad = None
        if s_["zd"] is not None:
            s_["zd"].grad = None
        do.micros += 1
        out.append((s_["acc"], s_["st"], s_["lrs"], s_["acc_z"]))
    return out


def _nanosteps_fsdp(dec, do, hv, depth, light):
    """the FSDP-sharded decoder path (PPM_DEC_MODE=fsdp, not used by ppm-v1): reference loop, static micro-batch count on every rank."""
    hd, n, perm_d, sizes, acc, st, _, _ = _nano_setup(do, hv, light)
    K, MB, lrs = CFG["dec_k"], CFG["dec_micro_frames"], []
    for j in range(K):
        m = sizes[j]
        hd.grad = None
        idx = perm_d[j::K] if m else perm_d[:0]
        nmb = -(-(-(-CFG["dec_max_frames"] // K)) // MB)
        assert m <= nmb * MB, ("decoder chunk larger than PPM_DEC_MAX_FRAMES / K", m, nmb * MB)
        for a0 in range(0, nmb * MB, MB):
            fl = light.index_select(0, idx[a0:a0 + MB]) if (m and a0 < m) else light[:1]
            mv = int(fl.numel()) if (m and a0 < m) else 0
            if fl.numel() < MB:
                fl = torch.cat([fl, light[:1].expand(MB - fl.numel())])
            wf = torch.zeros(MB, dtype=torch.float32, device=hd.device)
            if mv:
                wf[:mv] = 1.0 / (31.0 * m)
            l_, s_, num_, cb_ = dec(hd, depth, fl, wf, n_valid=mv)
            if mv:
                st["sum"] += s_; st["n"] += num_; st["cb"] += cb_.detach().float()
            l_.backward()
        if hd.grad is not None and m:
            acc += hd.grad.float() * (m / n)
        lrs.append(_dec_update_finish(do, j, K))
    hd.grad = None
    do.micros += 1
    return acc, st, lrs


# ------------------------------------------------------------------------------------------------ one micro
PROF = {}


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class DecFast:
    def __enter__(self):
        self.prev = M.DEC_FAST["active"]; M.DEC_FAST["active"] = True

    def __exit__(self, *a):
        M.DEC_FAST["active"] = self.prev


def _ticker(on):
    """PPM_PROFILE=1: per-phase wall times of a training micro (cuda-synchronised; off = no syncs)."""
    st = {"t0": time.time()}

    def tick(name):
        if on:
            torch.cuda.synchronize()
            t_ = time.time(); tick.t[name] = tick.t.get(name, 0.0) + t_ - st["t0"]; st["t0"] = t_
    tick.t = {}
    return tick



S8STAT = {}                                                          # 8,1,1: per-step decoder counters (updates, frames scored / available)


def run_micro(model, b, rot, dos, grad, info=None, keep=None, bbvar=False, cf=None):
    """-> (stats vector, loss value). grad: forward + nanosteps + ONE backward (loss / NM). keep (eval): dict <- the text head's stats [4, S].
    bbvar (D33 eval, pose rows): + the pose decoder on the same frames with the link term zeroed (tower-only) and with the tower term
    zeroed (backbone-only). cf (8,1,1 training micros only; None = the production micro exactly): COEF entry (mode, sp, c_bb, c_s, c_p)."""
    kind = b["kind"]
    prof = CFG["profile"] and grad
    tick = _ticker(prof)
    M.TPROF["on"], M.TPROF["ev"] = prof, []
    assert cf is None or grad, "8,1,1 coefficients are a training-micro option"
    with (torch.enable_grad() if grad else torch.no_grad()):
        if cf is None:
            out = model(b, rot, rot, info=info)
        else:                                                      # 8,1,1: bridges c_bb / c_side on full micros (model_ppm.GradScale)
            br = None if cf["mode"] == "frozen" else tuple((cf["c_bb"] / c_) if c_ > 0 else None for c_ in (cf["c_s"], cf["c_p"]))
            out = model(b, rot, rot, info=info, bb=cf["mode"], bridge=br, sp_parts=cf["sp"] or kind != "pose")   # (sp: pose rows only)
    if keep is not None:
        keep["text_stats"] = out["text_stats"]
    tick("fwd")
    wt = CFG["w_task"][kind]
    loss = out["text"] + CFG["w_cb0"] * out.get("cb0", 0.0) + CFG["w_pcb0"] * out.get("pcb0", 0.0)
    if "phead" in out:                                             # D32 auxiliary pose cb0 head (pose rows, flag on only)
        loss = loss + CFG["w_pose_head"] * out["phead"]
    v = text_vec(b, out["text_stats"])
    if cf is None:
        roots, grads = [wt * loss / NM], [None]
    else:                                                          # 8,1,1 roots: backbone-only text CE x c_bb (full micros), tower losses x c_side
        r_ = cf["c_bb"] * out["text"] if cf["mode"] == "full" else 0.0
        if "cb0" in out:
            r_ = r_ + cf["c_s"] * (CFG["w_cb0"] * out["cb0"])
        if "pcb0" in out:
            r_ = r_ + cf["c_p"] * (CFG["w_pcb0"] * out["pcb0"])
        roots, grads = [r_], [None]
        csd = {"dec_s": cf["c_s"], "dec_p": cf["c_p"]}
    dec_pairs = []
    if kind in ("speech", "pose") and "hv_s" in out:              # (8,1,1 PPM_POSE_SPEECH_PARTS=full: no speech tower on frozen pose micros)
        dec_pairs.append(("dec_s", model.dec_s, out["hv_s"], b["depth"][1], b["light"], CFG["w_cb"], None))
    zb_probe = None
    if kind == "pose":                                             # D33: zb_p = the backbone-link input of the pose decoder (flag on only)
        zb_ = out.get("zb_p")
        if BB_OFF:                                                 # LINK OFF: never computed in training; eval: the decp_linkon probe only
            assert not (grad and zb_ is not None), "LINK OFF: the removed link was computed in a training micro"
            zb_probe, zb_ = zb_, None
        dec_pairs.append(("dec_p", model.dec_p, out["hv_p"], b["pdepth"], b["plight"], CFG["w_pcb"], zb_))
    dvals, bbv = [], {}
    if grad and dec_pairs and dos[dec_pairs[0][0]].rep:           # replicated decoders: one call (pose rows: both decoders interleaved)
        with (DecFast() if CFG["dec_fast"] else _nullctx()):
            res = nanosteps_pair([(dec, dos[name], hv, depth, light, zb) for name, dec, hv, depth, light, wcb, zb in dec_pairs])
        tick("+".join(x[0] for x in dec_pairs))
        for (name, dec, hv, depth, light, wcb, zb), (up, acc, lrs, upz) in zip(dec_pairs, res):
            if cf is None:
                roots.append(hv); grads.append((up * (wt * wcb / NM)).to(hv.dtype))
                if zb is not None:                                 # D33: the link's accumulated input gradient, scaled exactly as hv's
                    roots.append(zb); grads.append((upz * (wt * wcb / NM)).to(zb.dtype))
            else:                                                  # 8,1,1: the decoder's side coefficient (c_s speech, c_p pose)
                roots.append(hv); grads.append((up * (csd[name] * wcb)).to(hv.dtype))
                if zb is not None:
                    roots.append(zb); grads.append((upz * (csd[name] * wcb)).to(zb.dtype))
                st_ = S8STAT.setdefault(name, [0, 0.0, 0.0])        # updates, frames scored, frames available (light positions)
                st_[0] += len(lrs); st_[1] += float(acc["n"]) / 31.0; st_[2] += float(light.numel())
            dvals.append((acc["sum"], float(acc["n"]), acc["cb"]))
        dec_pairs = []
    for name, dec, hv, depth, light, wcb, zb in dec_pairs:
        if grad:
            with (DecFast() if CFG["dec_fast"] else _nullctx()):
                up, acc, lrs, upz = nanosteps(dec, dos[name], hv, depth, light, zb)
            tick(name)
            roots.append(hv); grads.append((up * (wt * wcb / NM)).to(hv.dtype))
            if zb is not None:
                roots.append(zb); grads.append((upz * (wt * wcb / NM)).to(zb.dtype))
            dvals.append((acc["sum"], float(acc["n"]), acc["cb"]))
        else:
            with torch.no_grad():
                fl = light[torch.randperm(light.numel(), generator=torch.Generator().manual_seed(1234))[:CFG["eval_light_max"]].to(light.device)]
                wz = torch.zeros(fl.numel(), device=hv.device)
                _, s_, num_, cb_ = dec(hv, depth, fl, wz) if zb is None else dec(hv, depth, fl, wz, zb=zb)
                if name == "dec_p" and zb_probe is not None and bbvar:    # LINK OFF: the frozen link re-added (link-on path), same frames
                    _, s2_, n2_, c2_ = dec(hv, depth, fl, wz, zb=zb_probe)
                    bbv["linkon"] = (float(s2_), float(n2_), c2_.float())
                if zb is not None and bbvar:                       # D33 eval probes: the same frames, one input path each
                    _, s0_, n0_, c0_ = dec(hv, depth, fl, wz)                          # tower-only (link term zeroed)
                    bbv["toweronly"] = (float(s0_), float(n0_), c0_.float())
                    _, s1_, n1_, c1_ = dec(hv, depth, fl, wz, zb=zb, tower=False)      # backbone-only (tower term zeroed)
                    bbv["bbonly"] = (float(s1_), float(n1_), c1_.float())
            dvals.append((float(s_), float(num_), cb_.float()))
    if kind == "pose" and "hv_s" not in out:                       # 8,1,1 (no speech parts on this pose micro): zero sums, count 0
        z_ = torch.zeros((), device=out["text"].device)
        out.update(cb0_sum=z_, cb0_n=z_, cb0_hit=z_)
        dvals.insert(0, (z_, 0.0, torch.zeros(31, device=z_.device)))
    if kind in ("speech", "pose"):
        v += [out["cb0_sum"], out["cb0_n"], out["cb0_hit"], dvals[0][0], dvals[0][1]]
    if kind == "pose":
        v += [out["pcb0_sum"], out["pcb0_n"], out["pcb0_hit"], dvals[1][0], dvals[1][1]]
    if kind in ("speech", "pose"):                                 # per-codebook light CE sums (logging only): speech cb 8, 16, 31
        v += [dvals[0][2][c - 1] for c in CB_S]
    if kind == "pose":                                             #   pose cb 1, 8, 16, 31
        v += [dvals[1][2][c - 1] for c in CB_P]
    if "phead" in out:                                             # D32 pose head stats (appended last: codep_vals' indices unchanged)
        v += [out["phead_sum"], out["phead_n"], out["phead_hit"]]
    if kind == "pose" and CFG["decp_bb"]:                          # D33 (after D32's): per eval probe (DECP_VARS) the pose light CE sum, n,
        for vn in DECP_VARS:                                       #   cb 1/8/16/31 sums (micros without the probe: zeros, count 0)
            z_ = bbv.get(vn, (0.0, 0.0, torch.zeros(31, device=loss.device)))
            v += [z_[0], z_[1]] + [z_[2][c - 1] for c in CB_P]
    if grad:
        torch.autograd.backward(roots, grads)
        tick("bwd")
    if prof:
        torch.cuda.synchronize()
        for rec in M.TPROF["ev"]:                                  # tower-only forward / backward (CUDA events)
            tick.t[rec["name"] + "_fwd"] = rec["f0"].elapsed_time(rec["f1"]) / 1000.0
            if "b0" in rec and "b1" in rec:
                tick.t[rec["name"] + "_bwd"] = rec["b0"].elapsed_time(rec["b1"]) / 1000.0
        M.TPROF["on"], M.TPROF["ev"] = False, []
        PROF.setdefault(kind if cf is None else f"{kind}_{cf['mode']}", []).append(dict(tick.t))   # (8,1,1: per backbone mode)
    dv = loss.device
    v = torch.stack([torch.as_tensor(x, device=dv).double().reshape(()) for x in v]).tolist()   # one host sync, after the backward
    return v, float(loss)


# ------------------------------------------------------------------------------------------------ checkpoints
def opts_of(opt, dos):
    return {"opt": opt, "opt_dec_s": dos["dec_s"].opt, "opt_dec_p": dos["dec_p"].opt}


def save(model, opt, sched, dos, step, path, hosts):
    return SF.save_local(model, opts_of(opt, dos), sched, {"step": int(step), "cfg": CFG, "dec_micros": {k: d.micros for k, d in dos.items()},
                                                            "world": WORLD, "hosts": hosts, "time": now_pt()}, path, RANK, LOCAL, log, CFG["keep_ckpt"])


POSE_HEAD_KEYS = ("pose_head.weight", "pose_head.bias")
DECP_BB_KEYS = ("decp_bb.weight", "decp_bb.bias")


def decp_bb_alpha(step):
    """D33: the link's scale at optimizer step `step` = PPM_DECP_BB_ALPHA x (linear fade to 0 over fade_steps from fade_from, if set)."""
    a = CFG["decp_bb_alpha"]
    f0, fn = CFG["decp_bb_fade_from"], CFG["decp_bb_fade_steps"]
    if f0 >= 0 and fn > 0:
        a *= min(1.0, max(0.0, 1.0 - (int(step) - f0) / fn))
    return float(a)


def posehead_bbgrad(step):
    """D35: the D32 head's backbone-gradient scale at optimizer step `step`: None = off (FROM -1, or before FROM: the head reads hn as D32),
    else 1 -> 0 linearly over RAMP steps from FROM (1.0 at FROM), 0.0 = full detach after."""
    if not CFG["pose_head"]:
        return None
    f0, fn = CFG["pose_head_detach_from"], CFG["pose_head_detach_ramp"]
    if f0 < 0 or int(step) < f0:
        return None
    return float(min(1.0, max(0.0, 1.0 - (int(step) - f0) / fn))) if fn > 0 else 0.0


def load(model, opt, sched, dos, path, hosts):
    ts0 = json.load(open(f"{path}/train_state.json"))
    ph_old, ph_new = int(ts0["cfg"].get("pose_head", 0)), int(CFG["pose_head"])
    add_head = ph_new and not ph_old                               # D32 added at this resume: ONLY the head's keys may be missing
    off_old = bool(ph_old and ts0["cfg"].get("pose_head_off", 0))  # D36: the checkpoint's head was OFF (frozen, no optimizer state)
    rm_head = bool(ph_old and not ph_new)                          # D37 (10-16): the OFF head's module removed (PPM_POSE_HEAD=0)
    assert ph_new >= ph_old or off_old, ("resume: the checkpoint's D32 pose head is still TRAINING (not D36-off); this config removes it", path)
    drop_head = bool(ph_old and not off_old and PH_OFF)            # D36 applied at this resume: set the head's saved AdamW state aside
    revive_head = bool(ph_old and off_old and PH_ACTIVE)           # D36 reverted at this resume: weights as saved, AdamW fresh, re-warmup
    assert not (add_head and PH_OFF), ("D36 PPM_POSEHEAD_OFF=1: the checkpoint has no D32 head to freeze", path)
    bb_old, bb_new = int(ts0["cfg"].get("decp_bb", 0)), int(CFG["decp_bb"])
    assert bb_new >= bb_old, ("resume: the checkpoint has the D33 decp_bb link, this config turns it off (fade it with PPM_DECP_BB_ALPHA "
                              "instead)", path)
    add_bb = bb_new and not bb_old                                 # D33 added at this resume: ONLY the link's keys may be missing
    bboff_old = bool(bb_old and ts0["cfg"].get("decp_bb_off", 0))
    drop_bb = bool(BB_OFF and bb_old and not bboff_old)           # LINK OFF applied at this resume: the link's AdamW state set aside
    assert not (BB_OFF and add_bb), "PPM_DECP_BB_OFF=1: the checkpoint has no D33 link to remove"
    assert BB_OFF or not bboff_old, "the checkpoint's link is OFF (no AdamW state): reviving it needs its own code path"
    s2_old = ts0["cfg"].get("s2p")                                 # POSE-S2P (doc above): added at a resume -> ONLY s_emb may be missing
    assert not s2_old or s2_old == S2P, ("resume: the checkpoint has pose-s2p (tower_p.s_emb, mode "
                                         f"{s2_old}); this config has PPM_S2P {S2P} (removing it / changing the mode is not supported)", path)
    add_s2p = bool(S2P and not s2_old)
    g0_old = int(ts0["cfg"].get("g0cond", 0))                     # G0COND (doc above): added at a resume -> ONLY inp.g0.weight may be missing
    assert G0 or not g0_old, ("resume: the checkpoint has G0COND (inp.g0); this config has PPM_G0COND=0 (removing it is not supported)", path)
    add_g0 = bool(G0 and not g0_old)
    ts = SF.load_local(model, opts_of(opt, dos), sched, path, RANK, log,
                       allow_missing=(POSE_HEAD_KEYS if add_head else ()) + (DECP_BB_KEYS if add_bb else ()) + (S2P_KEYS if add_s2p else ())
                       + (G0_KEYS if add_g0 else ()),
                       drop_opt=(POSE_HEAD_KEYS if drop_head else ()) + (DECP_BB_KEYS if drop_bb else ()),
                       drop_model=(POSE_HEAD_KEYS if rm_head else ()))
    assert ts["world"] == WORLD and ts["hosts"] == hosts, ("resume needs the same world and rank -> host map", ts["world"], ts["hosts"], hosts)
    added = [g for g, a_ in (("pose_head", add_head), ("decp_bb", add_bb), ("g0cond", add_g0), ("s2p", add_s2p)) if a_]   # (M.GROUPS order)
    names_ = [pg["name"] for pg in opt.param_groups]               # D36: the saved LambdaLR's groups (M.GROUPS order, the saved cfg's LR keys;
    sg = [g for g in M.GROUPS if g in ts["cfg"]["lr"] and not (g == "pose_head" and off_old) and not (g == "decp_bb" and bboff_old)]
    assert len(sg) == len(sched.base_lrs) == len(sched._last_lr), ("resume: saved schedule groups", sg, len(sched.base_lrs))
    if drop_head:                                                  # D36 applied: remove the head's saved LambdaLR entry (by name)
        i_ = sg.index("pose_head")
        del sched.base_lrs[i_], sched._last_lr[i_], sg[i_]
    if drop_bb:                                                    # LINK OFF: remove the link's saved LambdaLR entry (by name)
        i_ = sg.index("decp_bb")
        del sched.base_lrs[i_], sched._last_lr[i_], sg[i_]
        log(f"[resume] LINK OFF: decp_bb link REMOVED at step {ts['step']}: weights restored as saved (frozen, never computed in training), "
            f"saved AdamW state set aside (digest incl. it == saved), its LambdaLR entry removed; groups now {names_}")
    if revive_head:                                                # D36 reverted: insert it at its group position (LR = lam at its first step)
        i_ = names_.index("pose_head")
        sched.base_lrs.insert(i_, opt.param_groups[i_]["initial_lr"]); sched._last_lr.insert(i_, opt.param_groups[i_]["lr"]); sg.insert(i_, "pose_head")
    assert sg == [n_ for n_ in names_ if n_ not in added], ("resume: optimizer groups != the saved schedule's groups", sg, names_, added)
    if rm_head:                                                    # D37: the OFF head had no group / LambdaLR entry (sg above excludes it)
        assert "pose_head" not in names_ and "pose_head" not in sg and not any(M.group_of(n_) == "pose_head" for n_, _ in model.named_parameters())
        log(f"[resume] D37 pose head module REMOVED at step {ts['step']} (user 10-16; it was D36-off since step "
            f"{ts['cfg'].get('pose_head_off_from_step')}): no pose_head module / weights / optimizer group from here; its saved weights set aside "
            f"(digest incl. them == saved), no optimizer state existed; every other tensor + optimizer state restored exactly; groups {names_}")
    if drop_head or revive_head:
        log(f"[resume] D36 pose head {'REMOVED from training' if drop_head else 'REVIVED'} at step {ts['step']}: weights restored as saved; "
            + ("saved AdamW state set aside (digest incl. it == saved), its LambdaLR entry removed; groups now " if drop_head else
               "AdamW state fresh, its LambdaLR entry inserted, LR warmup restarts from this step; groups now ") + f"{names_}")
    if added:                                                      # the saved LambdaLR state has one entry per OLD group: append the new
        names = [pg["name"] for pg in opt.param_groups]            #   groups' (they are the trailing groups, M.GROUPS order)
        assert names[-len(added):] == added and len(sched.base_lrs) == len(names) - len(added), (names, added, len(sched.base_lrs))
        for g in added:
            pg = opt.param_groups[names.index(g)]
            sched.base_lrs.append(pg["initial_lr"])
            if g in M.JOIN:                                        # g0cond / s2p: the JOIN group's lambda at the saved step (= its LR now, not lam(0))
                sched._last_lr.append(pg["initial_lr"] * sched.lr_lambdas[names.index(g)](sched.last_epoch))
            else:
                sched._last_lr.append(pg["lr"])
    if ph_new:
        CFG["pose_head_from_step"] = int(ts["step"]) if (add_head or revive_head) else int(ts["cfg"]["pose_head_from_step"])
        gi = [i for i, pg in enumerate(opt.param_groups) if pg["name"] == "pose_head"]
        assert len(gi) == (0 if PH_OFF else 1), gi
        if PH_OFF:                                                 # D36: the first step without the head (kept across later resumes)
            CFG["pose_head_off_from_step"] = int(ts["step"]) if drop_head else int(ts["cfg"]["pose_head_off_from_step"])
        if add_head:
            log(f"[resume] D32 pose head ADDED at step {ts['step']}: pose_head.weight / .bias = zeros (init), AdamW state fresh, LR group 'pose_head' "
                f"{CFG['lr']['pose_head']:g} (clip {CFG['clip']['pose_head']:g}, warmup {CFG['pose_head_warmup']} steps from step {CFG['pose_head_from_step']}), "
                f"W_POSE_HEAD {CFG['w_pose_head']:g}")
    if bb_new:
        oc = ts["cfg"]
        CFG["decp_bb_from_step"] = int(ts["step"]) if add_bb else int(oc["decp_bb_from_step"])
        assert sum(pg["name"] == "decp_bb" for pg in opt.param_groups) == (0 if BB_OFF else 1)
        if add_bb:
            log(f"[resume] D33 decp_bb link ADDED at step {ts['step']}: decp_bb.weight / .bias = zeros (init), AdamW state fresh, LR group 'decp_bb' "
                f"{CFG['lr']['decp_bb']:g} (clip {CFG['clip']['decp_bb']:g}, warmup {CFG['decp_bb_warmup']} steps from step {CFG['decp_bb_from_step']}), "
                f"alpha {CFG['decp_bb_alpha']:g} (fade from {CFG['decp_bb_fade_from']} over {CFG['decp_bb_fade_steps']})")
        else:                                                      # narrowly resume-changeable: alpha and its fade only; the link's LR /
            fix = {"lr": (oc["lr"]["decp_bb"], CFG["lr"]["decp_bb"]), "clip": (oc["clip"]["decp_bb"], CFG["clip"]["decp_bb"]),   # clip / warmup
                   "warmup": (oc["decp_bb_warmup"], CFG["decp_bb_warmup"])}                                                  # must match
            assert all(a_ == b_ for a_, b_ in fix.values()), ("resume: D33 decp_bb LR / clip / warmup differ from the checkpoint", fix)
            chg = {k: (oc.get(k), CFG[k]) for k in ("decp_bb_alpha", "decp_bb_fade_from", "decp_bb_fade_steps") if oc.get(k) != CFG[k]}
            if chg:
                log(f"[resume] D33 resume-changeable link scale changed (checkpoint -> now): {chg}")
    if S2P:                                                        # POSE-S2P: s_emb in its OWN group 's2p' (tower_p's LR / clip / schedule)
        CFG["s2p_from_step"] = int(ts["step"]) if add_s2p else int(ts["cfg"]["s2p_from_step"])
        if add_s2p:
            log(f"[resume] POSE-S2P ADDED at step {ts['step']}: tower_p.s_emb ({S2P}) = zeros (init), AdamW state fresh (bias correction from "
                f"its own step 1), in its own main-AdamW group 's2p' (LR {CFG['lr']['s2p']:g} x tower_p's schedule, clip {CFG['clip']['s2p']:g}); "
                f"every other tensor + optimizer state restored exactly")
    assert len(sched.base_lrs) == len(opt.param_groups) == len(sched.lr_lambdas), (len(sched.base_lrs), len(opt.param_groups))
    for pg, lr_ in zip(opt.param_groups, sched.get_last_lr()):    # FIX (10-12, found with D33): LambdaLR.load_state_dict does not touch the
        pg["lr"] = lr_                                             #   optimizer, whose groups kept the constructor's lam(0) LRs (= 0 with a
                                                                   #   warmup): the first step after every resume ran at LR 0. Now = the
                                                                   #   saved schedule's next LRs (added groups: their lam at their first step)
    if G0:                                                         # G0COND: inp.g0 in its OWN group 'g0cond' (audio's LR / clip / schedule)
        CFG["g0cond_from_step"] = int(ts["step"]) if add_g0 else int(ts["cfg"]["g0cond_from_step"])
        if add_g0:
            log(f"[resume] G0COND ADDED at step {ts['step']}: inp.g0.weight = zeros (init), AdamW state fresh (bias correction from its own "
                f"step 1), in its own main-AdamW group '{M.group_of('inp.g0.weight')}' (LR {CFG['lr']['g0cond']:g} x audio's schedule, clip "
                f"{CFG['clip']['g0cond']:g}); every other tensor + optimizer state restored exactly")
    def _rk(cfg, k):                                               # D31 (10-11 night): the decoder-update spec (dec_k, data.light_frac,
        v = cfg.get(k)                                             #   data.pose_light_frac) is resume-CHANGEABLE, nothing else: these only
        if k == "dec_k":                                           #   choose which frames / how many chunks the 0.6B decoders update on;
            return None                                            #   every other data key (and so the data stream) must still match
        if k == "data" and isinstance(v, dict):
            v = {a: b for a, b in v.items() if a not in ("light_frac", "pose_light_frac")}
        return v
    chg = {k: (ts["cfg"].get(k), CFG.get(k)) for k in ("dec_k",) if ts["cfg"].get(k) != CFG.get(k)}
    chg.update({f"data.{k}": (ts["cfg"].get("data", {}).get(k), CFG["data"].get(k)) for k in ("light_frac", "pose_light_frac")
                if ts["cfg"].get("data", {}).get(k) != CFG["data"].get(k)})
    if chg:
        log(f"[resume] D31 resume-changeable decoder-update keys changed (checkpoint -> now): {chg}")
    rkeys = RESUME_KEYS
    if S811:                                                       # 8,1,1: the switch from a PPM checkpoint changes ga_seq (only); from an
        so, s8 = ts["cfg"].get("s811"), CFG["s811"]                #   8,1,1 checkpoint everything data-defining must match (s811_init)
        if so is None:
            assert s8["from_step"] == int(ts["step"]) and s8["prev_ga_seq"] == list(ts["cfg"]["ga_seq"]), (s8["from_step"], ts["step"])
            rkeys = tuple(k for k in RESUME_KEYS if k != "ga_seq")
            log(f"[resume] 8,1,1 SWITCH at step {ts['step']}: GA {tuple(ts['cfg']['ga_seq'])} -> {SEQ} (the only resume key that changes); "
                f"every other resume key == the PPM checkpoint's")
        else:
            assert (int(so["from_step"]), list(so["prev_ga_seq"])) == (s8["from_step"], s8["prev_ga_seq"]), (so, s8)
            chg = {k: (so.get(k), s8[k]) for k in ("bb_mode", "bb_w", "tw_w", "w_norm", "tower_cadence", "pose_speech_parts", "protect_switch")
                   if so.get(k) != s8[k]}
            if chg:
                log(f"[resume] 8,1,1 resume-changeable knobs changed (checkpoint -> now): {chg}")
    bad = [k for k in rkeys if json.dumps(_rk(ts["cfg"], k), sort_keys=True) != json.dumps(_rk(CFG, k), sort_keys=True)]
    assert not bad, ("resume: config differs from the checkpoint", {k: (ts["cfg"].get(k), CFG.get(k)) for k in bad})
    for k, d in dos.items():
        d.micros = int(ts["dec_micros"][k])
    return int(ts["step"])


def digest(model, opts):
    h = hashlib.sha256()
    for n_, p_ in model.named_parameters():
        h.update(n_.encode()); h.update(M._loc(p_).detach().float().cpu().numpy().tobytes())
    for o in opts:
        for p_ in [q for g in o.param_groups for q in g["params"]]:
            st = o.state.get(p_, {})
            for k_ in sorted(st):
                t_ = st[k_]
                t_ = t_.to_local() if hasattr(t_, "to_local") else t_
                h.update(k_.encode()); h.update(t_.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def resume_source(dev):
    """the resume source by the production rule (PPM_RESUME_FROM '' | auto | <dcp dir>); 8,1,1 resolves it BEFORE the data plans."""
    if CFG["resume_from"] == "auto":
        m_ = SF.newest_common(CKPT, dev)
        log(f"[resume] auto: newest step COMPLETE on every node under {CKPT}: {m_ if m_ >= 0 else 'none (fresh start)'}")
        return f"{CKPT}/dcp-{m_}" if m_ >= 0 else None
    return CFG["resume_from"] or None


def s811_init(rsrc):
    """8,1,1: the switch step S0 + PPM's GA from the resume source's train_state.json (a PPM checkpoint: S0 = its step; an 8,1,1 checkpoint:
    carried; none: a fresh 8,1,1 run, S0 = 0). Data-defining 8,1,1 keys (epoch seeds, GA) must equal an 8,1,1 checkpoint's."""
    s8 = CFG["s811"]
    if rsrc is None:
        s8["from_step"], s8["prev_ga_seq"] = 0, list(SEQ)
        return
    oc = json.load(open(f"{rsrc}/train_state.json"))
    if "s811" in oc["cfg"]:
        so = oc["cfg"]["s811"]
        s8["from_step"], s8["prev_ga_seq"] = int(so["from_step"]), list(so["prev_ga_seq"])
        bad = {k: (so.get(k), s8[k]) for k in ("pose_ep_seed", "speech_ep_seed") if so.get(k) != s8[k]}
        assert not bad and list(oc["cfg"]["ga_seq"]) == list(SEQ), ("8,1,1 resume: data-defining keys differ from the 8,1,1 checkpoint", bad,
                                                                    oc["cfg"]["ga_seq"], SEQ)
    else:
        s8["from_step"], s8["prev_ga_seq"] = int(oc["step"]), list(oc["cfg"]["ga_seq"])
    fo_, fn_ = oc["cfg"].get("fresh"), CFG.get("fresh")         # 10-21 fresh-data resume: the fresh block is data-defining once a checkpoint carries it
    if FRESH_POSE2 or "pose_root2" in (fo_ or {}):               # 10-22 appended fresh pose plans (doc above): their own guard
        CFG["fresh"]["pose_append_at"], FRESH_SAVED_DIG[0] = fresh_append_guard(fo_, fn_, int(oc["step"]))
        log(f"[resume] FRESH POSE APPEND: plans {[FRESH_POSE] + list(FRESH_POSE2)} (tail {CFG['fresh']['pose_tail']}); checkpoint step {int(oc['step'])} "
            f"fresh block {json.dumps(fo_, sort_keys=True)[:600]}; appended plans added at step {CFG['fresh']['pose_append_at']}"
            + ("" if FRESH_SAVED_DIG[0] is None else "; plan digests recorded in the checkpoint (re-checked below)"))
    elif fo_ is not None:
        assert fo_ == fn_, ("fresh-data resume: the fresh keys differ from the checkpoint's", fo_, fn_)
    elif fn_ is not None:
        log(f"[resume] FRESH DATA from checkpoint step {int(oc['step'])} (checkpoint has no fresh block): {json.dumps(fn_, sort_keys=True)}")


# ------------------------------------------------------------------------------------------------ main
def make_opt(model):
    by = {g: [] for g in M.GROUPS}
    for n, p in model.named_parameters():
        g = M.group_of(n)
        if g in by and not (PH_OFF and g == "pose_head") and not (BB_OFF and g == "decp_bb"):   # D36 / LINK OFF: frozen, in no optimizer
            by[g].append(p)
    opt = torch.optim.AdamW([{"params": by[g], "lr": CFG["lr"][g], "name": g} for g in M.GROUPS if by[g]], betas=(0.9, CFG["adam_beta2"]), eps=1e-8,
                            weight_decay=0.0, fused=True)
    def lam_of(warm):
        if CFG["sched"] == "constant":
            w = CFG["warmup"] if warm else 0
            return lambda s: min(1.0, s / max(1, w)) if w else 1.0
        return SCH.make_lambda(warm, CFG)
    def lam_head(g="pose_head"):                                 # D32 / D33: the same warmup-then-constant rule, counted from the
        w = CFG[f"{g}_warmup"]                                     #   group's first step (<g>_from_step: 0 fresh, the resume step if
        if CFG["sched"] == "constant":                            #   added at a resume; read at call time, load() sets it)
            return lambda s: min(1.0, max(0, s - CFG[f"{g}_from_step"]) / max(1, w)) if w else 1.0
        base = SCH.make_lambda(True, dict(CFG, warmup=w))
        return lambda s: base(max(0, s - CFG[f"{g}_from_step"]))
    lams = [lam_head(pg["name"]) if pg["name"] in ("pose_head", "decp_bb") else lam_of(M.JOIN.get(pg["name"], pg["name"]) in CFG["warmup_groups"])
            for pg in opt.param_groups]                            # (g0cond / s2p: exactly their JOIN group's lambda)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lams), by


def _worker_init(_):
    """DataLoader workers: ONE intra-op thread each (the node's CPUs are shared by 8 ranks' workers, NCCL proxy threads and the main
    processes; oversubscription starves NCCL's proxies and exposes inter-node communication)."""
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"


def loaders_for(task_ds, workers):
    from torch.utils.data import DataLoader
    return iter(DataLoader(task_ds, batch_size=None, shuffle=False, num_workers=workers, prefetch_factor=2 if workers else None,
                           persistent_workers=bool(workers), pin_memory=True, worker_init_fn=_worker_init if workers else None))


def main():
    dev = torch.device("cuda", LOCAL)
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev, timeout=datetime.timedelta(seconds=int(E("INIT_TIMEOUT", "1800"))))
    os.makedirs(LOGDIR, exist_ok=True)
    log(f"PPM {CFG['run_tag']}: world {WORLD} (lw {LW}); GA {SEQ}; host {HOST}; cfg {json.dumps(CFG)}")
    rsrc0 = None
    if S811:                                                      # 8,1,1: the switch point decides the data streams -> resolve the resume source first
        rsrc0 = resume_source(dev)
        s811_init(rsrc0)
        s8_ = CFG["s811"]
        log(f"[811] SCHEDULE 8,1,1 from step {s8_['from_step']}: GA {SEQ} x backbone modes {s8_['bb_mode']} (PPM GA {tuple(s8_['prev_ga_seq'])} "
            f"before step {s8_['from_step']}); backbone weights {s8_['bb_w']}, tower weights {s8_['tw_w']}, norm {s8_['w_norm']}; tower cadence "
            f"{s8_['tower_cadence']}; speech parts on pose micros: {s8_['pose_speech_parts']}; light: {CFG['dec_k']} decoder updates per micro over "
            f"100% of its frames; per-micro coefficients (kind mode sp c_bb c_s c_p): "
            + "; ".join(f"{c['kind']} {c['mode']} {int(c['sp'])} {c['c_bb']:.6g} {c['c_s']:.6g} {c['c_p']:.6g}" for c in COEF))
    # ---- data plans (node-local stores)
    dcfg = CFG["data"]
    pcfg = dict(dcfg, g0cond=1) if G0 else dcfg                   # G0COND: the pose datasets' cfg (+ g0pos / g0f per micro); CFG['data'] unchanged
    pnode = D.PS.load_node(CFG["pose_root"])
    snode = D.R3.load_node(CFG["speech_root"])
    for nm_, nd in (("pose", pnode), ("speech", snode)):
        assert nd["node"] == HOST or CFG["node_swap"].get(nd["node"]) == HOST or CFG["allow_foreign_data"], (f"{nm_} store of another node", nd["node"], HOST)
    E_ = CFG["eval_per_task"]
    pplan = D.PS.plan(pnode, CFG["pose_seed"], LW, LOCAL, 1, LW, LOCAL, E_)
    sidx = D.DS.load_index(CFG["speech_root"])
    splan = D.speech_plan(snode, sidx, CFG["speech_seed"], LW, LOCAL, CFG["speech_eval_frac"], CFG["speech_eval_salt"], E_)
    hplan = None
    if CFG["extra_plan"]:                                         # D2: native + extra-slice rows (data_ppm CODES), exact continuation
        XP = D.load_extra_plan(CFG["extra_plan"])
        assert XP["world"] == WORLD and XP["lw"] == LW and XP["eval_salt"] == CFG["speech_eval_salt"] and XP["eval_frac"] == CFG["speech_eval_frac"], \
            ("extra plan made for another layout", XP["world"], XP["lw"], XP["eval_salt"], XP["eval_frac"])
        hplan = XP["hosts"][HOST]
        s0, nsm_ = CFG["extra_from_step"], SEQ.count("speech")
        assert s0 >= 0, "PPM_EXTRA_PLAN needs PPM_EXTRA_FROM_STEP (the step of the first resume with it)"
        if S811:                                                  # 8,1,1: the D2 stream was laid out with PPM's speech micros per step
            assert CFG["s811"]["from_step"] >= s0, ("8,1,1 must switch after the D2 point", CFG["s811"]["from_step"], s0)
            nsm_ = list(CFG["s811"]["prev_ga_seq"]).count("speech")
        splan["rows"], xs = D.d2_stream(np.asarray(splan["rows"], np.int64), sidx, hplan, LOCAL, RANK, s0, nsm_, CFG["train_exclude_salts"],
                                        CFG["speech_eval_frac"], CFG["speech_seed"])
        t_ = torch.tensor([float(xs[k]) for k in ("seen_holdout", "drop_native", "drop_extra", "extra_kept", "head", "native")], dtype=torch.float64, device=dev)
        dist.all_reduce(t_)
        lo_ = torch.tensor([float(len(splan["rows"]))], device=dev); hi_ = lo_.clone()
        dist.all_reduce(lo_, op=dist.ReduceOp.MIN); dist.all_reduce(hi_, op=dist.ReduceOp.MAX)
        log(f"[data] D2 extra slices {XP['slices']} from step {s0}: rows consumed before (kept, in order) {int(t_[4]):,}; native left + extra "
            f"{int(t_[5] - t_[4] - t_[1]):,} + {int(t_[3]):,} shuffled per rank; holdout salts {CFG['train_exclude_salts']}: dropped native "
            f"{int(t_[1]):,} + extra {int(t_[2]):,} rows, ALREADY SEEN before step {s0}: {int(t_[0]):,} rows; speech rows/rank {int(lo_):,}..{int(hi_):,} "
            f"(plan target {XP['target_rows_per_rank']:,}); this host's slots " + ", ".join(f"{x['node']} {len(x['rows']):,}" for x in hplan["slots"]))
    elif CFG["train_exclude_salts"]:
        raise AssertionError("PPM_TRAIN_EXCLUDE_SALTS is applied with the D2 stream (PPM_EXTRA_PLAN + PPM_EXTRA_FROM_STEP)")
    tstore = D.TXD.Store(CFG["text_root"])
    csplits = [int(x) for x in str(CFG["chat_split"]).split(",") if x]
    cpacks = CH.multi_split_packs(csplits, dcfg["txt_cap"], CFG["chat_seed"])
    xsplits_ = [int(x) for x in str(CFG.get("fresh", {}).get("chat_extra_splits", "")).split(",") if x]
    if xsplits_:                                                   # fresh: unused splits appended (doc above)
        assert not set(xsplits_) & set(csplits), (xsplits_, csplits)
        nc0_ = len(cpacks); cpacks = cpacks + CH.multi_split_packs(xsplits_, dcfg["txt_cap"], CFG["chat_seed"])
        log(f"[fresh] CHAT FRESH SPLITS {xsplits_} appended after split {CFG['chat_split']}: {nc0_:,} -> {len(cpacks):,} chat training packs (unused corpus15m packs instead of a 2nd pass)")
    ctb0, _, cho = CH.table(csplits[0])
    cev = [(csplits[0], p) for p in CH.pack_plan(cho, ctb0, dcfg["txt_cap"], 0, shuffle=False)]
    n_pose, n_speech, n_text, n_chat = len(pplan["rows"]), len(splan["rows"]), tstore.n_packs("train"), len(cpacks)
    npm, nsm, ntm = (SEQ.count(k) for k in ("pose", "speech", "text"))
    per = 2 if CFG["text_mix"] in ("alt", "rand") else 1          # rand: ~half the steps per source (x 0.98 margin below)
    big = 10 ** 9
    steps_avail = min(n_pose // npm if npm else big, n_speech // nsm if nsm else big,
                      per * ((n_text - CFG["text_offset"] - RANK) // (WORLD * ntm)) if ntm and CFG["text_mix"] != "chat" else big,
                      per * ((n_chat - CFG["chat_offset"] - RANK) // (WORLD * ntm)) if ntm and CFG["text_mix"] == "chat" else big)
    if CFG["text_mix"] == "rand" and ntm:                         # exact: this rank's per-source draw counts vs its world-strided packs
        cap_ = {"gemma": (n_text - CFG["text_offset"] - RANK - 1) // WORLD + 1, "chat": 10 ** 9}   # chat wraps (below)
        kc_, t_ = {"gemma": 0, "chat": 0}, 0
        while t_ < min(steps_avail, CFG["max_steps"]):
            s_ = text_src(t_)
            if (kc_[s_] + 1) * ntm > cap_[s_]:
                break
            kc_[s_] += 1; t_ += 1
        steps_avail = t_
    t_ = torch.tensor([steps_avail], device=dev); dist.all_reduce(t_, op=dist.ReduceOp.MIN); steps_avail = int(t_)
    n_pose0 = n_pose
    if hplan is not None and npm and not S811:                    # D2: speech epoch > the pose rows -> pose takes a 2nd pass (same order) (8,1,1: epochs, below)
        sp_steps = torch.tensor([n_speech // nsm], device=dev); dist.all_reduce(sp_steps, op=dist.ReduceOp.MIN); sp_steps = int(sp_steps)
        if sp_steps > n_pose // npm:
            need = sp_steps * npm
            r0 = np.asarray(pplan["rows"]); reps = -(-need // len(r0))
            pplan["rows"] = np.concatenate([r0] * reps)[:need]
            n_pose = len(pplan["rows"])
            log(f"[data] D2: pose rows/rank {len(r0):,} < speech steps {sp_steps:,} x {npm}: pose re-reads its first {need - len(r0):,} rows (2nd pass, same order)")
        steps_avail = min(n_pose // npm, n_speech // nsm if nsm else big,
                          per * ((n_text - CFG["text_offset"] - RANK) // (WORLD * ntm)) if ntm and CFG["text_mix"] != "chat" else big)
        if CFG["text_mix"] == "rand" and ntm:
            kc_, t2_ = {"gemma": 0, "chat": 0}, 0
            while t2_ < min(steps_avail, CFG["max_steps"]):
                s_ = text_src(t2_)
                if (kc_[s_] + 1) * ntm > cap_[s_]:
                    break
                kc_[s_] += 1; t2_ += 1
            steps_avail = t2_
        t_ = torch.tensor([steps_avail], device=dev); dist.all_reduce(t_, op=dist.ReduceOp.MIN); steps_avail = int(t_)
    if S811:                                                      # 8,1,1 streams (doc above): pose / speech run in epochs, only text can end the run
        fs_, ps_ = CFG["s811"]["from_step"], CFG["s811"]["prev_ga_seq"]
        cap_ = {"gemma": (n_text - CFG["text_offset"] - RANK - 1) // WORLD + 1, "chat": 10 ** 9}
        used_, t_ = {"gemma": 0, "chat": 0}, 0
        while t_ < CFG["max_steps"]:
            k_, s_ = S8.seq_at(t_, fs_, ps_, SEQ).count("text"), text_src(t_)
            if used_[s_] + k_ > cap_[s_]:
                break
            used_[s_] += k_; t_ += 1
        t_ = torch.tensor([t_], device=dev); dist.all_reduce(t_, op=dist.ReduceOp.MIN); steps_avail = int(t_)
        need_ = max(0, min(CFG["max_steps"], steps_avail) - fs_)
        r1_ = np.asarray(D.PS.plan(pnode, CFG["pose_seed"], LW, LOCAL, 1, LW, LOCAL, E_)["rows"], np.int64)   # PPM's epoch-1 order (unextended)
        pc_ = S8.consumed_rows("pose", fs_, fs_, ps_, SEQ)
        pstream, pmarks = S8.pose_stream(r1_, pc_, lambda e: D.PS.plan(pnode, CFG["s811"]["pose_ep_seed"] + e - 2, LW, LOCAL, 1, LW, LOCAL, E_)["rows"],
                                         need_ * npm)
        sp_rows = np.asarray(splan["rows"], np.int64)
        sp_set = sp_rows
        if hplan is not None and CFG["train_exclude_salts"]:      # the epoch's rows minus the holdout rows PPM saw before the D2 point
            h_ = sp_rows[:CFG["extra_from_step"] * list(ps_).count("speech")]
            hold_ = np.array([any(D.DS.is_eval_episode(D.DS.episode_key(sidx["dataset"][r], sidx["podcast_id"][r], sidx["episode_id"][r]),
                                                       CFG["speech_eval_frac"], sl) for sl in CFG["train_exclude_salts"]) for r in h_], bool)
            sp_set = np.concatenate([h_[~hold_], sp_rows[len(h_):]])
        sc_ = S8.consumed_rows("speech", fs_, fs_, ps_, SEQ)
        sstream, smarks = S8.speech_stream(sp_rows, sc_, sp_set, CFG["s811"]["speech_ep_seed"], RANK, need_ * nsm)
        if SPEECH_FLIP:                                           # fresh resume: epoch >= 2 rows (the repeats) flipped (FLIP_BIT; by stream position)
            sf0_ = min([i_ for e_, i_ in smarks if e_ >= 2] + [len(sstream)])
            if CFG["fresh"].get("speech_flip_from") is not None:      # 10-22: every row from step index speech_flip_from on (doc above)
                assert int(CFG["fresh"]["speech_flip_from"]) >= fs_, (CFG["fresh"]["speech_flip_from"], fs_)
                sf0_ = min(sf0_, (int(CFG["fresh"]["speech_flip_from"]) - fs_) * nsm)
            sstream = sstream.copy(); sstream[sf0_:] |= D.FLIP_BIT
            fs8_ = allsum([float(sf0_ < len(sstream)), float(len(sstream) - sf0_)], dev)
            lo8_ = torch.tensor([float(fs_ + sf0_ / max(nsm, 1))], device=dev); hi8_ = lo8_.clone()
            dist.all_reduce(lo8_, op=dist.ReduceOp.MIN); dist.all_reduce(hi8_, op=dist.ReduceOp.MAX)
            log(f"[fresh] SPEECH FLIP (PPM_SPEECH_FLIP=1): speech epoch >= 2 rows built with the sides flipped (B = the row's other speaker); "
                f"rank 0 flips from stream row {sf0_:,} (step {fs_ + sf0_ // max(nsm, 1) + 1}); first flipped step over ranks {int(lo8_) + 1}..{int(hi8_) + 1} "
                f"({int(fs8_[0])}/{WORLD} ranks flip within this stream; {int(fs8_[1]):,} flipped rows planned); "
                + (f"ALL rows from step index {CFG['fresh']['speech_flip_from']} flipped (PPM_SPEECH_FLIP_FROM)" if CFG["fresh"].get("speech_flip_from") is not None else "the unseen rest of epoch 1 unflipped"))
        if FRESH_POSE:                                            # fresh pose (doc above): the rank's fresh rows; the run ends where they do
            FP = _fresh_pose_module()
            fpl_ = FP.load_plan(FRESH_POSE)
            assert fpl_["host"] in (HOST, HOST_PHYS), ("fresh pose plan of another host", fpl_["host"], HOST, HOST_PHYS)
            Sf_ = int(CFG["fresh"]["pose_from"])
            assert Sf_ >= fs_, ("fresh pose must start at / after the 8,1,1 switch", Sf_, fs_)
            frows_ = FP.rank_rows(fpl_["n_rows"], LOCAL, LW)
            fmn_ = torch.tensor([float(len(frows_) // max(npm, 1))], device=dev); fmx_ = fmn_.clone()
            dist.all_reduce(fmn_, op=dist.ReduceOp.MIN); dist.all_reduce(fmx_, op=dist.ReduceOp.MAX)
            fx_ = allsum([float(len(frows_)), float(fpl_["kept_frames"]) if LOCAL == 0 else 0.0], dev)
            FRESH_END = Sf_ + int(fmn_)
            if FRESH_POSE2:                                       # 10-22 appended plans (doc above): the combined stream; ends where it does
                FRESH_END1 = FRESH_END
                fpls_ = [fpl_] + [FP.load_plan(r_) for r_ in FRESH_POSE2]
                assert all(q_["host"] in (HOST, HOST_PHYS) for q_ in fpls_), ("fresh pose plan of another host", [q_["host"] for q_ in fpls_], HOST, HOST_PHYS)
                assert len({q_.get("tag", "npl") for q_ in fpls_}) == len(fpls_), ("appended plans need distinct FRESH_TAGs", [q_.get("tag") for q_ in fpls_])
                tail_ = int(CFG["fresh"]["pose_tail"])
                ap_ = CFG["fresh"].setdefault("pose_append_at", [0] * len(FRESH_POSE2))    # (a fresh start: from step 0)
                ends_ = [FRESH_END1]
                for k_ in range(1, len(fpls_)):                   # the end after each plan (all_reduce MIN: every rank, every boundary)
                    assert int(ap_[k_ - 1]) <= ends_[k_ - 1], ("appended fresh pose plan added at a checkpoint past the end of the plans before it: "
                                                               "the steps in between would change", k_ + 1, ap_[k_ - 1], ends_[k_ - 1])
                    frows_ = FP.append_stream([q_["n_rows"] for q_ in fpls_[:k_ + 1]], LOCAL, LW, npm, Sf_, tail_, ends_)
                    fmn2_ = torch.tensor([float(len(frows_) // max(npm, 1))], device=dev); fmx2_ = fmn2_.clone()
                    dist.all_reduce(fmn2_, op=dist.ReduceOp.MIN); dist.all_reduce(fmx2_, op=dist.ReduceOp.MAX)
                    ends_.append(Sf_ + int(fmn2_))
                FRESH_END = ends_[-1]
                assert np.array_equal(frows_[:int(fmn_) * npm], FP.rank_rows(fpl_["n_rows"], LOCAL, LW)[:int(fmn_) * npm]), "plan-1 history changed"
                dg_ = [None] * WORLD; dist.all_gather_object(dg_, [fpl_["host"], [q_["digest"] for q_ in fpls_]])
                dig_ = {}
                for h_, d_ in dg_:
                    assert dig_.setdefault(h_, d_) == d_, ("ranks of one host loaded different fresh pose plans", h_, dig_[h_], d_)
                assert len(dig_) == WORLD // LW, ("fresh pose plans: one host per slot", sorted(dig_))
                sv_ = FRESH_SAVED_DIG[0]
                if sv_ is not None:                               # resume: the checkpoint's plans (a prefix of these) are the same files
                    assert set(sv_) == set(dig_) and all(dig_[h_][:len(sv_[h_])] == list(sv_[h_]) for h_ in dig_), \
                        ("resume: fresh pose plan digests differ from the checkpoint's", sv_, dig_)
                CFG["fresh"]["pose_digests"] = dig_
                fx2_ = allsum([float(len(frows_)), float(sum(q_["kept_frames"] for q_ in fpls_[1:])) if LOCAL == 0 else 0.0,
                               float(max(0, len(FP.rank_rows(fpl_["n_rows"], LOCAL, LW)) - int(fmn_) * npm)) if tail_ else 0.0], dev)
                log(f"[fresh] POSE APPEND: plans {[FRESH_POSE] + list(FRESH_POSE2)} (rank 0 host {fpl_['host']}: rows {[q_['n_rows'] for q_ in fpls_]}, "
                    f"digests {[q_['digest'] for q_ in fpls_]}); ends after each plan {ends_} (plan-1 end FRESH_END1 {FRESH_END1}); tail {tail_} "
                    f"({int(fx2_[2]):,} unused plan-1 rows {'carried' if tail_ else 'dropped'}); appended plans {fx2_[1] / 12.5 / 3600:.0f} h; "
                    f"combined rows/rank {len(frows_):,} (rank 0), over ranks {int(fx2_[0]):,}; steps supported {int(fmn2_)}..{int(fmx2_)} -> the "
                    f"fresh pose ends at step {FRESH_END}; plans added at steps {ap_}; digests "
                    + ("== the checkpoint's (prefix)" if sv_ is not None else "recorded in the fresh block from here"))
            steps_avail = min(steps_avail, FRESH_END)
            log(f"[fresh] POSE from step index {Sf_}: fresh rows (plan {FRESH_POSE}, host {fpl_['host']} digest {fpl_['digest']}, {fpl_['n_rows']:,} rows "
                f"{fpl_['kept_frames'] / 12.5 / 3600:.0f} h on this host); rows/rank {len(frows_):,} (rank 0), over ranks {int(fx_[0]):,} rows "
                f"{fx_[1] / 12.5 / 3600:.0f} h; steps supported {int(fmn_)}..{int(fmx_)} (min / max over ranks) -> the fresh pose ends at step {FRESH_END} "
                f"(pose rows never repeat); global dedup {json.dumps({k_: v_ for k_, v_ in fpl_['global'].items() if not k_.startswith(('kept_slinky', 'frames_kept_slinky'))})}")
        x_ = allsum([float(max(0, len(r1_) - pc_)), float(max(0, len(sp_rows) - sc_)), float(len(sp_rows) - len(sp_set))], dev)
        log(f"[811] data from step {fs_}: pose = the rest of PPM's epoch-1 order ({int(x_[0]):,} rows over all ranks, rank 0 {max(0, len(r1_) - pc_):,}) then "
            f"epoch 2.. = fresh node permutations (seed {CFG['s811']['pose_ep_seed']} + e - 2); rank-0 stream epochs at stream row {pmarks}; speech = the "
            f"rest of PPM's D2 stream ({int(x_[1]):,} rows, rank 0 {max(0, len(sp_rows) - sc_):,}) then epochs of the same rows minus {int(x_[2])} holdout "
            f"(seed {CFG['s811']['speech_ep_seed']} + e - 2, rank); rank-0 epochs at {smarks}; text = PPM's per-source pack order by global step; "
            f"{need_:,} 8,1,1 steps from {fs_} (text available to step {steps_avail:,})")
    max_steps = min(CFG["max_steps"], steps_avail)
    log(f"[data] pose node {pnode['node']}: train rows/rank {n_pose0:,}; speech node {snode['node']}: FULL train rows/rank {n_speech:,}; text packs "
        f"{n_text:,} Gemma + {n_chat:,} chat (split {CFG['chat_split']}, {CFG['text_mix']}) (world-strided); steps available {steps_avail:,}; max_steps {max_steps}")
    # ---- model
    c32, c4, c06 = M.cfg32(CFG["layers"]), M.cfg4b(CFG["tower_layers"]), M.cfg06(CFG["dec_layers"])
    partial = CFG["layers"] != 64
    model, info = M.build(c32, c4, c06, dev, log, partial=partial, proj_r=CFG["proj_r"], tower_vocab=CFG["tower_vocab"], dec_ac=CFG["dec_ac"], dec_mode=CFG["dec_mode"],
                          pose_head=bool(CFG["pose_head"]), **({"decp_bb": True} if CFG["decp_bb"] else {}), **({"s2p": S2P} if S2P else {}),
                          **({"g0cond": True} if G0 else {}))
    assert bool(S2P) == hasattr(model.tower_p, "s_emb"), ("POSE-S2P: module presence != PPM_S2P", S2P)
    assert G0 == hasattr(model.inp, "g0") == model.g0cond, ("G0COND: module presence != PPM_G0COND", G0)
    if G0:
        log(f"[g0cond] G0COND on: inp.g0 {tuple(model.inp.g0.weight.shape)} (zero init, no bias; main-AdamW group '{M.group_of('inp.g0.weight')}'): "
            f"trunk input at each pose clip's SLOT position += g0(frame-0 features, {D.G0_DIM}-d); pose rows only")
    if S2P:
        log(f"[s2p] POSE-S2P on ({S2P}): tower_p.s_emb {tuple(model.tower_p.s_emb.weight.shape)} (zero init; main-AdamW group "
            f"'{M.group_of('tower_p.s_emb.weight')}'): pose tower input at T += s_emb(channel B's Mimi codes at T); pose rows only")
    if PH_OFF:                                                     # D36: the head stays in the module (weights saved), skipped in forward
        model.pose_head_on = False
    if BB_OFF:                                                     # LINK OFF: the link is never computed in training
        model.decp_bb_on = False
    if CFG["dec_fast"]:
        for dec in (model.dec_s, model.dec_p):
            M.compile_decoder(dec, CFG["dec_compile_mode"], whole=CFG["dec_mode"] == "replicate", dynamic=CFG["dec_dyn"])
    if CFG["keep_last"] > 0:
        for L in list(model.layers)[-CFG["keep_last"]:]:
            L.ac = False
    rot = M.Rot(c32, dev)
    opt, sched, by = make_opt(model)
    dos = {"dec_s": DecOpt(model.dec_s, "dec_s"), "dec_p": DecOpt(model.dec_p, "dec_p")}
    n_main = sum(len(v) for v in by.values())
    frozen = sorted(n_ for n_, _ in model.named_parameters() if (PH_OFF and M.group_of(n_) == "pose_head") or (BB_OFF and M.group_of(n_) == "decp_bb"))
    assert frozen == sorted((POSE_HEAD_KEYS if PH_OFF else ()) + (DECP_BB_KEYS if BB_OFF else ())), frozen
    assert n_main + len(dos["dec_s"].ps) + len(dos["dec_p"].ps) + len(frozen) == len(list(model.parameters())), \
        "every parameter in exactly one optimizer (D36: except the frozen OFF head)"
    if BB_OFF:                                                     # LINK OFF startup asserts: in no optimizer, never computed in training
        ids_ = {id(p_) for o_ in [opt] + [d_.opt for d_ in dos.values()] for g_ in o_.param_groups for p_ in g_["params"]}
        lk_ = [(n_, p_) for n_, p_ in model.named_parameters() if M.group_of(n_) == "decp_bb"]
        assert lk_ and not any(id(p_) in ids_ for _, p_ in lk_) and not model.decp_bb_on, "LINK OFF: decp_bb must be frozen, in no optimizer, not computed"
        log(f"[link-off] D33 decp_bb link REMOVED from training (PPM_DECP_BB_OFF=1): {[n_ for n_, _ in lk_]} kept in the module / checkpoints, "
            f"frozen, in no optimizer, never computed in training (pose decoder zb=None, asserted per micro); gradient-free asserted every step; "
            f"eval main pose light = link off, probe decp_linkon_* = the frozen link re-added")
    if PH_OFF:
        log(f"[D36] pose head OFF (user OK 10-14): {frozen} kept in the module / checkpoints, frozen (no optimizer), skipped in forward; "
            f"groups {[pg['name'] for pg in opt.param_groups]}")
    hosts = [None] * WORLD; dist.all_gather_object(hosts, HOST)
    if HOST_ALIAS:                                                # placement log: logical slot <- physical host (PPM_HOST_ALIAS)
        phys_ = [None] * WORLD; dist.all_gather_object(phys_, HOST_PHYS)
        sw_ = sorted({(h_, p_) for h_, p_ in zip(hosts, phys_) if h_ != p_})
        log(f"[hosts] logical slots played by other hosts (PPM_HOST_ALIAS): {', '.join(f'{h_} <- {p_}' for h_, p_ in sw_)}; ranks "
            + ", ".join(f"{h_} {min(i for i, x in enumerate(hosts) if x == h_)}..{max(i for i, x in enumerate(hosts) if x == h_)}" for h_, _ in sw_))
    SF.install_usr1(log)
    ctl = SF.Control(CFG["control_dir"], RANK, dev, log)
    start, rsrc = 0, None
    if S811:                                                       # (resolved before the data plans: same rule)
        rsrc = rsrc0
    elif CFG["resume_from"] == "auto":
        m_ = SF.newest_common(CKPT, dev)
        rsrc = f"{CKPT}/dcp-{m_}" if m_ >= 0 else None
        log(f"[resume] auto: newest step COMPLETE on every node under {CKPT}: {m_ if m_ >= 0 else 'none (fresh start)'}")
    elif CFG["resume_from"]:
        rsrc = CFG["resume_from"]
    if rsrc:
        start = load(model, opt, sched, dos, rsrc, hosts)
        log(f"resumed {rsrc} at step {start}")
    if S811:
        assert start >= CFG["s811"]["from_step"], (start, CFG["s811"]["from_step"])
        if CFG["s811"]["protect_switch"]:                          # never delete the switch checkpoint (PPM's final state, the branch point)
            SF.PROTECT_STEPS.add(int(CFG["s811"]["from_step"]))
            log(f"[811] the switch checkpoint {CKPT}/dcp-{CFG['s811']['from_step']} is never deleted by this run (keep-last / space pruning); "
                f"other saves rotate as in PPM (keep last {CFG['keep_ckpt']})")
    log(f"[safety] checkpoints {CKPT}/dcp-N (node-local, keep last {CFG['keep_ckpt']}); control {CFG['control_dir']}/SAVE_NOW | STOP_AFTER_SAVE; "
        f"export every {CFG['export_every']} -> {CFG['export_root']} (floor {CFG['export_min_free_gb']:g} GB free after the write, parts {CFG['export_parts']})")
    # ---- loaders (rows consumed in step order: step t uses pose rows [t npm, ..), speech rows [t nsm, ..), text pack t WORLD + RANK)
    W_ = CFG["dl_workers"]
    ld = {}
    if S811:                                                       # 8,1,1 streams: index 0 = the first micro of step from_step
        fs_ = CFG["s811"]["from_step"]
        if npm and FRESH_POSE:                                    # fresh pose: the rank's fresh rows from step index S_f on (never repeated)
            assert start >= Sf_, ("resume before the fresh-pose start", start, Sf_)
            pstream = frows_[(start - Sf_) * npm:(max_steps - Sf_) * npm]
            assert len(pstream) == (max_steps - start) * npm, ("fresh pose rows short of max_steps", len(pstream), max_steps, start)
            ld["pose"] = loaders_for(FP.FreshPoseDS(fpl_, pstream, pcfg, RANK) if not FRESH_POSE2 else FP.FreshPoseMultiDS(fpls_, pstream, pcfg, RANK), W_)
            pstream = np.concatenate([np.full((start - fs_) * npm, -1, np.int64), pstream])   # (row log below: index (t - fs_) x npm; -1 = unused)
        elif npm:
            ld["pose"] = loaders_for(D.PoseDS(pnode, pstream[(start - fs_) * npm:(max_steps - fs_) * npm], pcfg, RANK), W_)
        if nsm:
            sr_ = sstream[(start - fs_) * nsm:(max_steps - fs_) * nsm]
            ld["speech"] = loaders_for(D.SpeechDS(CFG["speech_root"], sr_, dcfg, RANK) if hplan is None else
                                       D.MixedSpeechDS(CFG["speech_root"], hplan, sr_, dcfg, RANK), W_)
        if CFG["s811_log_rows"] and RANK in (0, WORLD - 1):
            for t_ in range(start, min(max_steps, start + CFG["s811_log_rows"])):
                print(f"[811 rows] rank {RANK} step {t_ + 1}: pose {pstream[(t_ - fs_) * npm:(t_ - fs_ + 1) * npm].tolist()} speech "
                      f"{sstream[(t_ - fs_) * nsm:(t_ - fs_ + 1) * nsm].tolist()} text {text_src(t_)}", flush=True)
    elif npm:
        ld["pose"] = loaders_for(D.PoseDS(pnode, pplan["rows"][start * npm:max_steps * npm], pcfg, RANK), W_)
    if nsm and not S811:
        ld["speech"] = loaders_for(D.SpeechDS(CFG["speech_root"], splan["rows"][start * nsm:max_steps * nsm], dcfg, RANK) if hplan is None else
                                   D.MixedSpeechDS(CFG["speech_root"], hplan, splan["rows"][start * nsm:max_steps * nsm], dcfg, RANK), W_)
    if ntm:
        kc = {"gemma": 0, "chat": 0}
        gp, cp_ = [], []
        mine = cpacks[CFG["chat_offset"] + RANK::WORLD]               # this rank's chat packs; a 2nd pass (same order) once exhausted
        if S811:                                                  # 8,1,1: packs consumed per source with each step's own text micro count
            used_ = {"gemma": 0, "chat": 0}                       #   (PPM's GA before the switch): = the production loop for a constant GA
            for t_ in range(max_steps):
                s_, k_ = text_src(t_), S8.seq_at(t_, CFG["s811"]["from_step"], CFG["s811"]["prev_ga_seq"], SEQ).count("text")
                if t_ >= start:
                    for j in range(k_):
                        if s_ == "gemma":
                            gp.append((CFG["text_offset"] + (used_[s_] + j) * WORLD + RANK)
                                      | (D.FLIP_BIT if TEXT_FLIP_FROM is not None and t_ >= TEXT_FLIP_FROM else 0))   # fresh: text flip
                        else:
                            cp_.append(mine[(used_[s_] + j) % len(mine)] + ((True,) if TEXT_FLIP_FROM is not None and TEXT_FLIP_CHAT and t_ >= TEXT_FLIP_FROM else ()))
                used_[s_] += k_
            kc = {k: -(-v // max(ntm, 1)) for k, v in used_.items()}  # (log line below: micros = kc x ntm)
            if TEXT_FLIP_FROM is not None:                         # fresh resume: text flip summary (rank 0's packs; chat pass position)
                nfg_ = sum(1 for g_ in gp if int(g_) & D.FLIP_BIT); nfc_ = sum(1 for c_ in cp_ if len(c_) > 2)
                uc0_ = sum(k_ for t_ in range(start) for k_ in [S8.seq_at(t_, CFG["s811"]["from_step"], CFG["s811"]["prev_ga_seq"], SEQ).count("text")]
                           if text_src(t_) == "chat")
                log(f"[fresh] TEXT FLIP (PPM_TEXT_FLIP=1, from step index {TEXT_FLIP_FROM}): rank 0 flips {nfg_:,}/{len(gp):,} Gemma packs (swap = 1 - "
                    f"side_of) and {nfc_:,}/{len(cp_):,} chat packs ({'user on B, <|im_end|> on the user turn ends' if TEXT_FLIP_CHAT else 'chat flip OFF: PPM_TEXT_FLIP_CHAT=0'}); Gemma packs never repeat "
                    f"(world-strided, the run ends when they run out); chat packs per rank {len(mine):,}, consumed before step {start + 1}: "
                    f"{uc0_:,} ({'2nd pass' if uc0_ >= len(mine) else '1st pass'}: pass {uc0_ // max(len(mine), 1) + 1}, pack {uc0_ % max(len(mine), 1):,})")
            if CFG["s811_log_rows"] and RANK in (0, WORLD - 1):
                print(f"[811 rows] rank {RANK} text packs of steps {start + 1}..: gemma {gp[:CFG['s811_log_rows']]} chat {cp_[:CFG['s811_log_rows']]}", flush=True)
        for t_ in (range(max_steps) if not S811 else ()):          # per-rank source counters from step 0 (exact on resume)
            s_ = text_src(t_)
            if t_ >= start:
                for j in range(ntm):
                    if s_ == "gemma":
                        gp.append(CFG["text_offset"] + (kc[s_] * ntm + j) * WORLD + RANK)
                    else:
                        cp_.append(mine[(kc[s_] * ntm + j) % len(mine)])
            kc[s_] += 1
        if gp:
            ld["gemma"] = loaders_for(D.TextDS(CFG["text_root"], "train", gp, dcfg), W_)
        nwrap = torch.tensor([float(max(0, kc["chat"] * ntm - len(mine)))], device=dev); dist.all_reduce(nwrap, op=dist.ReduceOp.MAX)
        log(f"[data] chat packs per rank {len(mine):,}; chat micros this run (rank 0) {kc['chat'] * ntm:,}; max over ranks re-read in a 2nd pass: {int(nwrap)}")
        if cp_:
            ld["chat"] = loaders_for(CH.ChatDS(cp_, dcfg), W_)
    ev = {"pose": [D.PoseDS(pnode, pplan["eval_rows"], pcfg, RANK)[i] for i in range(len(pplan["eval_rows"]))],
          "speech": [D.SpeechDS(CFG["speech_root"], splan["eval_rows"], dcfg, RANK)[i] for i in range(len(splan["eval_rows"]))],
          "gemma": [D.TextDS(CFG["text_root"], "eval", list(range(RANK * E_, (RANK + 1) * E_)), dcfg)[i] for i in range(E_)],
          "chat": [CH.ChatDS([cev[(RANK * E_ + i) % len(cev)] for i in range(E_)], dcfg)[i] for i in range(E_)]}
    assert all(len(v) == E_ for v in ev.values()), {k: len(v) for k, v in ev.items()}
    _sx = D.MixedSpeechDS(CFG["speech_root"], None, [], dcfg, RANK)         # eval_speech_A: the same eval rows, sides flipped (the channel-A
    ev["speech_A"] = [_sx.get(D.code_of(0, r), swap=1 - int(m["swap"])) for r, m in zip(splan["eval_rows"], ev["speech"])]   # speaker on the predicted B)
    evw = {k: [1.0] * len(v) for k, v in ev.items()}                # per-micro weight (0 = a filler micro: every rank runs as many forwards)
    ebase = {"pose": "pose", "speech": "speech", "gemma": "gemma", "chat": "chat", "speech_A": "speech"}
    epool = {}

    def _spread(n):                                               # global item i -> rank i % WORLD; fillers (weight 0) even the counts out
        k = -(-n // WORLD)
        return [((RANK + j * WORLD), 1.0) if RANK + j * WORLD < n else ((RANK + j * WORLD) % n, 0.0) for j in range(k)]
    if CFG["eval_s3_dir"]:                                        # stage-3 eval rows (Demeter 'ourrows'), lane B = stage-3 channel B, A = flipped
        man = json.load(open(f"{CFG['eval_s3_dir']}/MANIFEST.json"))
        _nl = lambda c: {a: b for a, b in c.items() if a not in ("light_frac", "pose_light_frac")}   # D31: the eval set stays EXACTLY as built
        assert json.dumps(_nl(man["cfg"]), sort_keys=True) == json.dumps(_nl(dcfg), sort_keys=True), ("speech_s3 micros built with another data cfg", man["cfg"], dcfg)
        if man["cfg"].get("light_frac") != dcfg.get("light_frac"):    #   (its stored light frames = a 10% subset; eval-only, comparable across D31)
            log(f"[eval] speech_s3: pre-built micros keep their light frames (built at light_frac {man['cfg'].get('light_frac')}; training now {dcfg.get('light_frac')})")
        for o in ("A", "B"):
            its = _spread(int(man["n"]))
            ev[f"speech_s3_{o}"] = [pickle.load(open(f"{CFG['eval_s3_dir']}/{o}-{i:03d}.pkl", "rb")) for i, _ in its]
            evw[f"speech_s3_{o}"] = [w for _, w in its]; ebase[f"speech_s3_{o}"] = "speech"
        epool["speech_s3"] = ("speech_s3_A", "speech_s3_B")
        log(f"[eval] speech_s3: {man['n']} rows x lanes A, B ({man.get('source', '')}); {len(its)} micro(s)/rank/lane")
    if CFG["eval_gemma_s3_root"]:                                 # gemma-txtdup-v1 eval packs 0..N-1, swap = parity of the eval-list position
        gst = D.TXD.Store(CFG["eval_gemma_s3_root"])
        gcfg = dict(dcfg, txt_cap=int(gst.cap))
        npk = int(CFG["eval_gemma_s3_packs"]); assert npk <= gst.n_packs("eval"), (npk, gst.n_packs("eval"))
        its = _spread(npk)
        for o, fl in (("A", 1), ("B", 0)):
            mic = []
            for p_, _ in its:
                ids_ = gst.pack("eval", p_); recs_ = [gst.get(j) for j in ids_]
                mic.append(D.packed(D.text_micro(recs_, gcfg, sws=[((int(gst.eval_ptr[p_]) + k) % 2) ^ fl for k in range(len(recs_))]), gcfg))
            ev[f"gemma_s3_{o}"] = mic; evw[f"gemma_s3_{o}"] = [w for _, w in its]; ebase[f"gemma_s3_{o}"] = "gemma"
        epool["gemma_s3"] = ("gemma_s3_A", "gemma_s3_B")
        log(f"[eval] gemma_s3: {CFG['eval_gemma_s3_root']} eval packs 0..{npk - 1} (cap {gst.cap}) x lanes A, B; {len(its)} micro(s)/rank/lane")
    # ---- 10-12 dependence probes (doc above codep_micro / xdep_micro)
    cdp = None
    if CFG["codep"]:
        t0c = time.time()
        prs_ = D.PS.PoseRows(pnode)
        rows_ = [prs_[int(i)] for i in pplan["eval_rows"]]
        pdc_ = {"new_word": dcfg["new_word"], "pose_delay": dcfg["pose_delay"], "light_seed": dcfg["light_seed"]}
        cdp = CDP.build(rows_, pdc_, RANK, CFG["codep_frames"], CFG["codep_seed"] + RANK, CFG["codep_light_frac"], CFG["codep_light_frac"])
        cdp["micros"] = [codep_micro(cdp, v, dcfg) for v in CDP.VARIANTS]
        if G0:                                                     # G0COND: the subset clips' frame-0 features per variant (codep_g0)
            codep_g0(cdp, rows_)
    if G0:                                                         # audit NIT: never a silent zero (counts of this rank's eval pose micros)
        g0s_ = allsum([float(D.G0_STATS["clips"]), float(D.G0_STATS["invalid"]), float(D.G0_STATS["missing_field"])], dev)
        log(f"[g0cond] startup: eval pose micros over all ranks: {int(g0s_[0])} clips, {int(g0s_[1])} without usable frame-0 fields "
            f"({int(g0s_[2])} missing a field) -> f = 0 there; training rows: every such clip is logged by the loader ([g0cond] WARNING)")
        nc_ = allsum([float(cdp["info"]["K"]), float(cdp["info"]["N"])], dev)
        log(f"[codep] codependence subset: {int(nc_[0])} held-out pose clips (this rank's {len(rows_)} pose eval rows), {int(nc_[1]):,} positions per variant "
            f"over all ranks ({cdp['info']['N']} on rank 0), variants {CDP.VARIANTS}, one forward each; seed {CFG['codep_seed']}+rank, light frac "
            f"{CFG['codep_light_frac']}; built in {time.time() - t0c:.1f}s")
    xdp = None
    if CFG["xdep"] and CFG["eval_s3_dir"]:
        nx_ = min(CFG["xdep_rows"], int(man["n"]))
        sil_ = np.asarray(json.load(open(SIL_CODES))["seq"], np.int64)
        assert sil_.shape[0] == D.DEPTH and sil_.min() >= 0 and sil_.max() < D.CBS, sil_.shape
        items_ = [(c_, o, i) for c_ in CFG["xdep_conds"] for o in ("A", "B") for i in range(nx_)]
        its_ = _spread(len(items_))
        xdp = {"n": nx_, "conds": list(CFG["xdep_conds"]), "w": [w for _, w in its_], "cond": [items_[j][0] for j, _ in its_],
               "mic": [xdep_micro(pickle.load(open(f"{CFG['eval_s3_dir']}/{items_[j][1]}-{items_[j][2]:03d}.pkl", "rb")), items_[j][0], sil_) for j, _ in its_],
               "sub": {f"speech_s3_{o}": [float(i < nx_) for i, _ in _spread(int(man["n"]))] for o in ("A", "B")}}   # intact subset mask per speech_s3 micro
        log(f"[xdep] B given A: speech_s3 rows 0..{nx_ - 1} x lanes A, B x conditions {xdp['conds']} = {len(items_)} forwards ({len(its_)}/rank); "
            f"ablation on channel A's conversation frames, trunk input only; intact = the same rows' speech_s3 forwards")
    jl = open(f"{LOGDIR}/metrics.jsonl", "a") if RANK == 0 else None

    def emit(d, step):
        if RANK == 0:
            jl.write(json.dumps({"step": step, "time_pt": now_pt(), **d}) + "\n"); jl.flush()

    def evaluate(step):
        if CFG["pose_head"]:                                      # D35: the head's backbone-grad scale (forward identical for every s)
            model.pose_head_bbgrad = posehead_bbgrad(step)
        if CFG["decp_bb"]:                                        # D33: the link's scale at this step (eval at `step` = before step + 1)
            model.decp_bb_alpha = decp_bb_alpha(step)
        if BB_OFF:                                                # LINK OFF: computed in eval for the decp_linkon probe only
            model.decp_bb_on = True
        model.eval()
        t0 = time.time()
        d, info0 = {}, {}
        sums = {}
        xreal = [0.0] * NV["speech"]                                  # xdep: intact sums of the probe rows (speech_s3 lanes A + B, rows < n)
        for kind in ev:
            acc = None
            for j_, (b, w) in enumerate(zip(ev[kind], evw[kind])):
                v, _ = run_micro(model, prep(b, dev), rot, dos, grad=False, info=info0 if kind == "speech" else None,
                                 **({"bbvar": True} if (CFG["decp_bb"] and kind == "pose") else {}))
                if xdp is not None and kind in xdp["sub"]:
                    xreal = [x + y * w * xdp["sub"][kind][j_] for x, y in zip(xreal, v)]
                v = [x * w for x in v]
                acc = v if acc is None else [x + y for x, y in zip(acc, v)]
            sums[kind] = allsum(acc, dev)
            d.update(summarize(kind, sums[kind], "eval", ebase[kind]))
        d.update({k.replace("eval_speech/", "eval_speech_B/"): v for k, v in list(d.items()) if k.startswith("eval_speech/")})   # B = the scored channel
        for kind, (ka, kb) in epool.items():                     # both lanes pooled (token sums), = the stage-3 / Demeter pooled mean
            d.update(summarize(kind, [x + y for x, y in zip(sums[ka], sums[kb])], "eval", ebase[ka]))
        torch.cuda.synchronize(); t_norm = time.time() - t0
        if cdp is not None:                                       # CODEPENDENCE (pose rows): 5 forwards, identical layout
            t1 = time.time()
            vals, tim = [], []
            for vi, mc in enumerate(cdp["micros"]):
                kp = {}
                v, _ = run_micro(model, prep(mc, dev), rot, dos, grad=False, keep=kp)
                vals += codep_vals(v)
                if vi in cdp["info"]["probe_v"]:
                    tim += codep_timing(cdp, kp["text_stats"])
            if CFG["abl_selftest"]:                               # self-test: the 'real' variant without the *_t overrides = bitwise the same
                b0 = {k: v_ for k, v_ in cdp["micros"][0].items() if not k.endswith("_t")}
                v0, _ = run_micro(model, prep(b0, dev), rot, dos, grad=False)
                dd_ = torch.tensor([max(abs(a - b_) for a, b_ in zip(codep_vals(v0), vals[:10]))], device=dev, dtype=torch.float64)
                dist.all_reduce(dd_, op=dist.ReduceOp.MAX)
                d["codependence/selftest_max_abs_diff"] = float(dd_)
                log(f"[codep] SELF-TEST: real variant via trunk overrides vs the plain micro: max |sum diff| over ranks {float(dd_):.3g}")
            d.update(CDP.summarize(allsum(vals + tim, dev)))
            torch.cuda.synchronize(); d["codependence/time_s"] = time.time() - t1
        if xdp is not None:                                       # XDEP (speech rows): B given A, the ablated forwards
            t2 = time.time()
            acc = {c_: [0.0] * NV["speech"] for c_ in xdp["conds"]}
            for b, c_, w in zip(xdp["mic"], xdp["cond"], xdp["w"]):
                v, _ = run_micro(model, prep(b, dev), rot, dos, grad=False)
                acc[c_] = [x + y * w for x, y in zip(acc[c_], v)]
            tot = allsum(xreal + [x for c_ in xdp["conds"] for x in acc[c_]], dev)
            K_ = NV["speech"]
            sv = {"real": tot[:K_], **{c_: tot[K_ * (i + 1):K_ * (i + 2)] for i, c_ in enumerate(xdp["conds"])}}
            for c_, s_ in sv.items():
                for mk, a_, n_ in XD_KEYS:
                    d[f"xdep/{'intact' if c_ == 'real' else 'a_' + c_}/{mk}"] = s_[a_] / max(s_[n_], 1.0)
            for c_ in xdp["conds"]:
                for mk, a_, n_ in XD_KEYS:
                    d[f"xdep/b_given_a/d_{mk}_{c_}"] = d[f"xdep/a_{c_}/{mk}"] - d[f"xdep/intact/{mk}"]
            d["xdep/words"], d["xdep/rows"] = sv["real"][2], float(2 * xdp["n"])
            if "none" in xdp["conds"]:                            # self-test: no ablation reproduces the intact speech_s3 sums
                dmx = max(abs(a - b_) for a, b_ in zip(sv["none"], sv["real"]))
                d["xdep/selftest_max_abs_diff"] = dmx
                log(f"[xdep] SELF-TEST: condition 'none' vs the intact speech_s3 forwards of the same rows: max |sum diff| {dmx:.3g} "
                    f"(word sums {sv['none'][1]:.6f} / {sv['real'][1]:.6f}, cb0 {sv['none'][10]:.6f} / {sv['real'][10]:.6f})")
            torch.cuda.synchronize(); d["xdep/time_s"] = time.time() - t2
        model.train()
        if BB_OFF:
            model.decp_bb_on = False
        d["eval/time_s_base"] = t_norm
        d["eval/time_s"] = time.time() - t0
        d.update({f"eval/{k}": v for k, v in info0.items()})
        emit(d, step)
        log(f"EVAL step {step}: " + " | ".join(f"{k} {v:.4f}" for k, v in d.items()))
        for k_, v_ in d.items():
            assert math.isfinite(v_), (k_, v_)
        return d

    if PH_ACTIVE and CFG["pose_head_detach_from"] >= 0:
        log(f"[resume] D35 pose head detach ADDED at step {CFG['pose_head_detach_from']}: the D32 head is detached from the backbone (user OK 10-13): backbone-grad scale s = 1 -> 0 linearly over {CFG['pose_head_detach_ramp']} "
            f"steps from optimizer step {CFG['pose_head_detach_from']} (head input hn.detach() + s x (hn - hn.detach()); forward / head loss / head "
            f"LR unchanged), then full detach; the head keeps training as a probe. Resumed at step {start}: s now {posehead_bbgrad(start)}")
    d_start = evaluate(start)
    if S811 and CFG["s811_eval_ref"] and RANK == 0:                # 8,1,1: the start eval vs PPM's eval at the same step (same weights / eval sets)
        import glob as _gl
        ref_ = {}
        for f_ in sorted(_gl.glob(CFG["s811_eval_ref"])):
            if os.path.abspath(f_) == os.path.abspath(f"{LOGDIR}/metrics.jsonl"):   # never this run's own record
                continue
            for ln_ in open(f_, errors="replace"):
                if f'"step": {start},' in ln_ and '"eval_' in ln_:
                    ref_ = json.loads(ln_)
        ks_ = sorted(k_ for k_ in d_start if k_ in ref_ and k_.startswith(("eval_", "codependence/", "xdep/")) and not k_.endswith("time_s"))
        if ks_:
            df_ = sorted(((abs(d_start[k_] - ref_[k_]), k_) for k_ in ks_), reverse=True)
            log(f"[811] START EVAL vs PPM's eval at step {start} ({CFG['s811_eval_ref']}): {len(ks_)} common metrics, max |diff| {df_[0][0]:.3g} "
                f"({df_[0][1]}); top: " + ", ".join(f"{k_} {x_:.3g}" for x_, k_ in df_[:6]) + f"; exact matches {sum(1 for x_, _ in df_ if x_ == 0.0)}")
        else:
            log(f"[811] START EVAL: no PPM eval record at step {start} in {CFG['s811_eval_ref']} (nothing to compare)")
    import gc                                                       # = the stage-3 / pose trainers: no automatic GC inside steps (a GC pause
    gc.collect(); gc.freeze(); gc.disable()                         #   on ANY of 128 ranks stalls every FSDP collective); manual GC at boundaries
    times, stopped = [], None
    T_LAST = {"save": time.time(), "export": time.time(), "eval": time.time()}
    if S811:                                                       # 8,1,1 step machinery (doc above)
        bb_ps = [p_ for pg in opt.param_groups if pg["name"] in BB_GROUPS for p_ in pg["params"]]
        tw_pgs = [pg for pg in opt.param_groups if pg["name"] in TW_GROUPS]
        side_ps = {"s": [p_ for pg in tw_pgs if pg["name"] == "tower_s" for p_ in pg["params"]],
                   "p": [p_ for pg in tw_pgs if pg["name"] in ("tower_p", "decp_bb", "s2p") for p_ in pg["params"]]}
        CAD = CFG["s811"]["tower_cadence"]

        def gsnap(ps):
            return [None if p_.grad is None else M._loc(p_.grad).detach().clone() for p_ in ps]

        def gsame(ps, snap):
            return all((g0 is None and p_.grad is None) or (g0 is not None and p_.grad is not None and torch.equal(g0, M._loc(p_.grad)))
                       for p_, g0 in zip(ps, snap))

        def tower_step(gacc):
            """PPM_TOWER_CADENCE=micro: clip + AdamW-step the tower-side groups on this micro's gradient (the backbone's accumulated gradient
            is set aside meanwhile: fused AdamW skips parameters without a gradient), then drop their gradients."""
            for pg in tw_pgs:
                if any(p_.grad is not None for p_ in pg["params"]):
                    g_ = torch.nn.utils.clip_grad_norm_([p_ for p_ in pg["params"] if p_.grad is not None], CFG["clip"][pg["name"]])
                    gacc.setdefault(pg["name"], []).append(float(g_.full_tensor() if hasattr(g_, "full_tensor") else g_))
            stash = [(p_, p_.grad) for p_ in bb_ps]
            for p_ in bb_ps:
                p_.grad = None
            opt.step()
            for p_, g_ in stash:
                p_.grad = g_
            for pg in tw_pgs:
                for p_ in pg["params"]:
                    p_.grad = None
        log(f"[811] step machinery: {len(bb_ps)} backbone tensors (groups {[pg['name'] for pg in opt.param_groups if pg['name'] in BB_GROUPS]}) step once "
            f"per optimizer step; tower-side groups {[pg['name'] for pg in tw_pgs]} cadence {CAD}; grad-isolation checks for the first "
            f"{CFG['s811_check_steps']} step(s)")
    for step in range(start, max_steps):
        torch.cuda.synchronize(); ts = time.time()
        torch.cuda.reset_peak_memory_stats()
        vs = {k: None for k in ("pose", "speech", "gemma", "chat")}
        tm = {k: 0.0 for k in ("pose", "speech", "text")}
        wait = 0.0
        tracer = None
        if CFG["decp_bb"]:                                        # D33: the link's scale for this step (PPM_DECP_BB_ALPHA x fade)
            model.decp_bb_alpha = decp_bb_alpha(step)
        if CFG["pose_head"]:                                      # D35: the head's backbone-grad scale for this step (None = off)
            model.pose_head_bbgrad = posehead_bbgrad(step)
            f0_ = CFG["pose_head_detach_from"]
            if PH_ACTIVE and 0 <= f0_ <= step <= f0_ + CFG["pose_head_detach_ramp"]:
                log(f"[D35] step {step + 1}: pose head backbone-grad scale s = {model.pose_head_bbgrad:.4f}"
                    + (" (ramp done: full detach from here)" if model.pose_head_bbgrad <= 0.0 else ""))
        if step == CFG["trace_step"] and RANK in (0, WORLD - 1):
            from torch.profiler import ProfilerActivity, profile
            tracer = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            tracer.__enter__()
        if S811:
            tm8, gacc, chk_ = {}, {}, step < start + CFG["s811_check_steps"]
            S8STAT.clear()
        for i_m, kind in enumerate(SEQ):
            tw = time.time()
            b = prep(next(ld[text_src(step) if kind == "text" else kind]), dev)
            wait += time.time() - tw
            t_m = time.time()
            if not S811:
                v, _ = run_micro(model, b, rot, dos, grad=True)
            else:
                cf = COEF[i_m]
                if chk_:
                    sb_, ss_, sp_ = gsnap(bb_ps), gsnap(side_ps["s"]), gsnap(side_ps["p"])
                v, _ = run_micro(model, b, rot, dos, grad=True, cf=cf)
                if chk_:                                           # backbone grads bit-identical across a frozen micro, changed by a full one;
                    same_ = gsame(bb_ps, sb_)                      #   the micro's tower sides accumulated a gradient
                    tw_ok = (not cf["sp"] or kind == "text" or not gsame(side_ps["s"], ss_)) and (kind != "pose" or not gsame(side_ps["p"], sp_))
                    ok_ = torch.tensor([float(same_ == (cf["mode"] == "frozen")), float(tw_ok)], device=dev)
                    dist.all_reduce(ok_, op=dist.ReduceOp.MIN)
                    log(f"[811 check] step {step + 1} micro {i_m} ({kind} {cf['mode']}): backbone grads bit-identical across the micro: {same_} "
                        f"(rank 0; expected {cf['mode'] == 'frozen'}) | all ranks as expected: {bool(ok_[0])}; tower side(s) accumulated: {bool(ok_[1])}")
                    assert float(ok_.min()) == 1.0, ("8,1,1 grad-isolation check FAILED", step + 1, i_m, kind, cf["mode"])
                    del sb_, ss_, sp_
                if CAD == "micro" and kind != "text":
                    tower_step(gacc)
            sk = b["src"]
            vs[sk] = v if vs[sk] is None else [x + y for x, y in zip(vs[sk], v)]
            torch.cuda.synchronize(); tm[kind] += time.time() - t_m
            if S811:
                k8_ = f"{kind}_{COEF[i_m]['mode']}"
                tm8[k8_] = tm8.get(k8_, 0.0) + time.time() - t_m; tm8[k8_ + "_n"] = tm8.get(k8_ + "_n", 0) + 1
        if tracer is not None:
            tracer.__exit__(None, None, None)
            ka = tracer.key_averages()
            tot = {"nccl": 0.0, "fa3": 0.0, "gemm": 0.0, "all": 0.0}
            for e in ka:
                dt_ = getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0.0)) / 1e6
                if e.key.startswith("ProfilerStep") or e.device_type.name != "CUDA" if hasattr(e, "device_type") else False:
                    pass
                k = e.key.lower()
                if "nccl" in k:
                    tot["nccl"] += dt_
                elif "flash" in k or "fa3" in k:
                    tot["fa3"] += dt_
                elif "gemm" in k or "sm90" in k or "cutlass" in k or "nvjet" in k:
                    tot["gemm"] += dt_
            print(f"[trace] rank {RANK} step {step}: GPU kernel seconds by family (summed, overlapping streams): {tot}\n" +
                  ka.table(sort_by="device_time_total" if hasattr(ka[0], "device_time_total") else "cuda_time_total", row_limit=22), flush=True)
        if PH_OFF:                                                # D36: the OFF head never gets a gradient
            assert model.pose_head.weight.grad is None and model.pose_head.bias.grad is None, "D36: the OFF pose head got a gradient"
        if BB_OFF:                                                # LINK OFF: the removed link never gets a gradient
            assert model.decp_bb.weight.grad is None and model.decp_bb.bias.grad is None, "LINK OFF: the removed link got a gradient"
        gn = {}
        t_opt = time.time()
        for pg in opt.param_groups:
            g_ = torch.nn.utils.clip_grad_norm_(pg["params"], CFG["clip"][pg["name"]])
            gn[pg["name"]] = float(g_.full_tensor() if hasattr(g_, "full_tensor") else g_)
        if S811 and CAD == "micro":                                # tower groups already stepped per micro: their mean per-micro norm
            gn.update({k_: float(np.mean(v_)) for k_, v_ in gacc.items()})
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); dt = time.time() - ts; times.append(dt); t_opt = time.time() - t_opt
        if S811 and chk_:                                          # AdamW step counters per group (cadence check: step = +1 / step, micro = +n)
            log(f"[811 check] step {step + 1} AdamW step counters: " + ", ".join(
                f"{pg['name']} {float(opt.state[pg['params'][0]]['step']) if pg['params'][0] in opt.state else 0:.0f}" for pg in opt.param_groups)
                + f"; decoder micros dec_s {dos['dec_s'].micros} dec_p {dos['dec_p'].micros} (x dec_k {CFG['dec_k']} updates); this step's decoder "
                f"updates / frame coverage: " + ", ".join(f"{k_} {v_[0]} upd, frames {v_[1]:.0f}/{v_[2]:.0f}" for k_, v_ in sorted(S8STAT.items())))
        d = {}
        for kind, v in vs.items():
            if v is not None:
                d.update(summarize(kind, allsum(v, dev), "train"))
        pk = torch.tensor([torch.cuda.max_memory_allocated() / GiB, dt, wait] + [tm[k] for k in ("pose", "speech", "text")], device=dev)
        dist.all_reduce(pk, op=dist.ReduceOp.MAX)
        if S811:                                                   # 8,1,1 timings (max over ranks) + decoder update / frame coverage counters
            k8s = sorted(k_ for k_ in tm8 if not k_.endswith("_n"))
            p8 = torch.tensor([tm8[k_] for k_ in k8s] + [t_opt], device=dev); dist.all_reduce(p8, op=dist.ReduceOp.MAX)
            for k_, x_ in zip(k8s, p8.tolist()):
                d[f"train/micro_s_{k_}"] = x_ / tm8[k_ + "_n"] if k_.endswith("frozen") else x_   # frozen: mean per micro; full: summed
            d["train/opt_s"] = float(p8[-1])
            for nm_ in ("dec_s", "dec_p"):
                if nm_ in S8STAT:
                    u_, fsc_, fav_ = S8STAT[nm_]
                    d[f"train/{nm_}_updates"] = float(u_); d[f"train/{nm_}_frame_frac"] = fsc_ / max(fav_, 1.0)
        d.update({f"train/gn_{k}": x for k, x in gn.items()})
        if CFG["decp_bb"]:                                        # D33: |decp_bb.weight| (Frobenius, after this step's update), alpha used
            w_ = model.decp_bb.weight
            s_ = M._loc(w_).detach().float().pow(2).sum().reshape(1)
            if hasattr(w_, "to_local"):
                dist.all_reduce(s_)
            d["train/decp_bb_wnorm"] = float(s_.sqrt()); d["train/decp_bb_alpha"] = float(model.decp_bb_alpha)
        if PH_ACTIVE and model.pose_head_bbgrad is not None:      # D35: the head's backbone-grad scale used this step
            d["train/pose_head_bbgrad_s"] = float(model.pose_head_bbgrad)
        if "text" in SEQ:
            fr_ = torch.tensor([float(text_src(step) == "gemma")], device=dev); dist.all_reduce(fr_)
            d["train/text_gemma_frac"] = float(fr_) / WORLD
        d.update({"train/step_time": dt, "train/peak_gib": float(pk[0]), "train/data_wait": float(pk[2]), "train/micro_s_pose": float(pk[3]),
                  "train/micro_s_speech": float(pk[4]), "train/micro_s_text": float(pk[5]),
                  "train/dec_s_lr": dos["dec_s"].lr(dos["dec_s"].micros * CFG["dec_k"]), "train/dec_p_lr": dos["dec_p"].lr(dos["dec_p"].micros * CFG["dec_k"]),
                  **{f"train/lr_{pg['name']}": lr_ for pg, lr_ in zip(opt.param_groups, sched.get_last_lr())}})
        for k_, v_ in d.items():
            assert math.isfinite(v_), (k_, v_, "non-finite: fail loudly")
        emit(d, step + 1)
        log(f"step {step + 1}/{max_steps} {dt:.1f}s (pose {float(pk[3]):.1f} speech {float(pk[4]):.1f} text {float(pk[5]):.1f} wait {float(pk[2]):.1f}) "
            f"peak {float(pk[0]):.1f} GiB | " + " ".join(f"{k.split('/', 1)[1] if k.startswith('train_') else k}={v:.4f}" for k, v in d.items()
                                                     if k.endswith(("word_ce_norm", "cb0_ce", "light_ce", "pad_ce")) or "/gn_" in k
                                                     or (S811 and k.startswith(("train/micro_s_pose_", "train/micro_s_speech_", "train/opt_s", "train/dec_p_",
                                                                                "train/dec_s_up", "train/dec_s_fr")))))
        if CFG["profile"] and RANK == 0:
            log("[profile] per-micro phase seconds (rank 0): " + "; ".join(f"{k}: " + ", ".join(f"{p} {np.mean([x.get(p, 0.0) for x in v]):.2f}" for p in v[0]) for k, v in PROF.items()))
        PROF.clear()
        done = step + 1
        gc.collect(2 if done % 100 == 0 else 1)                     # every rank at the same point (step boundary)
        if (CFG["eval_every"] and done % CFG["eval_every"] == 0) or done == max_steps:
            evaluate(done)
        if RANK == 0 and done in (CFG["test_stop_at"], CFG["test_savenow_at"]):       # smokes: exercise the control-file path
            open(f"{CFG['control_dir']}/{'STOP_AFTER_SAVE' if done == CFG['test_stop_at'] else 'SAVE_NOW'}", "w").write(now_pt())
        c_save, c_stop = ctl.poll()
        tdue = torch.tensor([0.0, 0.0, 0.0], device=dev)                # D22: wall-clock cadence, rank 0 decides, broadcast
        if RANK == 0:
            nowt = time.time()
            for i_, (k_, last_) in enumerate((("save_every_min", T_LAST["save"]), ("export_every_min", T_LAST["export"]), ("eval_every_min", T_LAST["eval"]))):
                tdue[i_] = float(CFG[k_] > 0 and nowt - last_ >= 60.0 * CFG[k_])
        dist.broadcast(tdue, 0)
        t_save, t_export, t_eval = (bool(x) for x in tdue.tolist())
        if t_eval:
            evaluate(done); T_LAST["eval"] = time.time()
        want_save = (CFG["save_every"] and done % CFG["save_every"] == 0) or c_save or c_stop or (done == max_steps and CFG["final_save"]) or t_save
        saved_ok = False
        if want_save:
            saved_ok = save(model, opt, sched, dos, done, f"{CKPT}/dcp-{done}", hosts)
            if saved_ok:
                SF.prune(CKPT, CFG["keep_ckpt"], LOCAL, log)
            T_LAST["save"] = time.time()
        if (CFG["export_every"] and (done % CFG["export_every"] == 0 or c_stop)) or t_export:
            T_LAST["export"] = time.time()
            SF.export_weights(model, done, CFG["export_root"], CFG["run_base"], CFG["export_min_free_gb"], CFG["export_parts"], RANK, log,
                              **({"side": {"step": done, "decp_bb": 1, "decp_bb_alpha": 0.0 if BB_OFF else decp_bb_alpha(done),
                                           **({"decp_bb_off": 1} if BB_OFF else {}), **({"s2p": S2P} if S2P else {}),
                                           **({"g0cond": 1} if G0 else {})}} if CFG["decp_bb"] else
                                 ({"side": {"step": done, **({"s2p": S2P} if S2P else {}), **({"g0cond": 1} if G0 else {})}} if (S2P or G0) else {})))
        if (c_save or c_stop) and not saved_ok:
            log(f"!!! [control] {'STOP_AFTER_SAVE' if c_stop else 'SAVE_NOW'} at step {done}: the save was SKIPPED (no /scratch space): NOT stopping; "
                f"control file kept (free space, it retries at the next step boundary)")
            c_stop = False
        elif c_save or c_stop:
            ctl.consume(done, c_stop)
            log(f"[control] {'STOP_AFTER_SAVE' if c_stop else 'SAVE_NOW'} at step {done}: saved {CKPT}/dcp-{done}" + (" -> exiting cleanly" if c_stop else ""))
        if c_stop:
            stopped = done
            break
    if CFG["ckpt_roundtrip"] and stopped is None:                  # step-0 check 3b: save -> fresh model + optimizers -> load (digests asserted)
        path = f"{CKPT}/dcp-{max_steps}"
        if not os.path.isfile(f"{path}/COMPLETE"):
            save(model, opt, sched, dos, max_steps, path, hosts)
        d0 = digest(model, [opt, dos["dec_s"].opt, dos["dec_p"].opt])
        m2, _ = M.build(c32, c4, c06, dev, log, partial=partial, proj_r=CFG["proj_r"], tower_vocab=CFG["tower_vocab"], init=False, dec_ac=CFG["dec_ac"], dec_mode=CFG["dec_mode"],
                          pose_head=bool(CFG["pose_head"]), **({"decp_bb": True} if CFG["decp_bb"] else {}), **({"s2p": S2P} if S2P else {}),
                          **({"g0cond": True} if G0 else {}))
        if PH_OFF:
            m2.pose_head_on = False
        if BB_OFF:
            m2.decp_bb_on = False
        o2, s2, _ = make_opt(m2)
        do2 = {"dec_s": DecOpt(m2.dec_s, "dec_s"), "dec_p": DecOpt(m2.dec_p, "dec_p")}
        st2 = load(m2, o2, s2, do2, path, hosts)
        d1 = digest(m2, [o2, do2["dec_s"].opt, do2["dec_p"].opt])
        ok = torch.tensor([float(d0 == d1 and st2 == max_steps and s2.state_dict()["last_epoch"] == sched.state_dict()["last_epoch"]
                                 and all(do2[k].micros == dos[k].micros for k in dos))], device=dev)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        log(f"[ckpt] ROUND TRIP {path}: weights + main AdamW + both decoder AdamWs + sched + counters bit-equal on every rank: {bool(float(ok))} "
            f"(rank 0 digest {d0[:16]})")
        assert float(ok) == 1.0, "checkpoint round trip FAILED"
    if stopped is not None:
        log(f"STOPPED at step {stopped} (clean exit): resume with PPM_RESUME_FROM=auto (or {CKPT}/dcp-{stopped}) on the same hosts")
    if FRESH_POSE and stopped is None and max_steps == FRESH_END:   # the fresh pose ran out: the final save above is the boundary
        log(f"[fresh] pose exhausted at step {max_steps}: every rank's fresh pose rows used once (none repeated); final save dcp-{max_steps} "
            f"(PPM_FINAL_SAVE={CFG['final_save']}); stopping here")
        if FRESH_POSE2:
            log(f"[fresh] (appended plans {list(FRESH_POSE2)}, tail {CFG['fresh']['pose_tail']}: the combined stream ended; ends after each plan {ends_})")
    log(f"done {max_steps} steps; step times {[round(x, 1) for x in times]}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
