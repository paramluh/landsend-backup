"""aloe SINGLE-SPEAKER FT TRAINER (10-24, user via main: "finetune our Aloe FT model on one speaker, all acoustic codes"; pick Ashley
TyypZEov3LSf5Gp8N6kSnBTZYXy2; "use all her acoustic codes over many micro optimiser steps for the light llm, get ready to train on 2 nodes",
"still use a voice prompt") = 29345's train_samsep_vox.py (job 29345, ALOE_INIT_VOX=1) with ONLY these changes, ALL gated on ALOE_SPK_ACCOUNT
(unset = 29345's trainer exactly; data module data_aloe_spk = data_aloe_v2vox when unset):
  DATA      data_aloe_spk (plan spk_plan.build): every chunk of every call of hers, her ALWAYS on B (flip), ~4 % of her calls held out;
            B voice clip = HER clip from another call (same 30 % draw rule, knob); training light_frac ALOE_SPK_LIGHT_FRAC (1.0 = every B frame
            trains the depth decoder). PROD checks run for logical world 16 or 64 (plan digest == env + a fresh rebuild + the plan file, the
            standard eval digest, per-plan A-FLAC ledgers DONE_spk_<digest>_node<l>, expected counts at THIS logical world).
  EVAL      two groups: the standard Aloe eval (316 micros, 29345's path exactly: eval / eval_asstB / eval_userB) + the speaker held-out calls
            (eval_spk / eval_spk_asstB = she was the data assistant / eval_spk_userB = she was the data user, i.e. flipped).
  CHECK     [same-as-v2] for speaker items: the CONVERSATION part of the micro (B lane, code masks, targets, light frames, A empty) == the
            reference v2 micro of the same row built from the old s2h-m1 stores with the same cfg (code values excluded); the prefix differs
            by design (voice rule). Standard eval micros: 29345's check unchanged.
  DECODER   QQA_DEC_K nanosteps per main step (existing PPM / 21980 path: K interleaved frame chunks, one all-reduce + AdamW update each);
            ALOE_SPK_DEC_PER_SLOT=1 (needed when slots > 1): every slot runs its own K decoder updates over ITS logical micros' chunk frames
            (decoder loss normalised by that update's frames, tower gradient weighted by those frames / the main step's frames = the step's
            token mean as before); a slot whose micros are all dummies (last step) takes no decoder update and uses no LR index. Decoder
            schedule (WSD, same shape) counts updates: warmup / decay start = steps x K x slots, horizon = K x (real slot updates); LR index
            of update j of slot g at step t = K x (real slot updates before (t, g)) + j. Without per-slot: = 29345's K-nanostep rule.
  CONFIG    + a report-only diff vs 29345's logged config (ALOE_REF_CFG_29345); the 26232 abort rule allows the SPK keys listed in SPK_KEYS.
29345's doc follows.
aloe-ft-32-samsep TRAINER (10-24, user 10-23 ~22:00 PT: "On the spare nodes, take down all demos and fine-tune our latest 32B PPM, no
cotrain", Aloe handover rows with the Mimi codes swapped for the SAM-separated ones) = aloe-zoom-cotrain's train_cotrain.py (job 26467) in
its ALOE-ONLY mode (COTRAIN_GROUP_MODE=aloe_only: every rank is an Aloe rank, no zoom data / eval / no-op passes; = job 26232's recipe,
26467's Aloe side) with ONLY these changes:
  RANKS     uneven hosts (slinky-18 x 8 + slinky-4 GPUs 0-5 x 6 = world 14): PPM_HOST_NRANKS gives each host's rank count (rank -> host map
            = PPM_HOSTS order); 64 LOGICAL ranks as 26232's, physical rank i runs logical ranks i + 14 s (5 slots, 6 pad-slot dummies).
  DATA      ALOE_SAMSEP_STORES (data_aloe_v2vox): the sam-sep stores, the v2 plan builder + seed over their rows (15,898 micros, 249 steps);
            the [same-as-v2] startup check compares every logical rank's step-0 / first eval micro with the reference v2 micro of the SAME
            row (old stores, row map) on everything but the Mimi code VALUES (code masks, text, targets, light frames, voice clip identical).
  INIT      QQA_INIT may be a ppm-local-bf16-shards-v1 copy (26232's load_init, ported verbatim) or a consolidated DCP.
  OFFLOAD   ALOE_OPT_OFFLOAD=1: main AdamW m / v in pinned host RAM between steps (optoff.py; bit-identical, checked in process at steps 1-2).
  EXPORTS   node-local exports write NODE_COMPLETE per host (HF watcher); an intermediate export is SKIPPED (loudly) when a host's /scratch
            would fall below ALOE_EXPORT_SCRATCH_FLOOR_GB + the final export's size (the final export is never starved).
  CONFIG    resolved CFG diffed against 26232's logged config (abort outside the allowed keys) and, report only, against 26467's.
Original aloe-zoom-cotrain doc follows.
aloe-zoom-cotrain TRAINER (10-21, user 10-20 23:35 PT: "prep a finetune which will run on all nodes which will be cotraining aloe and
InternalCan/zoom-strangers-5k 1:1 per step. When we run this on 16 nodes make 8 run the speech steps, 8 run these, nooping modules not fsdp
detachable") = aloe-v2vox-asr-w15973's train_aloe_v2vox_raw.py (job 26232) with ONLY these changes:
  GROUPS    the W ranks split into two RANK GROUPS for the WHOLE run: 'aloe' ranks run 26232's Aloe micros (its plan, its 64 logical ranks, its
            rows / seeds / A audio / decoder draws, bit-identical micros), 'zoom' ranks run zoom-strangers-5k chat-text micros (data_zoom.py:
            PPM's corpus15m chat-row rendering, assistant = B, B text CE only) AT THE SAME TIME, every optimizer step. Production: 16 nodes,
            the 8 hosts that ran 26232 (stores staged there) = aloe (64 ranks = 26232's logical ranks 0..63, one micro each), the other 8 =
            zoom (64 ranks, one 7,500-cap chat pack each). Split configurable (COTRAIN_ALOE_HOSTS | COTRAIN_GROUP_MODE=ranks); a group with
            fewer ranks than its logical world runs several micros per step (logical rank = group index + n_group x slot, = 26232's
            accumulation); every rank runs the same number of micros per step (GMAX; missing slots = zero-weight dummies).
  1:1       loss_step = W_ALOE x L_aloe + W_ZOOM x L_zoom (COTRAIN_W_ALOE = COTRAIN_W_ZOOM = 1): each group's loss is ITS OWN token mean over
            the step, exactly as a run on that data alone would normalise it (26232's tokens_step rule): L_aloe = 26232's loss (B text CE with
            PPM's factorised weights + W_CB0 cb0 CE, each normalised by the step's Aloe totals; + the decoder's token mean through the
            towers' input gradient), L_zoom = B text CE with the same factorised weights normalised by the step's zoom B-text-target total.
            Per micro the normaliser is the group total / W (W = ALL ranks): FSDP2's reduce-scatter averages over W, so the gradient of
            every shared tensor = W_ALOE x grad(L_aloe) + W_ZOOM x grad(L_zoom); the Aloe-only tensors (speech tower, cb0 head, audio tables,
            vadapter, vgain) get exactly grad(L_aloe) x W_ALOE, the trunk / text head the 1:1 sum. The replicated 0.6B decoder keeps 26232's
            own nanostep update (all-reduce over W of chunk losses weighted W / (31 M_j), M_j = the step's Aloe chunk frames): exactly the
            Aloe decoder mean (zoom ranks add exact zeros). With W_ALOE = 1 every Aloe-side number is 26232's (x 1.0 is exact).
  NO-OP     zoom ranks run EVERY FSDP unit the Aloe ranks run, in the same order, with zero weight (never detached from FSDP): the zoom micro
            is speech-shaped (data_zoom.as_noop_speech): inp gets depth -1 everywhere (code_feedback of -1 codes = exactly 0: the trunk input
            is bit-identical to PPM's depth-None chat micro, and inp.aud gets an exactly-zero gradient), vadapter + vgain run on a zero
            Voxtral latent (one adapter frame) injected x vmul = 0.0 at position 0 (model_ppm 'vmul' edit), the backbone + text head run the
            chat micro for real, tower_s runs on the micro's hn with every cb0 label -100 and weight 0 and its output hv gets a zero root
            gradient (same autograd graph as an Aloe micro: same FSDP all-gather / reduce-scatter sequence, nothing detached), and the
            replicated decoder (not an FSDP unit) joins its one per-step gradient all-reduce with an all-zero buffer, then takes the same
            AdamW update as every rank (identical replicas). All of it is the template's DUMMY-micro mechanism (zero loss weights).
            Asserted / tested: zoom micros give exactly 0 to tower_s / vadapter / vgain / inp.aud / dec_s (tests/: COTRAIN_TEST_GRADCHECK),
            and the step gradient == W_ALOE x (aloe-only gradient) + W_ZOOM x (zoom-only gradient).
  LENGTH    COTRAIN_STEPS_MODE aloe (default) = 26232's 253 steps = one Aloe epoch (zoom: 253 x 64 = 16,192 packs = 3.30 epochs, each epoch
            its own seeded order) | zoom_epoch = ceil(4,900 / 64) = 77 steps = one zoom epoch (the last step's 28 extra zoom slots are dummies;
            Aloe: its first 77 x 64 plan items). QQA_STEPS / QQA_WARMUP / QQA_EXPORT_STEPS are set explicitly by the env (asserted).
  EVAL      26232's Aloe eval (324 micros, eval / eval_asstB / eval_userB) on the aloe ranks + the zoom held-out eval (100 eval conversations,
            one pack each) on the zoom ranks (eval_zoom), at step 0, every QQA_EVAL_EVERY steps, every export step and the end.
  RESUME    NEW (26232 had none): STOP_AFTER_SAVE / SIGUSR1 (or COTRAIN_RESUME_EVERY) writes a node-local full-state save (safety_ppm.save_local:
            per-rank fp32 shards + main AdamW state, the replicated decoder + its AdamW once per node, LambdaLR, decoder update count, run
            counters; per-rank sha256 digests), COTRAIN_RESUME=<dir> restores it exactly (digests asserted) and continues at the next step
            with the same data (every micro is a pure function of (step, slot, logical rank)). No old save is ever deleted by the run.
  CONFIG    the resolved CFG is diffed against job 26232's logged config (plan/ref_cfg_26232.json): only run dirs, accumulation, the cotrain /
            resume / zoom keys and (zoom_epoch mode only) the length keys may differ, else abort.
  TEST      COTRAIN_TEST_DEPTH=L,Lt,Ld (reduced backbone / tower / decoder depth; init keys of deeper layers skipped), COTRAIN_GROUP_MODE=ranks,
            ALOE_LOGICAL_WORLD / ZOOM_LOGICAL_WORLD < 64, COTRAIN_TEST_STORES=src (Aloe stores read from /data), COTRAIN_TEST_GRADCHECK=1.
Original aloe-v2vox-raw doc follows.
aloe-v2vox-raw TRAINER (10-17, user): = aloe-v2vox-ft's train_aloe_v2vox.py (job 22420) with ONE maths change and bookkeeping only:
  A GAIN    VOX_A_GAIN=raw (data_aloe_v2vox.a_gain): channel A's waveform into Voxtral at the RAW recording level (WLT FLAC 0 dB; mimidec
            tracks x 10^(-recorded LUFS gain / 20)) instead of the recorded Mimi LUFS gain. Training and eval. Everything else = 22420.
  ACCUM     22420's 64 ranks become 64 LOGICAL ranks (plan, items, eval split, rank-seeded data / decoder draws, normalisers = 22420's): the
            physical rank (P, j) of a W-GPU job runs logical ranks 8 l + j for l in ALOE_LNODES[P] as consecutive micros of ONE optimizer
            step (G = max |ALOE_LNODES[P]| slots; a slot without a logical rank = a dummy micro, zero weight). Same rows in every step.
            tokens_step normalisers = the step totals over all 64 logical micros / W (FSDP and the decoder all-reduce average over W):
            gradient = the token mean over the step's 64 micros, exactly 22420's. Decoder (K = 1): its gradient accumulates over the G
            micros, ONE all-reduce + update per step (22420: one per step), each micro's frame permutation seeded with (step, logical rank)
            as 22420. s calibration, step-0 loss check, step-0 B-channel check and the eval adapter-RMS stat run per logical rank.
  CONFIG    resolved CFG diffed against 22420's logged config: only run dirs, the A-gain keys and the accumulation keys may differ (else abort).
(older docs: aloe-v2vox-asr-w15973/code/train_aloe_v2vox_raw.py)"""
import datetime
import hashlib
import json
import math
import os
import shutil
import signal
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
import data_qqa as Q  # noqa: E402
import data_aloe_spk as A  # noqa: E402   (speaker FT: = data_aloe_v2vox when ALOE_SPK_ACCOUNT is unset)
import spk_plan as SPL  # noqa: E402
import data_zoom as Z  # noqa: E402
import audio_s4q as AU  # noqa: E402
import sched_s3q as SCH  # noqa: E402
import safety_ppm as SAFE  # noqa: E402
import optoff as OO  # noqa: E402

D = Q.D


def E(k, d=None):
    v = os.environ.get(k, d)
    assert v is not None, ("missing env", k)
    return v


CFG = {
    "run_tag": E("QQA_RUN"), "log_root": E("QQA_LOG_ROOT"), "init": E("QQA_INIT"), "plan_dir": E("QQA_PLAN_DIR"),
    "export": E("QQA_EXPORT"), "control_dir": E("QQA_CONTROL_DIR"), "export_min_free_gb": float(E("QQA_EXPORT_MIN_FREE_GB", "300")),
    "export_steps": [int(x) for x in E("QQA_EXPORT_STEPS", "").split(",") if x], "export_wait_min": float(E("QQA_EXPORT_WAIT_MIN", "0")),
    "steps": int(E("QQA_STEPS")), "warmup": int(E("QQA_WARMUP")), "eval_every": int(E("QQA_EVAL_EVERY")),
    "w_text": float(E("QQA_W_TEXT_CE")), "w_cat_word": float(E("QQA_W_CAT_WORD")), "w_tim": float(E("QQA_W_TIM_CE")),
    "w_nw": float(E("QQA_W_NW_CE")), "w_eob": float(E("QQA_W_EOB_CE")), "w_cb0": float(E("QQA_W_CB0")), "w_cb": float(E("QQA_W_CB")),
    "lr": {"trunk": float(E("QQA_LR_TRUNK")), "spec": float(E("QQA_LR_SPEC")), "proj": float(E("QQA_LR_PROJ")),
           "audio": float(E("QQA_LR_AUDIO")), "tower_s": float(E("QQA_LR_TOWER"))},
    "clip": {"trunk": float(E("QQA_CLIP_TRUNK")), "spec": float(E("QQA_CLIP_SPEC")), "proj": float(E("QQA_CLIP_PROJ")),
             "audio": float(E("QQA_CLIP_AUDIO")), "tower_s": float(E("QQA_CLIP_TOWER"))},
    "adam_beta2": float(E("QQA_ADAM_BETA2")),
    "dec_k": int(E("QQA_DEC_K")), "dec_lr": float(E("QQA_DEC_LR")), "dec_clip": float(E("QQA_DEC_CLIP")),
    "dec_seed": int(E("QQA_DEC_SEED")),
    "dec_micro_frames": int(E("QQA_DEC_MICRO_FRAMES", "300")), "dec_mb_max": int(E("QQA_DEC_MB_MAX", "470")),
    "dec_even": E("QQA_DEC_EVEN", "1") == "1", "dec_dyn": E("QQA_DEC_DYN", "1") == "1", "dec_fast": E("QQA_DEC_FAST", "1") == "1",
    "dec_compile_mode": E("QQA_DEC_COMPILE_MODE", "default"), "eval_light_max": int(E("QQA_EVAL_LIGHT_MAX", "2048")),
    "losscheck": int(E("QQA_LOSSCHECK", "1")), "losscheck_tol": float(E("QQA_LOSSCHECK_TOL", "2e-3")),
    "dl_workers": int(E("QQA_DL_WORKERS", "3")), "data": A.aloe_cfg(), "cd": D.CD,
    "local_export_root": E("ALOE_LOCAL_EXPORT_ROOT"), "stores": A.STORES, "flac_root": A.FLAC_ROOT, "m1vox_index": A.INDEX,
    "vadapter": {"init": E("VOX_ADAPTER"), "enc_hidden": int(E("VOX_ENC_HIDDEN")), "hidden": int(E("VOX_VADAPT_HIDDEN")), "stack": int(E("VOX_VADAPT_STACK")),
                 "norm": E("VOX_VADAPT_NORM"), "sha_expect": E("VOX_ADAPTER_SHA", "")},
    "vox_a_norm": E("VOX_A_NORM"), "vox_a_scale_init": float(E("VOX_A_SCALE_INIT")), "vox_a_scale_learn": E("VOX_A_SCALE_LEARN") == "1",
    "vox_model": E("VOX_MODEL"), "vox_attn": os.environ.get("S4_VOXTRAL_ATTN", "flex_attention"),
    "test_steps": int(E("QQA_TEST_STEPS", "0")), "test_no_main_opt": E("QQA_TEST_NO_MAIN_OPT", "0") == "1",
    "eval_data": A.eval_cfg(), "model_nopose": True, "first_chunks_only": not A.SPK,
    "ref_cfg": {"m1vox": E("ALOE_REF_CFG_M1VOX"), "v2": E("ALOE_REF_CFG_V2"), "v2vox": E("ALOE_REF_CFG_V2VOX")},
    "light_ratio_lo": float(E("ALOE_LIGHT_RATIO_LO", "0.09")), "light_ratio_hi": float(E("ALOE_LIGHT_RATIO_HI", "0.11")),
}
SPK = bool(A.SPK)                                                  # speaker FT (module doc); False = 29345's trainer
CFG["vox_calib_rows"] = int(E("VOX_CALIB_ROWS", "1"))
CFG["vox_a_gain"] = A.GAIN_MODE
assert E("VOX_A_GAIN") == A.GAIN_MODE
if A.GAIN_MODE == "raw":
    CFG["rawgain_index"] = A.RAWGAIN
    CFG["rawgain_sha"] = hashlib.sha256(open(A.RAWGAIN, "rb").read()).hexdigest()[:16]
CFG["lr"]["vadapter"] = float(E("QQA_LR_VADAPT"))
CFG["clip"]["vadapter"] = float(E("QQA_CLIP_VADAPT"))
assert CFG["vox_a_norm"] == "none" and CFG["vox_a_scale_init"] > 0 and CFG["vox_a_scale_learn"], \
    ("user 10-16: no per-frame norm; ONE learnable global scalar s = exp(vgain.theta), init VOX_A_SCALE_INIT", CFG["vox_a_norm"], CFG["vox_a_scale_init"])
CFG["lr"]["vgain"] = float(E("QQA_LR_VGAIN"))
CFG["clip"]["vgain"] = float(E("QQA_CLIP_VGAIN"))
assert CFG["vadapter"]["norm"] == "none", ("the ASR adapter export to use is the no-norm one (VADAPT_NORM=none)", CFG["vadapter"]["norm"])
CFG["sched"] = {"warmup": CFG["warmup"], "horizon": CFG["steps"], "min_lr_rate": 0.0, "s3_sched": "wsd", "wsd_decay_start": CFG["warmup"],
                "wsd_decay_end": CFG["steps"], "wsd_floor": 0.0, "wsd_shape": "linear"}
CFG["dec_sched"] = {"warmup": CFG["warmup"] * CFG["dec_k"], "horizon": CFG["steps"] * CFG["dec_k"], "min_lr_rate": 0.0, "s3_sched": "wsd",
                    "wsd_decay_start": CFG["warmup"] * CFG["dec_k"], "wsd_decay_end": CFG["steps"] * CFG["dec_k"], "wsd_floor": 0.0, "wsd_shape": "linear"}
TRAIN_GROUPS = ("trunk", "spec", "proj", "audio", "tower_s", "vadapter", "vgain")    # NOPOSE: no pose groups (no pose modules built)
FROZEN_PREFIX = ()
POSE_PREFIX = ("pose_head.", "tower_p.", "dec_p.", "decp_bb.", "inp.pose.")
ALOE_ONLY_PREFIX = ("tower_s.", "vadapter.", "vgain.", "inp.aud.")             # FSDP tensors only Aloe micros give a non-zero gradient (+ dec_s)
RANK, WORLD, LOCAL = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
LW = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
HOST = socket.gethostname().split(".")[0]

# ---- cotrain: rank groups + logical ranks (module doc GROUPS)
ALOE_ONLY = E("COTRAIN_GROUP_MODE", "hosts") == "aloe_only"         # aloe-ft-32-samsep: no cotrain (every rank Aloe, no zoom anything)
if ALOE_ONLY:
    CO = {"group_mode": "aloe_only", "aloe_logical_world": int(E("ALOE_LOGICAL_WORLD", "64")), "w_aloe": 1.0, "steps_mode": "aloe",
          "test_depth": E("COTRAIN_TEST_DEPTH", ""), "test_stores": E("COTRAIN_TEST_STORES", ""), "test_gradcheck": False}
    assert float(E("COTRAIN_W_ALOE", "1")) == 1.0 and E("COTRAIN_STEPS_MODE", "aloe") == "aloe" and E("COTRAIN_TEST_GRADCHECK", "0") == "0"
else:
    CO = {"group_mode": E("COTRAIN_GROUP_MODE", "hosts"), "aloe_hosts": [h for h in E("COTRAIN_ALOE_HOSTS", "").split(",") if h],
          "aloe_ranks": int(E("COTRAIN_ALOE_RANKS", "0")), "aloe_logical_world": int(E("ALOE_LOGICAL_WORLD", "64")),
          "zoom_logical_world": int(E("ZOOM_LOGICAL_WORLD", "64")), "w_aloe": float(E("COTRAIN_W_ALOE", "1")), "w_zoom": float(E("COTRAIN_W_ZOOM", "1")),
          "steps_mode": E("COTRAIN_STEPS_MODE", "aloe"), "zoom_src": Z.SRC, "zoom_hf_repo": Z.HF_REPO, "zoom_sha256": Z.SHA256, "zoom_cap": Z.CAP,
          "zoom_plan_seed": Z.PLAN_SEED, "zoom_seed_off": Z.SEED_OFF, "zoom_table": Z.TABLE, "zoom_chat_side": "assistant", "zoom_chat_eob": True,
          "test_depth": E("COTRAIN_TEST_DEPTH", ""), "test_stores": E("COTRAIN_TEST_STORES", ""), "test_gradcheck": E("COTRAIN_TEST_GRADCHECK", "0") == "1"}
    assert CO["w_zoom"] > 0
assert CO["steps_mode"] in ("aloe", "zoom_epoch"), CO["steps_mode"]
assert CO["w_aloe"] > 0
HOSTS = os.environ["PPM_HOSTS"].split(",")
# aloe-ft-32-samsep: uneven hosts. PPM_HOST_NRANKS = ranks per host in PPM_HOSTS order (default 8 each = the reference's rank // 8 rule)
NR_HOST = [int(x) for x in os.environ.get("PPM_HOST_NRANKS", ",".join(["8"] * len(HOSTS))).split(",")]
assert len(NR_HOST) == len(HOSTS) and sum(NR_HOST) == WORLD, ("PPM_HOST_NRANKS vs PPM_HOSTS / WORLD_SIZE", NR_HOST, HOSTS, WORLD)
RANK_HOST = [h for h, n_ in zip(HOSTS, NR_HOST) for _ in range(n_)]
assert HOST == RANK_HOST[RANK] or os.environ.get("COTRAIN_CFG_DRY") == "1", ("rank -> host order differs from PPM_HOSTS", HOST, RANK_HOST[RANK])
assert LW == NR_HOST[HOSTS.index(RANK_HOST[RANK])] and LOCAL == RANK - sum(NR_HOST[:HOSTS.index(RANK_HOST[RANK])]), ("local rank layout", LW, LOCAL)
if ALOE_ONLY:
    _is_aloe = lambda r: True  # noqa: E731
elif CO["group_mode"] == "hosts":
    assert CO["aloe_hosts"] and set(CO["aloe_hosts"]) < set(HOSTS), ("COTRAIN_ALOE_HOSTS must be a proper subset of the job's hosts", CO["aloe_hosts"], HOSTS)
    _is_aloe = lambda r: RANK_HOST[r] in CO["aloe_hosts"]  # noqa: E731
else:
    assert CO["group_mode"] == "ranks" and 0 < CO["aloe_ranks"] < WORLD, CO
    _is_aloe = lambda r: r < CO["aloe_ranks"]  # noqa: E731
ARANKS = [r for r in range(WORLD) if _is_aloe(r)]
ZRANKS = [r for r in range(WORLD) if not _is_aloe(r)]
GROUP = "aloe" if RANK in ARANKS else "zoom"
IS_ALOE = GROUP == "aloe"
NA, NZ = len(ARANKS), len(ZRANKS)
LWORLD, ZL = CO["aloe_logical_world"], (0 if ALOE_ONLY else CO["zoom_logical_world"])
GA, GZ = -(-LWORLD // NA), (-(-ZL // NZ) if NZ else 0)
GMAX = max(GA, GZ)
GI = (ARANKS if IS_ALOE else ZRANKS).index(RANK)                   # this rank's index inside its group
NG, LWG = (NA, LWORLD) if IS_ALOE else (NZ, ZL)
LRANKS = [GI + NG * s if GI + NG * s < LWG else None for s in range(GMAX)]   # slot s -> logical rank of this rank's group (None: pad = dummy)
assert LRANKS[0] is not None
PROD = ((LWORLD == 64 or (SPK and LWORLD in (16, 64))) and (ALOE_ONLY or ZL == 64) and not CO["test_depth"] and not CO["test_stores"])
DEC_PER_SLOT = SPK and E("ALOE_SPK_DEC_PER_SLOT", "0") == "1"   # speaker FT: K decoder updates per SLOT (module doc DECODER)
assert GMAX == 1 or int(E("QQA_DEC_K")) == 1 or DEC_PER_SLOT, "decoder accumulation over slots implemented for K = 1 (26232's) only (or ALOE_SPK_DEC_PER_SLOT=1)"
if SPK:                                                            # speaker FT: plan world, LR-decay knob, decoder schedule units, bookkeeping
    assert ALOE_ONLY and A.SAMSEP, "the speaker FT runs on the sam-sep stores, aloe-only"
    A.set_world(LWORLD)
    _ds = int(E("ALOE_SPK_DECAY_START", str(CFG["warmup"])))
    CFG["sched"]["wsd_decay_start"] = _ds
    DEC_UNITS = CFG["dec_k"] * (GMAX if DEC_PER_SLOT else 1)       # decoder updates per (full) main step
    _hz = CFG["steps"] * DEC_UNITS
    if DEC_PER_SLOT:                                               # slot g of step t has a real micro iff its lowest logical rank (NA x g) does:
        _ni = int(E("ALOE_EXPECT_TRAIN_MICROS"))                   #   t x LWORLD + NA x g < train micros (items fill logical ranks in order);
        SLOT_REAL = [[t * LWORLD + NA * g < _ni for g in range(GMAX)] for t in range(CFG["steps"])]   # dummy-only slots take no update and
        SLOT_BASE, _c = [], 0                                      #   use no LR index: the decoder schedule counts REAL slot updates only
        for t in range(CFG["steps"]):
            SLOT_BASE.append([])
            for g in range(GMAX):
                SLOT_BASE[-1].append(_c); _c += int(SLOT_REAL[t][g])
        _hz = _c * CFG["dec_k"]
    CFG["dec_sched"] = {"warmup": CFG["warmup"] * DEC_UNITS, "horizon": _hz, "min_lr_rate": 0.0, "s3_sched": "wsd",
                        "wsd_decay_start": _ds * DEC_UNITS, "wsd_decay_end": _hz, "wsd_floor": 0.0, "wsd_shape": "linear"}
    CFG["spk"] = {"account": A.SPK, "epochs": A.SPK_EPOCHS, "steps_knob": A.SPK_STEPS, "eval_frac": A.SPK_EVAL_FRAC, "light_frac": A.SPK_LIGHT_FRAC,
                  "voice": [A.VOICE_NUM, A.VOICE_DEN], "plan_file": E("ALOE_SPK_PLAN"), "plan_digest": E("QQA_REF_PLAN_DIGEST"),
                  "expect_train_micros": int(E("ALOE_EXPECT_TRAIN_MICROS")), "expect_eval_micros": int(E("ALOE_EXPECT_EVAL_MICROS")),
                  "expect_eval_spk_micros": int(E("ALOE_SPK_EXPECT_EVAL_MICROS")), "dec_per_slot": DEC_PER_SLOT, "dec_updates_per_step": DEC_UNITS}
else:
    DEC_UNITS = CFG["dec_k"]
# production plan: 16 nodes = 8 aloe + 8 zoom (NA = NZ = 64, GMAX 1); other splits run the same maths with slots / dummies (module doc GROUPS)
CFG["accum"] = {"logical_world": LWORLD, "physical_world": WORLD, "slots": GMAX, "aloe_ranks": NA, "zoom_ranks": NZ,
                "rule": "aloe / zoom rank i of its group (sorted by global rank) runs logical ranks i + n_group x s, s < slots; others = dummies"}
if os.environ.get("PPM_HOST_NRANKS"):
    CFG["accum"]["host_ranks"] = dict(zip(HOSTS, NR_HOST))
# aloe-ft-32-samsep: data source + run-length bookkeeping keys (the plan's own counts, asserted at start)
SAMSEP = bool(A.SAMSEP)
if SAMSEP:
    CFG["samsep"] = {"stores_src": A.SRC_STORES[0].rsplit("/", 1)[0], "digests": list(A.A2.DIGESTS), "row_map": E("ALOE_SAMSEP_ROW_MAP"),
                     "rawgain_plan_digest": A.RAWGAIN_PLAN_DIGEST, "expect_train_micros": int(E("ALOE_EXPECT_TRAIN_MICROS")),
                     "expect_eval_micros": int(E("ALOE_EXPECT_EVAL_MICROS"))}
OFFLOAD = E("ALOE_OPT_OFFLOAD", "0") == "1"
if OFFLOAD:
    CFG["opt_offload"] = {"main_adamw_states": "pinned host RAM between steps (optoff.OffloadAdamW: same fused AdamW step, chunked)",
                          "chunk_gb": float(E("ALOE_OPT_OFFLOAD_CHUNK_GB", "2"))}
CFG["export_scratch_floor_gb"] = float(E("ALOE_EXPORT_SCRATCH_FLOOR_GB", "30"))
CFG["cotrain"] = CO
CFG["resume"] = {"from": os.environ.get("COTRAIN_RESUME", ""), "every": int(E("COTRAIN_RESUME_EVERY", "0")),
                 "root": E("COTRAIN_RESUME_ROOT", "/scratch/amartyavarma/aloe-zoom-cotrain/resume")}
assert CFG["resume"]["root"].startswith("/scratch/") or (os.environ.get("COTRAIN_TEST_SHM") == "1" and CFG["resume"]["root"].startswith("/dev/shm/")), \
    "resume saves are node-local (/scratch; tests: /dev/shm)"
if CO["test_stores"] == "src":                                    # TEST ONLY: the Aloe stores read in place from /data (no node-local copy)
    A.A2.STORES[:] = A.SRC_STORES
    CFG["stores"] = A.STORES
PT = zoneinfo.ZoneInfo("America/Los_Angeles")
LOGDIR = f"{CFG['log_root']}/{CFG['run_tag']}"
GiB = 2 ** 30
assert CFG["cd"] == 2, ("PPM code delay must be 2 (PPM_CD)", CFG["cd"])


def now_pt():
    return datetime.datetime.now(PT).strftime("%m-%d %H:%M:%S PT")


def log(*a):
    if RANK == 0:
        print(f"[{now_pt()}]", *a, flush=True)


def allsum(v, dev):
    t = torch.tensor(v, dtype=torch.float64, device=dev)
    dist.all_reduce(t)
    return t.tolist()


def frozen_name(n):
    n = n.replace("_checkpoint_wrapped_module.", "")
    return n.startswith(FROZEN_PREFIX)


def group_of(n):
    return None if frozen_name(n) else M.group_of(n)


# ------------------------------------------------------------------------------------------------ micro -> device + weights
ARR = ("ids", "depth", "present", "pid", "sel", "y", "word", "pad", "eob", "nw", "cb0", "light")
ZERO_LAT = {}


def zero_lat(dev):
    """the zoom ranks' no-op Voxtral latent: 4 frames x 1280 of exact zeros (one adapter frame)."""
    if dev not in ZERO_LAT:
        ZERO_LAT[dev] = torch.zeros(1, 4, CFG["vadapter"]["enc_hidden"], dtype=torch.bfloat16, device=dev)
    return ZERO_LAT[dev]


def prep(b, dev, S_text, S_cb0, lat=None, lam=1.0):
    """= the template's prep with the normalisers given + the cotrain group weight lam (W_ALOE / W_ZOOM; 1.0 = the template exactly).
    Aloe micro: unchanged. Zoom micro (b['zoom']): the text weights over its group total, cb0 weights 0 (no labels), the no-op adapter input
    (zero latent, vpos [0], vmul 0.0)."""
    assert b["kind"] == "speech", b["kind"]
    zoom = bool(b.get("zoom", 0))
    t = {"kind": "speech", "src": "zoom" if zoom else "speech", "segs": [int(x) for x in b["segs"]], "zoom": zoom}
    for k in ARR:
        t[k] = torch.as_tensor(np.asarray(b[k])).to(dev, non_blocking=True)
    t["pid"] = t["pid"][None]
    w = {k: t[k].float() for k in ("word", "pad", "eob", "nw")}
    S = float(S_text)
    t["wt"] = ((CFG["w_cat_word"] * w["word"] + CFG["w_tim"] * w["pad"] + CFG["w_eob"] * w["eob"] + CFG["w_nw"] * w["nw"]) / S).float()
    t["wn"] = (CFG["w_text"] * w["word"] / S).float()
    m = (t["cb0"] >= 0).float()
    t["cb0_w"] = m / float(S_cb0)
    if lam != 1.0:
        t["wt"], t["wn"], t["cb0_w"] = t["wt"] * lam, t["wn"] * lam, t["cb0_w"] * lam
    if zoom:
        assert bool((t["depth"] < 0).all()) and bool((t["cb0"] < 0).all()) and t["light"].numel() == 0, "zoom micro: no codes / labels / frames"
        assert float(m.sum()) == 0.0
        t["vpos"] = torch.zeros(1, dtype=torch.int64, device=dev)
        t["vlat"] = zero_lat(dev)
        t["vmul"] = 0.0
    else:
        P_, n_ = (int(x) for x in np.asarray(b["vin"]).reshape(2))
        t["vpos"] = torch.arange(P_, P_ + n_, device=dev)
        assert lat is not None and tuple(lat.shape) == (1, 4 * n_, CFG["vadapter"]["enc_hidden"]), ("Voxtral latents of the micro", None if lat is None else tuple(lat.shape), n_)
        t["vlat"] = lat
        assert float(t["present"][0, P_:].abs().sum()) == 0.0 and bool((t["depth"][0] < 0).all()) and bool((t["ids"][0, P_:] == M.PAD).all()), \
            "channel A must carry nothing but the adapter in the conversation (no text / Mimi / present)"
    t["dummy"] = bool(b.get("dummy", 0))
    if t["dummy"]:
        t["wt"], t["wn"], t["cb0_w"] = t["wt"] * 0.0, t["wn"] * 0.0, t["cb0_w"] * 0.0
    return t


TK = ("word_ce", "word_norm", "n_word", "word_hit", "pad_ce", "n_pad", "nw_ce", "n_nw", "eob_ce", "n_eob")
CB_S = (8, 16, 31)
NVA = len(TK) + 5 + len(CB_S)                                     # Aloe stats vector (= the template's)
NVZ = len(TK)                                                    # zoom stats vector (text only)


def text_vec(t, st):
    w, p, n_, e = (t[k].float() for k in ("word", "pad", "nw", "eob"))
    return [((st[0] * w).sum()), ((st[3] * w).sum()), (w.sum()), ((st[1] * w).sum()), ((st[0] * p).sum()), (p.sum()),
            ((st[0] * n_).sum()), (n_.sum()), ((st[0] * e).sum()), (e.sum())]


def summarize(v, pre):
    d = dict(zip(TK, v[:len(TK)]))
    m = lambda a, b: d[a] / max(d[b], 1.0)  # noqa: E731
    k = f"{pre}_speech"
    out = {f"{k}/word_ce_norm": m("word_norm", "n_word"), f"{k}/word_ce": m("word_ce", "n_word"), f"{k}/word_top1": m("word_hit", "n_word"),
           f"{k}/pad_ce": m("pad_ce", "n_pad"), f"{k}/nw_ce": m("nw_ce", "n_nw"),
           f"{k}/text_ce": (d["word_ce"] + d["pad_ce"] + d["nw_ce"] + d["eob_ce"]) / max(d["n_word"] + d["n_pad"] + d["n_nw"] + d["n_eob"], 1.0),
           f"{k}/n_word": d["n_word"]}
    x = v[len(TK):]
    out[f"{k}/cb0_ce"] = x[0] / max(x[1], 1.0); out[f"{k}/cb0_top1"] = x[2] / max(x[1], 1.0)
    out[f"{k}/light_ce"] = x[3] / max(x[4], 1.0)
    for j, c in enumerate(CB_S):
        out[f"{k}/light_ce_cb{c}"] = x[5 + j] / max(x[4] / 31.0, 1.0)
    assert len(x) == 5 + len(CB_S), len(x)
    return out


def summarize_zoom(v, pre):
    """zoom chat micros: B text only (no NEW_WORD in chat rows; EOB = the assistant's <|im_end|>)."""
    d = dict(zip(TK, v[:len(TK)]))
    m = lambda a, b: d[a] / max(d[b], 1.0)  # noqa: E731
    k = f"{pre}_zoom"
    return {f"{k}/word_ce_norm": m("word_norm", "n_word"), f"{k}/word_ce": m("word_ce", "n_word"), f"{k}/word_top1": m("word_hit", "n_word"),
            f"{k}/pad_ce": m("pad_ce", "n_pad"), f"{k}/eob_ce": m("eob_ce", "n_eob"),
            f"{k}/text_ce": (d["word_ce"] + d["pad_ce"] + d["nw_ce"] + d["eob_ce"]) / max(d["n_word"] + d["n_pad"] + d["n_nw"] + d["n_eob"], 1.0),
            f"{k}/n_word": d["n_word"]}


# ------------------------------------------------------------------------------------------------ nanosteps (token-weighted over the step)
def dec_lam():
    return SCH.make_lambda(True, CFG["dec_sched"])


class DecOpt:
    """= the template's DecOpt (replicated decoder: grads = views of ONE flat fp32 buffer, one all-reduce per update)."""

    def __init__(self, dec, peak, clip):
        self.ps = list(dec.parameters())
        assert not hasattr(self.ps[0], "to_local"), "replicated decoder expected (PPM_DEC_MODE=replicate)"
        self.flat = torch.zeros(sum(p.numel() for p in self.ps), dtype=torch.float32, device=self.ps[0].device)
        o = 0
        for p in self.ps:
            p.grad = self.flat[o:o + p.numel()].view_as(p); o += p.numel()
        self.opt = torch.optim.AdamW([{"params": self.ps, "lr": 0.0}], betas=(0.9, CFG["adam_beta2"]), eps=1e-8, weight_decay=0.0, fused=True)
        self.micros, self.peak, self.clip = 0, float(peak), float(clip)
        self.lam = dec_lam()

    def lr(self, s):
        return self.peak * self.lam(int(s))


def _mb_plan(m):
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
    return [(a0, max(z, 16)) for a0, z in plan]


def _dec_update(do, j, K, nano):
    """the step's ONE decoder update (= the template's tail of nanosteps_tw): all-reduce mean over W, LR, clip, AdamW. TEST (gradcheck):
    nano['no_step'] = capture the reduced buffer, no AdamW step."""
    dist.all_reduce(do.flat)
    do.flat.div_(WORLD)
    lr = do.lr((do.micros * K if "lr_base" not in nano else int(nano["lr_base"])) + j)   # speaker per-slot: (step x slots + slot) x K
    if nano.get("no_step"):
        nano["flat_out"] = do.flat.detach().clone()
        return lr
    for g in do.opt.param_groups:
        g["lr"] = lr
    torch.nn.utils.clip_grad_norm_(do.ps, do.clip)
    do.opt.step()
    return lr


def nanosteps_tw(dec, do, hv, depth, light, dummy=False, zb=None, nano=None):
    """= the template's nanosteps_tw (Aloe micros) with the update factored into _dec_update (same ops, same order)."""
    K, dev = CFG["dec_k"], hv.device
    assert nano is not None and do.micros == int(nano["step"]), ("decoder update count != step", do.micros, nano)
    hd = hv.detach().requires_grad_()
    zd = zb.detach().requires_grad_() if zb is not None else None
    n = int(light.numel())
    perm = torch.randperm(max(n, 1), generator=torch.Generator().manual_seed(CFG["dec_seed"] * 1000003 + int(nano["step"]) * 1009 + int(nano["lrank"])))
    sizes = [int(perm[j::K].numel()) if n else 0 for j in range(K)]
    Mj = [float(x) for x in nano["Mj"]]; N = float(nano["N"]) if "N" in nano else float(sum(Mj))   # speaker per-slot: N = the main step's frames
    perm_d = perm.pin_memory().to(dev, non_blocking=True)
    acc = torch.zeros(hd.shape, dtype=torch.float32, device=dev)
    acc_z = torch.zeros(zd.shape, dtype=torch.float32, device=dev) if zd is not None else None
    st = {"sum": torch.zeros((), device=dev, dtype=torch.float32), "n": 0, "cb": torch.zeros(31, device=dev, dtype=torch.float32)}
    lrs = []
    for j in range(K):
        m = sizes[j]
        idx = perm_d[j::K] if m else perm_d[:0]
        hd.grad = None
        if zd is not None:
            zd.grad = None
        if nano["first"]:
            do.flat.zero_()
        den = 31.0 * Mj[j] / WORLD
        for a0, sz in _mb_plan(m):
            mv = max(0, min(sz, m - a0))
            fl = light.index_select(0, idx[a0:a0 + mv]) if mv else light[:1]
            if fl.numel() < sz:
                fl = torch.cat([fl, light[:1].expand(sz - fl.numel())])
            wf = torch.zeros(sz, dtype=torch.float32, device=dev)
            if mv and not dummy:
                wf[:mv] = 1.0 / den
            l_, s_, num_, cb_ = dec(hd, depth, fl, wf, n_valid=mv) if zd is None else dec(hd, depth, fl, wf, n_valid=mv, zb=zd)
            if mv and not dummy:
                st["sum"] += s_; st["n"] += num_; st["cb"] += cb_.detach().float()
            l_.backward()
        if hd.grad is not None and m and not dummy:
            acc += hd.grad.float() * (Mj[j] / N)
        if zd is not None and zd.grad is not None and m and not dummy:
            acc_z += zd.grad.float() * (Mj[j] / N)
        if nano["last"] and not nano.get("skip_update"):          # speaker per-slot: a slot of dummies only takes no decoder update
            lrs.append(_dec_update(do, j, K, nano))
    hd.grad = None
    if zd is not None:
        zd.grad = None
    if nano["last"] and not nano.get("no_step") and nano.get("inc", True):
        do.micros += 1
    return acc, st, lrs, (Mj, N), acc_z


def dec_noop(do, nano):
    """zoom ranks: the replicated decoder (NOT an FSDP unit) is never run; its per-step collective and update are joined exactly as an Aloe
    rank does them: the flat gradient buffer zeroed at the first slot (contributes exact zeros), the step's ONE all-reduce + clip + AdamW at
    the last slot (identical reduced buffer on every rank -> identical replicas)."""
    K = CFG["dec_k"]
    assert do.micros == int(nano["step"]), ("decoder update count != step", do.micros, nano)
    lrs = []
    for j in range(K):
        if nano["first"]:
            do.flat.zero_()
        if nano["last"]:
            lrs.append(_dec_update(do, j, K, nano))
    if nano["last"] and not nano.get("no_step"):
        do.micros += 1
    return lrs


class DecFast:
    def __enter__(self):
        self.prev = M.DEC_FAST["active"]; M.DEC_FAST["active"] = CFG["dec_fast"]

    def __exit__(self, *a):
        M.DEC_FAST["active"] = self.prev


def run_micro(model, b, rot, dos, grad, info=None, nano=None):
    """Aloe micro = the template's run_micro (forward, loss = text + W_CB0 cb0, decoder nanosteps, ONE backward with roots (loss, hv_s)).
    Zoom micro = the SAME forward (every FSDP unit, module doc NO-OP), loss = text + W_CB0 x cb0 (cb0 exactly 0: weights 0), the decoder's
    per-step collective / update joined via dec_noop, ONE backward with roots (loss, hv_s) and a ZERO root gradient on hv_s (the Aloe graph).
    -> (aloe stats [NVA], zoom stats [NVZ], loss value, extra)."""
    with (torch.enable_grad() if grad else torch.no_grad()):
        out = model(b, rot, rot, info=info)
    assert "phead" not in out and "hv_p" not in out and "zb_p" not in out, "no-pose run: the pose path must not run"
    loss = out["text"] + CFG["w_cb0"] * out["cb0"]
    v = text_vec(b, out["text_stats"])
    extra = {}
    dummy = b.get("dummy", False)
    dvc = loss.device
    if b.get("zoom"):
        assert float(out["cb0_n"]) == 0.0, "zoom micro with tower labels"
        if grad:
            lrs = dec_noop(dos["dec_s"], nano)
            extra.update({"dec_s_lrs": lrs, "dec_s_M": [float(x) for x in nano["Mj"]], "dec_s_N": float(sum(nano["Mj"]))})
            torch.autograd.backward([loss, out["hv_s"]], [None, torch.zeros_like(out["hv_s"])])
        vz = torch.stack([torch.as_tensor(x, device=dvc).double().reshape(()) for x in v]).tolist()
        if dummy:
            assert float(loss) == 0.0, ("dummy micro with a non-zero loss", float(loss))
            vz = [0.0] * len(vz)
        return [0.0] * NVA, vz, float(loss.detach()), extra
    pairs = [("dec_s", model.dec_s, out["hv_s"], b["depth"][1], b["light"], CFG["w_cb"], None)]
    lam = float(b.get("lam", 1.0))
    dvals = []
    roots, grads = [loss], [None]
    for name, dec, hv, depth, light, wcb, zb in pairs:
        if grad:
            with DecFast():
                up, acc, lrs, (Mj, N), upz = nanosteps_tw(dec, dos[name], hv, depth, light, dummy=dummy, zb=zb, nano=nano)
            g_ = up * wcb if lam == 1.0 else up * (wcb * lam)            # W_ALOE on the decoder's backbone term (1.0 = the template)
            roots.append(hv); grads.append(g_.to(hv.dtype))
            dvals.append((acc["sum"], float(acc["n"]), acc["cb"]))
            extra.update({f"{name}_lrs": lrs, f"{name}_M": Mj, f"{name}_N": N})
        else:
            with torch.no_grad():
                fl = light[torch.randperm(light.numel(), generator=torch.Generator().manual_seed(1234))[:CFG["eval_light_max"]].to(light.device)]
                wz = torch.zeros(fl.numel(), device=light.device)
                _, s_, num_, cb_ = dec(hv, depth, fl, wz) if zb is None else dec(hv, depth, fl, wz, zb=zb)
            dvals.append((float(s_), float(num_), cb_.float()))
    v += [out["cb0_sum"], out["cb0_n"], out["cb0_hit"], dvals[0][0], dvals[0][1]]
    v += [dvals[0][2][c - 1] for c in CB_S]
    if grad:
        torch.autograd.backward(roots, grads)
    v = torch.stack([torch.as_tensor(x, device=dvc).double().reshape(()) for x in v]).tolist()
    if dummy:
        assert float(loss) == 0.0, ("dummy micro with a non-zero loss", float(loss))
        v = [0.0] * len(v)
    return v, [0.0] * NVZ, float(loss.detach()), extra


@torch.no_grad()
def micro_losses(model, mic, rot, dev, lat):
    """= the template's (one Aloe micro with MICRO-level token normalisers)."""
    oc = np.asarray(mic["own"]).sum(0)
    b = prep(mic, dev, oc[0], oc[1], lat=lat)
    out = model(b, rot, rot)
    res = []
    for dec, hv, depth, light, zb in ((model.dec_s, out["hv_s"], b["depth"][1], b["light"], None),):
        s_tot, n_tot = 0.0, 0
        for a0 in range(0, light.numel(), 1024):
            fl = light[a0:a0 + 1024]
            wz = torch.zeros(fl.numel(), device=dev)
            _, s_, num_, _ = dec(hv, depth, fl, wz) if zb is None else dec(hv, depth, fl, wz, zb=zb)
            s_tot += float(s_); n_tot += int(num_)
        res.append((s_tot / max(n_tot, 1), n_tot))
    return [float(out["text"]), float(out["cb0"]), res[0][0]], [float(oc[0]), float(oc[1]), float(res[0][1])]


def pack_loss_check(model, b, rot, dev, lat):
    """= the template's (Aloe ranks). Same number of forwards on every rank (FSDP)."""
    model.eval()
    singles = b["singles"]
    Pv, _ = micro_losses(model, b, rot, dev, lat)
    K = len(singles)
    t = torch.tensor([float(K)], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MAX); Kmax = int(t)
    num, den = [0.0] * 3, [0.0] * 3
    for k in range(Kmax):
        L, N = micro_losses(model, singles[k if k < K else 0], rot, dev, lat)
        if k < K:
            for i in range(3):
                num[i] += N[i] * L[i]; den[i] += N[i]
    model.train()
    exp = [num[i] / den[i] for i in range(3)]
    err = [abs(Pv[i] - exp[i]) / max(abs(exp[i]), 1e-12) for i in range(3)]
    return Pv, exp, err, K


@torch.no_grad()
def pack_loss_check_zoom(model, b, rot, dev):
    """zoom ranks' FSDP call parity with pack_loss_check: the same forwards (1 + Kmax, no grad) and the same all-reduce. The zoom micro's
    text loss with its own micro-level normaliser is returned (the zoom group's step-0 micro loss, logged)."""
    model.eval()
    bt = prep(b, dev, max(float(b["own_zoom"]), 1.0), 1.0)
    out = model(bt, rot, rot)
    lz = float(out["text"])
    t = torch.tensor([1.0], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MAX); Kmax = int(t)
    for _ in range(Kmax):
        model(bt, rot, rot)
    model.train()
    return lz


# ------------------------------------------------------------------------------------------------ init / export
@torch.no_grad()
def load_init_dcp(model, path, dev):
    """= the template's load_init_dcp; TEST (COTRAIN_TEST_DEPTH): keys of layers beyond the reduced depth are skipped too."""
    import pickle
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    assert os.path.isfile(f"{path}/COMPLETE") and os.path.isfile(f"{path}/.metadata"), path
    keys = set(pickle.load(open(f"{path}/.metadata", "rb")).state_dict_metadata)
    msd = get_model_state_dict(model)
    want = {k: v for k, v in msd.items() if not k.startswith(("vadapter.", "vgain."))}
    skipped = sorted(keys - set(want))
    deep = [k for k in skipped if CO["test_depth"] and k.startswith(("layers.", "tower_s.layers.", "dec_s.model.layers."))]
    bad = [k for k in skipped if not k.startswith(POSE_PREFIX) and k not in set(deep)]
    assert not bad, ("init keys skipped that are not pose parts", bad[:6])
    SKIPPED_INIT.update({p_: sorted(k for k in skipped if k.startswith(p_)) for p_ in POSE_PREFIX})
    if deep:
        SKIPPED_INIT["TEST_deeper_layers"] = deep
    miss = sorted(set(want) - keys)
    assert not miss, ("init DCP lacks model tensors", miss[:6])
    assert not any(k.startswith("vadapter.") for k in keys), "the PPM init must not carry a vadapter"
    before = {k: (M._loc(v).data_ptr() if hasattr(v, "to_local") else v.data_ptr()) for k, v in want.items()}
    dcp.load(want, checkpoint_id=path)
    params = {n.replace("_checkpoint_wrapped_module.", ""): p for n, p in model.named_parameters()}
    for k, v in want.items():
        p = params[k]
        assert (M._loc(p).data_ptr() == (M._loc(v).data_ptr() if hasattr(v, "to_local") else v.data_ptr()) == before[k]), ("not in place", k)
    nonfin = torch.tensor([float(sum(int(not torch.isfinite(M._loc(p)).all()) for n_, p in params.items() if not n_.startswith(("vadapter.", "vgain."))))], device=dev)
    dist.all_reduce(nonfin)
    assert float(nonfin) == 0, "non-finite init weights"
    js = {}
    for f in ("train_cfg.json", "side.json", "meta.json"):
        if os.path.isfile(f"{path}/{f}"):
            js = json.load(open(f"{path}/{f}")); break
    step = js.get("step", open(f"{path}/COMPLETE").read().strip())
    return {"step": step, "run_tag": js.get("run", js.get("run_tag", "?")), "world": "dcp", "save_time": "?", "format": "dcp"}, \
        sum(v.numel() for v in want.values()) * 2 / 1e9


@torch.no_grad()
def load_init(model, path, dev):
    """aloe-ft-32-samsep: = 26232's train_aloe_v2vox_raw.load_init VERBATIM (a ppm-local-bf16-shards-v1 copy, world-W0 dim-0 FSDP2 shards
    concatenated in rank order and resharded onto this world, each rank reading only its rows; else the consolidated DCP path)."""
    if not os.path.isfile(f"{path}/meta.json"):
        return load_init_dcp(model, path, dev)
    meta = json.load(open(f"{path}/meta.json"))
    assert meta["format"] == "ppm-local-bf16-shards-v1" and os.path.isfile(f"{path}/COMPLETE"), path
    W0 = int(meta["world"])
    params = {n.replace("_checkpoint_wrapped_module.", ""): p for n, p in model.named_parameters()
              if not n.startswith(("vgain.",) if INITVOX else ("vadapter.", "vgain."))}
    srcs = {}

    def src(r):
        if r not in srcs:
            srcs[r] = torch.load(f"{path}/rank{r}.pt", map_location="cpu", mmap=True, weights_only=True)
        return srcs[r]
    rep = torch.load(f"{path}/rep.pt", map_location="cpu", mmap=True, weights_only=True)
    ck = set(src(0).keys()) | set(rep.keys())
    skipped = sorted(ck - set(params))
    bad = [k for k in skipped if not k.startswith(POSE_PREFIX) and not (INITVOX and k.startswith("vgain."))]
    assert not bad, ("init keys skipped that are not pose parts", bad[:6])
    if INITVOX:
        assert any(k.startswith("vadapter.") for k in ck), "EVALVOX: the export carries no vadapter"
        assert "vgain_theta_fp32" in meta, "EVALVOX: the export's meta.json has no vgain_theta_fp32"
    SKIPPED_INIT.update({p_: sorted(k for k in skipped if k.startswith(p_)) for p_ in POSE_PREFIX})
    done, nbytes = set(), 0
    for n, p in params.items():
        if n in rep:
            p.copy_(rep[n].to(device=dev, dtype=p.dtype)); done.add(n); continue
        R = p.shape[0]
        c0 = -(-R // W0)
        a, r = M._rows(p)
        loc = M._loc(p)
        got = 0
        if r:
            for s in range(a // c0, min(W0, -(-(a + r) // c0))):
                lo, hi = s * c0, min((s + 1) * c0, R)
                t = src(s)[n]
                assert t.shape[0] == hi - lo and tuple(t.shape[1:]) == tuple(p.shape[1:]), (n, s, tuple(t.shape), lo, hi)
                x0, x1 = max(lo, a), min(hi, a + r)
                if x1 > x0:
                    loc[x0 - a:x1 - a].copy_(t[x0 - lo:x1 - lo].to(device=dev, dtype=loc.dtype)); got += x1 - x0
                    nbytes += (x1 - x0) * t[0].numel() * 2
        assert got == r, ("rows not covered", n, got, r)
        done.add(n)
    miss = sorted(set(params) - done)
    assert not miss, ("init: parameters not in the checkpoint", miss[:6])
    srcs.clear()
    nonfin = torch.tensor([float(sum(int(not torch.isfinite(M._loc(p)).all()) for p in params.values()))], device=dev)
    dist.all_reduce(nonfin)
    assert float(nonfin) == 0, "non-finite init weights"
    t = torch.tensor([float(nbytes)], device=dev); dist.all_reduce(t)
    return meta, float(t) / 1e9


SKIPPED_INIT = {}


def tensor_digest(model, names):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        if n not in names:
            continue
        h.update(n.encode()); h.update(M._loc(p).detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def export_weights(model, step, path, min_free_gb):
    """= the template's (consolidated bf16 DCP on /data; refused below the floor; never deletes)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    msd = get_model_state_dict(model)
    size_gb = sum(v.numel() * 2 for v in msd.values()) / 1e9
    ok = [None]
    if RANK == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tw = time.time()
        while True:
            free = shutil.disk_usage(os.path.dirname(path)).free / 1e9
            if free - size_gb >= min_free_gb or time.time() - tw > 60.0 * CFG["export_wait_min"] or os.path.exists(path):
                break
            time.sleep(15.0)
        ok = [(free - size_gb >= min_free_gb and not os.path.exists(path), free)]
    dist.broadcast_object_list(ok, src=0)
    good, free = ok[0]
    if not good:
        log(f"!!! [export] REFUSED {path}: {size_gb:.1f} GB with {free:.0f} GB free (floor {min_free_gb}) or the path exists")
        return None
    t0 = time.time()
    dcp.save({k: v.to(torch.bfloat16) for k, v in msd.items()}, checkpoint_id=path)
    dist.barrier()
    if RANK == 0:
        json.dump({"step": int(step), "decp_bb": 0, "pose": 0, "nopose_model": 1, "run": CFG["run_tag"], "init": CFG["init"], "cfg": CFG},
                  open(f"{path}/train_cfg.json", "w"), indent=1, default=str)
        open(f"{path}/COMPLETE", "w").write(str(step))
    dist.barrier()
    log(f"[export] {path}: consolidated bf16 weights ({size_gb:.1f} GB, {time.time() - t0:.0f}s); /data free before {free:.0f} GB")
    return path


def export_local(model, step, path, final=False):
    """= the template's node-local weights-only export (ppm-local-bf16-shards-v1, world = WORLD). aloe-ft-32-samsep: an INTERMEDIATE export
    is skipped (loudly) unless every host keeps >= ALOE_EXPORT_SCRATCH_FLOOR_GB + one more export of its own size free after writing it (the
    final export is never starved); each host's local rank 0 writes NODE_COMPLETE once every rank of the job has written its file."""
    t0 = time.time()
    mine = sum(M._loc(p).numel() * 2 for p in model.parameters() if hasattr(p, "to_local"))
    if RANK == 0:
        mine += sum(p.numel() * 2 for p in model.parameters() if not hasattr(p, "to_local"))
    allb = [None] * WORLD; dist.all_gather_object(allb, (HOST, float(mine)))
    node_b = sum(b for h, b in allb if h == HOST)
    root = os.path.dirname(path); os.makedirs(root, exist_ok=True)
    free = shutil.disk_usage(root).free
    need = node_b + (0 if final else node_b) + CFG["export_scratch_floor_gb"] * 1e9
    ok = torch.tensor([float(final or free >= need)], device=next(model.parameters()).device); dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if LOCAL == 0 and not final and free < need:
        print(f"!!! [export-local] {HOST}: /scratch free {free / 1e9:.0f} GB < {need / 1e9:.0f} GB (this export {node_b / 1e9:.1f} GB + the final's reserve + floor "
              f"{CFG['export_scratch_floor_gb']:.0f} GB)", flush=True)
    if float(ok) < 1.0:
        log(f"!!! [export-local] SKIPPED intermediate {path}: some host's /scratch cannot hold it and still keep the final export + floor (see the per-host lines)")
        return None
    os.makedirs(path, exist_ok=True)
    th_ = model.vgain.theta                                         # speaker FT: the exact fp32 vgain theta into meta.json (29345's had none)
    th_full = float((th_.full_tensor() if hasattr(th_, "full_tensor") else th_).detach().float().reshape(-1)[0])   # collective: every rank
    sh, rp = {}, {}
    for n, p in model.named_parameters():
        k = n.replace("_checkpoint_wrapped_module.", "")
        if hasattr(p, "to_local"):
            sh[k] = p.to_local().detach().to(torch.bfloat16).cpu().contiguous().clone()
        elif RANK == 0:
            rp[k] = p.detach().to(torch.bfloat16).cpu().contiguous().clone()
    torch.save(sh, f"{path}/rank{RANK}.pt.tmp"); os.replace(f"{path}/rank{RANK}.pt.tmp", f"{path}/rank{RANK}.pt")
    nb = sum(t.numel() * 2 for t in sh.values()) + sum(t.numel() * 2 for t in rp.values())
    hosts = [None] * WORLD; dist.all_gather_object(hosts, HOST)
    if RANK == 0:
        torch.save(rp, f"{path}/rep.pt")
        json.dump({"format": "ppm-local-bf16-shards-v1", "src": "aloe-zoom-cotrain train_cotrain export_local", "step": int(step), "world": WORLD, "hosts": hosts,
                   "run_tag": CFG["run_tag"], "init": CFG["init"], "cfg": CFG, "save_time": now_pt(), "vgain_theta_fp32": th_full, "vgain_s_fp32": math.exp(th_full),
                   "vgain_theta_dtype": "float32"}, open(f"{path}/meta.json", "w"), indent=1, default=str)
    dist.barrier()
    if RANK == 0:
        open(f"{path}/COMPLETE", "w").write(str(step))
    if LOCAL == 0:                                                  # every rank's file of the job is on disk (barrier above): this host's part
        json.dump({"step": int(step), "host": HOST, "ranks": [r for r, h in enumerate(hosts) if h == HOST], "world": WORLD, "final": bool(final)},
                  open(f"{path}/NODE_COMPLETE", "w"))
    t = torch.tensor([float(nb)], device=next(model.parameters()).device); dist.all_reduce(t)
    log(f"[export-local] {path} on every node: {float(t) / 1e9:.1f} GB bf16 shards (world {WORLD}), {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------------ VOX: adapter init + Voxtral front
def adapter_sha(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].contiguous().view(torch.uint8).numpy().tobytes() if sd[k].dim() else sd[k].reshape(1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def init_adapter(model, path, dev):
    """= the template's (vadapter.* <- the ASR adapter's bf16 DCP export; sha asserted; theta = log(VOX_A_SCALE_INIT))."""
    import pickle
    import torch.distributed.checkpoint as dcp
    assert os.path.isfile(f"{path}/COMPLETE"), path
    md = pickle.load(open(f"{path}/.metadata", "rb")).state_dict_metadata
    keys = sorted(k for k in md if k.startswith("vadapter."))
    sd = {k: torch.empty(tuple(md[k].size), dtype=md[k].properties.dtype) for k in keys}
    dcp.load(sd, checkpoint_id=path, no_dist=True)
    ts = json.load(open(f"{path}/train_state.json"))
    assert ts["cfg"].get("s4_vadapt_norm") == "none" and ts["cfg"].get("vadapt_norm") == "none", "not the no-norm ASR adapter export"
    assert "vadapter.target_rms" not in sd and "vadapter.gain" not in sd
    names = {n for n, _ in model.vadapter.named_parameters(prefix="vadapter")}
    assert set(sd) == names, ("adapter export keys != the model's vadapter", sorted(set(sd) ^ names))
    sha = adapter_sha({k: v.to(torch.bfloat16) for k, v in sd.items()})
    if CFG["vadapter"]["sha_expect"]:
        assert sha.startswith(CFG["vadapter"]["sha_expect"]), ("adapter export sha differs from VOX_ADAPTER_SHA (the ASR run's logged export sha)", sha[:16], CFG["vadapter"]["sha_expect"])
    for n, p in model.vadapter.named_parameters(prefix="vadapter"):
        assert tuple(p.shape) == tuple(sd[n].shape), (n, tuple(p.shape), tuple(sd[n].shape))
        a, r = M._rows(p)
        M._loc(p).copy_(sd[n][a:a + r].to(device=dev, dtype=M._loc(p).dtype))
        assert M._loc(p).dtype == torch.float32, ("fp32 adapter masters", n)
    model.vadapter.target_rms.fill_(float("nan"))
    set_s(model, CFG["vox_a_scale_init"])
    log(f"[init] vadapter <- {path} (ASR run {ts['cfg'].get('run_tag')} step {ts['step']}, norm none): {len(sd)} tensors, "
        f"{sum(v.numel() for v in sd.values()) / 1e6:.1f}M params, sha256(bf16) {sha[:16]}; TRAINABLE (lr {CFG['lr']['vadapter']:g}, clip {CFG['clip']['vadapter']:g})")
    return sha


def s_value(model):
    g = model.vgain.theta
    return math.exp(float((g.full_tensor() if hasattr(g, "full_tensor") else g).detach().float().reshape(-1)[0]))


@torch.no_grad()
def set_s(model, s):
    g = model.vgain.theta
    a_, r_ = M._rows(g)
    if r_:
        M._loc(g).fill_(math.log(float(s)))
    assert abs(s_value(model) / float(s) - 1.0) < 1e-6, (s_value(model), s)


def make_front(dev):
    """the frozen Voxtral Realtime encoder (Aloe ranks only: zoom ranks never encode audio; it is outside FSDP)."""
    stub = os.environ.get("VOX_TEST_STUB_FRONT", "0") == "1"
    assert not stub or CFG["test_steps"] > 0
    return AU.make_front({"s4_audio": "stub" if stub else "flac", "s4_vox_model": CFG["vox_model"], "s4_enc_hidden": CFG["vadapter"]["enc_hidden"]}, dev)


def encode(front, b):
    ref = json.loads(b["aref"])
    w = torch.from_numpy(np.ascontiguousarray(b["wav"]))
    lat = front.encode([w], [{"row_key": ref["row_key"], "n": int(ref["n"])}])
    assert torch.isfinite(lat).all(), ("non-finite Voxtral latents", ref["row_key"])
    return lat


def adapter_call(model, front, m, dev):
    """one vadapter forward (eval stat / calibration): Aloe = the micro's real latents; zoom = the zero latent (FSDP parity, discarded)."""
    with torch.no_grad():
        return model.vadapter(encode(front, m) if (IS_ALOE and m is not None) else zero_lat(dev))[0].float()


# ------------------------------------------------------------------------------------------------ main
STOP = {"flag": False}


def _usr1(signum, frame):
    STOP["flag"] = True


def make_opt(model):
    by = {g: [] for g in TRAIN_GROUPS}
    frozen, dec = [], []
    for n, p in model.named_parameters():
        g = group_of(n)
        if g is None:
            frozen.append(n)
        elif g in ("dec_s", "dec_p"):
            dec.append(n)
        else:
            assert g in by, (n, g)
            by[g].append(p)
    pgs = [{"params": by[g], "lr": CFG["lr"][g], "name": g} for g in TRAIN_GROUPS if by[g]]
    if OFFLOAD:                                                    # aloe-ft-32-samsep: same AdamW, m / v parked in pinned host RAM (optoff doc)
        opt = OO.OffloadAdamW(pgs, chunk_gb=CFG["opt_offload"]["chunk_gb"], betas=(0.9, CFG["adam_beta2"]), eps=1e-8, weight_decay=0.0, fused=True)
    else:
        opt = torch.optim.AdamW(pgs, betas=(0.9, CFG["adam_beta2"]), eps=1e-8, weight_decay=0.0, fused=True)
    lam = SCH.make_lambda(True, CFG["sched"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lam for _ in opt.param_groups])
    return opt, sched, by, frozen, dec


# ------------------------------------------------------------------------------------------------ resolved-config diff vs job 26232
REF_26232 = os.environ.get("COTRAIN_REF_CFG", "/data/home/amartyavarma/ppm/aloe-zoom-cotrain/plan/ref_cfg_26232.json")
ALLOWED_26232 = ("run_tag", "log_root", "control_dir", "export", "local_export_root", "accum", "cotrain", "resume")
# aloe-ft-32-samsep: + init (QQA_INIT), the sam-sep data source (stores / A index / RAWGAIN / FLAC root = the same A FLACs via symlinks, samsep),
#   the run length the new plan gives (steps / warmup / schedules / export steps), the offload + export-floor bookkeeping
SAMSEP_KEYS = ("init", "stores", "m1vox_index", "rawgain_index", "rawgain_sha", "flac_root", "samsep", "steps", "warmup", "sched", "dec_sched",
               "export_steps", "opt_offload", "export_scratch_floor_gb")
# rerun with the matched ASR adapter (user 10-24, ALOE_ADAPTER_SWAP=1): + ONLY the adapter export (VOX_ADAPTER / VOX_ADAPTER_SHA); s is
#   recalibrated at start exactly as in 28249 / 26232
# EVAL-ONLY of an external export (ALOE_EVALVOX=1, 10-24, main: "eval PPMVOX exports on Aloe"): the init is a ppm-local-bf16-shards-v1 export that
#   CARRIES its own trained vadapter.* (loaded from the shards like every other tensor) and its fp32 vgain theta in meta.json (vgain_theta_fp32; the
#   bf16 vgain.theta in the shards is ignored); no VOX_ADAPTER load, no s recalibration; the step-0 eval runs exactly as the FT's, then exit.
EVALVOX = os.environ.get("ALOE_EVALVOX", "0") == "1"
# FT FROM A PPMVOX EXPORT (ALOE_INIT_VOX=1, user 10-24 07:18 PT): the same init handling as EVALVOX (trunk + decoders + the export's own vadapter
#   + its fp32 vgain theta; no VOX_ADAPTER load, no s recalibration), then the normal training loop. End save (ALOE_END_SAVE=1) and SAVE_NOW added.
INITVOX = EVALVOX or os.environ.get("ALOE_INIT_VOX", "0") == "1"
# ALOE_EVAL_ONLY=1 (control evals, 10-24): the normal init + VOX_ADAPTER + s recalibration path (= the FT's step 0), then the step-0 eval and exit
EVAL_ONLY = EVALVOX or os.environ.get("ALOE_EVAL_ONLY", "0") == "1"
ADAPTER_SWAP_KEYS = ("vadapter.init", "vadapter.sha_expect") if os.environ.get("ALOE_ADAPTER_SWAP") == "1" else ()
LENGTH_KEYS = ("steps", "warmup", "sched", "dec_sched", "export_steps")              # zoom_epoch mode only
TEST_KEYS = ("stores", "test_steps", "test_no_main_opt")                             # reduced-depth tests only
# speaker FT (ALOE_SPK_ACCOUNT): its own keys + the data / decoder knobs main specified 10-24 (light_frac 1.0, decoder nanosteps K, light-ratio
#   band, every chunk, the her-other-call voice rule keys) + the speaker init / FLAC root / RAWGAIN of its plan (diffed and listed at start)
SPK_KEYS = ("spk", "data.light_frac", "data.spk_voice", "data.spk_voice_num", "data.spk_voice_den", "eval_data.spk_voice", "eval_data.spk_voice_num",
            "eval_data.spk_voice_den", "first_chunks_only", "dec_k", "light_ratio_lo", "light_ratio_hi", "vadapter.init", "vadapter.sha_expect")


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = json.loads(json.dumps(v, default=str))
    return out


def cfg_diff_26232():
    """the resolved CFG vs job 26232's logged config: only run dirs, accumulation, cotrain / resume keys (+ the length keys in zoom_epoch mode,
    + test keys in tests) may differ, else abort. Written to <logdir>/cfg_diff_26232.txt."""
    cur = _flat(json.loads(json.dumps(CFG, default=str)))
    ref = _flat(json.load(open(REF_26232)))
    allowed = ALLOWED_26232 + (LENGTH_KEYS if CO["steps_mode"] == "zoom_epoch" else ()) + (TEST_KEYS if not PROD else ()) + (SAMSEP_KEYS if SAMSEP else ()) + ADAPTER_SWAP_KEYS \
        + (SPK_KEYS if SPK else ())
    rows, bad = [], []
    for k in sorted(set(ref) | set(cur)):
        a, b = ref.get(k, "<absent>"), cur.get(k, "<absent>")
        if a != b:
            ok = any(k == x or k.startswith(x + ".") for x in allowed)
            rows.append(f"  {'ok ' if ok else 'BAD'} {k}: 26232 {a} -> cotrain {b}")
            if not ok:
                bad.append(k)
    txt = (f"[cfg-diff] resolved config vs 26232 ({REF_26232}): {len(rows)} keys differ ({len(bad)} outside run dirs / accumulation / cotrain / "
           f"resume{' / length (zoom_epoch)' if CO['steps_mode'] == 'zoom_epoch' else ''}{' / TEST keys' if not PROD else ''}"
           f"{' / samsep data, init, plan length, offload' if SAMSEP else ''}), every other key of "
           f"{len(set(ref) | set(cur))} identical\n" + "\n".join(rows))
    log(txt)
    if RANK == 0:
        open(f"{LOGDIR}/cfg_diff_26232.txt", "w").write(txt + f"\nVERDICT {'PASS' if not bad else 'FAIL ' + str(bad)}\n")
    assert not bad or not PROD, ("config differs from 26232 outside the expected keys", bad)   # tests (not PROD): reported only
    return len(rows), bad


def cfg_diff_report(path, tag):
    """aloe-ft-32-samsep: REPORT-ONLY diff of the resolved CFG vs another logged config (26467 = the co-train run behind the demo)."""
    if not path:
        return
    cur = _flat(json.loads(json.dumps(CFG, default=str)))
    ref = _flat(json.load(open(path)))
    rows = [f"  {k}: {tag} {ref.get(k, '<absent>')} -> this {cur.get(k, '<absent>')}" for k in sorted(set(ref) | set(cur))
            if ref.get(k, "<absent>") != cur.get(k, "<absent>")]
    txt = f"[cfg-diff {tag}] resolved config vs {tag} ({path}): {len(rows)} keys differ (report only), every other key of {len(set(ref) | set(cur))} identical\n" + "\n".join(rows)
    log(txt)
    if RANK == 0:
        open(f"{LOGDIR}/cfg_diff_{tag}.txt", "w").write(txt + "\n")
    return rows


class Offset:
    """a dataset view starting at element `off` (resume: element i of the run = (step, slot) -> starts at start_step x GMAX)."""

    def __init__(self, ds, off):
        self.ds, self.off = ds, int(off)

    def __len__(self):
        return len(self.ds) - self.off

    def __getitem__(self, i):
        return self.ds[int(i) + self.off]


def grads_local(model, dos):
    """TEST: {name: this rank's local gradient shard (fp32, CPU clone)} + the decoder's reduced flat buffer is captured separately."""
    out = {}
    for n, p in model.named_parameters():
        k = n.replace("_checkpoint_wrapped_module.", "")
        if k.startswith("dec_s."):
            continue
        g = p.grad
        out[k] = None if g is None else M._loc(g).detach().float().cpu().clone()
    return out


def main():
    dev = torch.device("cuda", LOCAL)
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev, timeout=datetime.timedelta(seconds=int(E("INIT_TIMEOUT", "1800"))))
    os.makedirs(LOGDIR, exist_ok=True)
    signal.signal(signal.SIGUSR1, _usr1)
    log(f"aloe-ft-32-samsep {CFG['run_tag']}: world {WORLD} (hosts {dict(zip(HOSTS, NR_HOST))}); "
        + (f"ALOE ONLY (no cotrain): {NA} ranks, logical world {LWORLD}, {GA} slots" if ALOE_ONLY else
           f"groups: aloe {NA} ranks (logical world {LWORLD}, {GA} slots) | zoom {NZ} ranks (logical world {ZL}, {GZ} slots)")
        + f"; {GMAX} micro slots per GPU per step; host {HOST}; PROD {PROD}; main AdamW offload {OFFLOAD}; cfg {json.dumps(CFG, default=str)}")
    cfg_diff_26232()
    cfg_diff_report(os.environ.get("ALOE_REF_CFG_26467", ""), "26467")
    cfg_diff_report(os.environ.get("ALOE_REF_CFG_28511", ""), "28511")
    cfg_diff_report(os.environ.get("ALOE_REF_CFG_29345", ""), "29345")
    resumed = None
    if CFG["resume"]["from"]:
        resumed = json.load(open(f"{CFG['resume']['from']}/train_state.json"))
        assert resumed["world"] == WORLD and resumed["hosts"] == HOSTS and resumed.get("rank_hosts", RANK_HOST) == RANK_HOST \
            and resumed["groups"] == {"aloe": ARANKS, "zoom": ZRANKS}, \
            ("resume needs the same world, rank -> host map and rank groups", resumed.get("world"), resumed.get("hosts"))
        cur_, old_ = cfg_resume_view(), resumed.get("cfg_view", {})
        dif_ = {k: (old_.get(k, "<absent>"), cur_.get(k, "<absent>")) for k in sorted(set(cur_) | set(old_)) if old_.get(k, "<absent>") != cur_.get(k, "<absent>")}
        assert resumed["steps"] == CFG["steps"] and resumed["cfg_flat_sha"] == cfg_sha() and not dif_, \
            ("resume: the resolved config differs from the saved run's (saved -> now)", dif_)
    start = int(resumed["step"]) if resumed else 0
    # ---- data: Aloe ranks = 26232's (byte-identical data modules, the v2 plan over its 64 logical ranks); zoom ranks = data_zoom
    import importlib.util
    ref_dm = "/data/home/amartyavarma/ppm/aloe-v2-ft/code/data_aloe_v2.py"
    assert hashlib.sha256(open(ref_dm, "rb").read()).digest() == hashlib.sha256(open(A.A2.__file__, "rb").read()).digest(), "data_aloe_v2 != v2's"
    for f_ in ("data_qqa.py", "data_ppm.py", "data_aloe.py"):
        assert all(hashlib.sha256(open(f"{d_}/{f_}", "rb").read()).digest() == hashlib.sha256(open(f"{HERE}/{f_}", "rb").read()).digest()
                   for d_ in ("/data/home/amartyavarma/ppm/aloe-v2-ft/code", "/data/home/amartyavarma/ppm/aloe-m1vox-ft/code")), f_
    if not ALOE_ONLY:
        tbz, zmeta = Z.load_table()                                # every rank (small, /data): the zoom steps are known everywhere
        n_ep0 = len(Z.epoch_packs(tbz["train"], 0))
        zoom_epoch_steps = -(-n_ep0 // ZL)
    else:
        tbz, zmeta, n_ep0, zoom_epoch_steps = None, None, 0, -1
    PL, RIT, fallback, leds, dg = None, None, None, {}, None
    RIT_SPK = [[] for _ in range(GMAX)]
    if IS_ALOE:
        PL = A.make_plan(LWORLD)
        if SPK:                                                     # speaker FT: her side of every row (DataLoader workers inherit it)
            assert PL["account"] == A.SPK and PL["lworld"] == LWORLD and PL["eval_digest"] == SPL.STD_EVAL_DIGEST, (PL["account"], PL["lworld"], PL["eval_digest"])
            A.her_side(PL)
        for k in range(4):
            A.Stores().open(k)
        dg = hashlib.sha256(json.dumps(PL["items"]).encode()).hexdigest()[:16]
        aloe_steps = int(PL["steps"])
    else:
        dg = hashlib.sha256(json.dumps([[int(x) for x in p] for p in Z.epoch_packs(tbz["train"], 0)]).encode()).hexdigest()[:16]
        aloe_steps = -1
    ast = [None] * WORLD; dist.all_gather_object(ast, (GROUP, aloe_steps, dg))
    aloe_steps = max(x[1] for x in ast)
    assert len({x[2] for x in ast if x[0] == "aloe"}) == 1 and len({x[2] for x in ast if x[0] == "zoom"}) == (0 if ALOE_ONLY else 1), "plan differs across ranks of a group"
    steps_mode = {"aloe": aloe_steps, "zoom_epoch": zoom_epoch_steps}
    assert CFG["steps"] == steps_mode[CO["steps_mode"]], ("QQA_STEPS != the steps of COTRAIN_STEPS_MODE", CFG["steps"], steps_mode)
    assert CFG["steps"] <= aloe_steps, ("more steps than the Aloe plan has", CFG["steps"], aloe_steps)
    if IS_ALOE:
        RIT = []
        for L in LRANKS:
            if L is None:
                RIT.append((None, []))
            else:
                tr, ev_ = A.rank_items(PL, L, LWORLD)
                RIT.append((tr[:CFG["steps"]], ev_))
        fallback = PL["items"][0]
        RIT_SPK = [A.rank_items_spk(PL, L, LWORLD) if (SPK and L is not None) else [] for L in LRANKS]   # speaker held-out eval items per slot
        if PROD:                                                    # tools/stage_raw.py ledgers: A tracks of every logical node this host runs
            for l in sorted({L // 8 for L in LRANKS if L is not None}):
                lf_ = f"{A.FLAC_ROOT}/_ledger/DONE_spk_{dg}_lw{LWORLD}_node{l}.json" if SPK else f"{A.FLAC_ROOT}/_ledger/DONE_v2vox_node{l}.json"
                leds[l] = json.load(open(lf_))
                assert leds[l]["node"] == l and leds[l]["plan_digest"] == dg, ("logical node's A FLACs not staged + verified for THIS plan", l)
        real = [g for g, L in enumerate(LRANKS) if L is not None]
        ndum = sum(sum(x is None for x in RIT[g][0]) for g in real)
        nreal = sum(len(RIT[g][0]) for g in real) - ndum
        cnt = [float(nreal), float(sum(len(RIT[g][1]) for g in real)), float(ndum), float(GMAX - len(real)), float(len(real)), 0.0, 0.0, 0.0]
    else:
        nzr = sum(1 for z in LRANKS if z is not None)
        zmax = n_ep0 if CO["steps_mode"] == "zoom_epoch" else None
        zreal = sum(1 for t in range(CFG["steps"]) for z in LRANKS if z is not None and (zmax is None or t * ZL + z < zmax))
        zev_n = sum(len(Z.eval_packs(tbz["eval"])[z::ZL]) for z in LRANKS if z is not None)
        cnt = [0.0, 0.0, 0.0, 0.0, 0.0, float(zreal), float(zev_n), float(nzr)]
    nconv = allsum(cnt, dev)
    assert int(nconv[4]) == LWORLD and int(nconv[7]) == ZL, ("every logical rank of both groups exactly once", nconv)
    if PROD:
        exp_tr, exp_ev = (CFG["samsep"]["expect_train_micros"], CFG["samsep"]["expect_eval_micros"]) if SAMSEP else (16134, 324)
        assert int(nconv[0]) == min(exp_tr, CFG["steps"] * (LWORLD if SPK else 64)) and int(nconv[1]) == exp_ev, (nconv, exp_tr, exp_ev)
    if SPK:                                                         # speaker held-out eval micros: every one exactly once over the logical ranks
        nspk = allsum([float(sum(len(RIT_SPK[g]) for g, L in enumerate(LRANKS) if L is not None)) if IS_ALOE else 0.0], dev)
        assert int(nspk[0]) == len(PL["eval_spk"]) == CFG["spk"]["expect_eval_spk_micros"], (nspk, len(PL["eval_spk"]))
        st_ = PL["stats"]
        log(f"[spk] account {A.SPK}: {PL['convs']} calls, {st_['all']['rows']} rows (every chunk), her on B (flipped rows {st_['all']['flipped_rows']}); "
            f"train {PL['n_train_rows']} rows x {PL['epochs']:.3f} epochs = {len(PL['items'])} micros in {PL['steps']} steps x {LWORLD}; held-out "
            f"{len(PL['convs_heldout'])} calls = {len(PL['eval_spk'])} eval_spk micros (+ the standard eval {len(PL['eval'])}); light_frac "
            f"{A.SPK_LIGHT_FRAC}; voice {A.VOICE_NUM}/{A.VOICE_DEN} her clip from another call; decoder K {CFG['dec_k']} x "
            f"{'slots ' + str(GMAX) if DEC_PER_SLOT else '1'} = {DEC_UNITS} updates / step; LR decay from step {CFG['sched']['wsd_decay_start']} to 0 at {CFG['steps']}")
    if ALOE_ONLY:
        log(f"[data] ALOE plan {[x[2] for x in ast if x[0] == 'aloe'][0]}: {int(nconv[0]):,} train micros in {CFG['steps']} steps x {LWORLD} logical ranks "
            f"({int(nconv[2])} dummy micros on the last step, {int(GMAX * NA - LWORLD)} pad-slot dummies per step) on {NA} GPUs x {GA} slots; eval "
            f"{int(nconv[1])} micros; stores {A.STORES[0].rsplit('/', 1)[0]} (digests {[d[:8] for d in A.A2.DIGESTS]}); NO cotrain")
    else:
      log(f"[data] ALOE plan {[x[2] for x in ast if x[0] == 'aloe'][0]}: {int(nconv[0]):,} train micros in {CFG['steps']} steps x {LWORLD} logical ranks "
        f"({int(nconv[2])} dummy micros on the last step) on {NA} GPUs x {GA} slots; eval {int(nconv[1])} micros | ZOOM: {Z.N_CONV['train']:,} train "
        f"conversations -> {n_ep0:,} packs / epoch (cap {Z.CAP}, 1 conversation per pack: every conversation 3,998-7,017 frames), {int(nconv[5]):,} zoom "
        f"micros in {CFG['steps']} steps x {ZL} logical ranks = {nconv[5] / n_ep0:.3f} epochs (mode {CO['steps_mode']}: aloe {aloe_steps} | zoom_epoch "
        f"{zoom_epoch_steps} steps); zoom eval {int(nconv[6])} packs; table key {zmeta['cfg_key']}; source {Z.SRC} (= private HF {Z.HF_REPO}, sha256 asserted)")
    if IS_ALOE and leds and LOCAL == 0:
        print(f"[data] {HOST} rank {RANK}: A tracks of logical nodes {sorted(leds)} verified ({', '.join(str(d['gcs_files']) for d in leds.values())} FLACs)", flush=True)
    # ---- = the template's v2 check (aloe ranks): plan digest vs v2's logged one, a fresh plan from v2's own data module on the /data source
    #      stores, per-step items at steps 0..5, the eval list (PROD: 64 logical ranks; tests use another logical world -> plan checks skipped),
    #      and every logical rank's step-0 + first eval micro B channel (prefix included) vs v2's micro for the same item (always)
    loc = []
    if IS_ALOE:
        spec = importlib.util.spec_from_file_location("data_aloe_v2_22383", ref_dm)
        R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
        if PROD and SAMSEP and SPK:                                # speaker FT: plan == a fresh spk_plan rebuild == the plan file == env digest
            ref_dg = E("QQA_REF_PLAN_DIGEST")
            PR = SPL.build(A.SPK, LWORLD, epochs=A.SPK_EPOCHS, steps=None if A.SPK_STEPS == "auto" else int(A.SPK_STEPS), eval_frac=A.SPK_EVAL_FRAC,
                           root=A.SRC_STORES[0].rsplit("/", 1)[0])
            PF = json.load(open(CFG["spk"]["plan_file"]))
            assert dg == ref_dg == PR["plan_digest"] == PF["plan_digest"] and PL["items"] == PR["items"] == [tuple(x) for x in PF["items"]], "plan != spk plan"
            assert PL["eval"] == PR["eval"] and PL["eval_spk"] == PR["eval_spk"] == [tuple(x) for x in PF["eval_spk"]] and PL["steps"] == PR["steps"] == PF["steps"] == CFG["steps"]
            assert SPL.digest(PL["eval"]) == SPL.STD_EVAL_DIGEST and PL["pool_digest"] == PR["pool_digest"] == PF["pool_digest"]
            assert len(PL["items"]) == CFG["spk"]["expect_train_micros"] and len(PL["eval"]) == CFG["spk"]["expect_eval_micros"], (len(PL["items"]), len(PL["eval"]))
            assert fallback == PR["items"][0]
            del PR, PF
        elif PROD and SAMSEP:                                      # aloe-ft-32-samsep: the plan = v2's builder over the sam-sep rows
            PR = A.make_plan(LWORLD, roots=A.SRC_STORES)
            ref_dg = E("QQA_REF_PLAN_DIGEST")
            assert dg == ref_dg and PL["items"] == PR["items"] and PL["eval"] == PR["eval"] and PL["steps"] == PR["steps"] == CFG["steps"], "plan != samsep plan"
            assert len(PL["items"]) == CFG["samsep"]["expect_train_micros"] and len(PL["eval"]) == CFG["samsep"]["expect_eval_micros"], (len(PL["items"]), len(PL["eval"]))
            assert fallback == PR["items"][0]
            del PR
        if PROD and not SAMSEP:
            PR = R.make_plan(LWORLD, roots=R.SRC_STORES)
            ref_dg = E("QQA_REF_PLAN_DIGEST")
            st_dg = [(hashlib.sha256(json.dumps(PL["items"][t * LWORLD:(t + 1) * LWORLD]).encode()).hexdigest()[:12],
                      hashlib.sha256(json.dumps(PR["items"][t * LWORLD:(t + 1) * LWORLD]).encode()).hexdigest()[:12]) for t in range(6)]
            assert dg == ref_dg and PL["items"] == PR["items"] and PL["eval"] == PR["eval"] and PL["steps"] == PR["steps"] == 253, "plan != v2's"
            assert len(PL["items"]) == 16134 and len(PL["eval"]) == 324, (len(PL["items"]), len(PL["eval"]))
            assert all(a_ == b_ for a_, b_ in st_dg) and fallback == PR["items"][0]
            del PR

        if SAMSEP:                                                  # new (k, r) -> the reference store row of the same conversation chunk
            rm_ = json.load(open(CFG["samsep"]["row_map"]))
            OLD_OF_NEW = {k_: {int(nw_): int(od_) for od_, nw_ in rm_["map"][f"m1n{k_}"]["new_of_old"].items()} for k_ in range(4)}
            R_ST = R.Stores(roots=R.SRC_STORES)
            del rm_

        def same_b_r(it_, cfg_, rk_):
            if SAMSEP:                                              # reference micro of the SAME row (old stores); only code VALUES may differ
                it_ref = (int(it_[0]), OLD_OF_NEW[int(it_[0])][int(it_[1])], int(it_[2]))
                m_ref = R_ST.conv(it_ref, cfg_, rk_)
            else:
                m_ref = R.Stores().conv(it_, cfg_, rk_)               # v2's micro for this item (its exact code path: words + Mimi on A)
            m_new = A.Stores().conv(it_, cfg_, rk_, audio=False, spk=False)   # speaker FT: the standard (29345) path for these items
            P_ = int(m_new["P"])
            cv_ = (lambda a_, b_: np.array_equal(np.asarray(a_) >= 0, np.asarray(b_) >= 0)) if SAMSEP else (lambda a_, b_: np.array_equal(a_, b_))
            c_ = {"P": int(m_ref["P"]) == P_, "row_key": (not SAMSEP) or m_ref["row_key"] == m_new["row_key"],
                  "ids_B": np.array_equal(m_ref["ids"][1], m_new["ids"][1]), "depth_B": cv_(m_ref["depth"][1], m_new["depth"][1]),
                  "present_B": np.array_equal(m_ref["present"][1], m_new["present"][1]),
                  "prefix_A": np.array_equal(m_ref["ids"][0][:P_], m_new["ids"][0][:P_]) and np.array_equal(m_ref["present"][0][:P_], m_new["present"][0][:P_]),
                  "targets": all(np.array_equal(np.asarray(m_ref[k_]), np.asarray(m_new[k_])) for k_ in ("sel", "y", "word", "pad", "eob", "nw", "light"))
                  and cv_(m_ref["cb0"], m_new["cb0"]),
                  "voice": (m_ref["voice_clip"], m_ref["voice_drawn"]) == (m_new["voice_clip"], m_new["voice_drawn"]),
                  "A_empty": bool((m_new["ids"][0][P_:] == D.PAD_ID).all() and (m_new["depth"][0] < 0).all() and not m_new["present"][0][P_:].any()
                                  and tuple(m_new["vin"]) == (P_, int(m_new["vin"][1])))}
            return c_, int(m_new["voice_clip"])

        def same_b_spk(it_, cfg_, rk_):
            """speaker item: the CONVERSATION part (positions >= P) of the speaker micro == the reference v2 micro of the SAME row (old s2h-m1
            stores, data_aloe_v2.speech_micro_v2 with the same cfg; any chunk); code VALUES excluded (masks compared); the prefix differs by
            design (voice rule): only the 30 % draw is compared (default rule) and the clip must come from another call of hers."""
            k_, r_, sw_ = int(it_[0]), int(it_[1]), int(it_[2])
            ro_ = OLD_OF_NEW[k_][r_]
            _, st_o, sib_o = R_ST.open(k_)
            m_ref = R.speech_micro_v2(st_o[ro_], ro_, sib_o, cfg_, rk_, sw_)
            m_new = A.Stores().conv(it_, cfg_, rk_, audio=False, spk=True)
            Pr, Pn = int(m_ref["P"]), int(m_new["P"])
            mk_ = lambda x_: np.asarray(x_) >= 0  # noqa: E731
            pv_ = json.loads(m_new["prov"])
            conv_ = "/".join(m_new["row_key"].split("/")[2:4])
            c_ = {"row_key": json.loads(m_ref["prov"])["row_key"] == m_new["row_key"], "T_minus_P": len(m_ref["pid"]) - Pr == len(m_new["pid"]) - Pn,
                  "ids_B": np.array_equal(m_ref["ids"][1][Pr:], m_new["ids"][1][Pn:]), "depth_B": np.array_equal(mk_(m_ref["depth"][1][Pr:]), mk_(m_new["depth"][1][Pn:])),
                  "present_B": np.array_equal(m_ref["present"][1][Pr:], m_new["present"][1][Pn:]),
                  "targets": all(np.array_equal(np.asarray(m_ref[k2_]), np.asarray(m_new[k2_])) for k2_ in ("y", "word", "pad", "eob", "nw"))
                  and np.array_equal(np.asarray(m_ref["sel"]) - Pr, np.asarray(m_new["sel"]) - Pn)
                  and np.array_equal(np.asarray(m_ref["light"]) - Pr, np.asarray(m_new["light"]) - Pn)
                  and np.array_equal(mk_(np.asarray(m_ref["cb0"])[Pr:]), mk_(np.asarray(m_new["cb0"])[Pn:])),
                  "voice_draw": (A.VOICE_NUM, A.VOICE_DEN) != (3, 10) or m_ref["voice_drawn"] == m_new["voice_drawn"],
                  "clip_other_call": all("/".join(c2_["src_key"].split("/")[2:4]) != conv_ for c2_ in pv_["clips"]) and pv_["voice_clip"] == pv_["voice_drawn"],
                  "A_empty": bool((m_new["ids"][0][Pn:] == D.PAD_ID).all() and (m_new["depth"][0] < 0).all() and not m_new["present"][0][Pn:].any()
                                  and tuple(m_new["vin"]) == (Pn, int(m_new["vin"][1])))}
            return c_, int(m_new["voice_clip"])
        for g in [g for g, L in enumerate(LRANKS) if L is not None]:
            L, rows_g, ev_g = LRANKS[g], RIT[g][0], RIT[g][1]
            it0 = rows_g[0] if rows_g[0] is not None else fallback
            chk, vc0 = (same_b_spk if SPK else same_b_r)(it0, CFG["data"], L)
            chk_e, _ = same_b_r(ev_g[0], CFG["eval_data"], L) if ev_g else ({"none": True}, 0)
            if SPK and RIT_SPK[g]:
                chk_s, _ = same_b_spk(RIT_SPK[g][0], CFG["eval_data"], L)
                chk_e = dict(chk_e, **{f"spk_{k2_}": v2_ for k2_, v2_ in chk_s.items()})
            loc.append((all(chk.values()) and all(chk_e.values()), vc0, it0, L, chk, chk_e))
        del R
    allc = [None] * WORLD; dist.all_gather_object(allc, [x[:4] for x in loc])
    allc = [x for c in allc for x in c]
    log(f"[same-as-v2] {('SAMSEP: plan digest ' + E('QQA_REF_PLAN_DIGEST') + ' == a fresh v2-builder plan over the /data sam-sep stores (items, eval list, steps); CODE VALUES EXCLUDED below (masks compared); ') if (PROD and SAMSEP) else ''}"
        f"{'plan digest ' + E('QQA_REF_PLAN_DIGEST') + ' == v2 logged + v2 data module fresh, steps 0..5 items, eval list; ' if (PROD and not SAMSEP) else ('' if PROD else '(TEST: plan checks skipped) ')}"
        f"step-0 train micro + first eval micro: B channel (prefix incl. voice clip), targets, light frames == v2's micro of the same item on "
        f"{sum(c[0] for c in allc)} / {len(allc)} Aloe logical ranks (step-0 items with a B voice clip: {sum(c[1] for c in allc)}); A = Voxtral only"
        + (f"; rank 0 slot 0 checks train {loc[0][4]} eval {loc[0][5]}" if loc and RANK == 0 else ""))
    assert len(allc) == LWORLD and sorted(c[3] for c in allc) == list(range(LWORLD)) and all(c[0] for c in allc), [c for c in allc if not c[0]][:3]
    # ---- model (= the template's; TEST: reduced depth)
    nl, nt, nd = (int(x) for x in (CO["test_depth"] or "64,36,28").split(","))
    c32, c4, c06 = M.cfg32(nl), M.cfg4b(nt), M.cfg06(nd)
    model, _ = M.build(c32, c4, c06, dev, log, partial=False, proj_r=2048, tower_vocab=M.TOWER_VOCAB, init=False, dec_ac=False,
                       dec_mode="replicate", pose_head=False, decp_bb=False, vadapter=CFG["vadapter"], nopose=True)
    assert not any(n.replace("_checkpoint_wrapped_module.", "").startswith(POSE_PREFIX) for n, _ in model.named_parameters()), "pose parts built"
    t0 = time.time()
    meta, gb = load_init(model, CFG["init"], dev)
    log(f"[init] {CFG['init']}: PPM step {meta['step']}, {gb:.1f} GB read; {time.time() - t0:.0f}s; SKIPPED init keys: "
        + ", ".join(f"{p_}* {len(v_)} keys" for p_, v_ in SKIPPED_INIT.items()))
    assert SKIPPED_INIT.get("pose_head.") or SKIPPED_INIT.get("decp_bb."), "expected the PPM pose-head keys (pose_head.* or decp_bb.*) in the init (skipped)"
    if INITVOX:
        th = float(meta["vgain_theta_fp32"])
        g_ = model.vgain.theta
        with torch.no_grad():
            if M._rows(g_)[1]:
                M._loc(g_).fill_(th)
            model.vadapter.target_rms.fill_(float("nan"))
        ad_sha = "from-init"
        log(f"[evalvox] vadapter <- the init export's own shards; vgain theta {th:.9g} (s {math.exp(th):.9g}; meta vox_a_scale {meta.get('vox_a_scale')}) "
            f"from {CFG['init']}/meta.json (run {meta.get('run_tag')}, step {meta.get('step')}, vox_step {meta.get('vox_step')}); no s recalibration")
        assert abs(s_value(model) / math.exp(th) - 1.0) < 1e-6
    else:
        ad_sha = init_adapter(model, CFG["vadapter"]["init"], dev)
    front = make_front(dev) if IS_ALOE else None
    if CFG["dec_fast"]:
        M.compile_decoder(model.dec_s, CFG["dec_compile_mode"], whole=True, dynamic=CFG["dec_dyn"])
    with torch.no_grad():
        ew = model.inp.emb[0].weight
        a_, r_ = M._rows(ew)
        ewl = M._loc(ew).float()
        wm = torch.arange(a_, a_ + r_, device=dev) < M.SP.CTRL_LO
        s_w = allsum([float((ewl[wm] ** 2).sum()), float(wm.sum()) * ewl.shape[1]], dev)
        EMB_RMS = math.sqrt(s_w[0] / max(s_w[1], 1.0))
        del ewl
    rot = M.Rot(c32, dev)
    tfz = tuple(x for x in os.environ.get("COTRAIN_TEST_FREEZE", "").split(",") if x)
    if tfz:                                                       # TEST ONLY (resume test on a node with little /scratch): these tensors get
        assert not PROD                                           #   requires_grad False (no grad, no AdamW state) before the first forward
        for n_, p_ in model.named_parameters():
            if n_.replace("_checkpoint_wrapped_module.", "").startswith(tfz):
                p_.requires_grad_(False)
        global FROZEN_PREFIX
        FROZEN_PREFIX = tfz
    opt, sched, by, frozen, decn = make_opt(model)
    dos = {"dec_s": DecOpt(model.dec_s, CFG["dec_lr"], CFG["dec_clip"])}
    n_all = len(list(model.parameters()))
    assert all(n.replace("_checkpoint_wrapped_module.", "").startswith("dec_s.") for n in decn), decn[:4]
    assert sum(len(v) for v in by.values()) + sum(len(d.ps) for d in dos.values()) + len(frozen) == n_all, "every parameter in exactly one optimizer or frozen"
    assert not frozen or tfz, ("full fine-tune: no model parameter may be frozen", frozen[:4])
    fz_set = set(frozen)
    fz0 = tensor_digest(model, fz_set)
    gcount = {g: [len(by[g]), int(sum(p_.numel() for p_ in by[g]))] for g in TRAIN_GROUPS}
    gcount["dec_s (nanostep AdamW)"] = [len(dos["dec_s"].ps), int(sum(p_.numel() for p_ in dos["dec_s"].ps))]
    log("[params] TRAINABLE groups (tensors, params): " + ", ".join(f"{g} ({a_}, {b_ / 1e6:.2f}M)" for g, (a_, b_) in gcount.items())
        + f" | model {sum(p_.numel() for p_ in model.parameters()) / 1e9:.4f}B; Voxtral encoder frozen, per Aloe rank, outside the model")
    log(f"[opt] main AdamW groups {[(pg['name'], len(pg['params']), pg['lr']) for pg in opt.param_groups]}; decoder nanostep AdamW dec_s (peak "
        f"{CFG['dec_lr']:g}, clip {CFG['dec_clip']:g}, K {CFG['dec_k']}); schedule {SCH.describe(dict(CFG['sched'], warmup_groups=list(TRAIN_GROUPS)))}; "
        + (f"ALOE ONLY (no cotrain); main AdamW m / v offloaded to host: {OFFLOAD}" if ALOE_ONLY else f"group weights W_ALOE {CO['w_aloe']:g} W_ZOOM {CO['w_zoom']:g}"))
    if resumed:                                                   # exact restore (per-rank digests asserted) of weights + both optimizers + sched
        SAFE.load_local(model, {"opt": opt, "dec_s": dos["dec_s"].opt}, sched, CFG["resume"]["from"], RANK, log)
        dos["dec_s"].micros = int(resumed["dec_micros"])
        assert dos["dec_s"].micros == start, ("decoder update count != resume step", dos["dec_s"].micros, start)
        for pg, lr_ in zip(opt.param_groups, sched.get_last_lr()):  # = train_ppm's 10-12 fix: LambdaLR.load_state_dict leaves the groups at
            pg["lr"] = lr_                                           #   the constructor's lam(0) LRs (0 in warmup) -> restore the saved ones
        model.vadapter.target_rms.fill_(float("nan"))                # (buffer, not a parameter: = init_adapter's value)
        log(f"[resume] continuing at step {start} (s = {s_value(model):.8g}, lr_trunk {opt.param_groups[0]['lr']:.4g})")
    # ---- loaders + eval sets
    from torch.utils.data import DataLoader
    W_ = CFG["dl_workers"]

    def _wi(_):
        torch.set_num_threads(1); os.environ["OMP_NUM_THREADS"] = "1"
        os.nice(19)
    if IS_ALOE:
        ds = A.AccDS([RIT[g][0] for g in range(GMAX)], LRANKS, CFG["data"], fallback, singles_at=0 if (CFG["losscheck"] and not resumed) else -1)
    else:
        ds = Z.ZoomAccDS(tbz["train"], LRANKS, ZL, CFG["steps"], max_packs=(n_ep0 if CO["steps_mode"] == "zoom_epoch" else None))
    assert len(ds) == CFG["steps"] * GMAX, (len(ds), CFG["steps"], GMAX)
    ld = iter(DataLoader(Offset(ds, start * GMAX), batch_size=None, shuffle=False, num_workers=W_, prefetch_factor=2 if W_ else None,
                         persistent_workers=bool(W_), pin_memory=False, worker_init_fn=_wi if W_ else None, collate_fn=lambda x: x))
    ev, ev_first = [], [None] * GMAX
    ecap = int(os.environ.get("COTRAIN_TEST_EVAL_MAX", "0"))         # TEST ONLY: cap the eval micros per rank (built only up to the cap)
    assert not (ecap and PROD)
    if IS_ALOE:
        for g in [g for g, L in enumerate(LRANKS) if L is not None]:
            eg = A.eval_micros(RIT[g][1][:ecap] if ecap else RIT[g][1], CFG["eval_data"], LRANKS[g])
            ev_first[g] = eg[0] if eg else None
            ev += eg
            if SPK and RIT_SPK[g]:                                  # speaker held-out calls (eval group 1: eval_spk*)
                es_ = A.eval_micros(RIT_SPK[g], CFG["eval_data"], LRANKS[g], spk=True)
                for m_ in es_:
                    m_["egroup"] = 1
                ev += es_
    else:
        for z in LRANKS:
            if z is not None:
                ev += Z.eval_micros(tbz["eval"], z, ZL)
        assert ev, "every zoom rank needs one eval micro (FSDP parity source)"
    if ecap:
        ev = ev[:ecap]
    evc = allsum([float(sum(int(m["voice_clip"]) for m in ev if not m.get("zoom"))), float(sum(int(m["voice_drawn"]) for m in ev if not m.get("zoom"))),
                  float(sum(1 for m in ev if not m.get("zoom")))], dev)
    t_ = torch.tensor([float(len(ev))], device=dev); dist.all_reduce(t_, op=dist.ReduceOp.MAX); n_ev = int(t_)
    if IS_ALOE:
        ev += [dict(ev[0], dummy=1, own=np.zeros_like(ev[0]["own"])) for _ in range(n_ev - len(ev))]
    else:
        ev += [dict(ev[0], dummy=1, own_zoom=0.0) for _ in range(n_ev - len(ev))]
    log(f"[eval] {n_ev} eval micros per rank (dummies pad the short ranks): Aloe held-out voice clips {int(evc[0])} / {int(evc[2])}; zoom eval packs on the zoom ranks")
    jl = open(f"{LOGDIR}/metrics.jsonl", "a") if RANK == 0 else None

    def emit(d, step):
        if RANK == 0:
            jl.write(json.dumps({"step": step, "time_pt": now_pt(), **d}) + "\n"); jl.flush()

    def evaluate(step):
        model.eval()
        t0_ = time.time()
        acc = {0: [0.0] * NVA, 1: [0.0] * NVA, 2: [0.0] * NVA, 3: [0.0] * NVA}   # 2 x egroup + swap (speaker FT: group 1 = speaker held-out)
        accz = [0.0] * NVZ
        for m in ev:
            if m.get("zoom"):
                _, vz, _, _ = run_micro(model, prep(m, dev, max(float(m["own_zoom"]), 1.0), 1.0), rot, dos, grad=False)
                if not m.get("dummy"):
                    accz = [x + y for x, y in zip(accz, vz)]
            else:
                oc = np.asarray(m["own"]).sum(0)
                v, _, _, _ = run_micro(model, prep(m, dev, max(oc[0], 1.0), max(oc[1], 1.0), lat=encode(front, m)), rot, dos, grad=False)
                if not m.get("dummy"):
                    sw = 2 * int(m.get("egroup", 0)) + int(m["swap"])
                    acc[sw] = [x + y for x, y in zip(acc[sw], v)]
        ad_ = [0.0, 0.0, 0.0, 0.0]                                 # adapter output scale on the first eval micro of every Aloe logical rank
        for g in range(GMAX):                                      # GMAX vadapter calls on every GPU (FSDP parity; zoom: zero latents, discarded)
            m0 = (ev_first[g] if ev_first[g] is not None else ev[0]) if IS_ALOE else None
            ya = adapter_call(model, front, m0, dev)
            fr = ya.pow(2).mean(-1).sqrt()
            if IS_ALOE and ev_first[g] is not None:
                ad_ = [ad_[0] + float(ya.pow(2).sum()), ad_[1] + float(ya.numel()), ad_[2] + float(fr.sum()), ad_[3] + float(fr.numel())]
            del ya, fr
        ad = allsum(ad_, dev)
        g0, g1, gz = allsum(acc[0], dev), allsum(acc[1], dev), allsum(accz, dev)
        d = summarize([x + y for x, y in zip(g0, g1)], "eval")
        d.update(summarize(g0, "eval_asstB")); d.update(summarize(g1, "eval_userB"))
        if SPK:                                                    # speaker held-out calls: all / she was the data assistant / she was the user (flipped)
            s0, s1 = allsum(acc[2], dev), allsum(acc[3], dev)
            d.update(summarize([x + y for x, y in zip(s0, s1)], "eval_spk"))
            d.update(summarize(s0, "eval_spk_asstB")); d.update(summarize(s1, "eval_spk_userB"))
        if not ALOE_ONLY:
            d.update(summarize_zoom(gz, "eval"))
        d["eval/adapter_rms"] = math.sqrt(ad[0] / ad[1]); d["eval/adapter_frame_rms_mean"] = ad[2] / ad[3]; d["eval/emb_A_word_rms"] = EMB_RMS; d["eval/vox_a_scale"] = s_value(model)
        model.train()
        d["eval/time_s"] = time.time() - t0_
        emit(d, step)
        log(f"EVAL step {step}: " + " | ".join(f"{k} {v:.4f}" for k, v in d.items()))
        for k_, v_ in d.items():
            assert math.isfinite(v_), (k_, v_)
        return d

    ncal = (int(os.environ.get("VOX_TEST_CALIB", "0")) or CFG["vox_calib_rows"]) if not (resumed or INITVOX) else 0
    if ncal:                                                      # = the template's s recalibration over the Aloe logical ranks' first train items
        m_ = ev[0]
        if IS_ALOE:
            oc = np.asarray(m_["own"]).sum(0)
            run_micro(model, prep(m_, dev, max(oc[0], 1.0), max(oc[1], 1.0), lat=encode(front, m_)), rot, dos, grad=False)   # FSDP2 root lazy init
        else:
            run_micro(model, prep(m_, dev, max(float(m_["own_zoom"]), 1.0), 1.0), rot, dos, grad=False)
        acc_ = [0.0, 0.0]
        stc = A.Stores() if IS_ALOE else None
        it_pad = [x for x in RIT[0][0] if x is not None][0] if IS_ALOE else None
        for g in range(GMAX):
            L = LRANKS[g]
            rl = ([x for x in RIT[g][0] if x is not None][:ncal] if L is not None else []) if IS_ALOE else []
            for j in range(ncal):
                mm = (A.as_micro(stc.conv(rl[j] if j < len(rl) else (rl[0] if rl else it_pad), CFG["data"], L if L is not None else LRANKS[0]))
                      if IS_ALOE else None)
                fr = adapter_call(model, front, mm, dev).pow(2).mean(-1).sqrt()
                if IS_ALOE and j < len(rl):
                    acc_[0] += float(fr.sum()); acc_[1] += float(fr.numel())
        g_ = allsum(acc_, dev)
        raw = g_[0] / g_[1]
        s_cal = EMB_RMS / raw
        log(f"[calib] raw adapter per-frame RMS mean {raw:.6f} over {int(g_[1])} frames of {ncal} train rows x {LWORLD} Aloe logical ranks; emb_A word-row RMS "
            f"{EMB_RMS:.8f} -> s_init = {s_cal:.8g} (26232: 0.0030343957 from 6.572227 over 388502 frames; ratio {s_cal / 0.0030343957:.6f})")
        CFG["vox_calib_raw_rms"] = raw
        assert 0.25 < s_cal / CFG["vox_a_scale_init"] < 4.0, ("recalibrated s far from the 10-16 calibration: check before training", s_cal)
        set_s(model, s_cal)
        CFG["vox_a_scale_init_recal"] = s_cal
    if not resumed or os.environ.get("COTRAIN_EVAL_AT_RESUME", "1") == "1":
        evaluate(start)
    if EVAL_ONLY:
        log(f"[evalvox] done: eval-only of {CFG['init']} (no training step taken)")
        dist.barrier()
        dist.destroy_process_group()
        return
    import gc
    gc.collect(); gc.freeze(); gc.disable()
    times = list(resumed.get("times", [])) if resumed else []
    stopped = None
    vc_cum = list(resumed["vc_cum"]) if resumed else [0.0, 0.0, 0.0]
    for step in range(start, CFG["steps"]):
        torch.cuda.synchronize(); ts = time.time()
        torch.cuda.reset_peak_memory_stats()
        bs, wait = [], 0.0
        for g in range(GMAX):
            tw = time.time()
            b = next(ld)
            wait += time.time() - tw
            assert int(b["step"]) == step and int(b["slot"]) == g and int(b["lrank"]) == (LRANKS[g] if LRANKS[g] is not None else -1), \
                ("packed micro order", b["step"], b["slot"], b["lrank"], step, g)
            assert bool(b.get("zoom", 0)) == (not IS_ALOE), "micro of the wrong group"
            bs.append(b)
        ocs = [np.asarray(b["own"]).sum(0) for b in bs]
        assert all(len(oc) == 3 for oc in ocs), ("no-pose own counts (text, cb0, light)", [len(oc) for oc in ocs])
        assert all(not (int(b["padslot"]) and not b.get("dummy", 0)) for b in bs), "a pad slot must be a dummy micro"
        # the step's totals: Aloe (text targets, cb0 labels, light frames) over its logical micros, zoom B text targets over its micros
        tot = allsum([float(sum(oc[i] for oc in ocs)) for i in range(3)] + [float(sum(float(b.get("own_zoom", 0.0)) for b in bs))], dev)
        K_ = CFG["dec_k"]
        Mj = allsum([float(sum(0 if (b.get("dummy", 0) or b.get("zoom", 0)) else len(range(j, len(b["light"]), K_)) for b in bs)) for j in range(K_)], dev)
        if DEC_PER_SLOT:                                          # speaker per-slot decoder updates: each slot's chunk frames over all ranks
            Mj_slot = allsum([float(0 if (b.get("dummy", 0) or b.get("zoom", 0)) else len(range(j, len(b["light"]), K_))) for b in bs for j in range(K_)], dev)
            Mj_slot = [Mj_slot[g * K_:(g + 1) * K_] for g in range(GMAX)]
            assert all(abs(sum(Mj_slot[g][j] for g in range(GMAX)) - Mj[j]) < 0.5 for j in range(K_))
        if step == start:
            log(f"[pack] tokens_step per group: step {step} ALOE totals over {LWORLD} logical micros: text targets {int(tot[0])}, cb0 labels {int(tot[1])}, "
                f"light frames {int(tot[2])} | ZOOM B text targets {int(tot[3])} over {ZL} logical micros; per-micro normaliser = group total / {WORLD}")

        def one_pass(bs_, nano_extra=None):
            """the step's micros of this rank (forward + decoder + backward) -> (aloe stats, zoom stats, loss aloe, loss zoom, extra of the last slot)."""
            va, vz, la, lz, ex_ = None, None, 0.0, 0.0, None
            lrs_run = []
            for g, b in enumerate(bs_):
                lat = encode(front, b) if not b.get("zoom") else None
                if b.get("zoom"):
                    bt = prep(b, dev, tot[3] / WORLD, 1.0, lam=CO["w_zoom"])
                else:
                    bt = prep(b, dev, tot[0] / WORLD, tot[1] / WORLD, lat=lat, lam=CO["w_aloe"])
                    bt["lam"] = CO["w_aloe"]
                L_ = LRANKS[g] if LRANKS[g] is not None else LRANKS[0]
                nano = {"Mj": Mj, "first": g == 0, "last": g == GMAX - 1, "lrank": L_, "step": step}
                if DEC_PER_SLOT:                                  # module doc DECODER: K updates on this slot's frames, tower weight / step frames
                    assert (float(sum(Mj_slot[g])) == 0.0) == (not SLOT_REAL[step][g]), ("dummy-only slot map", step, g, Mj_slot[g])
                    nano.update({"Mj": Mj_slot[g], "N": float(sum(Mj)), "first": True, "last": True, "inc": g == GMAX - 1,
                                 "lr_base": SLOT_BASE[step][g] * K_, "skip_update": not SLOT_REAL[step][g]})
                if nano_extra:
                    nano.update(nano_extra)
                v_, z_, lv_, ex_g = run_micro(model, bt, rot, dos, grad=True, nano=nano)
                if DEC_PER_SLOT:                                  # every slot's decoder LRs (the log line reads the first / last)
                    lrs_run += list(ex_g.get("dec_s_lrs", []))
                    if g == GMAX - 1:
                        ex_g["dec_s_lrs"] = list(lrs_run)
                va = v_ if va is None else [x + y for x, y in zip(va, v_)]
                vz = z_ if vz is None else [x + y for x, y in zip(vz, z_)]
                if b.get("zoom"):
                    lz += lv_
                else:
                    la += lv_
                ex_ = (ex_g, nano) if g == GMAX - 1 else ex_
                del lat, bt
            return va, vz, la, lz, ex_

        if step == start and CO["test_gradcheck"]:               # TEST: gradient identities on this step's micros (no update)
            gradcheck(model, dos, bs, one_pass, dev)
        t_enc, t_micro, ea_all = 0.0, 0.0, []
        if step == 0 and CFG["losscheck"]:                        # = the template's step-0 packed-loss gate (Aloe); zoom ranks: FSDP call parity
            lz0 = []
            for g, b in enumerate(bs):
                if b.get("zoom"):
                    lz0.append(pack_loss_check_zoom(model, b, rot, dev))
                else:
                    lat = encode(front, b)
                    bc = b if not int(b["padslot"]) else dict(b["singles"][0], singles=b["singles"])
                    Pv, ex, er, Kc = pack_loss_check(model, bc, rot, dev, lat)
                    if not int(b["padslot"]):
                        ea_all.append(er)
                    if g == 0:
                        Pv0, ex0, K0 = Pv, ex, Kc
                    del lat
            ea = [None] * WORLD; dist.all_gather_object(ea, ea_all)
            ea = [e for x in ea for e in x]
            worst = [max(e[i] for e in ea) for i in range(3)]
            log(f"[pack] LOSS CHECK step 0 (Aloe ranks): packed micro loss vs token-weighted standalone conversations, max rel err over "
                f"{len(ea)} logical ranks: text {worst[0]:.2e} cb0 {worst[1]:.2e} light {worst[2]:.2e}")
            assert len(ea) == LWORLD and max(worst) < CFG["losscheck_tol"], ("packed loss != token-weighted standalone losses", worst, len(ea))
        t_m = time.time()
        va, vz, loss_a, loss_z, (ex_, _) = one_pass(bs)
        torch.cuda.synchronize(); t_micro = time.time() - t_m
        assert abs(ex_["dec_s_N"] - tot[2]) < 0.5, ("speech decoder frames != the step's light frames", ex_["dec_s_N"], tot[2])
        lr_ratio = tot[2] / max(tot[1], 1.0)
        assert CFG["light_ratio_lo"] <= lr_ratio <= CFG["light_ratio_hi"], ("light frames / B coded frames outside the 10 % band", lr_ratio, tot)
        vc = allsum([float(sum(b.get("voice_clip", 0) for b in bs if not b.get("zoom"))), float(sum(b.get("voice_drawn", 0) for b in bs if not b.get("zoom"))),
                     float(sum(0.0 if (b.get("dummy", 0) or b.get("zoom", 0)) else 1.0 for b in bs))], dev)
        vc_cum = [x + y for x, y in zip(vc_cum, vc)]
        for n_, p_ in model.named_parameters():
            if n_ in fz_set:
                assert p_.grad is None, ("a frozen parameter got a gradient", n_)
        if step == 0:
            miss_ = [n_ for n_, p_ in model.vadapter.named_parameters(prefix="vadapter") if M._rows(p_)[1] and p_.grad is None]
            assert not miss_, ("vadapter tensors without a gradient", miss_)
            assert all(p_.grad is not None for p_ in model.vgain.parameters() if M._rows(p_)[1]), "theta (s) has no gradient"
            gs_ = model.vgain.theta.grad
            log(f"[vox] step 0: theta (s = exp theta) gradient {float((gs_.full_tensor() if hasattr(gs_, 'full_tensor') else gs_).reshape(-1)[0]):.4e}")
        gn = {}
        for pg in opt.param_groups:
            g_ = torch.nn.utils.clip_grad_norm_(pg["params"], CFG["clip"][pg["name"]])
            gn[pg["name"]] = float(g_.full_tensor() if hasattr(g_, "full_tensor") else g_)
        lr_used = {pg["name"]: pg["lr"] for pg in opt.param_groups}
        oo_ref = offload_ref(opt) if (OFFLOAD and step < start + 2) else None
        if not CFG["test_no_main_opt"]:
            opt.step()
        if oo_ref is not None:
            offload_check(opt, oo_ref, step)
        sched.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); dt = time.time() - ts; times.append(dt)
        sv = allsum(list(va) + list(vz) + [loss_a, loss_z], dev)
        d = summarize(sv[:NVA], "train")
        if not ALOE_ONLY:
            d.update(summarize_zoom(sv[NVA:NVA + NVZ], "train"))
        d["train/loss_aloe"], d["train/loss_zoom"] = sv[-2] / WORLD, sv[-1] / WORLD
        d["train/loss"] = d["train/loss_aloe"] + d["train/loss_zoom"]
        if ALOE_ONLY:
            del d["train/loss_zoom"]
        d["train/dummies"] = float(nconv[2]) if step == CFG["steps"] - 1 else 0.0
        pk = torch.tensor([torch.cuda.max_memory_allocated() / GiB, dt, wait, t_micro if IS_ALOE else 0.0, t_micro if not IS_ALOE else 0.0], device=dev)
        dist.all_reduce(pk, op=dist.ReduceOp.MAX)
        d.update({f"train/gn_{k}": x for k, x in gn.items()})
        d["train/vox_a_scale"] = s_value(model)
        d.update({f"train/lr_{k}": x for k, x in lr_used.items()})
        d.update({"train/dec_lr_first": ex_["dec_s_lrs"][0], "train/dec_lr_last": ex_["dec_s_lrs"][-1], "train/step_time": dt, "train/peak_gib": float(pk[0]),
                  "train/data_wait": float(pk[2]), "train/micro_s_aloe": float(pk[3]), "train/micro_s_zoom": float(pk[4]), "train/text_targets": tot[0],
                  "train/cb0_labels": tot[1], "train/light_frames": tot[2], "train/zoom_text_targets": tot[3], "train/light_ratio": lr_ratio,
                  "train/voice_clip_frac_cum": vc_cum[0] / max(vc_cum[2], 1.0), "train/voice_drawn_frac_cum": vc_cum[1] / max(vc_cum[2], 1.0)})
        for k_, v_ in d.items():
            assert math.isfinite(v_), (k_, v_, "non-finite: fail loudly")
        done = step + 1
        emit(d, done)
        eta = (CFG["steps"] - done) * float(np.mean(times[-20:]))
        log(f"step {done}/{CFG['steps']} {dt:.1f}s (aloe {float(pk[3]):.1f} zoom {float(pk[4]):.1f} wait {float(pk[2]):.1f}) peak {float(pk[0]):.1f} GiB | "
            + " ".join(f"{k.split('/', 1)[1]}={v:.4f}" for k, v in d.items() if k.startswith(("train_speech/", "train_zoom/")) and k.endswith(("word_ce_norm", "cb0_ce", "light_ce", "pad_ce")) and "light_ce_cb" not in k)
            + f" loss_aloe={d['train/loss_aloe']:.4f}" + ("" if ALOE_ONLY else f" loss_zoom={d['train/loss_zoom']:.4f}") + " "
            + " ".join(f"{k.split('/')[1]}={v:.4g}" for k, v in d.items() if "/gn_" in k or "/lr_trunk" in k or "/lr_vadapter" in k or "vox_a_scale" in k or "dec_lr_last" in k)
            + f" | light {lr_ratio:.4f} clip {vc_cum[0] / max(vc_cum[2], 1.0):.4f} | ETA {eta / 60:.1f} min")
        gc.collect(1)
        if (CFG["eval_every"] and done % CFG["eval_every"] == 0) or done == CFG["steps"] or done in CFG["export_steps"]:
            evaluate(done)
        if done in CFG["export_steps"] and done < CFG["steps"]:
            export_local(model, done, f"{CFG['local_export_root']}/export-{done}")
        if CFG["test_steps"] > 0 and done >= CFG["test_steps"]:
            log(f"[test] QQA_TEST_STEPS {CFG['test_steps']} reached: stop (no final export); main opt step {'SKIPPED' if CFG['test_no_main_opt'] else 'run'}")
            stopped = done
            break
        stop_at = int(os.environ.get("COTRAIN_TEST_STOP_AT", "0"))     # TEST ONLY: behave as if STOP_AFTER_SAVE appeared after step stop_at
        assert not stop_at or CFG["test_steps"] > 0
        st_ = torch.tensor([float(STOP["flag"] or os.path.exists(f"{CFG['control_dir']}/STOP_AFTER_SAVE") or done == stop_at)], device=dev)
        dist.all_reduce(st_, op=dist.ReduceOp.MAX)
        periodic = CFG["resume"]["every"] and done % CFG["resume"]["every"] == 0 and done < CFG["steps"]
        sn_ = torch.tensor([float(os.path.exists(f"{CFG['control_dir']}/SAVE_NOW"))], device=dev)
        dist.all_reduce(sn_, op=dist.ReduceOp.MAX)
        periodic = periodic or (float(sn_) > 0 and float(st_) == 0)                 # SAVE_NOW: full-state save, training continues
        saved = None
        if (float(st_) > 0 or periodic) and done < CFG["steps"]:
            saved = save_resume(model, opt, sched, dos, done, times, vc_cum)
            if float(sn_) > 0 and RANK == 0 and os.path.exists(f"{CFG['control_dir']}/SAVE_NOW"):
                os.rename(f"{CFG['control_dir']}/SAVE_NOW", f"{CFG['control_dir']}/SAVE_NOW.done-{done}")
                log(f"[control] SAVE_NOW at step {done}: resume state {'saved' if saved else 'NOT saved (no /scratch room: see above)'}; training continues")
        if float(st_) > 0 and done < CFG["steps"]:
            log(f"[control] STOP_AFTER_SAVE at step {done}: resume state {'saved' if saved else 'NOT saved (no /scratch room: see above; the run cannot be resumed from here)'}; "
                f"weights-only export, then exit")
            if RANK == 0 and os.path.exists(f"{CFG['control_dir']}/STOP_AFTER_SAVE"):
                os.rename(f"{CFG['control_dir']}/STOP_AFTER_SAVE", f"{CFG['control_dir']}/STOP_AFTER_SAVE.done-{done}")
            if not CFG["test_steps"]:
                export_local(model, done, f"{CFG['local_export_root']}/export-stop{done}", final=True)
                export_weights(model, done, f"{CFG['export']}-stop{done}", CFG["export_min_free_gb"])
            stopped = done
            break
    fz1 = tensor_digest(model, fz_set)
    same = torch.tensor([float(fz1 == fz0)], device=dev); dist.all_reduce(same, op=dist.ReduceOp.MIN)
    assert float(same) == 1.0, "a frozen parameter changed"
    if stopped is None and not CFG["test_steps"] and os.environ.get("ALOE_END_SAVE", "0") == "1":
        saved = save_resume(model, opt, sched, dos, CFG["steps"], times, vc_cum)
        log(f"[end-save] full state at step {CFG['steps']}: {'saved' if saved else 'NOT saved (no /scratch room: see above)'}")
    if stopped is None and not CFG["test_steps"]:
        export_local(model, CFG["steps"], f"{CFG['local_export_root']}/export-{CFG['steps']}", final=True)
        export_weights(model, CFG["steps"], CFG["export"], CFG["export_min_free_gb"])
    log(f"done {stopped or CFG['steps']} steps; mean step {np.mean(times) if times else float('nan'):.2f}s; step times {[round(x, 1) for x in times]}")
    dist.barrier()
    dist.destroy_process_group()


RESUME_FREE = ("resume.", "run_tag", "control_dir", "log_root", "test_steps", "vox_calib_raw_rms", "vox_a_scale_init_recal", "cotrain.gradcheck")


def cfg_resume_view():
    """the resolved config a resume must match exactly (everything but run bookkeeping: run tag / control + log dirs / resume keys / the
    measured s calibration, which the resume restores from the saved weights)."""
    return {k: v for k, v in _flat(json.loads(json.dumps(CFG, default=str))).items() if not k.startswith(RESUME_FREE)}


def cfg_sha():
    return hashlib.sha256(json.dumps(cfg_resume_view(), sort_keys=True).encode()).hexdigest()[:16]


def save_resume(model, opt, sched, dos, done, times, vc_cum):
    """node-local full state (safety_ppm.save_local: per-rank shards + main AdamW, decoder + its AdamW once per node, digests) + run counters."""
    path = f"{CFG['resume']['root']}/{CFG['run_tag']}/dcp-{done}"
    extra = {"step": done, "steps": CFG["steps"], "world": WORLD, "hosts": HOSTS, "rank_hosts": RANK_HOST, "groups": {"aloe": ARANKS, "zoom": ZRANKS},
             "dec_micros": dos["dec_s"].micros, "times": times, "vc_cum": vc_cum, "cfg_flat_sha": cfg_sha(), "cfg_view": cfg_resume_view(), "run_tag": CFG["run_tag"],
             "save_time": now_pt()}
    ok = SAFE.save_local(model, {"opt": opt, "dec_s": dos["dec_s"].opt}, sched, extra, path, RANK, LOCAL, log, keep=10 ** 6)
    log(f"[resume-save] {'WROTE' if ok else 'SKIPPED (no /scratch room)'} {path} at step {done}; resume with COTRAIN_RESUME={path}")
    return ok


OO_CHECK_GROUPS = ("spec", "proj", "vgain")


def offload_ref(opt):
    """ALOE_OPT_OFFLOAD exactness check, part 1 (before opt.step): clones of this rank's local shard, gradient and AdamW state of every tensor
    of the small groups spec / proj / vgain."""
    ref = []
    for pg in opt.param_groups:
        if pg["name"] not in OO_CHECK_GROUPS:
            continue
        for p in pg["params"]:
            if p.grad is None or not M._rows(p)[1]:
                continue
            q = M._loc(p).detach().clone()
            q.grad = M._loc(p.grad).detach().clone()
            st = {k: M._loc(v).detach().clone().to(q.device) for k, v in opt.state.get(p, {}).items()}
            ref.append((p, q, st, pg["lr"]))
    return ref


def offload_check(opt, ref, step):
    """part 2 (after opt.step): a fresh torch fused AdamW (same betas / eps / weight decay / lr) on the clones must give BIT-identical weights
    and m / v; asserted on every rank (step 1: fresh state; step 2: the state after its host round trip)."""
    bad, n = 0, 0
    for p, q, st, lr in ref:
        o = torch.optim.AdamW([q], lr=lr, betas=(0.9, CFG["adam_beta2"]), eps=1e-8, weight_decay=0.0, fused=True)
        if st:
            o.state[q] = st
        o.step()
        sp = opt.state[p]
        n += 1
        bad += int(not (torch.equal(q, M._loc(p)) and torch.equal(o.state[q]["exp_avg"], M._loc(sp["exp_avg"]).to(q.device))
                        and torch.equal(o.state[q]["exp_avg_sq"], M._loc(sp["exp_avg_sq"]).to(q.device))))
    t = torch.tensor([float(bad), float(n)], device=torch.device("cuda", LOCAL)); dist.all_reduce(t)
    log(f"[offload] step {step + 1}: main AdamW with host-offloaded m / v vs a fresh fused AdamW on clones ({'/'.join(OO_CHECK_GROUPS)}, {int(t[1])} "
        f"local tensors over {WORLD} ranks): {int(t[0])} differ (want 0: bit-identical); host-pinned state {opt.host_bytes() / 2 ** 30:.1f} GiB on rank 0")
    assert float(t[0]) == 0.0, "offloaded AdamW step is not bit-identical"


def gradcheck(model, dos, bs, one_pass, dev):
    """TEST (COTRAIN_TEST_GRADCHECK=1): on this step's micros and the current weights, three passes with no update: BOTH groups real, ALOE only
    (zoom micros as zero-weight dummies), ZOOM only (Aloe micros as dummies). Checks: (1) the zoom-only gradient of every Aloe-only tensor
    (tower_s, vadapter, vgain, inp.aud) and the reduced decoder buffer are EXACTLY 0; (2) grad(both) == W_ALOE grad(aloe only) + W_ZOOM
    grad(zoom only) for every tensor (relative error, fp32 reduce order + FA3 backward nondeterminism); (3) the group losses: loss_zoom of the
    zoom-only pass == the zoom group's token mean (sum of per-rank losses / W)."""
    assert all(p.grad is None for p in model.parameters() if hasattr(p, "to_local")), "gradcheck must start from clean grads"

    def dummies(group):
        out = []
        for b in bs:
            if (group == "zoom") == bool(b.get("zoom", 0)):
                b2 = dict(b, dummy=1)
                if b.get("zoom"):
                    b2["own_zoom"] = 0.0
                else:
                    b2["own"] = np.zeros_like(b["own"])
                out.append(b2)
            else:
                out.append(b)
        return out
    res = {}
    for tag, bs_ in (("both", bs), ("aloe", dummies("zoom")), ("zoom", dummies("aloe")), ("both2", bs)):
        cap = {"no_step": True}
        va, vz, la, lz, (ex_, nano) = one_pass(bs_, cap)
        flat = nano.get("flat_out")
        res[tag] = {"g": grads_local(model, dos), "flat": flat.cpu() if flat is not None else None,
                    "loss": allsum([la, lz], dev)}
        for p in model.parameters():
            if hasattr(p, "to_local"):
                p.grad = None
        dos["dec_s"].flat.zero_()
    # (1) exact zeros of the zoom-only pass on the Aloe-only tensors + decoder
    nz_local = 0.0
    for k, g in res["zoom"]["g"].items():
        if k.startswith(ALOE_ONLY_PREFIX) and g is not None:
            nz_local = max(nz_local, float(g.abs().max()) if g.numel() else 0.0)
    flat_z = float(res["zoom"]["flat"].abs().max()) if res["zoom"]["flat"] is not None else -1.0
    none_local = sum(1 for k, g in res["zoom"]["g"].items() if k.startswith(ALOE_ONLY_PREFIX) and g is None)
    t = torch.tensor([nz_local, flat_z, float(none_local)], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MAX)
    # (2) both == w_a aloe + w_z zoom
    num, den, worst = 0.0, 0.0, (0.0, "")
    for k, gb in res["both"]["g"].items():
        if gb is None:
            continue
        ga, gz = res["aloe"]["g"][k], res["zoom"]["g"][k]
        comb = ga + gz                                             # each pass already carries its group weight (prep lam): both = W_A g_a + W_Z g_z
        dd = float((gb - comb).pow(2).sum()); nn = float(gb.pow(2).sum())
        num += dd; den += nn
        if nn > 0 and math.sqrt(dd / nn) > worst[0]:
            worst = (math.sqrt(dd / nn), k)
    fb, fa = res["both"]["flat"], res["aloe"]["flat"]
    dflat = float((fb - fa).abs().max()) if fb is not None else 0.0
    num2 = 0.0                                                     # nondeterminism floor: the same pass twice (FA3 / embedding backward atomics)
    for k, gb in res["both"]["g"].items():
        if gb is not None:
            num2 += float((gb - res["both2"]["g"][k]).pow(2).sum())
    dflat2 = float((fb - res["both2"]["flat"]).abs().max()) if fb is not None else 0.0
    sums = allsum([num, den, dflat, num2, dflat2], dev)
    wl = [None] * WORLD; dist.all_gather_object(wl, worst)
    wk = max(wl)
    rel = math.sqrt(sums[0] / max(sums[1], 1e-30))
    floor = math.sqrt(sums[3] / max(sums[1], 1e-30))
    la_b, lz_b = res["both"]["loss"][0] / WORLD, res["both"]["loss"][1] / WORLD
    la_a, lz_z = res["aloe"]["loss"][0] / WORLD, res["zoom"]["loss"][1] / WORLD
    log(f"[gradcheck] (1) zoom-only pass: max |grad| over the Aloe-only tensors (tower_s, vadapter, vgain, inp.aud) on every rank = {float(t[0]):.3e}, "
        f"reduced decoder buffer max |g| = {float(t[1]):.3e}, Aloe-only tensors without a grad tensor = {int(t[2])} (want 0 / 0 / 0)")
    log(f"[gradcheck] (2) grad(both) vs W_ALOE x grad(aloe only) + W_ZOOM x grad(zoom only): global relative L2 error {rel:.3e}; worst tensor "
        f"{wk[1]} {wk[0]:.3e}; decoder buffer both vs aloe-only max |diff| {sums[2]:.3e} | nondeterminism floor (the both pass run twice): relative "
        f"L2 {floor:.3e}, decoder buffer max |diff| {sums[4]:.3e}")
    log(f"[gradcheck] (3) group losses: both pass aloe {la_b:.6f} zoom {lz_b:.6f} | aloe-only pass aloe {la_a:.6f} | zoom-only pass zoom {lz_z:.6f}")
    assert float(t[0]) == 0.0 and float(t[1]) == 0.0 and int(t[2]) == 0, "zoom ranks contribute a non-zero gradient to an Aloe-only module"
    assert rel < max(float(os.environ.get("COTRAIN_GRADCHECK_TOL", "2e-3")), 3.0 * floor), ("grad(both) != weighted sum of the group gradients", rel, floor, wk)
    assert abs(la_b - la_a) <= 1e-6 * max(1.0, abs(la_a)) and abs(lz_b - lz_z) <= 1e-6 * max(1.0, abs(lz_z)), "group losses depend on the other group"
    CO["gradcheck"] = {"zero_max": float(t[0]), "flat_zero": float(t[1]), "rel_l2": rel, "floor_rel_l2": floor, "worst": wk, "dflat": sums[2],
                       "loss": {"aloe": la_b, "zoom": lz_b}}
    if os.environ.get("COTRAIN_TEST_GRADCHECK_EXIT", "0") == "1":
        log("[gradcheck] PASS; COTRAIN_TEST_GRADCHECK_EXIT=1: exit")
        dist.barrier(); dist.destroy_process_group(); sys.exit(0)
    log("[gradcheck] PASS; continuing with the real step")


if __name__ == "__main__":
    main()
