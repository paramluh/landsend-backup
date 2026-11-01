"""pose6 (10-27, workstream B1): the monoplex Aloe pose FT (= aloe-pose-mono-8020's trainer, via aloe-pose-rosie's env-gated fixes and
pose-s2p's PPM_S2P) with BOTH mods of the 8,1,1 pose run (job 31101, pose5/code-811-s2p). Flags off = mono8020's trainer exactly (TESTS.md).
  POSE-S2P  PPM_S2P=cb0 | full (pose-s2p): tower_p.s_emb in its OWN main-AdamW group 's2p' (decision D4 = 31101's audit S7; pose-s2p had it in
            tower_p): LR / clip value = tower_p's (model_ppm.JOIN; cfg keys lr.s2p / clip.s2p only when on), the same schedule lambda as every
            group (this trainer warms every group up), its own clip norm.
  G0COND    PPM_G0COND=1 (+ ALOE_G0_SIDECAR = pose6/g0aloe/g0aloe.parquet): model_ppm inp.g0 (Linear 8 -> 5120, no bias) adds g0(f) to the
            TRUNK input at each packed item's SLOT = its own P - 1 (data_aloe_pose G0COND doc: decision D1); own group 'g0cond' (LR / clip =
            audio's, JOIN); cfg keys g0cond / g0_sidecar / lr.g0cond / clip.g0cond only when on. Coverage of every train + eval item asserted
            at startup (fail loud on a missing item).
  INIT (D5) QQA_INIT = a ppm-local-bf16-shards-v1 dir (meta.json, as mono8020's local-11488) OR a consolidated bf16 DCP export (.metadata +
            COMPLETE: 31101's ckpt-weka/ppm-v1/w-<step>, or an FT export): EVERY model tensor loaded exactly (bf16 -> fp32 masters by copy),
            incl. tower_p.s_emb.weight / inp.g0.weight when the flags are on (the base trained them; never re-initialised). Flags on + a key
            missing in the base = FAIL, except under the TEST-ONLY override ALOE_TEST_NEWPARAM_RANDOM=<seed> (seeded N(0, 0.02 / sqrt 32)
            for s_emb, N(0, 0.02) for g0, world-invariant; cfg key test_newparam_random so it can never pass a resolved-cfg diff unseen).
            Flags off + base with those keys = FAIL (never dropped silently). Unloaded base keys: only the D32 pose head's (QQA_POSE_HEAD_BUILD=0).
            AdamW state fresh (as mono8020). Exports record s2p / g0cond in train_cfg.json (absent when off).
  LINK OFF  QQA_DECP_BB_OFF=1 (port of pose5/code-811-s2p's PPM_DECP_BB_OFF, the base run 31101's setting; default 0 = mono8020 exactly):
            the D33 backbone -> pose-decoder link decp_bb stays in the module (weights loaded from the base exactly), is FROZEN (in no
            optimizer: group decp_bb not built; frozen set + digest checked), is never computed (model.decp_bb_on False: no zb_p; the pose
            decoder runs with zb=None = the pre-D33 graph; asserted per micro, train and eval) and never gets a gradient (asserted every step).
            cfg key decp_bb_off (only when set). Exports record decp_bb_alpha 0 + decp_bb_off 1 (DCP train_cfg.json; node-local meta.json
            'side') so inference matches. 811's eval probe decp_linkon_* is not ported (eval main pose light = the link-off path, as 811's).
aloe-pose-mono-8020 (10-18): = job 24198's trainer (aloe-v2pure-clip80 = 23051's) run with env ALOE_DATA=aloe-mono: the v2pure machinery
over the Aloe handover pose stores of job 22091 (one monoplex item per qualifying track; data_aloe_pose module doc + its last section) with
the 10-18 prompt rule (voice clip p 0.8, persona text p 0.2, independent per-item draws). Trainer additions (logging / checks only): the
resolved-config diff is logged against BOTH references (QQA_REF_CFG = 23051, QQA_REF_CFG2 = 24198); the [prompt] line logs the realised
both / clip only / persona only / empty fractions (data_aloe_pose.prompt_stats_mono); the step-0 eval under 23051's prompt rule must
reproduce job 22317's step-0 eval (= 23051's rule on these very eval items). With ALOE_DATA unset this file runs 24198 unchanged.
aloe-v2pure-clip80 (10-18): = job 23051's trainer (aloe-v2pure-ft) with ONE recipe change, the system-prompt rule, which lives in the
data module (data_aloe_pose.aloe_cfg / voice_draw: no persona text, voice clip on a seeded 80% of items via env ALOE_*). Trainer additions,
checks / logging only: the step composition is asserted = 23051's (rows digest, env ALOE_ROWS_DIGEST); the realised prompt fractions are
logged at launch ([prompt], data_aloe_pose.prompt_stats); a ONE-OFF step-0 eval of the init under 23051's prompt rule
(eval_ref23051rule_*) checks the pipeline against 23051's step-0 eval. Everything else = 23051 (doc below).
aloe-uncond-ft (10-16): UNCONDITIONAL channel-B generation = job 22091 (QQA_POSE_B=1: 22091's pose maths) with ONLY:
  DATA      QQA_A_EMPTY=1: channel A empty (data_aloe_pose module doc: A's clip / persona dropped from the system block, B's token-identical;
            A at every position = PPM's one-sided-row representation ids PAD / depth -1 / present 0). Same items / plan / split / B targets.
  LOSS      per-STREAM multipliers on top of 22091's weights (env): QQA_LW_WORD x the whole B word-lane loss (text CE: word pieces, NEW_WORD
            timing, PAD, EOB = 22091's out['text'] with W_TEXT_CE / W_CAT_WORD / W_TIM / W_NW / W_EOB unchanged), QQA_LW_SPEECH x (W_CB0 cb0 +
            the speech decoder's gradient into the tower x W_CB), QQA_LW_POSE x (W_POSE_CB0 pose cb0 + the pose decoder's x W_POSE_CB).
            Default 1 / 1 / 1 = 22091. Logged per step: uloss_* (each stream's loss as 22091 weighs it) and wloss_* (x the multiplier).
  MODEL     QQA_POSE_HEAD_BUILD=0: the D32 pose head (22091: built, frozen, unused, D36 off) is NOT built and its checkpoint keys are not loaded.
  LOG       rank 0 prints the resolved-config diff vs 22091's (QQA_REF_CFG).
aloe-nopose-ft (10-16): the NO-POSE ABLATION of job 22091 = THIS file (22091's trainer) with env QQA_POSE_B=0 (default 1 = 22091's
maths unchanged). QQA_POSE_B=0 changes ONLY the pose path, exactly as 21980's no-pose aloe trainer (aloe-ft/code/train_aloe.py) runs it:
  DATA      data_aloe_pose with QQA_POSE_B=0: same stores / items / plan / split / rank sharding / speech micro, no pose arrays, kind 'speech'.
  MODEL     no pose input on B (inp.pose never called: pdepth absent), no pose tower, no pose decoder, no decp_bb link; loss = text + W_CB0 cb0
            + the speech decoder nanosteps (x W_CB). Pose knobs W_POSE_CB0 / W_POSE_CB / LR_TOWER_P / DEC_P_LR / LR_DECP_BB must be 0 (asserted).
  OPT       groups trunk / spec / proj / audio / tower_s (21980's); speech decoder nanostep AdamW only.
  FROZEN    pose parts (tower_p, dec_p, pose_head, decp_bb, inp.pose) in no optimizer, never run, asserted grad-free and bit-identical (= 21980).
  LOG       rank 0 prints the resolved-config diff vs 22091's (QQA_REF_CFG = 22091's export-430/train_cfg.json).
aloe-pose-ft TRAINER (10-16): 21980's aloe-ft trainer (train_aloe.py) with POSE ON CHANNEL B, the pose parts TRAINING.
= aloe-ft/code/train_aloe.py with ONLY these changes (everything else, incl. the qqa recipe / tokens_step normalisers / plan rule, unchanged):
  DATA      data_aloe_pose: one micro per QUALIFYING B track per row (pose ok, no crossbleed into B), kind 'pose' = 21980's speech micro +
            B's pose at PPM's pose delay (= CD) exactly as PPM's pose rows (pdepth summed into the backbone input via inp.pose, pose cb0
            labels, pose 0.6B frames = all pose frames).
  MODEL     PPM's pose path as train_ppm.run_micro runs it on pose rows: speech tower on Mimi B + POSE TOWER on pdepth (cb0 CE x
            W_POSE_CB0), speech decoder + POSE DECODER nanosteps (x W_POSE_CB), D33 backbone link decp_bb (alpha 1.0) into the pose
            decoder's slot 0 (detached + accumulated input gradient fed back as hv_p's). D32 pose head stays OFF (PPM main D36): frozen.
  OPT       + groups tower_p (QQA_LR_TOWER_P) and decp_bb (QQA_LR_DECP_BB); inp.pose stays in 'audio' (model_ppm.group_of, as PPM);
            + the pose decoder's own nanostep AdamW (QQA_DEC_P_LR, same shape: warmup x K, linear decay to 0 at steps x K).
  NORM      pose cb0 and pose light normalised like cb0 / light: the STEP's totals over all ranks / WORLD (tokens_step).
  EVAL      + pose cb0 CE / top-1, pose light CE (eval_speech/pose_* keys; combined + per orientation as before).
  FROZEN    only pose_head (asserted grad-free and bit-identical).
Original aloe-ft doc follows.
aloe-ft TRAINER (10-16): PPM full fine-tune on Model 1's Aloe handover rows, BOTH sides as B (data_aloe: every train row twice per
epoch, once per orientation; loss on B only; PPM's sysprompt rule with both speakers' persona text), with qqa-ft's recipe (qqa.env).
= qqa-ft/code/train_qqa.py (job 21823 / 21832) with ONLY these changes:
  DATA      data_aloe plan (one 7,500-frame row = one micro per rank per step, identical plan on all ranks, node-local copies of all 4 stores
            on every node); the last step's spare ranks run DUMMY micros (zero loss weights, zero counts, zero decoder chunk weights: they add
            nothing to any loss / gradient / statistic; asserted zero).
  EVAL      the stores' 240 eval rows x both orientations (480 micros, round-robin over ranks, dummies on the short ranks), reported per
            orientation (eval_asstB = data assistant on B, eval_userB = data user on B) and combined (eval), at step 0, every
            QQA_EVAL_EVERY steps, every export step and the final step.
  EXPORTS   intermediate weights-only exports at QQA_EXPORT_STEPS -> NODE-LOCAL bf16 shards (ppm-local-bf16-shards-v1, world = this run's,
            rank<R>.pt on each node's ALOE_LOCAL_EXPORT_ROOT/export-<step>, rep.pt + meta.json on rank 0's node; = ckpt_local_export's
            format, readable by train_qqa.load_init / infer_ppm load_weights), so /data never holds them; final = qqa's consolidated DCP.
  LOGGING   train/loss (the optimised per-step token-mean loss, text + cb0, mean over ranks) emitted.
Original qqa-ft doc follows.
qqa-ft TRAINER: run C's speech-duplex quick-QA fine-tune (s4-qqa job 21169) on a PPM checkpoint, in PPM's own model / speech-row maths.
A trimmed copy of ppm/code/train_ppm.py (job 21765's code, snapshot in qqa-ft/code): same model (model_ppm), same per-micro forward / loss
terms (B text CE with PPM's factorised weights, speech-tower cb0 CE, 0.6B speech decoder by NANOSTEPS), FSDP2, bf16 compute / fp32 masters.
Differences from train_ppm (all forced by the run-C recipe, each listed in qqa-ft/STATUS.md's mapping table):
  DATA      speech micros only: ONE packed micro per rank per step = run C's conversations of that step (qqa-ft/plan/PLAN-<host>.json, run C's
            order), each conversation a PPM speech row (data_qqa: swap 0, empty system block, CD 2) in its own attention segment.
  LOSS NORM run C's S4_PACK_NORM=tokens_step: every target token of the step weighs the same. Per loss term the normaliser = the STEP's total
            over all ranks / WORLD (FSDP averages over ranks): text (B text targets), cb0 (B cb0 labels), and for the 0.6B decoder update j
            the step's total chunk-j frames over ranks / WORLD (decoder all-reduce = mean over ranks); its input gradient into the tower is the
            chunk gradients weighted M_j / N (M_j = chunk j frames over all ranks, N = all light frames) = d(token mean)/d(hv).
            Step-0 PACKED-LOSS GATE (run C's S4_PACK_LOSSCHECK): with micro-level token normalisers, each packed loss term (text / cb0 / light)
            == the token-weighted mean of the conversations' standalone losses (max rel err over ranks < 2e-3, asserted).
  INIT      every weight from a PPM local inference copy (ckpt_local_export.py: bf16 dim-0 FSDP shards of world 128 + rep.pt), resharded to
            this world; fresh AdamW (run C: fresh Adam from its init export).
  LR        run C's groups / peaks mapped (env QQA_LR_*), run C's schedule: linear warmup over QQA_WARMUP steps then linear decay to 0 at the
            last step (sched_s3q wsd, floor 0); decoder nanostep updates on the same shape in update units (warmup x K, decay to 0 at steps x K).
  FROZEN    pose parts (tower_p, dec_p, pose_head, decp_bb, inp.pose) are in no optimizer, never run, asserted grad-free and bit-identical.
  EVAL      B normalised word CE (+ text / cb0 / light CE) on run C's 96 held-out conversations (4 per rank, unpacked) at step 0, every
            QQA_EVAL_EVERY steps and the final step.
  SAVE      no optimizer checkpoint; weights-only consolidated bf16 DCP export at the final step (QQA_EXPORT, readable by infer_ppm) and on
            STOP_AFTER_SAVE (control file / SIGUSR1: export at the next step boundary, then exit cleanly)."""
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
import data_aloe_pose as A  # noqa: E402
import sched_s3q as SCH  # noqa: E402

D = Q.D


def E(k, d=None):
    v = os.environ.get(k, d)
    assert v is not None, ("missing env", k)
    return v


CFG = {
    "run_tag": E("QQA_RUN"), "log_root": E("QQA_LOG_ROOT"), "init": E("QQA_INIT"), "plan_dir": E("QQA_PLAN_DIR"),
    "export": E("QQA_EXPORT"), "control_dir": E("QQA_CONTROL_DIR"), "export_min_free_gb": float(E("QQA_EXPORT_MIN_FREE_GB", "300")),
    # 10-16 scaling (export keys only): weights-only exports at these optimizer steps -> <dirname(export)>/export-<step>; an export waits
    # (polling every 15 s, up to QQA_EXPORT_WAIT_MIN) for /data space above the floor instead of being refused (training maths unchanged)
    "export_steps": [int(x) for x in E("QQA_EXPORT_STEPS", "").split(",") if x], "export_wait_min": float(E("QQA_EXPORT_WAIT_MIN", "0")),
    "steps": int(E("QQA_STEPS")), "warmup": int(E("QQA_WARMUP")), "eval_every": int(E("QQA_EVAL_EVERY")),
    "w_text": float(E("QQA_W_TEXT_CE")), "w_cat_word": float(E("QQA_W_CAT_WORD")), "w_tim": float(E("QQA_W_TIM_CE")),
    "w_nw": float(E("QQA_W_NW_CE")), "w_eob": float(E("QQA_W_EOB_CE")), "w_cb0": float(E("QQA_W_CB0")), "w_cb": float(E("QQA_W_CB")),
    "w_pcb0": float(E("QQA_W_POSE_CB0")), "w_pcb": float(E("QQA_W_POSE_CB")), "decp_bb_alpha": float(E("QQA_DECP_BB_ALPHA")),
    "lr": {"trunk": float(E("QQA_LR_TRUNK")), "spec": float(E("QQA_LR_SPEC")), "proj": float(E("QQA_LR_PROJ")),
           "audio": float(E("QQA_LR_AUDIO")), "tower_s": float(E("QQA_LR_TOWER")), "tower_p": float(E("QQA_LR_TOWER_P")),
           "decp_bb": float(E("QQA_LR_DECP_BB"))},
    "clip": {"trunk": float(E("QQA_CLIP_TRUNK")), "spec": float(E("QQA_CLIP_SPEC")), "proj": float(E("QQA_CLIP_PROJ")),
             "audio": float(E("QQA_CLIP_AUDIO")), "tower_s": float(E("QQA_CLIP_TOWER")), "tower_p": float(E("QQA_CLIP_TOWER_P")),
             "decp_bb": float(E("QQA_CLIP_DECP_BB"))},
    "adam_beta2": float(E("QQA_ADAM_BETA2")),
    "dec_k": int(E("QQA_DEC_K")), "dec_lr": float(E("QQA_DEC_LR")), "dec_clip": float(E("QQA_DEC_CLIP")),
    "dec_p_lr": float(E("QQA_DEC_P_LR")), "dec_p_clip": float(E("QQA_DEC_P_CLIP")), "dec_seed": int(E("QQA_DEC_SEED")),
    "dec_micro_frames": int(E("QQA_DEC_MICRO_FRAMES", "300")), "dec_mb_max": int(E("QQA_DEC_MB_MAX", "470")),
    "dec_even": E("QQA_DEC_EVEN", "1") == "1", "dec_dyn": E("QQA_DEC_DYN", "1") == "1", "dec_fast": E("QQA_DEC_FAST", "1") == "1",
    "dec_compile_mode": E("QQA_DEC_COMPILE_MODE", "default"), "eval_light_max": int(E("QQA_EVAL_LIGHT_MAX", "2048")),
    "losscheck": int(E("QQA_LOSSCHECK", "1")), "losscheck_tol": float(E("QQA_LOSSCHECK_TOL", "2e-3")),
    "dl_workers": int(E("QQA_DL_WORKERS", "3")), "data": A.aloe_cfg(), "cd": D.CD,
    "local_export_root": E("ALOE_LOCAL_EXPORT_ROOT"), "stores": A.STORES,
    "pose_b": E("QQA_POSE_B", "1") == "1",                        # aloe-nopose-ft: 0 = the no-pose ablation (module doc)
    # aloe-uncond-ft (module doc); defaults = 22091
    "a_empty": E("QQA_A_EMPTY", "0") == "1", "lw_word": float(E("QQA_LW_WORD", "1")), "lw_speech": float(E("QQA_LW_SPEECH", "1")),
    "lw_pose": float(E("QQA_LW_POSE", "1")), "pose_head_build": E("QQA_POSE_HEAD_BUILD", "1") == "1",
    # aloe-v2pure-ft (10-17): release-v2 clip rows packed 64 rows / step (data_aloe_pose v2pure section), rounds + FULL resumable saves
    "v2pure": True, "rows_per_step": A.ROWS_PER_STEP, "row_cap": A.ROW_CAP, "micro_tok": A.MICRO_TOK, "eval_pack_frames": A.EVAL_PACK,
    "save_root": E("ALOE_SAVE_ROOT"), "save_every_min": float(E("ALOE_SAVE_EVERY_MIN", "60")), "resume": E("ALOE_RESUME", "auto"),
}
if A.MONO:                                                        # aloe-pose-mono-8020: the data source + its micro layout (data module)
    CFG["data_src"] = "aloe-mono"
    CFG["micro_layout"] = "row_per_rank" if A.ROWS_PER_STEP <= int(os.environ["WORLD_SIZE"]) else "lpt"
S2P = M.s2p_mode(os.environ.get("PPM_S2P", "0"))                  # pose-s2p (10-27): speech codes at T -> pose tower at T (model_ppm.S2PEmb)
if S2P:                                                           # flag off: no cfg key, no module, today's trainer exactly
    CFG["s2p"] = S2P
assert os.environ.get("PPM_G0COND", "0") in ("0", "1"), ("PPM_G0COND must be 0 | 1", os.environ.get("PPM_G0COND"))
G0 = os.environ.get("PPM_G0COND", "0") == "1"                    # pose6 G0COND (module doc)
assert G0 == A.G0, "trainer and data must agree on PPM_G0COND"
if G0:                                                            # keys only when on (off = mono8020's resolved cfg exactly)
    CFG["g0cond"] = 1
    CFG["g0_sidecar"] = A.G0_SIDECAR
for g_, on_ in (("g0cond", G0), ("s2p", bool(S2P))):            # pose6 D4: OWN groups, LR / clip value = the JOIN group's (keys only when on)
    if on_:
        CFG["lr"][g_], CFG["clip"][g_] = CFG["lr"][M.JOIN[g_]], CFG["clip"][M.JOIN[g_]]
assert os.environ.get("QQA_DECP_BB_OFF", "0") in ("0", "1"), ("QQA_DECP_BB_OFF must be 0 | 1", os.environ.get("QQA_DECP_BB_OFF"))
BB_OFF = os.environ.get("QQA_DECP_BB_OFF", "0") == "1"          # pose6 LINK OFF (module doc; = 811's PPM_DECP_BB_OFF)
if BB_OFF:                                                        # key only when set (unset = mono8020's resolved cfg exactly)
    CFG["decp_bb_off"] = 1
NEW_KEYS = (("inp.g0.weight",) if G0 else ()) + (("tower_p.s_emb.weight",) if S2P else ())   # loaded from the base (D5)
TEST_FILL = os.environ.get("ALOE_TEST_NEWPARAM_RANDOM", "")      # TEST ONLY (module doc): random non-zero new params when the base lacks them
if TEST_FILL:
    assert NEW_KEYS, "ALOE_TEST_NEWPARAM_RANDOM without PPM_S2P / PPM_G0COND"
    CFG["test_newparam_random"] = int(TEST_FILL)
POSE = CFG["pose_b"]
assert POSE == A.POSE_B, "trainer and data must agree on QQA_POSE_B"
assert CFG["a_empty"] == A.A_EMPTY, "trainer and data must agree on QQA_A_EMPTY"
assert CFG["pose_head_build"] or POSE, "pose-head build flag only for the pose path"
assert POSE or not (G0 or S2P or BB_OFF), ("PPM_G0COND / PPM_S2P / QQA_DECP_BB_OFF need the pose path", G0, S2P, BB_OFF)
if not POSE:
    _pk = {"w_pcb0": CFG["w_pcb0"], "w_pcb": CFG["w_pcb"], "lr_tower_p": CFG["lr"]["tower_p"], "dec_p_lr": CFG["dec_p_lr"], "lr_decp_bb": CFG["lr"]["decp_bb"]}
    assert all(v == 0.0 for v in _pk.values()), ("QQA_POSE_B=0 needs every pose weight / LR at 0", _pk)
CFG["sched"] = {"warmup": CFG["warmup"], "horizon": CFG["steps"], "min_lr_rate": 0.0, "s3_sched": "wsd", "wsd_decay_start": CFG["warmup"],
                "wsd_decay_end": CFG["steps"], "wsd_floor": 0.0, "wsd_shape": "linear"}
CFG["dec_sched"] = {"warmup": CFG["warmup"] * CFG["dec_k"], "horizon": CFG["steps"] * CFG["dec_k"], "min_lr_rate": 0.0, "s3_sched": "wsd",
                    "wsd_decay_start": CFG["warmup"] * CFG["dec_k"], "wsd_decay_end": CFG["steps"] * CFG["dec_k"], "wsd_floor": 0.0, "wsd_shape": "linear"}
if POSE:
    TRAIN_GROUPS = ("trunk", "spec", "proj", "audio", "tower_s", "tower_p") + (() if BB_OFF else ("decp_bb",)) + (("g0cond",) if G0 else ()) \
        + (("s2p",) if S2P else ())
    FROZEN_PREFIX = ("pose_head.",) + (("decp_bb.",) if BB_OFF else ())   # LINK OFF: the link frozen (in no optimizer)                               # PPM main D36: the D32 pose head is off (frozen, unused); aloe-uncond: not built
else:                                                             # no-pose ablation = 21980 (train_aloe.py): pose parts frozen, never run
    TRAIN_GROUPS = ("trunk", "spec", "proj", "audio", "tower_s")
    FROZEN_PREFIX = ("tower_p.", "dec_p.", "pose_head.", "decp_bb.", "inp.pose.")
RANK, WORLD, LOCAL = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
LW = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
HOST = socket.gethostname().split(".")[0]
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
ARR = ("ids", "depth", "present", "pdepth", "pid", "sel", "y", "word", "pad", "eob", "nw", "cb0", "light", "pcb0", "plight") if POSE else \
    ("ids", "depth", "present", "pid", "sel", "y", "word", "pad", "eob", "nw", "cb0", "light")
if G0:
    ARR = ARR + ("g0pos", "g0f")                                  # pose6 G0COND: each packed item's SLOT position + its frame-0 features


def prep(b, dev, S_text, S_cb0, S_pcb0=None):
    """= train_ppm.prep with the normalisers given: S_text (text targets), S_cb0 (cb0 labels), S_pcb0 (pose cb0 labels). tokens_step:
    the step totals / WORLD; micro-level ('tokens', loss check): this micro's own totals."""
    kd = "pose" if POSE else "speech"
    assert b["kind"] == kd, (b["kind"], kd)
    t = {"kind": kd, "src": kd, "segs": [int(x) for x in b["segs"]]}
    for k in ARR:
        t[k] = torch.as_tensor(np.asarray(b[k])).to(dev, non_blocking=True)
    t["pid"] = t["pid"][None]
    w = {k: t[k].float() for k in ("word", "pad", "eob", "nw")}
    S = float(S_text)
    t["wt"] = ((CFG["w_cat_word"] * w["word"] + CFG["w_tim"] * w["pad"] + CFG["w_eob"] * w["eob"] + CFG["w_nw"] * w["nw"]) / S).float()
    t["wn"] = (CFG["w_text"] * w["word"] / S).float()
    m = (t["cb0"] >= 0).float()
    t["cb0_w"] = m / float(S_cb0)
    if POSE:
        m = (t["pcb0"] >= 0).float()
        t["pcb0_w"] = m / float(S_pcb0)
    t["dummy"] = bool(b.get("dummy", 0))
    if t["dummy"]:                                                # aloe: a dummy micro carries zero weight in every loss term
        t["wt"], t["wn"], t["cb0_w"] = t["wt"] * 0.0, t["wn"] * 0.0, t["cb0_w"] * 0.0
        if POSE:
            t["pcb0_w"] = t["pcb0_w"] * 0.0
    return t


TK = ("word_ce", "word_norm", "n_word", "word_hit", "pad_ce", "n_pad", "nw_ce", "n_nw", "eob_ce", "n_eob")
CB_S, CB_P = (8, 16, 31), (1, 8, 16, 31)                            # per-codebook light CE logged (speech decoder / pose decoder), as PPM


def text_vec(t, st):
    w, p, n_, e = (t[k].float() for k in ("word", "pad", "nw", "eob"))
    return [((st[0] * w).sum()), ((st[3] * w).sum()), (w.sum()), ((st[1] * w).sum()), ((st[0] * p).sum()), (p.sum()),
            ((st[0] * n_).sum()), (n_.sum()), ((st[0] * e).sum()), (e.sum())]


NSTREAM = 3                                                       # aloe-uncond: run_micro appends (text, cb0, pose cb0) loss values


def summarize(v, pre):
    v = v[:-NSTREAM]
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
    if not POSE:
        return out
    o = 5 + len(CB_S)                                             # pose (PPM summarize's pose keys): cb0 sum / n / hits, light sum / n, cb 1/8/16/31
    out[f"{k}/pose_cb0_ce"] = x[o] / max(x[o + 1], 1.0); out[f"{k}/pose_cb0_top1"] = x[o + 2] / max(x[o + 1], 1.0)
    out[f"{k}/pose_light_ce"] = x[o + 3] / max(x[o + 4], 1.0)
    for j, c in enumerate(CB_P):
        out[f"{k}/pose_light_ce_cb{c}"] = x[o + 5 + j] / max(x[o + 4] / 31.0, 1.0)
    return out


# ------------------------------------------------------------------------------------------------ nanosteps (token-weighted over the step)
def dec_lam():
    return SCH.make_lambda(True, CFG["dec_sched"])


class DecOpt:
    """= train_ppm.DecOpt (replicated decoder: grads = views of ONE flat fp32 buffer, one all-reduce per update), LR = peak x the run-C
    shaped schedule over decoder updates (micros x K + j). aloe-pose: one per decoder (dec_s: QQA_DEC_LR / QQA_DEC_CLIP, dec_p: QQA_DEC_P_LR /
    QQA_DEC_P_CLIP), the same schedule shape (both decoders run on every micro)."""

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
    """= train_ppm._mb_plan (dec_even / dec_dyn speed path; same chunk gradient)."""
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


def nanosteps_tw(dec, do, hv, depth, light, dummy=False, zb=None):
    """train_ppm.nanosteps (replicated, one decoder) with tokens_step weights: chunk j's frames weigh 1 / (31 x M_j / WORLD) (M_j = chunk-j
    frames summed over ranks) -> after the all-reduce mean the update's gradient = the mean CE over ALL ranks' chunk-j frames; the input
    gradient accumulates chunk gradients x M_j / N (N = all ranks' light frames) = d(token mean x WORLD / ...)/d hv, i.e. the tower /
    backbone receive the token-weighted decoder loss. zb (pose decoder, D33 link input): detached + accumulated exactly as hv (train_ppm
    _nano_setup / nanosteps_pair). -> (acc, stats, lrs, (M, N), acc_zb or None)."""
    K, dev = CFG["dec_k"], hv.device
    hd = hv.detach().requires_grad_()
    zd = zb.detach().requires_grad_() if zb is not None else None
    n = int(light.numel())
    perm = torch.randperm(max(n, 1), generator=torch.Generator().manual_seed(CFG["dec_seed"] * 1000003 + do.micros * 1009 + RANK))
    sizes = [int(perm[j::K].numel()) if n else 0 for j in range(K)]
    Mg = torch.tensor([0] * K if dummy else sizes, dtype=torch.float64, device=dev)      # aloe: a dummy micro adds no frames to any update
    dist.all_reduce(Mg)
    Mj = Mg.tolist(); N = float(sum(Mj))
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
        dist.all_reduce(do.flat)
        do.flat.div_(WORLD)
        lr = do.lr(do.micros * K + j)
        for g in do.opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(do.ps, do.clip)
        do.opt.step()
        lrs.append(lr)
    hd.grad = None
    if zd is not None:
        zd.grad = None
    do.micros += 1
    return acc, st, lrs, (Mj, N), acc_z


class DecFast:
    def __enter__(self):
        self.prev = M.DEC_FAST["active"]; M.DEC_FAST["active"] = CFG["dec_fast"]

    def __exit__(self, *a):
        M.DEC_FAST["active"] = self.prev


def run_micro(model, b, rot, dos, grad, info=None):
    """= train_ppm.run_micro for a POSE micro (NM = 1, w_task 1): forward (speech + pose towers, D33 link), loss = text + W_CB0 cb0 +
    W_POSE_CB0 pose cb0; nanosteps of BOTH decoders (train: speech decoder on hv_s / Mimi B / light, then pose decoder on hv_p / pdepth /
    plight with the link zb_p; each decoder's own update sequence, = train_ppm with dec_pair's interleave off: same maths) or a seeded
    <= eval_light_max frame decoder eval each; ONE backward. -> (stats vector, loss value, extra)."""
    with (torch.enable_grad() if grad else torch.no_grad()):
        out = model(b, rot, rot, info=info)
    assert "phead" not in out, "D32 pose head must stay off (PPM main D36)"
    if not POSE:
        assert "hv_p" not in out and "zb_p" not in out, "no-pose ablation: the pose path must not run"
    lw_w, lw_s, lw_p = CFG["lw_word"], CFG["lw_speech"], CFG["lw_pose"]     # aloe-uncond: per-stream multipliers (1 / 1 / 1 = 22091)
    loss = lw_w * out["text"] + lw_s * CFG["w_cb0"] * out["cb0"] + (lw_p * CFG["w_pcb0"] * out["pcb0"] if POSE else 0.0)
    v = text_vec(b, out["text_stats"])
    extra = {}
    dummy = b.get("dummy", False)
    pairs = [("dec_s", model.dec_s, out["hv_s"], b["depth"][1], b["light"], lw_s * CFG["w_cb"], None)]
    if POSE:
        pairs.append(("dec_p", model.dec_p, out["hv_p"], b["pdepth"], b["plight"], lw_p * CFG["w_pcb"], out.get("zb_p")))
        assert (pairs[1][6] is None) == BB_OFF, "D33 decp_bb link: on (PPM main) unless QQA_DECP_BB_OFF=1 (then never computed)"
    dvals = []
    roots, grads = [loss], [None]
    for name, dec, hv, depth, light, wcb, zb in pairs:
        if grad:
            with DecFast():
                up, acc, lrs, (Mj, N), upz = nanosteps_tw(dec, dos[name], hv, depth, light, dummy=dummy, zb=zb)
            roots.append(hv); grads.append((up * wcb).to(hv.dtype))
            if zb is not None:
                roots.append(zb); grads.append((upz * wcb).to(zb.dtype))
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
    if POSE:
        v += [out["pcb0_sum"], out["pcb0_n"], out["pcb0_hit"], dvals[1][0], dvals[1][1]]
        v += [dvals[1][2][c - 1] for c in CB_P]
    # aloe-uncond: per-stream optimised losses WITHOUT the stream multipliers (tokens_step: sum over ranks / WORLD = the step's token mean)
    v += [out["text"].detach(), out["cb0"].detach(), (out["pcb0"].detach() if POSE else 0.0)]
    if grad:
        torch.autograd.backward(roots, grads)
    dvc = loss.device
    v = torch.stack([torch.as_tensor(x, device=dvc).double().reshape(()) for x in v]).tolist()
    if dummy:                                                     # aloe: a dummy micro contributes nothing to any statistic
        assert float(loss) == 0.0, ("dummy micro with a non-zero loss", float(loss))
        v = [0.0] * len(v)
    return v, float(loss), extra


@torch.no_grad()
def micro_losses(model, mic, rot, dev):
    """one micro with MICRO-level token normalisers -> (text, cb0, light mean CE over ALL its light frames, pose cb0, pose light mean CE
    over ALL its pose frames), counts."""
    oc = np.asarray(mic["own"]).sum(0)
    b = prep(mic, dev, oc[0], oc[1], oc[3] if POSE else None)
    out = model(b, rot, rot)
    res = []
    decs = [(model.dec_s, out["hv_s"], b["depth"][1], b["light"], None)]
    if POSE:
        assert ("zb_p" in out) != BB_OFF, "LINK OFF: zb_p must not exist (on: must)"
        decs.append((model.dec_p, out["hv_p"], b["pdepth"], b["plight"], out.get("zb_p")))
    for dec, hv, depth, light, zb in decs:
        s_tot, n_tot = 0.0, 0
        for a0 in range(0, light.numel(), 1024):
            fl = light[a0:a0 + 1024]
            wz = torch.zeros(fl.numel(), device=dev)
            _, s_, num_, _ = dec(hv, depth, fl, wz) if zb is None else dec(hv, depth, fl, wz, zb=zb)
            s_tot += float(s_); n_tot += int(num_)
        res.append((s_tot / max(n_tot, 1), n_tot))
    if not POSE:                                                  # = 21980's micro_losses
        return [float(out["text"]), float(out["cb0"]), res[0][0]], [float(oc[0]), float(oc[1]), float(res[0][1])]
    return ([float(out["text"]), float(out["cb0"]), res[0][0], float(out["pcb0"]), res[1][0]],
            [float(oc[0]), float(oc[1]), float(res[0][1]), float(oc[3]), float(res[1][1])])


def pack_loss_check(model, b, rot, dev):
    """run C's S4_PACK_LOSSCHECK: each packed loss term == the token-weighted mean of the standalone conversations' losses. Same number of
    forwards on every rank (FSDP): ranks with fewer conversations repeat conversation 0 (ignored)."""
    model.eval()
    singles = b["singles"]
    Pv, _ = micro_losses(model, b, rot, dev)
    K = len(singles)
    t = torch.tensor([float(K)], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MAX); Kmax = int(t)
    NT = 5 if POSE else 3
    num, den = [0.0] * NT, [0.0] * NT
    for k in range(Kmax):
        L, N = micro_losses(model, singles[k if k < K else 0], rot, dev)
        if k < K:
            for i in range(NT):
                num[i] += N[i] * L[i]; den[i] += N[i]
    model.train()
    exp = [num[i] / den[i] for i in range(NT)]
    err = [abs(Pv[i] - exp[i]) / max(abs(exp[i]), 1e-12) for i in range(NT)]
    return Pv, exp, err, K


# ------------------------------------------------------------------------------------------------ init / export
NEW_ALL = ("inp.g0.weight", "tower_p.s_emb.weight")             # pose6: the two mods' parameters (D5)


def _init_new_missing(params, miss, meta):
    """pose6 D5: a new parameter (NEW_KEYS) the base lacks -> FAIL, except under the TEST-ONLY override ALOE_TEST_NEWPARAM_RANDOM=<seed>:
    seeded N(0, std) per NAME and global row chunk (heads_s2q.init_rows_: world-size invariant; s_emb 0.02 / sqrt 32 = the code_emb rule,
    g0 0.02), never zero. -> the remaining missing names (asserted empty by the caller)."""
    new = [n for n in miss if n in NEW_KEYS]
    assert not new or TEST_FILL, ("init: the base lacks the new parameters; D5: they must be loaded from the base (31101 export), never "
                                  "re-initialised (test-only override: ALOE_TEST_NEWPARAM_RANDOM)", new, CFG["init"])
    assert not TEST_FILL or new, ("ALOE_TEST_NEWPARAM_RANDOM set but the base HAS the new parameters: unset it", CFG["init"])
    for n in new:
        p = params[n]
        a, _ = M._rows(p)
        M.H2.init_rows_(M._loc(p).data, a, tuple(p.shape), "normal", (0.02 / 32 ** 0.5) if n == "tower_p.s_emb.weight" else 0.02, n, int(TEST_FILL))
    if new:
        log(f"!!! [init] TEST ONLY (ALOE_TEST_NEWPARAM_RANDOM={TEST_FILL}): {new} are NOT in {CFG['init']} -> seeded RANDOM non-zero values "
            f"(never a real run: cfg key test_newparam_random)")
    meta["newparam_random_fill"] = new
    return [n for n in miss if n not in new]


def _init_unloaded(unl, meta):
    """checkpoint tensors this model does not build: never one of the mods' parameters (a base that trained them with the flag off here would
    silently drop them: FAIL); otherwise only the D32 pose head's (QQA_POSE_HEAD_BUILD=0) or none."""
    stale = [k for k in unl if k in NEW_ALL]
    assert not stale, ("init: the base carries trained new parameters this config does not build (set PPM_S2P / PPM_G0COND as the base)", stale)
    if not CFG["pose_head_build"]:
        assert unl in ([], ["pose_head.bias", "pose_head.weight"]), ("unloaded checkpoint keys must be exactly the pose head's (or none: an init from a pose-FT export, which has no pose head)", unl[:8])
    else:
        assert not unl, ("checkpoint keys not loaded", unl[:8])
    meta["unloaded_keys"] = unl


@torch.no_grad()
def load_init_dcp(model, path, dev):
    """pose6 D5: QQA_INIT = a consolidated bf16 DCP export (safety_ppm.export_weights w-<step> of the 8,1,1 run 31101 / this trainer's
    export_weights: .metadata + COMPLETE [+ train_cfg.json]): every model tensor loaded IN PLACE into this world's FSDP2 shards (DCP reshards;
    bf16 -> fp32 masters by copy = exact), the replicated decoders whole on every rank. = aloe-v2vox-asr-w15973's load_init_dcp pattern."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    assert os.path.isfile(f"{path}/COMPLETE") and os.path.isfile(f"{path}/.metadata"), ("init: not a COMPLETE DCP export", path)
    smd = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
    keys = set(smd)
    side = json.load(open(f"{path}/train_cfg.json")) if os.path.isfile(f"{path}/train_cfg.json") else {}
    if S2P and side.get("s2p") is not None:
        assert side["s2p"] == S2P, ("init: the base's s2p mode differs from PPM_S2P", side["s2p"], S2P)
    meta = {"format": "dcp", "step": side.get("step", open(f"{path}/COMPLETE").read().strip()), "run_tag": side.get("run", "?"), "world": "dcp",
            "save_time": "?", "side": side}
    params = {n.replace("_checkpoint_wrapped_module.", ""): p for n, p in model.named_parameters()}
    msd = get_model_state_dict(model)
    assert set(params) <= set(msd), ("state dict lacks parameters", sorted(set(params) - set(msd))[:6])
    miss = _init_new_missing(params, sorted(set(msd) - keys), meta)
    assert not miss, ("init: parameters not in the checkpoint", miss[:6])
    want = {k: v for k, v in msd.items() if k in keys}
    for k, v in want.items():
        assert tuple(smd[k].size) == tuple(v.shape), ("init: shape differs from the base", k, tuple(smd[k].size), tuple(v.shape))
    before = {k: M._loc(v).data_ptr() for k, v in want.items()}
    dcp.load(want, checkpoint_id=path)
    for k, v in want.items():                                      # get_model_state_dict returns the live parameters: loaded in place
        assert M._loc(v).data_ptr() == before[k] and (k not in params or M._loc(params[k]).data_ptr() == before[k]), ("not in place", k)
    _init_unloaded(sorted(keys - set(msd)), meta)
    nonfin = torch.tensor([float(sum(int(not torch.isfinite(M._loc(p)).all()) for p in params.values()))], device=dev)
    dist.all_reduce(nonfin)
    assert float(nonfin) == 0, "non-finite init weights"
    return meta, sum(v.numel() for v in want.values()) * 2 / 1e9


@torch.no_grad()
def load_init(model, path, dev):
    """every parameter <- a ppm-local-bf16-shards-v1 copy (world-W0 dim-0 FSDP2 shards rank<r>.pt, concatenated in rank order = infer_ppm
    load_weights; rep.pt = the replicated decoders), resharded onto this world's FSDP2 shards (each rank reads only the rows it holds).
    pose6: no meta.json -> a consolidated DCP export (load_init_dcp); the mods' parameters loaded like every other tensor (D5)."""
    if not os.path.isfile(f"{path}/meta.json"):
        return load_init_dcp(model, path, dev)
    meta = json.load(open(f"{path}/meta.json"))
    assert meta["format"] == "ppm-local-bf16-shards-v1" and os.path.isfile(f"{path}/COMPLETE"), path
    W0 = int(meta["world"])
    params = {n.replace("_checkpoint_wrapped_module.", ""): p for n, p in model.named_parameters()}
    # global row offsets of every tensor in each source rank file (FSDP2 = torch.chunk along dim 0)
    srcs = {}

    def src(r):
        if r not in srcs:
            srcs[r] = torch.load(f"{path}/rank{r}.pt", map_location="cpu", mmap=True, weights_only=True)
        return srcs[r]
    rep = torch.load(f"{path}/rep.pt", map_location="cpu", mmap=True, weights_only=True)
    done, nbytes = set(), 0
    for n, p in params.items():
        if n in rep:
            p.copy_(rep[n].to(device=dev, dtype=p.dtype)); done.add(n); continue
        if n in NEW_KEYS and n not in src(0):                         # pose6 D5: a new parameter the base lacks -> _init_new_missing (FAIL / test fill)
            continue
        R = p.shape[0]
        c0 = -(-R // W0)                                          # source chunk rows (torch.chunk)
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
    miss = _init_new_missing(params, sorted(set(params) - done), meta)   # pose6 D5 (pose-s2p's zero-fill of a missing s_emb removed)
    assert not miss, ("init: parameters not in the checkpoint", miss[:6])
    ck = set(rep) | set(src(0))                                   # aloe-uncond: checkpoint tensors the model does not have (not loaded)
    _init_unloaded(sorted(ck - set(params)), meta)
    srcs.clear()
    nonfin = torch.tensor([float(sum(int(not torch.isfinite(M._loc(p)).all()) for p in params.values()))], device=dev)
    dist.all_reduce(nonfin)
    assert float(nonfin) == 0, "non-finite init weights"
    t = torch.tensor([float(nbytes)], device=dev); dist.all_reduce(t)
    return meta, float(t) / 1e9


def tensor_digest(model, names):
    """sha256 over THIS rank's local values (FSDP shard / replicated tensor) of the named parameters -> hex (compared per rank)."""
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        if n not in names:
            continue
        h.update(n.encode()); h.update(M._loc(p).detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def export_weights(model, step, path, min_free_gb):
    """consolidated bf16 DCP of every model tensor (= safety_ppm.export_weights' format, infer_ppm 'dcp' kind) at `path`; never deletes."""
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
        if time.time() - tw > 1.0:
            log(f"[export] waited {time.time() - tw:.0f}s for /data space ({free:.0f} GB free)")
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
        json.dump({"step": int(step), "decp_bb": 1, "decp_bb_alpha": 0.0 if BB_OFF else 1.0, **({"decp_bb_off": 1} if BB_OFF else {}),
                   **({"s2p": S2P} if S2P else {}), **({"g0cond": 1} if G0 else {}),
                   "run": CFG["run_tag"], "init": CFG["init"], "cfg": CFG},
                  open(f"{path}/train_cfg.json", "w"), indent=1, default=str)
        open(f"{path}/COMPLETE", "w").write(str(step))
    dist.barrier()
    log(f"[export] {path}: consolidated bf16 weights ({size_gb:.1f} GB, {time.time() - t0:.0f}s); /data free before {free:.0f} GB")
    return path


def export_local(model, step, path):
    """aloe: weights-only NODE-LOCAL export (ppm-local-bf16-shards-v1 = ckpt_local_export's format, world = WORLD): each rank writes its
    FSDP2 dim-0 shards (names without the checkpoint wrapper) to <path>/rank<R>.pt on its own node; rank 0 also writes rep.pt (the
    replicated tensors) + meta.json; COMPLETE on rank 0's node once every rank is done. Nothing on /data."""
    t0 = time.time()
    os.makedirs(path, exist_ok=True)
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
        json.dump({"format": "ppm-local-bf16-shards-v1", "src": "aloe-ft train_aloe export_local", "step": int(step), "world": WORLD, "hosts": hosts,
                   "run_tag": CFG["run_tag"], "init": CFG["init"], "cfg": CFG, "save_time": now_pt(),
                   **({"side": {"decp_bb": 1, "decp_bb_alpha": 0.0 if BB_OFF else 1.0, **({"decp_bb_off": 1} if BB_OFF else {}),   # pose6: inference
                                **({"s2p": S2P} if S2P else {}), **({"g0cond": 1} if G0 else {})}} if (BB_OFF or S2P or G0) else {})},   # settings
                  open(f"{path}/meta.json", "w"), indent=1, default=str)
    dist.barrier()
    if RANK == 0:
        open(f"{path}/COMPLETE", "w").write(str(step))
    t = torch.tensor([float(nb)], device=next(model.parameters()).device); dist.all_reduce(t)
    log(f"[export-local] {path} on every node: {float(t) / 1e9:.1f} GB bf16 shards (world {WORLD}), {time.time() - t0:.0f}s")


# ================================================================================================ aloe-v2pure-ft (10-17): rounds + saves
def nanosteps_multi(dec, do, mics):
    """nanosteps_tw over ALL of this rank's micros of the step (rounds), jointly: one decoder update sequence per STEP (K updates, = 22317),
    update j = chunk j of EVERY micro of every rank (M_j summed over ranks and rounds), each micro's input gradient accumulated x M_j / N
    exactly as nanosteps_tw. mics = [(hv, depth, light, zb or None, dummy)]. -> (acc per micro, stats, lrs, (M, N), acc_zb per micro)."""
    K, dev = CFG["dec_k"], mics[0][0].device
    hds = [m[0].detach().requires_grad_() for m in mics]
    zds = [m[3].detach().requires_grad_() if m[3] is not None else None for m in mics]
    perms, sizes = [], []
    for r, (hv, depth, light, zb, dm) in enumerate(mics):
        n = int(light.numel())
        g = torch.Generator().manual_seed((CFG["dec_seed"] * 1000003 + do.micros * 1009 + RANK) * 64 + r)
        perm = torch.randperm(max(n, 1), generator=g)
        sizes.append([int(perm[j::K].numel()) if n else 0 for j in range(K)])
        perms.append(perm.pin_memory().to(dev, non_blocking=True))
    Mg = torch.tensor([float(sum(0 if mics[r][4] else sizes[r][j] for r in range(len(mics)))) for j in range(K)], dtype=torch.float64, device=dev)
    dist.all_reduce(Mg)
    Mj = Mg.tolist(); N = float(sum(Mj))
    accs = [torch.zeros(h.shape, dtype=torch.float32, device=dev) for h in hds]
    acczs = [torch.zeros(z.shape, dtype=torch.float32, device=dev) if z is not None else None for z in zds]
    st = {"sum": torch.zeros((), device=dev, dtype=torch.float32), "n": 0, "cb": torch.zeros(31, device=dev, dtype=torch.float32)}
    lrs = []
    for j in range(K):
        do.flat.zero_()
        den = 31.0 * Mj[j] / WORLD
        for r, (hv, depth, light, zb, dm) in enumerate(mics):
            hd, zd, m = hds[r], zds[r], sizes[r][j]
            idx = perms[r][j::K] if m else perms[r][:0]
            hd.grad = None
            if zd is not None:
                zd.grad = None
            for a0, sz in _mb_plan(m):
                mv = max(0, min(sz, m - a0))
                fl = light.index_select(0, idx[a0:a0 + mv]) if mv else light[:1]
                if fl.numel() < sz:
                    fl = torch.cat([fl, light[:1].expand(sz - fl.numel())])
                wf = torch.zeros(sz, dtype=torch.float32, device=dev)
                if mv and not dm:
                    wf[:mv] = 1.0 / den
                l_, s_, num_, cb_ = dec(hd, depth, fl, wf, n_valid=mv) if zd is None else dec(hd, depth, fl, wf, n_valid=mv, zb=zd)
                if mv and not dm:
                    st["sum"] += s_; st["n"] += num_; st["cb"] += cb_.detach().float()
                l_.backward()
            if hd.grad is not None and m and not dm:
                accs[r] += hd.grad.float() * (Mj[j] / N)
            if zd is not None and zd.grad is not None and m and not dm:
                acczs[r] += zd.grad.float() * (Mj[j] / N)
        dist.all_reduce(do.flat)
        do.flat.div_(WORLD)
        lr = do.lr(do.micros * K + j)
        for g_ in do.opt.param_groups:
            g_["lr"] = lr
        torch.nn.utils.clip_grad_norm_(do.ps, do.clip)
        do.opt.step()
        lrs.append(lr)
    for hd, zd in zip(hds, zds):
        hd.grad = None
        if zd is not None:
            zd.grad = None
    do.micros += 1
    return accs, st, lrs, (Mj, N), acczs


def train_step_multi(model, bts, rot, dos):
    """ONE optimizer step over this rank's R micros (rounds) = run_micro(grad=True) over the union: PASS A forwards every micro without a graph
    (hv_s / hv_p / zb_p kept), the decoders take their K joint nanostep updates over all rounds (nanosteps_multi), PASS B re-forwards each
    micro WITH a graph and backpropagates loss + (hv, zb) . (decoder input gradients) (= run_micro's single backward per micro); FSDP sums
    the rounds' gradients. Same weights in both passes (no optimizer step between them). -> (stats vector = run_micro's layout, loss, extra)."""
    assert POSE, "aloe-v2pure: pose path only"
    lw_w, lw_s, lw_p = CFG["lw_word"], CFG["lw_speech"], CFG["lw_pose"]
    keep, tv, c0, p0, sl, loss_sum, la = [], None, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, []
    with torch.no_grad():
        for b in bts:
            out = model(b, rot, rot)
            assert "phead" not in out and (out.get("zb_p") is None) == BB_OFF, "D32 off; D33 link on (PPM main) unless QQA_DECP_BB_OFF=1"
            loss = lw_w * out["text"] + lw_s * CFG["w_cb0"] * out["cb0"] + lw_p * CFG["w_pcb0"] * out["pcb0"]
            keep.append((out["hv_s"].detach(), out["hv_p"].detach(), None if BB_OFF else out["zb_p"].detach()))
            la.append(float(out["cb0"]) + float(out["pcb0"]))
            del out, loss
    with DecFast():
        up_s, st_s, lrs_s, (Mjs, Ns), _ = nanosteps_multi(model.dec_s, dos["dec_s"],
                                                           [(k[0], b["depth"][1], b["light"], None, b.get("dummy", False)) for k, b in zip(keep, bts)])
        up_p, st_p, lrs_p, (Mjp, Np), upz = nanosteps_multi(model.dec_p, dos["dec_p"],
                                                            [(k[1], b["pdepth"], b["plight"], k[2], b.get("dummy", False)) for k, b in zip(keep, bts)])
    ws, wp = lw_s * CFG["w_cb"], lw_p * CFG["w_pcb"]
    dl, dh = 0.0, 0.0
    for r, b in enumerate(bts):
        out = model(b, rot, rot)
        loss = lw_w * out["text"] + lw_s * CFG["w_cb0"] * out["cb0"] + lw_p * CFG["w_pcb0"] * out["pcb0"]
        lb = float(out["cb0"]) + float(out["pcb0"])
        dl = max(dl, abs(lb - la[r]) / max(abs(la[r]), 1e-12))                         # pass A vs pass B towers' cb0 CE: same forward
        dh = max(dh, max(float((out[k_].detach().float() - keep[r][i_].float()).abs().max()) for i_, k_ in enumerate(("hv_s", "hv_p", "zb_p"))
                         if not (BB_OFF and k_ == "zb_p")))
        keep[r] = None
        if b.get("dummy", False):
            assert float(loss) == 0.0, ("dummy micro with a non-zero loss", float(loss))
        else:                                                     # = run_micro's stats / loss values (grad-mode forward, as 22317)
            v_ = [float(x) for x in text_vec(b, out["text_stats"])]
            tv = v_ if tv is None else [x + y for x, y in zip(tv, v_)]
            c0 = [c0[0] + float(out["cb0_sum"]), c0[1] + float(out["cb0_n"]), c0[2] + float(out["cb0_hit"])]
            p0 = [p0[0] + float(out["pcb0_sum"]), p0[1] + float(out["pcb0_n"]), p0[2] + float(out["pcb0_hit"])]
            sl = [sl[0] + float(out["text"]), sl[1] + float(out["cb0"]), sl[2] + float(out["pcb0"])]
            loss_sum += float(loss)
        roots = [loss, out["hv_s"], out["hv_p"]] + ([] if BB_OFF else [out["zb_p"]])
        grads = [None, (up_s[r] * ws).to(out["hv_s"].dtype), (up_p[r] * wp).to(out["hv_p"].dtype)] + ([] if BB_OFF else [(upz[r] * wp).to(out["zb_p"].dtype)])
        assert not BB_OFF or ("zb_p" not in out and upz[r] is None), "LINK OFF: the removed link was computed / fed back"
        torch.autograd.backward(roots, grads)
        del out, loss, roots, grads
    tv = tv if tv is not None else [0.0] * len(TK)
    v = tv + c0 + [float(st_s["sum"]), float(st_s["n"])] + [float(st_s["cb"][c - 1]) for c in CB_S]
    v += p0 + [float(st_p["sum"]), float(st_p["n"])] + [float(st_p["cb"][c - 1]) for c in CB_P]
    v += sl
    extra = {"dec_s_lrs": lrs_s, "dec_s_M": Mjs, "dec_s_N": Ns, "dec_p_lrs": lrs_p, "dec_p_M": Mjp, "dec_p_N": Np, "passab_loss_rel": dl, "passab_hv_abs": dh}
    return v, loss_sum, extra


def _sum2(t):
    t = t.detach()
    if not t.is_floating_point():
        t = t.double()
    return torch.stack([t.sum(dtype=torch.float64), (t.double() ** 2).sum()])


def state_digest(model, opt, dos):
    """float64 (sum, sum of squares) of every LOCAL tensor of the resumable state (this rank's FSDP shards + main AdamW state, the replicated
    decoders + their AdamW state, scheduler step) -> sha256 hex (compared save vs load, per rank)."""
    vals = []
    for n, p in model.named_parameters():
        vals.append(_sum2(M._loc(p)))
        for k in sorted(opt.state.get(p, {})):
            vals.append(_sum2(M._loc(opt.state[p][k]).reshape(-1).to(p.device)))
    for nm in sorted(dos):
        for p in dos[nm].ps:
            for k in sorted(dos[nm].opt.state.get(p, {})):
                vals.append(_sum2(dos[nm].opt.state[p][k].reshape(-1).to(p.device)))
    a = torch.stack(vals).cpu().numpy()
    extra = np.asarray([float(dos[nm].micros) for nm in sorted(dos)], np.float64)
    return hashlib.sha256(a.tobytes() + extra.tobytes()).hexdigest()[:16]


def save_full(model, opt, sched, dos, step, why, plan_dg):
    """FULL RESUMABLE save on NODE-LOCAL scratch: <save_root>/step-NNNNN/rank<R>.pt (this rank's fp32 FSDP shards + its AdamW state +
    scheduler + group LRs), node.pt per node (replicated decoders fp32 + both decoder AdamW states + update counters), NODE_COMPLETE per node
    (all ranks' state digests), meta.json on rank 0's node. Keeps the last 2 complete saves per node (the older ones are removed before writing)."""
    import glob
    t0 = time.time()
    root = CFG["save_root"]
    d = f"{root}/step-{step:05d}"
    if LOCAL == 0:
        os.makedirs(root, exist_ok=True)
        comp = sorted(x for x in glob.glob(f"{root}/step-*") if os.path.isfile(f"{x}/NODE_COMPLETE") and x != d)
        for x in glob.glob(f"{root}/step-*"):
            if x != d and x not in comp[-1:]:
                shutil.rmtree(x, ignore_errors=True)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    dist.barrier()
    names = {p: n.replace("_checkpoint_wrapped_module.", "") for n, p in model.named_parameters()}
    sh = {names[p]: M._loc(p).detach().cpu() for p in names if hasattr(p, "to_local")}
    ost = {}
    for g in opt.param_groups:
        for p in g["params"]:
            if p in opt.state and opt.state[p]:
                ost[names[p]] = {k: M._loc(v).detach().cpu() for k, v in opt.state[p].items()}
    blob = {"step": int(step), "world": WORLD, "rank": RANK, "host": HOST, "plan": plan_dg, "model": sh, "opt": ost,
            "sched": sched.state_dict(), "lrs": [g["lr"] for g in opt.param_groups], "why": why, "time": now_pt()}
    torch.save(blob, f"{d}/rank{RANK}.pt.tmp"); os.replace(f"{d}/rank{RANK}.pt.tmp", f"{d}/rank{RANK}.pt")
    del blob, sh, ost
    if LOCAL == 0:
        rp = {names[p]: p.detach().cpu() for p in names if not hasattr(p, "to_local")}
        dec = {nm: {"opt": do.opt.state_dict(), "micros": int(do.micros)} for nm, do in dos.items()}
        torch.save({"rep": rp, "dec": dec, "step": int(step)}, f"{d}/node.pt.tmp"); os.replace(f"{d}/node.pt.tmp", f"{d}/node.pt")
        del rp, dec
    dg = state_digest(model, opt, dos)
    dgs = [None] * WORLD; dist.all_gather_object(dgs, dg)
    hosts = [None] * WORLD; dist.all_gather_object(hosts, HOST)
    dist.barrier()
    if LOCAL == 0:
        json.dump({"step": int(step), "digests": dgs, "hosts": hosts, "world": WORLD, "plan": plan_dg, "why": why, "time": now_pt()},
                  open(f"{d}/NODE_COMPLETE", "w"))
    if RANK == 0:
        json.dump({"format": "aloe-v2pure-resumable-v1", "step": int(step), "world": WORLD, "hosts": hosts, "digests": dgs, "plan": plan_dg,
                   "why": why, "run": CFG["run_tag"], "time": now_pt()}, open(f"{d}/meta.json", "w"), indent=1)
    dist.barrier()
    fr = shutil.disk_usage(root).free / 1e9
    fmin = torch.tensor([fr], device=next(model.parameters()).device); dist.all_reduce(fmin, op=dist.ReduceOp.MIN)
    log(f"[save] {why}: FULL resumable save step {step} -> {d} on every node ({time.time() - t0:.0f}s); digests rank0 {dgs[0]} "
        f"(all {len(set(dgs))} distinct over {WORLD} ranks); min /scratch free after {float(fmin):.0f} GB")


def find_resume():
    """latest step whose save is complete for EVERY rank on its own node (rank<R>.pt + node.pt + NODE_COMPLETE), else None."""
    import glob
    have = []
    for x in glob.glob(f"{CFG['save_root']}/step-*"):
        if all(os.path.isfile(f"{x}/{f}") for f in ("NODE_COMPLETE", f"rank{RANK}.pt", "node.pt")):
            have.append(int(os.path.basename(x)[5:]))
    allh = [None] * WORLD; dist.all_gather_object(allh, sorted(have))
    common = set(allh[0]).intersection(*[set(h) for h in allh[1:]])
    return max(common) if common else None


def load_full(model, opt, sched, dos, step, plan_dg, dev):
    d = f"{CFG['save_root']}/step-{step:05d}"
    t0 = time.time()
    blob = torch.load(f"{d}/rank{RANK}.pt", map_location="cpu", weights_only=False)
    assert (blob["step"], blob["world"], blob["rank"], blob["host"], blob["plan"]) == (step, WORLD, RANK, HOST, plan_dg), \
        ("resume save from another world / rank / host / plan", blob["step"], blob["world"], blob["rank"], blob["host"], blob["plan"])
    nd = torch.load(f"{d}/node.pt", map_location="cpu", weights_only=False)
    assert nd["step"] == step
    names = {p: n.replace("_checkpoint_wrapped_module.", "") for n, p in model.named_parameters()}
    with torch.no_grad():
        for p, n in names.items():
            src = blob["model"][n] if hasattr(p, "to_local") else nd["rep"][n]
            M._loc(p).copy_(src.to(dev))
        for g in opt.param_groups:
            for p in g["params"]:
                s = blob["opt"].get(names[p])
                if s is None:
                    continue
                stt = {}
                for k, v in s.items():
                    if k == "step":
                        stt[k] = v.to(device=dev, dtype=torch.float32)
                    else:
                        z = torch.zeros_like(p)
                        M._loc(z).copy_(v.to(dev)); stt[k] = z
                opt.state[p] = stt
    sched.load_state_dict(blob["sched"])
    for g, lr in zip(opt.param_groups, blob["lrs"]):
        g["lr"] = lr
    for nm, do in dos.items():
        do.opt.load_state_dict(nd["dec"][nm]["opt"]); do.micros = int(nd["dec"][nm]["micros"])
    del blob, nd
    dg = state_digest(model, opt, dos)
    ref = json.load(open(f"{d}/NODE_COMPLETE"))["digests"][RANK]
    ok = torch.tensor([float(dg == ref)], device=dev); dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    log(f"[resume] step {step} from {d}: state digest after load == digest at save on every rank: {bool(float(ok))} (rank0 {dg} vs {ref}); "
        f"decoder updates {[dos[k].micros for k in sorted(dos)]}; lrs {[round(g['lr'], 12) for g in opt.param_groups]}; {time.time() - t0:.0f}s")
    assert float(ok) == 1.0, "resumed state differs from the saved state"


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
    if os.environ.get("ALOE_OPT_OFFLOAD", "0") == "1":                # aloe-pose-rosie (10-27): = aloe-ft-32-samsep's optoff.OffloadAdamW (AdamW m / v in
        import optoff as OO                                           # pinned host RAM between steps, the same fused step chunked: bit-identical), the only
        opt = OO.OffloadAdamW([{"params": by[g], "lr": CFG["lr"][g], "name": g} for g in TRAIN_GROUPS if by[g]],   # way 38.5B params fit world 8
                              chunk_gb=float(os.environ.get("ALOE_OPT_OFFLOAD_CHUNK_GB", "2")), betas=(0.9, CFG["adam_beta2"]),
                              eps=1e-8, weight_decay=0.0, fused=True)
    else:
        opt = torch.optim.AdamW([{"params": by[g], "lr": CFG["lr"][g], "name": g} for g in TRAIN_GROUPS if by[g]], betas=(0.9, CFG["adam_beta2"]),
                                eps=1e-8, weight_decay=0.0, fused=True)
    lam = SCH.make_lambda(True, CFG["sched"])                     # every trained group warms up (run C: warmup_groups = all its groups)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lam for _ in opt.param_groups])
    return opt, sched, by, frozen, dec


def main():
    dev = torch.device("cuda", LOCAL)
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev, timeout=datetime.timedelta(seconds=int(E("INIT_TIMEOUT", "1800"))))
    os.makedirs(LOGDIR, exist_ok=True)
    signal.signal(signal.SIGUSR1, _usr1)
    log(f"aloe-ft {CFG['run_tag']}: world {WORLD} (lw {LW}); host {HOST}; POSE_B {int(POSE)}; cfg {json.dumps(CFG, default=str)}")
    for ref in [x for x in (os.environ.get("QQA_REF_CFG"), os.environ.get("QQA_REF_CFG2")) if x and RANK == 0]:   # mono-8020: both refs
        rc = json.load(open(ref))["cfg"]

        def flat(d, pre=""):
            o = {}
            for k_, v_ in d.items():
                if isinstance(v_, dict):
                    o.update(flat(v_, f"{pre}{k_}."))
                else:
                    o[f"{pre}{k_}"] = v_
            return o
        a_, b_ = flat(json.loads(json.dumps(rc, default=str))), flat(json.loads(json.dumps(CFG, default=str)))
        diff = [(k_, a_.get(k_, "<absent>"), b_.get(k_, "<absent>")) for k_ in sorted(set(a_) | set(b_)) if a_.get(k_, "<absent>") != b_.get(k_, "<absent>")]
        log(f"[cfgdiff] resolved config vs {ref} ({len(diff)} keys differ; every other key identical):")
        for k_, x_, y_ in diff:
            log(f"[cfgdiff]   {k_}: ref {x_!r} -> this {y_!r}")
    # ---- data (aloe plan: identical on every rank; every train row x both orientations, one row per rank per step)
    assert POSE and CFG["a_empty"], "aloe-v2pure: QQA_POSE_B=1 and QQA_A_EMPTY=1"
    PL = A.make_plan_v2(WORLD, CFG["data"])
    assert PL["steps"] == CFG["steps"], ("QQA_STEPS != the plan's steps (one epoch, 64 rows of <= 7,500 frames per step)", CFG["steps"], PL["steps"])
    for k in range(A.NST):
        A.Stores().open(k)                                        # store digests == the build's (asserted)
    my_steps = [PL["step_micros"][t][RANK] for t in range(PL["steps"])]
    ev_groups = PL["eval_micros"][RANK::WORLD]
    fallback = PL["items"][0]
    nclips = sum(len(g) for st_ in my_steps for g in st_)
    ndum = sum(1 for st_ in my_steps for g in st_ if not g)
    nconv = allsum([float(nclips), float(sum(len(g) for g in ev_groups)), float(ndum)], dev)
    assert int(nconv[0]) == PL["n_train_items"] and int(nconv[1]) == PL["n_eval_items"], nconv
    dg = A.plan_digest(PL)
    dga = [None] * WORLD; dist.all_gather_object(dga, dg)
    assert len(set(dga)) == 1, "plan differs across ranks"
    PLAN_DG = dg
    log(f"[data] aloe-v2pure plan {dg}: {PL['n_train_items']:,} train clips ({PL['train_frames']:,} frames = {PL['train_frames'] / 12.5 / 3600:.1f} h) "
        f"-> {PL['n_rows']:,} rows of <= {A.ROW_CAP} frames -> {CFG['steps']} steps x {A.ROWS_PER_STEP} rows; micros per rank per step (rounds) "
        f"min {min(PL['R'])} mean {np.mean(PL['R']):.2f} max {max(PL['R'])} (mean <= {A.MICRO_TOK} positions, max position bound {PL['max_micro_tok_ub']}); "
        f"{int(nconv[2])} dummy micros; eval {PL['n_eval_items']:,} clips ({PL['eval_frames']:,} frames) in {len(PL['eval_micros'])} packed micros; "
        f"1-frame clips skipped (train, eval) {PL['skipped_lt2']}; "
        f"data cfg {CFG['data']}")
    # aloe-v2pure-clip80 (10-18): the step composition (items in plan order, the 64-row groups, eval packing) must be 23051's exactly
    rdg = A.rows_digest(PL)
    want_rdg = os.environ.get("ALOE_ROWS_DIGEST")
    log(f"[data] rows digest {rdg} (items + 64-row step groups + eval packing; 23051's = {want_rdg}); eval micro position bound "
        f"{PL['max_eval_tok_ub']} <= {A.BUDGET}")
    assert want_rdg and rdg == want_rdg, ("step composition differs from 23051's", rdg, want_rdg)
    if G0:                                                        # pose6 G0COND: every train + eval item has a frame-0 record (FAIL LOUD otherwise)
        n_tr, inv_tr = A.g0_coverage(PL["items"])
        n_ev_, inv_ev = A.g0_coverage(PL["eval_items"])
        log(f"[g0cond] sidecar {A.G0_SIDECAR}: coverage train {n_tr:,} / {PL['n_train_items']:,}, eval {n_ev_:,} / {PL['n_eval_items']:,} items "
            f"(missing items fail loud); f = 0 (frame0_ok false, the base's rule) on train {inv_tr}, eval {inv_ev}")
    t0 = time.time()
    if A.MONO:                                                    # aloe-pose-mono-8020: realised prompt cells (data_aloe_pose.prompt_stats_mono)
        ref_cfg0 = Q.qqa_cfg()
        for nm_, its_ in (("train", PL["items"]), ("eval", PL["eval_items"])):
            s_ = allsum(list(A.prompt_stats_mono(its_, CFG["data"], ref_cfg0, RANK, WORLD)), dev)
            f_ = {k_: s_[i_] / s_[0] for i_, k_ in enumerate(A.PSTAT_MONO_KEYS)}
            assert int(s_[0]) == len(its_), (nm_, s_[0])
            log(f"[prompt] realised system blocks, {nm_} ({int(s_[0]):,} items): THIS RUN drawn: voice clip {f_['voice_drawn']:.4f}, persona "
                f"{f_['persona_drawn']:.4f} | present: clip {f_['clip']:.4f}, persona text {f_['persona']:.4f} | BOTH {f_['both']:.4f}, CLIP ONLY "
                f"{f_['clip_only']:.4f}, PERSONA ONLY {f_['persona_only']:.4f}, EMPTY {f_['empty']:.4f} || 23051's rule on the same items: both "
                f"{f_['ref_both']:.4f}, clip only {f_['ref_clip_only']:.4f}, persona only {f_['ref_persona_only']:.4f}, empty {f_['ref_empty']:.4f}")
            if RANK == 0:
                json.dump({"step": 0, "time_pt": now_pt(), "n": int(s_[0]), **{f"prompt_{nm_}/{k_}": float(v_) for k_, v_ in f_.items() if k_ != "items"}},
                          open(f"{LOGDIR}/prompt_stats_{nm_}.json", "w"), indent=1)
        log(f"[prompt] stats computed in {time.time() - t0:.0f}s")
    ps_tr = allsum(list(A.prompt_stats(PL["items"], CFG["data"], RANK, WORLD)), dev) if not A.MONO else None
    ps_ev = allsum(list(A.prompt_stats(PL["eval_items"], CFG["data"], RANK, WORLD)), dev) if not A.MONO else None
    for nm_, s_ in ((("train", ps_tr), ("eval", ps_ev)) if not A.MONO else ()):
        f_ = {k_: s_[i_] / s_[0] for i_, k_ in enumerate(A.PSTAT_KEYS)}
        assert int(s_[0]) == (PL["n_train_items"] if nm_ == "train" else PL["n_eval_items"]), (nm_, s_[0])
        log(f"[prompt] realised system-prompt fractions, {nm_} ({int(s_[0]):,} clips): THIS RUN voice clip drawn {f_['voice_drawn']:.4f}, "
            f"clip valid {f_['voice_valid']:.4f} ({f_['voice_valid'] / max(f_['voice_drawn'], 1e-9):.4f} of drawn), empty system block "
            f"{f_['empty_block']:.4f}, persona text on B {f_['persona_B']:.4f} | 23051's rule on the same clips: drawn {f_['ref_voice_drawn']:.4f}, "
            f"valid {f_['ref_voice_valid']:.4f}, B persona text available {f_['ref_persona_B_avail']:.4f}, empty block {f_['ref_empty_block']:.4f}")
        emit_pre = {f"prompt_{nm_}/{k_}": float(v_) for k_, v_ in f_.items() if k_ != "items"}
        if RANK == 0:
            json.dump({"step": 0, "time_pt": now_pt(), "n": int(s_[0]), **emit_pre}, open(f"{LOGDIR}/prompt_stats_{nm_}.json", "w"), indent=1)
    if CFG["data"]["persona_sides"] == "none" and not A.MONO:
        assert ps_tr[4] == 0 and ps_ev[4] == 0, "persona text present although persona_sides is none"
    if not A.MONO:
        log(f"[prompt] stats computed in {time.time() - t0:.0f}s")
    # ---- model
    c32, c4, c06 = M.cfg32(64), M.cfg4b(36), M.cfg06(28)
    model, _ = M.build(c32, c4, c06, dev, log, partial=False, proj_r=2048, tower_vocab=M.TOWER_VOCAB, init=False, dec_ac=False,
                       dec_mode="replicate", pose_head=CFG["pose_head_build"], decp_bb=True, s2p=S2P, **({"g0cond": True} if G0 else {}))
    assert bool(S2P) == hasattr(model.tower_p, "s_emb") and (not S2P or POSE), ("pose-s2p needs the pose path", S2P, POSE)
    assert G0 == hasattr(model.inp, "g0") == model.g0cond and (not G0 or POSE), ("G0COND: module presence != PPM_G0COND", G0)
    for n_ in NEW_KEYS:
        log(f"[pose6] {n_} -> main-AdamW group '{M.group_of(n_)}' (LR {CFG['lr'][M.group_of(n_)]:g}, clip {CFG['clip'][M.group_of(n_)]:g}; JOIN "
            f"{M.JOIN[M.group_of(n_)]}); loaded from the base (D5)" + (f"; G0COND SLOT = each item's P - 1, sidecar {A.G0_SIDECAR}" if n_ == "inp.g0.weight" else ""))
    if CFG["pose_head_build"]:
        model.pose_head_on = False                                # PPM main D36: the D32 pose head is off (frozen)
    assert hasattr(model, "pose_head") == CFG["pose_head_build"], "pose head built iff QQA_POSE_HEAD_BUILD"
    model.decp_bb_alpha = CFG["decp_bb_alpha"]                    # PPM main D33: alpha 1.0, no fade
    if BB_OFF:                                                    # pose6 LINK OFF: never computed (train and eval), frozen
        model.decp_bb_on = False
    t0 = time.time()
    meta, gb = load_init(model, CFG["init"], dev)
    log(f"[init] {CFG['init']}: PPM step {meta['step']} (run {meta.get('run_tag')}, saved {meta.get('save_time')}), world {meta['world']} shards "
        f"resharded to {WORLD}; {gb:.1f} GB read; {time.time() - t0:.0f}s; checkpoint keys NOT loaded (module not built): {meta['unloaded_keys']}"
        + (f"; pose6 new params loaded from the base: {[n_ for n_ in NEW_KEYS if n_ not in meta.get('newparam_random_fill', [])]}" if NEW_KEYS else ""))
    if CFG["dec_fast"]:
        for dec_ in ((model.dec_s, model.dec_p) if POSE else (model.dec_s,)):
            M.compile_decoder(dec_, CFG["dec_compile_mode"], whole=True, dynamic=CFG["dec_dyn"])
    rot = M.Rot(c32, dev)
    opt, sched, by, frozen, decn = make_opt(model)
    dos = {"dec_s": DecOpt(model.dec_s, CFG["dec_lr"], CFG["dec_clip"])}
    if POSE:
        dos["dec_p"] = DecOpt(model.dec_p, CFG["dec_p_lr"], CFG["dec_p_clip"])
    else:                                                         # no-pose: dec_p params are frozen (no nanostep AdamW over them)
        dec = [n for n in decn if not frozen_name(n)]
        assert all(n.replace("_checkpoint_wrapped_module.", "").startswith("dec_s.") for n in dec), dec[:4]
    n_all = len(list(model.parameters()))
    assert sum(len(v) for v in by.values()) + sum(len(d.ps) for d in dos.values()) + len(frozen) == n_all, "every parameter in exactly one optimizer or frozen"
    if POSE:
        assert sorted({n.replace("_checkpoint_wrapped_module.", "").split(".")[0] for n in frozen}) == \
            sorted((["pose_head"] if CFG["pose_head_build"] else []) + (["decp_bb"] if BB_OFF else [])), ("frozen set", frozen[:4])
        if BB_OFF:
            assert not model.decp_bb_on and sorted(n.replace("_checkpoint_wrapped_module.", "") for n in frozen if "decp_bb" in n) == \
                ["decp_bb.bias", "decp_bb.weight"], "LINK OFF: decp_bb frozen, in no optimizer, never computed"
            log("[link-off] D33 decp_bb link OFF (QQA_DECP_BB_OFF=1, = the base run's PPM_DECP_BB_OFF): decp_bb.weight / .bias loaded from the "
                "base, frozen (no optimizer), never computed (pose decoder zb=None); gradient-free asserted every step; exports record alpha 0")
    else:
        fpre = {p_ for p_ in FROZEN_PREFIX for n in frozen if n.replace("_checkpoint_wrapped_module.", "").startswith(p_)}
        assert fpre == set(FROZEN_PREFIX), ("no-pose frozen set must cover every pose part", sorted(fpre))
    fz_set = set(frozen)
    fz0 = tensor_digest(model, fz_set)
    log(f"[opt] main AdamW groups {[(pg['name'], len(pg['params']), pg['lr']) for pg in opt.param_groups]}; decoder nanostep AdamW "
        f"dec_s {len(dos['dec_s'].ps)} tensors (peak {CFG['dec_lr']:g}, clip {CFG['dec_clip']:g}), "
        + (f"dec_p {len(dos['dec_p'].ps)} tensors (peak {CFG['dec_p_lr']:g}, clip {CFG['dec_p_clip']:g}) " if POSE else "dec_p NONE (frozen) ")
        + f"(K {CFG['dec_k']}); inp.pose {'in audio' if POSE else 'FROZEN'} ({sum('inp.pose' in n for n, _ in model.named_parameters())} tensor); FROZEN {len(frozen)} tensors ({sorted({n.split('.')[0] for n in frozen})}) digest {fz0[:16]}; "
        f"schedule {SCH.describe(dict(CFG['sched'], warmup_groups=list(TRAIN_GROUPS)))}; decoder schedule = the same shape over updates "
        f"(warmup {CFG['dec_sched']['warmup']}, decay to 0 at {CFG['dec_sched']['wsd_decay_end']})")
    # ---- loaders + eval set
    from torch.utils.data import DataLoader
    W_ = CFG["dl_workers"]

    def _wi(_):
        torch.set_num_threads(1); os.environ["OMP_NUM_THREADS"] = "1"
    # ---- resume (ALOE_RESUME=auto: the newest save complete for every rank on its own node; same world / hosts / plan asserted)
    start = 0
    if CFG["resume"] == "auto":
        rs = find_resume()
        if rs is not None:
            load_full(model, opt, sched, dos, rs, PLAN_DG, dev)
            start = rs
    assert CFG["resume"] in ("auto", "never"), CFG["resume"]
    log(f"[resume] start step {start} ({'fresh from the init' if start == 0 else 'resumed'}); saves {CFG['save_root']} every "
        f"{CFG['save_every_min']:g} min (keep 2) + on STOP_AFTER_SAVE / USR1 + final")
    ld = iter(DataLoader(A.StepDS(PL["items"], my_steps, CFG["data"], RANK, fallback, start=start, singles_at=0 if CFG["losscheck"] else -1),
                         batch_size=None, shuffle=False, num_workers=W_, prefetch_factor=2 if W_ else None, persistent_workers=bool(W_),
                         pin_memory=False, worker_init_fn=_wi if W_ else None, collate_fn=lambda x: x))
    ev = A.eval_micros_v2(PL["eval_items"], ev_groups, CFG["data"], RANK)
    t_ = torch.tensor([float(len(ev))], device=dev); dist.all_reduce(t_, op=dist.ReduceOp.MAX); n_ev = int(t_)
    ev0 = ev[0] if ev else A.eval_micros_v2(PL["eval_items"], PL["eval_micros"][:1], CFG["data"], RANK)[0]   # aloe-pose-rosie: fewer eval micros
    ev += [dict(ev0, dummy=1, own=np.zeros_like(ev0["own"])) for _ in range(n_ev - len(ev))]   # same forwards on every rank (FSDP)  # than ranks
    log(f"[eval] {n_ev} eval micros per rank (dummies pad the short ranks); P mean {np.mean([int(m['P'][0]) for m in ev]):.0f}; "
        + (f"pose frames / micro {np.mean([len(m['plight']) for m in ev]):.0f}" if POSE else "no pose (POSE_B 0)"))
    jl = open(f"{LOGDIR}/metrics.jsonl", "a") if RANK == 0 else None

    def emit(d, step):
        if RANK == 0:
            jl.write(json.dumps({"step": step, "time_pt": now_pt(), **d}) + "\n"); jl.flush()

    def evaluate(step, mics=None, pre="eval"):
        model.eval()
        t0_ = time.time()
        acc = {0: None, 1: None}
        for m in (ev if mics is None else mics):
            oc = np.asarray(m["own"]).sum(0)
            v, _, _ = run_micro(model, prep(m, dev, max(oc[0], 1.0), max(oc[1], 1.0), max(oc[3], 1.0) if POSE else None), rot, dos, grad=False)
            if m.get("dummy"):
                v = [0.0] * len(v)
            sw = int(m["swap"])
            acc[sw] = v if acc[sw] is None else [x + y for x, y in zip(acc[sw], v)]
        nv = len(next(a for a in acc.values() if a is not None))
        acc = {k: (a if a is not None else [0.0] * nv) for k, a in acc.items()}
        g0, g1 = allsum(acc[0], dev), allsum(acc[1], dev)
        d = summarize([x + y for x, y in zip(g0, g1)], pre)
        if pre == "eval":
            d.update(summarize(g0, "eval_asstB")); d.update(summarize(g1, "eval_userB"))
        model.train()
        d[f"{pre}/time_s"] = time.time() - t0_
        emit(d, step)
        log(f"EVAL{'' if pre == 'eval' else ' (' + pre + ')'} step {step}: " + " | ".join(f"{k} {v:.4f}" for k, v in d.items()))
        for k_, v_ in d.items():
            assert math.isfinite(v_), (k_, v_)
        return d

    if start == 0:
        evaluate(0)
        # aloe-v2pure-clip80: ONE-OFF identity check at step 0 (the init): the same eval clips under 23051's prompt rule (its data cfg =
        # aloe_cfg without the 10-18 env) -> must reproduce 23051's step-0 eval (eval_speech/word_ce_norm 2.2348). Not repeated later.
        # aloe-pose-mono-8020: on the Aloe eval items 23051's rule = job 22317's rule -> must reproduce 22317's step-0 eval (word_ce_norm 1.9574).
        ref_cfg = Q.qqa_cfg()
        assert {k_: v_ for k_, v_ in ref_cfg.items()} == json.load(open(os.environ["QQA_REF_CFG"]))["cfg"]["data"], "ref data cfg != 23051's"
        evr = A.eval_micros_v2(PL["eval_items"], ev_groups, ref_cfg, RANK)
        evr0 = evr[0] if evr else A.eval_micros_v2(PL["eval_items"], PL["eval_micros"][:1], ref_cfg, RANK)[0]   # aloe-pose-rosie (as ev0)
        evr += [dict(evr0, dummy=1, own=np.zeros_like(evr0["own"])) for _ in range(n_ev - len(evr))]
        log(f"[eval] step-0 identity check under 23051's prompt rule: P mean {np.mean([int(m['P'][0]) for m in evr]):.0f} (this run's eval: "
            f"{np.mean([int(m['P'][0]) for m in ev]):.0f})")
        evaluate(0, evr, "eval_ref23051rule")
        del evr
    import gc
    gc.collect(); gc.freeze(); gc.disable()
    times = []
    stopped = None
    t_save = time.time()
    for step in range(start, CFG["steps"]):
        torch.cuda.synchronize(); ts = time.time()
        torch.cuda.reset_peak_memory_stats()
        tw = time.time()
        bl = next(ld)
        wait = time.time() - tw
        assert all(int(b_["step"]) == step for b_ in bl) and len(bl) == PL["R"][step], ("micro order / rounds", [b_["step"] for b_ in bl], step)
        b = bl[0]
        if step == 0 and CFG["losscheck"]:
            Pv, ex, er, K = pack_loss_check(model, b, rot, dev)
            ea = [None] * WORLD; dist.all_gather_object(ea, er)
            worst = [max(e[i] for e in ea) for i in range(len(er))]
            log(f"[pack] LOSS CHECK step 0: packed micro loss (micro token-normalised) vs token-weighted standalone conversations, max rel err over "
                f"{WORLD} ranks: text {worst[0]:.2e} cb0 {worst[1]:.2e} light {worst[2]:.2e} "
                + (f"pose_cb0 {worst[3]:.2e} pose_light {worst[4]:.2e} " if POSE else "")
                + f"(rank 0: {K} conversations, packed {[round(x, 5) for x in Pv]}, "
                f"weighted standalone {[round(x, 5) for x in ex]})")
            assert max(worst) < CFG["losscheck_tol"], ("packed loss != token-weighted standalone losses", worst)
        oc = np.sum([np.asarray(b_["own"]).sum(0) for b_ in bl], 0)
        tot = allsum([float(x) for x in oc[:5]], dev)
        bts = [prep(b_, dev, tot[0] / WORLD, tot[1] / WORLD, tot[3] / WORLD) for b_ in bl]   # tokens_step normalisers (step totals, all rounds)
        if step == start:
            log(f"[pack] tokens_step: step {step} totals over {WORLD} ranks x {len(bl)} rounds: text targets {int(tot[0])}, cb0 labels {int(tot[1])}, "
                f"light frames {int(tot[2])}, pose cb0 labels {int(tot[3])}, pose frames {int(tot[4])}; per-micro normaliser = total / {WORLD}")
        t_m = time.time()
        v, loss_v, ex_ = train_step_multi(model, bts, rot, dos)
        torch.cuda.synchronize(); t_micro = time.time() - t_m
        assert abs(ex_["dec_s_N"] - tot[2]) < 0.5, ("speech decoder frames != the step's light frames", ex_["dec_s_N"], tot[2])
        if POSE:
            assert abs(ex_["dec_p_N"] - tot[4]) < 0.5, ("pose decoder frames != the step's pose frames", ex_["dec_p_N"], tot[4])
        for n_, p_ in model.named_parameters():
            if n_ in fz_set:
                assert p_.grad is None, ("a frozen parameter got a gradient", n_)
        gn = {}
        for pg in opt.param_groups:
            g_ = torch.nn.utils.clip_grad_norm_(pg["params"], CFG["clip"][pg["name"]])
            gn[pg["name"]] = float(g_.full_tensor() if hasattr(g_, "full_tensor") else g_)
        lr_used = {pg["name"]: pg["lr"] for pg in opt.param_groups}
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); dt = time.time() - ts; times.append(dt)
        sv = allsum(list(v) + [loss_v], dev)
        d = summarize(sv[:-1], "train")
        d["train/loss"] = sv[-1] / WORLD                          # the optimised (weighted) loss
        ul = [x / WORLD for x in sv[-1 - NSTREAM:-1]]             # aloe-uncond: per-stream losses (22091's weights) and x the stream multipliers
        lt_s, lt_p = d["train_speech/light_ce"], d.get("train_speech/pose_light_ce", 0.0)
        d.update({"train/uloss_word": ul[0], "train/uloss_cb0": CFG["w_cb0"] * ul[1], "train/uloss_light": CFG["w_cb"] * lt_s,
                  "train/wloss_word": CFG["lw_word"] * ul[0], "train/wloss_cb0": CFG["lw_speech"] * CFG["w_cb0"] * ul[1],
                  "train/wloss_light": CFG["lw_speech"] * CFG["w_cb"] * lt_s})
        if POSE:
            d.update({"train/uloss_pose_cb0": CFG["w_pcb0"] * ul[2], "train/uloss_pose_light": CFG["w_pcb"] * lt_p,
                      "train/wloss_pose_cb0": CFG["lw_pose"] * CFG["w_pcb0"] * ul[2], "train/wloss_pose_light": CFG["lw_pose"] * CFG["w_pcb"] * lt_p})
        d["train/loss_unweighted"] = ul[0] + CFG["w_cb0"] * ul[1] + (CFG["w_pcb0"] * ul[2] if POSE else 0.0)
        d["train/dummies"] = float(nconv[2]) if step == CFG["steps"] - 1 else 0.0
        pk = torch.tensor([torch.cuda.max_memory_allocated() / GiB, dt, wait, t_micro], device=dev)
        dist.all_reduce(pk, op=dist.ReduceOp.MAX)
        d.update({f"train/gn_{k}": x for k, x in gn.items()})
        d.update({f"train/lr_{k}": x for k, x in lr_used.items()})
        d.update({"train/dec_lr_first": ex_["dec_s_lrs"][0], "train/dec_lr_last": ex_["dec_s_lrs"][-1], "train/step_time": dt, "train/peak_gib": float(pk[0]),
                  "train/data_wait": float(pk[2]), "train/micro_s": float(pk[3]), "train/text_targets": tot[0], "train/cb0_labels": tot[1],
                  "train/light_frames": tot[2], "train/rounds": float(len(bl)), "train/passab_loss_rel": ex_["passab_loss_rel"],
                  "train/passab_hv_abs": ex_["passab_hv_abs"]})
        if POSE:
            d.update({"train/dec_p_lr_first": ex_["dec_p_lrs"][0], "train/dec_p_lr_last": ex_["dec_p_lrs"][-1],
                      "train/pose_cb0_labels": tot[3], "train/pose_frames": tot[4]})
        for k_, v_ in d.items():
            assert math.isfinite(v_), (k_, v_, "non-finite: fail loudly")
        done = step + 1
        emit(d, done)
        eta = (CFG["steps"] - done) * float(np.mean(times[-20:]))
        log(f"step {done}/{CFG['steps']} {dt:.1f}s (micro {float(pk[3]):.1f} wait {float(pk[2]):.1f}) peak {float(pk[0]):.1f} GiB | "
            + " ".join(f"{k.split('/', 1)[1]}={v:.4f}" for k, v in d.items() if k.startswith("train_speech/") and k.endswith(("word_ce_norm", "cb0_ce", "light_ce", "pad_ce")) and "light_ce_cb" not in k)
            + " " + " ".join(f"{k.split('/')[1]}={v:.4g}" for k, v in d.items() if "/gn_" in k or "/lr_trunk" in k or "/lr_tower_p" in k or "dec_lr_last" in k or "dec_p_lr_last" in k)
            + " " + " ".join(f"{k.split('/')[1]}={v:.4f}" for k, v in d.items() if "/uloss_" in k or "/wloss_" in k or k in ("train/loss", "train/loss_unweighted"))
            + f" | ETA {eta / 60:.1f} min")
        gc.collect(1)
        if (CFG["eval_every"] and done % CFG["eval_every"] == 0) or done == CFG["steps"] or done in CFG["export_steps"]:
            evaluate(done)
        if done in CFG["export_steps"] and done < CFG["steps"]:
            export_local(model, done, f"{CFG['local_export_root']}/export-{done}")
        st_ = torch.tensor([float(STOP["flag"] or os.path.exists(f"{CFG['control_dir']}/STOP_AFTER_SAVE")),
                            float(time.time() - t_save >= 60.0 * CFG["save_every_min"])], device=dev)
        dist.all_reduce(st_, op=dist.ReduceOp.MAX)
        if float(st_[0]) > 0 and done < CFG["steps"]:
            log(f"[control] STOP_AFTER_SAVE at step {done}: FULL resumable save (node-local), then exit cleanly")
            save_full(model, opt, sched, dos, done, "stop", PLAN_DG)
            if RANK == 0:
                open(f"{CFG['control_dir']}/STOPPED", "w").write(f"{done}\n")
            stopped = done
            break
        if float(st_[1]) > 0 and done < CFG["steps"]:
            save_full(model, opt, sched, dos, done, "hourly", PLAN_DG)
            t_save = time.time()
    fz1 = tensor_digest(model, fz_set)
    same = torch.tensor([float(fz1 == fz0)], device=dev); dist.all_reduce(same, op=dist.ReduceOp.MIN)
    log(f"[frozen] {('pose_head (PPM D36 off)' if CFG['pose_head_build'] else 'nothing (pose head not built)') if POSE else 'pose parts (tower_p, dec_p, pose_head, decp_bb, inp.pose)'} bit-identical to the init on every rank: {bool(float(same))} (rank 0 {fz1[:16]})")
    assert float(same) == 1.0, "a frozen parameter changed"
    if stopped is None:
        if os.environ.get("ALOE_SKIP_FINAL_FULL", "0") == "1":        # aloe-pose-rosie (10-27): world 8 + AdamW offload: the final FULL save (masters
            log("[save] final FULL save skipped (ALOE_SKIP_FINAL_FULL=1); freeing the offloaded AdamW host buffers before the weight exports")   # + m/v)
            opt.state.clear()                                         # gathered to host on top of the pinned m/v ran the node out of RAM (job 30839);
            if hasattr(opt, "host"):                                  # the weights-only exports below are all this run needs
                opt.host.clear()
            del ld                                                    # (job 30844: the cleared buffers stayed cached -> OOM again) shut the DataLoader
            gc.collect()                                              # workers down, then hand the freed PINNED blocks back to the OS: torch's
            torch.cuda.empty_cache()                                  # CachingHostAllocator keeps them otherwise
            torch._C._host_emptyCache()
            log(f"[save] host memory after freeing: RSS {open('/proc/self/status').read().split('VmRSS:')[1].split()[0]} kB on rank {RANK}")
        else:
            save_full(model, opt, sched, dos, CFG["steps"], "final", PLAN_DG)                          # v2pure: final FULL save (node-local)
        export_local(model, CFG["steps"], f"{CFG['local_export_root']}/export-{CFG['steps']}")   # aloe: node-local copy first (never refused)
        export_weights(model, CFG["steps"], CFG["export"], CFG["export_min_free_gb"])
    log(f"done {stopped or CFG['steps']} steps; mean step {np.mean(times):.2f}s; step times {[round(x, 1) for x in times]}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
