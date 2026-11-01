# aloe-nextvox-6242: Aloe fine-tune of the 14B next-arch w-6242 with Voxtral latents in place of the user text stream

User (10-18): "finetune this on aloe using voxtral latents instead of the user text stream". This is the 14B counterpart of PPM's
Voxtral-on-A Aloe FT, the run the live demo serves.

## Reference run copied
- **aloe-v2vox-raw, job 23081**: "Aloe Voxtral v2 raw-A", export-253. The live demo (aloe-m1vox-live/LOADED.json) serves
  /data/home/amartyavarma/ppm/aloe-v2vox-raw/export-253.
- It is aloe-v2vox-ft (job 22420) with raw channel-A gain into Voxtral. 22420 is aloe-m1vox-ft (22335) with the aloe-v2 changes (22383).
- Its logged resolved cfg is in `plan/ref_cfg_v2voxraw_23081.json`. At start, the trainer diffs this run's resolved cfg against it and aborts on any
  key outside the allowed set: init, run dirs, ref_cfg, arch, accum, vadapter.init / sha_expect, vox_a_scale_init, asr_adapter.
  - CPU check 16:28 PT: 34 keys differ, 0 outside the allowed set.

Copied exactly:
- **Data:** data_aloe_v2vox, byte-identical. Stage-2 handover "none" bucket stores m1n0..3 (s2h-m1), first chunks only: 8,067 train rows x 2 orientations
  = 16,134 micros. Eval: 162 rows x 2 = 324 micros.
  - Plan digest ba2d70729ea437ab, seed 20261011, 1 row per rank per step on 64 ranks.
  - System prompt with no text. B voice clip on the seeded 30 % rule. light_frac 0.1 in training, 1.0 in eval.
  - The trainer re-checks the same-as-v2 plan / items / B channel at start.
- **Channel A:** no text (PAD), no Mimi, present 0. A's audio is the partner's WLT FLAC at RAW gain (VOX_A_GAIN=raw, RAWGAIN.json), from the
  shared /data FLAC copy, through the frozen Voxtral Realtime encoder.
- **Schedule:** 253 steps, warmup 8, linear decay to 0 at 253. tokens_step loss normalisation.
- **Losses:** factorised text CE with word x5, cb0 1, light 1. Decoder K 1 on about 10 % of B's frames (light-ratio band [0.09, 0.11]).
- **Eval and exports:** eval at step 0, every 25 steps, at the export steps and at 253. Exports at 42, 84, 126, 169, 211 (node-local),
  plus 253 (node-local and a consolidated /data DCP).
- **Adapter:** trained, lr 5e-4, clip 1.0.
- **Scale s:** s = exp(theta), its own group 'vgain' (lr 5e-4, clip 1.0), fp32. Recalibrated at start by the recipe's rule: emb_A word-row RMS /
  raw adapter per-frame RMS mean over 1 train row per rank.
- **Decoder:** compile_decoder on.
- **Precision and control:** bf16 only. STOP_AFTER_SAVE.

## What differs (each forced by the 14B architecture)
| item | PPM reference (23081) | this run | reason |
|---|---|---|---|
| model | PPM main 32B + 4B tower + 0.6B decoder (local-11488) | 14B next-arch w-6242 (model_ilp.ILP verbatim: Qwen3-14B 40 layers, interleaved [user_text, asst_text, asst_speech], cb0 head, 14-layer decoder, read v2, RoPE / 3) | the task |
| where A's latent enters | summed into the one position of frame p (channel A: present 0 + s x vadapter) | the USER_TEXT slot 3p of A's row frames (P..P+n-1) = s x vadapter, emb_A masked there (mA 0); emb_A(PAD) on the prefix / CD-tail frames | the 14B has one slot per stream; this is exactly where the 14B ASR adapter was trained to enter (asr14b: `mA emb_A + vin`, mA = 0). An absent user is emb_A(PAD), never a zero vector, in the 14B (zero-history NaN rule). |
| adapter init | PPM ASR adapter asrppm-nonorm step 1972 (trained into PPM channel A) | 14B ASR adapter asr14b final save w-480 (trained into this 14B's user_text slot; sha b6ca8779f8d0b1bd; HF InternalCan/asr-adapter-14b/asr14b/w-480; local copy adapter-asr14b-w480) | the PPM adapter was trained for PPM's backbone; user decision: the 14B adapter at its final save (480) |
| VOX_A_SCALE_INIT (anchor for the calibration sanity band 0.25-4x) | 0.0030930437 (PPM calibration) | 0.00085735058 = 14B emb_A word-row RMS 0.018880 / asr14b w-480 eval per-frame adapter RMS 22.02133 | anchor only; s itself is recalibrated at start by the same rule |
| frozen | nothing (inp.aud.0 had zero gradient: A has no Mimi) | inp.aud.0 (user audio table, no input slot in the interleaved layout) | = aloe-next-6242 |
| cb0 module LR | 4B tower (tower_s) 1.5e-5 | cb0_head in group tower_s, 1.5e-5 | the cb0 head replaces the tower's cb0 role (the 14B pretraining trains it in tower_s) |
| ranks | 64 logical ranks on 24 GPUs (3 micros per GPU per step) | 64 physical ranks (8 nodes), 1 micro per GPU = 22420's layout | same rows per step; the recipe's accumulation code reduces to one slot |
| cpus/task | 5 | 7 | engineering only |
| SAC | none | next.env (attn all layers, attn_mm 0..9) | same maths (= aloe-next-6242) |

## Per-module peak LR mapping (recipe value -> 14B module)
| 14B module | group | PPM recipe counterpart | peak LR | clip |
|---|---|---|---|---|
| inp.emb.0/1, layers.* (40), head.norm, head.lm_head | trunk | trunk (32B backbone, embeddings, lm_head) | 3e-6 | 0.5 |
| head.spec_out / nw_out / eob_out | spec | spec | 6e-5 | 1.0 |
| head.proj.* | proj | proj | 3e-5 | 1.0 |
| inp.aud.1 (asst_speech slot) | audio | audio (inp.aud) | 6e-5 | 1.0 |
| cb0_head (5120 -> 2048) | tower_s | 4B speech tower (cb0 role) | 1.5e-5 | 1.0 |
| dec_s.* (14 layers + 5120-wide link) | decoder nanostep AdamW | 0.6B light decoder | 1.5e-5 | 1.0 |
| vadapter.* (6 tensors, 62.9M) | vadapter | vadapter | 5e-4 | 1.0 |
| vgain.theta (s = exp theta) | vgain | vgain | 5e-4 | 1.0 |
| inp.aud.0 | frozen | (zero-gradient in the reference) | - | - |
Every group has warmup 8 and linear decay to 0 at 253. AdamW betas 0.9 / 0.95, eps 1e-8, wd 0. The decoder uses the same shape in update units (K 1).

## Checks in the real run
- **PARITY** (every rank, before the step-0 eval): this forward with the adapter output x 0 and emb_A kept, vs model_ilp.ILP.forward itself on the
  rank's first eval micro. Text loss / stats, cb0 loss / stats and the decoder conditioning must be bit-identical (asserted; `runs/<run>/parity.json`).
  It also logs B normalised word CE with no user input vs with the latents.
- **DATA CHECK** (step 1, every rank): the user lane has 0 word ids, and the user slots on A's row frames equal s x adapter output exactly
  (asserted; `data_check.json`).
- Also checked: il_check on every micro, with A's lane PAD everywhere; the step-0 packed-loss check; the light-ratio band.
- The frozen inp.aud.0 is bit-identical at the end.
- CPU pre-check (`tools/cpu_check.py`, 16:28 PT): cfg diff ok, VOX grouping ok, real micros: user lane non-PAD ids 0, latent frames = T - P - 2.

## Files
- `aloe_nextvox.env`: the reference env verbatim plus the changes above.
- `aloe_nextvox.sbatch`: the reference sbatch (pre-flight: stores, FLAC symlink and ledgers, export root, GPU clocks), 8 nodes.
- `code/`: the reference code, plus next-arch model_ppm / model_ilp / rope_frame (verbatim) and model_ilp_vox.py (ILPVox).
  - The trainer is `train_aloe_nextvox.py` = train_aloe_v2vox_raw.py plus the docstring changes.
- Stop: `touch control/STOP_AFTER_SAVE` (weights-only export, then exit). Never a bare scancel.
- Nodes: slinky-9, 10, 11, 12, 13, 14, 16, 19, drained from the TTS via `si20k-tts/control/DRAIN_FOR_FT.14bvox`. Delete that file when the FT ends.
- Exports go to private HF only (InternalCan/aloe-nextvox-6242), never GCS.

## Log
- 10-18 16:30 PT: drain file written. Job 24098 submitted, pending on the 8 nodes.
- 10-18 16:47 PT: job 24098 aborted in pre-flight: slinky-9 and slinky-12 had REAL /scratch/amartyavarma/aloe-m1vox-ft/flac dirs (complete
  22420-era stagings of logical nodes 6 / 7, ledgers verified). These were moved aside to `flac.aside-node6-1018` / `flac.aside-node7-1018`
  (not deleted, 67 GB each) and replaced by the symlink to the shared copy.
- 10-18 16:51 / 16:53 PT: jobs 24185 and 24191 crashed at step 1 in the data-check LOG code only, not in the maths:
  (1) "item" arrives as a JSON string; (2) FSDP2's root forward passes a copy of the info dict, so ILPVox.forward now records the check
  on the module. Both are fixed.
- **10-18 16:54 PT: job 24193 RUNNING** on slinky-9, 10, 11, 12, 13, 14, 16, 19. Run dir runs/aloe-nextvox-6242-j24193.
  - cfg diff vs j23081: 34 keys, 0 outside the allowed set.
  - PARITY bit-identical on 64/64 ranks (max |d| 0). On the parity micros, B normalised word CE is 2.0706 for the base 14B with no user input
    vs 2.2084 with the latents.
  - s calibrated to 0.0030047 (raw adapter per-frame RMS 6.2835, 14B emb_A word-row RMS 0.018880). PPM's was 0.0031081.
  - **Step-0 eval B normalised word CE 2.1949. PPM reference j23081 step 0: 2.0372.**
  - DATA CHECK OK: user lane word ids 0 on 64 ranks; the user slots equal s x adapter output exactly on 388,502 frames; emb_A(PAD) on the 609
    prefix / CD-tail frames.
  - Step-0 loss check: rel err 0. Steps 1-5: 6.1-6.4 s/step, peak 60.7 GiB.
- Export watcher: `tools/watch.sbatch`, job 24194, CPU on a non-training node. After each node-local export-N it copies and verifies
  /data/.../exports/export-N, then runs the private-HF upload (`tools/hf_up.sbatch`). On the verified final export it does three things
  immediately:
  - deletes si20k-tts/control/DRAIN_FOR_FT.14bvox;
  - writes FINAL_EXPORT_READY (containing the path) and a README line;
  - starts the HF upload of export-253-dcp.
  The FT job never waits on uploads.
- Final export path (when complete): **/data/home/amartyavarma/ppm/aloe-nextvox-6242/export-253** (consolidated bf16 DCP, every tensor incl. vadapter.* and
  vgain.theta). The demo agent waits for `FINAL_EXPORT_READY`.
- 10-18 17:34:27 PT: FINAL EXPORT (consolidated bf16 DCP, step 253, verified: 64 shard files, none empty, train_cfg step 253, 622 tensors incl. vadapter (6 weights + target_rms buffer), vgain.theta, cb0_head): /data/home/amartyavarma/ppm/aloe-nextvox-6242/export-253 -> FINAL_EXPORT_READY written; DRAIN_FOR_FT.14bvox deleted. (The watcher's check expected 6 vadapter keys and missed the buffer; finalised by hand.)
