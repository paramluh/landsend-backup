# aloe-zoom-cotrain (prepared 10-21 PT, not launched)

User, 10-20 23:35 PT: "prep a finetune which will run on all nodes which will be cotraining aloe and InternalCan/zoom-strangers-5k 1:1 per
step. When we run this on 16 nodes make 8 run the speech steps, 8 run these, nooping modules not fsdp detachable".

The run is a copy of the latest Aloe FT, job 26232 (aloe-v2vox-asr-w15973). Its trainer, env and data are reused unchanged, plus three things:
- rank groups;
- a 1:1 loss between the two groups;
- zero-weight no-op passes on the zoom ranks.

The resolved config is diffed against 26232's logged config at start: 35 keys differ, all of them run dirs, accumulation, cotrain or resume
keys (`plan/cfg_diff_26232_aloe253.txt`). Every recipe key is identical: init, adapter, LRs, clips, loss weights, data cfg, eval data, seeds
and decoder. The env diff is in `plan/env_diff_26232.txt`. Launch procedure: `LAUNCH.txt`.

## Files
| what | path |
|---|---|
| trainer | `code/train_cotrain.py` (doc at the top); `code/model_ppm.py` has one edit (the `vmul` key); `code/data_zoom.py`; `code/safety_ppm.py` (copy of PPM's save/resume, plus a test-only /dev/shm option) |
| env / sbatch | `aloe_zoom_cotrain.env`, `aloe_zoom_cotrain.sbatch` (16 nodes), `aloe_zoom_cotrain_zoom77.env` (77-step variant), `hf_local_exports.sbatch` (private HF `InternalCan/ppm-aloe-zoom-cotrain`, no deletions; can run as nice'd overlap steps of the relaunched PPM job) |
| zoom table / stats | `plan/zoom_table.npz`, `plan/zoom_stats.json` (`tools/scan_zoom.py`, job 26408) |
| config diffs vs 26232 | `plan/cfg_diff_26232_aloe253.txt`, `plan/cfg_diff_26232_zoom77.txt` (`tests/cfg_diff_dry.py`), `plan/env_diff_26232.txt` |
| tests | `tests/` (`run_test.sbatch`, `t_resume.sbatch`, `t1.env` / `t2.env` / `t3.env` / `t4.env`, `compare_resume.py`, `cpu_data_check.py`, `hf_local_dry.sbatch`, `cfg_diff_dry.py`), logs in `tests/logs/` |

## Data
**Aloe:** exactly 26232's data.
- The handover none-crossbleed bucket (s2h-m1 stores), first chunks only.
- 8,067 train rows x both orientations = 16,134 micros (~940 h).
- 162 eval rows x 2 = 324 eval micros.
- Voxtral latents of the partner's raw-gain audio on channel A; B's own voice clip on the seeded 30 % rule.
- Plan digest ba2d70729ea437ab. The startup check `[same-as-v2]` is kept.

**zoom-strangers-5k:** already on disk at `/data/home/amartyavarma/zoom-charisma/full-5k`.
- These are the files pushed to the private HF repo (push log sizes match; their sha256 is asserted at load). Nothing was downloaded.
- Text only. 4,900 train and 100 eval conversations between two strangers on a first Zoom call. 6.65 M words in train, about 36 turns per conversation.
- No audio, no word timings, no system prompt. The user opens and the assistant closes every conversation.
- Synthetic: made with MIT-licensed generators (glm-5.3-flash, mimo-v2.6-flash) and an Apache-2.0 judge (gpt-oss-120b), so it is commercially usable.

**Mapping: PPM's own chat text rows.** These are the corpus15m rows, i.e. the text half of PPM pretraining (`chat_ppm.py`), used unchanged. The zoom data has the same shape: untimed user/assistant messages.
- **Render:** qwen-1ch `render_q.build_row_q` puts both speakers on one frame timeline. It uses E28 build_row4 timing4 (within-turn gaps capped at 6, turn gap U{8..12}). There is no NEW_WORD (text rows never carry it). The plain Qwen3 chat template is asserted per row.
- **Channels:** the assistant (the friendly person) is model channel B on every row (PPM D19); the user is channel A, input only. The assistant's `<|im_end|>` is kept as B's EOB target (PPM D24).
- **Loss:** every position predicts B's next token. B text CE only, with PPM's factorised weights (word x5).
- **Packing:** packs of at most 7,500 frames, one plain-causal segment (PPM 10-11).
- **No Whisper,** and no timings were invented: the renderer is the one PPM already trained on.
- The zoom micros are byte-identical to PPM's chat-micro path on the same conversations (`tests/cpu_data_check.py`).
- Why this mapping: the speech-row and text-duplex formats both need word timings, which zoom does not have. Voxtral-on-A needs audio, which it also does not have.

| zoom (rendered) | train | eval |
|---|---|---|
| conversations | 4,900 (all renderable) | 100 |
| frames per conversation (mean / min / max) | 5,457 / 3,998 / 7,017 | 5,471 |
| total frames (= h at 12.5 Hz) | 26.7 M (594 h) | 0.55 M (12 h) |
| B targets: word / EOB / PAD | 4.08 M / 89.7 k / 22.6 M | 84 k / 1.8 k / 461 k |
| packs (1 conversation each: two never fit in 7,500) | 4,900 per epoch, each epoch its own seeded order | 100 |

## Parallel design
- **Groups:** there are 16 nodes and 128 ranks. The 8 hosts that ran 26232 (slinky-5,6,8,10,11,16,17,19) form the aloe group, because the stores are already node-local there. Its 64 ranks are 26232's logical ranks 0..63, one micro each, so the micros are bit-identical to 26232's. The other 8 hosts are the zoom group, one 7,500-cap chat pack per rank. Both groups run at the same time in every optimizer step. The split is configurable (`COTRAIN_ALOE_HOSTS`, or `COTRAIN_GROUP_MODE=ranks`). If a group is smaller than its logical world, it accumulates micros (slots). Every rank runs the same number of micros; missing slots are zero-weight dummies.
- **1:1 normalisation:** the step loss is `W_ALOE x L_aloe + W_ZOOM x L_zoom`, with both weights 1.
  - Each L is that group's own token mean over the step, normalised the way a run on that data alone would be (26232's `tokens_step` rule).
  - `L_aloe` is 26232's loss: B text CE + cb0 CE, plus the decoder's token mean through the towers.
  - `L_zoom` is B text CE over the step's zoom B targets.
  - Per micro, the normaliser is the group total / 128. FSDP2 averages over 128, so:
    - every shared tensor's gradient = `grad L_aloe + grad L_zoom`;
    - every Aloe-only tensor gets exactly `grad L_aloe` (26232's gradient).
  - The replicated 0.6B decoder keeps 26232's own update: an all-reduce of chunk losses weighted W / (31 M_j), which gives exactly the Aloe decoder mean.
  - Under Adam, sum and average (0.5 / 0.5) differ only through clipping (decision 3).
- **No-op modules (never detached from FSDP):** a zoom micro is speech-shaped, so every FSDP unit runs on zoom ranks in the same order as on aloe ranks.
  - `inp` gets depth = -1 everywhere. The code tables output exactly 0, so the trunk input is bit-identical to PPM's chat micro, and `inp.aud` gets an exactly-zero gradient.
  - `vadapter` and `vgain` run on a zero Voxtral latent (one frame), added x 0.0 (`vmul`).
  - The backbone and text head run the chat micro for real.
  - `tower_s` runs on the micro's own hn with every cb0 label at -100 and weight 0. Its output gets a zero root gradient, so the autograd graph is the same as an Aloe micro's. This matters because FSDP2's reduce-scatter only includes tensors that have a grad, and post-backward hooks fire only on inputs that require grad.
  - The replicated decoder is not an FSDP unit and is not run. Its one per-step all-reduce gets an all-zero buffer, then every rank takes the same AdamW step, so the replicas stay identical.
  - Evals, the s calibration and the step-0 loss check keep the same forward and collective count on both groups.

## Tests (reduced depth 2,2,2 from the real w-15973 + w-1441 init, real Aloe and zoom micros)
These ran on the only spare GPUs: slinky-4 GPUs 6-7, which are throttled. slinky-7/15/18 were in use by the text-B Aloe FT, and nothing was preempted.
- **T1** (`tests/logs/cotrain-t1-26429.log`): 2 GPUs, 1 aloe rank x 2 slots and 1 zoom rank + 1 pad slot.
  - gradcheck (1): with only zoom micros live, every grad of tower_s, vadapter, vgain and inp.aud is exactly 0, and so is the reduced decoder buffer; no Aloe-only tensor lacks a grad tensor.
  - gradcheck (2): grad(both) vs grad(aloe only) + grad(zoom only) has relative L2 error 9.9e-5. The worst tensor is a k_norm weight at 1.1e-3, from FA3 / embedding backward atomics.
  - gradcheck (3): group losses are identical across passes.
  - The step-0 loss check is exact (0 error); 3 steps and evals ran with no hang.
- **T2** (resume on slinky-4): the save was skipped by design because the node's /scratch (57 GB) and /dev/shm (16 GB) were too small for a 61 GB state; the run logged it loudly and kept going.
- **T3** (`tests/logs/t3-26459.log`): 2 nodes, slinky-4 GPUs 6-7 = 2 aloe ranks and slinky-18 GPUs 6-7 = 2 zoom ranks. Those were not demo GPUs; the job was submitted from another session with this directory's `tests/t_resume.sbatch`.
  - gradcheck, one micro per rank (the production structure): (1) exact 0 / 0 / 0; (2) relative L2 error 9.1e-5, against a nondeterminism floor of 9.4e-5 (the same pass run twice); (3) group losses identical across passes.
  - Pad-slot case (zoom logical world 4: 2 zoom slots, and each aloe rank's 2nd slot is a template dummy): gradcheck (1) exact zeros; (2) 1.0e-4 against a floor of 9.2e-5.
  - Step-0 loss check exact; 6 steps; evals; a node-local export at world 4 (10.1 GB of bf16 shards).
  - Its resume leg could not save: slinky-18's /scratch has 12 GB free (filled by the text-B FT's exports).
- **T4 resume** (`tests/logs/t4-26463.log`): slinky-4, 1 aloe + 1 zoom rank. The big embedding / lm_head / code tables (3.57 B of 4.88 B params) were frozen in this test only, so the state fits the node's 57 GB.
  - A: 6 steps uninterrupted. B: stopped after step 3 and wrote the node-local save (57 s). C: resumed from B's save.
  - **PASS.** The per-rank digests of weights, both AdamWs, the decoder and the scheduler equal the saved ones.
  - C's EVAL at step 3 is bit-identical to B's (50 / 50 values). Data counts and LRs at steps 4-6 are identical to A's.
  - Losses at steps 4-6 are within 1.9e-4 relative of A's. Pure backward nondeterminism already gives 4.7e-5 at steps 1-3.
- **Other checks:**
  - `tests/cpu_data_check.py`: the zoom micro is identical to PPM's chat micro, epochs are fresh permutations, and the zoom_epoch dummies are correct.
  - HF backup dry runs pass (`tests/hf_local_dry.sbatch`: no deletions by default, the PPM-overlap mode, `HF_BACKUP.json`).
  - All test files on /scratch were deleted afterwards.
- **The gn_* identity across tests:** with the same 2 Aloe logical ranks, the step-1 Aloe-only gradient norms (gn_tower_s 40.15, gn_audio 14.6, gn_vadapter 5.182, gn_vgain 0.1232) are the same in T1, T2 and T3 under three different zoom setups. That is the zero-contribution property seen directly.

## Step time (16 nodes) and duration
- **Step time:** about 7.5 s per step (range 7-8.5).
  - The aloe ranks bound the step. 26232's Aloe micro took 6.6 s on 8 nodes: micro 6.6 + Voxtral 0.3 + wait 0.2, for a 7.46 s mean step. PPM's 16-node speech micro is 7.2 s (`steptime/components.json`, K 8 on 100 % of frames, heavier than Aloe's K 1 on 10 %).
  - The per-rank work on aloe ranks is unchanged. Per-rank FSDP traffic is the same, and the optimizer shard is halved.
  - The zoom ranks are lighter: a PPM text micro is 4.3 s, plus about 0.5 s for the no-op tower. They are paced by the collectives.
- **253 steps (default):** about 40 min. That is 32 min of steps, 16 evals x about 14 s, 6 node-local exports, and about 3.5 min of startup.
- **77 steps:** about 17 min.
- **Not included:** the PPM pause (save, about 2 min) and the PPM relaunch afterwards (about 10 min).

## Decisions for the user (defaults set; each one is a recipe call)
1. **Init:** PPM w-15973 + ASR adapter w-1441. This is 26232's pair; the adapter was trained on w-15973's frozen backbone. The main job is now at step ~16.9k, but nothing newer is exported to /data, and no adapter has been trained for it.
2. **Length:** 253 steps (one Aloe epoch; zoom seen 3.3x, about 40 min) or 77 steps (one zoom epoch; Aloe sees 30 % of its epoch, about 17 min). The knob is `COTRAIN_STEPS_MODE`; the 77-step block is `tests/zoom_epoch.env`.
3. **1:1:** the sum of the two token means (default; the Aloe term is exactly 26232's gradient) or their average (`COTRAIN_W_*` = 0.5). 26232's trunk gradient norm (0.5-1.6) already exceeds its 0.5 clip, so the sum clips the trunk harder.
4. **Prompt rule:** the Aloe side keeps 26232's rule (B voice clip 30 %, no persona), and zoom rows have no prompt, as PPM chat rows have none. The 10-18 rule ("80 % clip / 20 % persona for all future runs") would change the Aloe micros. Zoom's character cards could serve as the 20 % persona.
