# aloe-next-6242: Aloe fine-tune of the 14B next-arch at w-6242

Purpose: compare two architectures fine-tuned on Aloe from pretraining checkpoints at EQUAL pretraining loss.
- This run: 14B next-arch w-6242, pretraining eval speech word CE norm ~1.9986 (PT step 6164).
- Partner run: aloe-v2init-8869 (PPM main w-8869, ~1.999).

Both runs use the same schedule, data, split, seeds, order and loss weights. Only the model differs.

## What it is
- Recipe: aloe-v2init-12529 / -10351 / -8869 (the aloe-v2 recipe, job 22383), unchanged.
  - Data: data_aloe_v2. Stage2-handover stores m1n0..3, first chunks only, 8,067 train rows x 2 orientations = 16,134 micros. Eval: 162 rows x 2 = 324 micros.
  - Plan digest ba2d70729ea437ab, seed 20261011, 1 row / rank / step on 64 ranks.
  - Schedule: 253 steps, warmup 8, linear decay to 0. tokens_step loss normalisation.
  - Loss weights: factorised text CE with word x5, cb0 1, light 1.
  - Decoder: K 1 on ~10 % of B's frames.
  - Eval: step 0, every 25 steps, at each export step and at 253.
  - Exports: 42, 84, 126, 169, 211 and 253 (node-local), plus the final consolidated DCP on /data.
  - Control: STOP_AFTER_SAVE.
  - bf16 only.
  - Orientation: both orientations as in the recipe (eval reports asstB and userB). The recipe does not swap randomly.
- Model: the 14B next-arch, using the pretraining's own code copied verbatim (model_ilp.py, model_ppm.py, rope_frame.py from ppm/next-arch/code).
  - Qwen3-14B backbone, 40 layers.
  - Interleaved layout [user_text_p, asst_text_p, asst_speech_p]; the speech slot holds B's codes at CD 2. The partner (A) enters as words only.
  - cb0 head on the backbone (no 4B tower). 14-layer light decoder.
  - Read positions v2. Backbone RoPE / 3. ILP_HIST=0.
  - SAC = next.env (attn on all layers, attn_mm on layers 0..9).
  - train_aloe_next.py asserts that every architecture knob equals next.env.
- Rows: the recipe's own PPM-native micros, rebuilt into the interleaved layout inside the model. This is the path aloe_tf_eval used for the 14B, and train_ilp's il_check runs on every micro.
- Init: /data/home/amartyavarma/ppm/next-arch/ckpt-weka/ppm-next/w-6242 (consolidated bf16 DCP, every tensor). AdamW starts fresh, as in the recipe.
- Config check: the resolved cfg is diffed at start against aloe-v2init-12529's (ref_cfg_12529.json). Only init, run dirs, ref_cfg and the `arch` block may differ, otherwise the run aborts.

## Per-module peak LR mapping (recipe value -> matching 14B module)
| 14B module (parameters) | optimizer group | PPM recipe counterpart | peak LR | clip |
|---|---|---|---|---|
| backbone: inp.emb.0/1, layers.*, head.norm, head.lm_head | trunk | trunk (32B backbone, embeddings, lm_head) | 3e-6 | 0.5 |
| text-head extra rows: head.spec_out, head.nw_out, head.eob_out | spec | spec | 6e-5 | 1.0 |
| text-head projection head.proj.* | proj | proj | 3e-5 | 1.0 |
| B audio input table inp.aud.1 (asst_speech slot) | audio | audio (inp.aud) | 6e-5 | 1.0 |
| cb0_head (Linear 5120 -> 2048 on the backbone) | tower_s | **no direct counterpart**: uses the 4B speech tower's LR (QQA_LR_TOWER) | 1.5e-5 | 1.0 |
| light decoder dec_s.* (14 layers, its backbone link dec_s.proj, code tables / heads) | decoder nanostep AdamW | 0.6B light decoder dec_s (QQA_DEC_LR) | 1.5e-5 | 1.0 |
| inp.aud.0 (A audio table) | frozen | (no input slot in the interleaved layout) | - | - |

- Why the cb0 head gets the tower LR: in PPM the 4B speech tower (group tower_s) produces the cb0 logits. In the 14B the cb0 head replaces that role.
  - The 14B's own pretraining already trains cb0_head in the tower_s group (model_ilp.group_of_next: same LR, clip and warmup as the old tower).
  - So the recipe's tower value (1.5e-5, clip 1.0) is the one that maps. Its warmup and decay match every other group.
- Every other value (betas 0.9/0.95, eps 1e-8, wd 0, warmup 8, linear decay to 0 at 253, decoder schedule in update units with K 1) is the recipe's.

## Files
- aloe_next.env: the recipe env (= aloe-v2init-8869/aloe_v2init.env with only the run names / paths changed) plus the next.env arch block.
- aloe_next.sbatch: store pre-flight (stage_check.sh) and GPU clock check, then run_train.sh.
- run_train.sh: init path, port 29871, export guard, then srun of code/train_aloe_next.py.
- code/: the recipe's code (data_aloe_v2 / data_qqa / data_ppm / lib / vendor, identical) plus next-arch model_ppm.py, model_ilp.py, rope_frame.py.
  The trainer is train_aloe_next.py (= train_aloe_v2.py plus the changes listed in its docstring).
- ref_cfg_12529.json: aloe-v2init-12529's resolved cfg (from its job log).

## Stop / resume
- Save-and-exit: `touch /data/home/amartyavarma/ppm/aloe-next-6242/control/STOP_AFTER_SAVE` or `scancel -b -s USR1 <job>`. The run writes a weights-only export, then exits. Never use a bare scancel.
- Exports: /scratch/amartyavarma/aloe-next-6242/exports/export-N on every node, copied to /data/home/amartyavarma/ppm/aloe-next-6242/exports/export-N. The final consolidated DCP goes to /data/home/amartyavarma/ppm/aloe-next-6242/export-253. HF backup is private (HF only, never GCS).
