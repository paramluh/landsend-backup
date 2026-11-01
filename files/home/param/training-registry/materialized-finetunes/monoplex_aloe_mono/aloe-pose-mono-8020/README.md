# aloe-pose-mono-8020: PPM 32B MONOPLEX pose FT on the Aloe handover pose rows, 80% voice clip / 20% persona

The user asked on 10-18 to "ft the pose with the proper persona scheme" in "monoplex - using the same rows as the aloe but the pose equivalent".

- **Recipe:** job 23051's (`ppm/aloe-v2pure-ft`, v2pure monoplex). Channel A is empty. Channel B is one speaker: speech, words and pose.
- **Init:** PPM local-11488.
- **Data:** the Aloe handover pose rows of job 22091 (`ppm/aloe-pose-ft`). Every qualifying track (pose ok, no crossbleed into it) becomes one monoplex item.
- **Prompt:** the new 10-18 rule. Voice clip with p 0.8 and persona text with p 0.2, as two independent draws per item.
- **Code:** `code/` is a copy of job 24198's (`aloe-v2pure-clip80`). The changes are env-gated (`ALOE_DATA=aloe-mono`, `ALOE_PERSONA_RULE`, `ALOE_DRAW_KEY`). With that env unset, the code reproduces 23051 and 24198 exactly. This was checked by `tests/plan_mono_check.py ref`: 23051's plan digest `bfdf5ca5beab72db` and 24198's rows digest `e606adf327cb0f9d` both reproduce.

## Data (`tests/plan_mono_check.json`)
- **Stores:** `ppm/aloe-pose-ft/stores` p0..p7, read in place and read only. There are no downloads.
- **Items:** 17,527 train rows give **27,517 train items**: 3,033 h, 136.5M frames, median 4,630 frames, 37% are full 7,500-frame rows. 345 held-out rows give **531 eval items**. These are exactly 22091's and 22317's qualifying B tracks and eval tracks.
  - A conversation whose two tracks both qualify gives two items, each with that speaker alone.
- **Rows:** as in 23051, the permuted items (seed 20261011) are cut in order into rows of at most 7,500 frames and at most 9,502 positions (the position bound counts each item's own system block). That gives **22,825 rows**, with 1.21 items per row on average.
- **Steps:** at 64 rows per step, one epoch is **357 steps**. 23051 did 200 steps over its data.
  - Warmup = 23051's rule round(10 x 357 / 306) = **12**.
  - LR decays linearly to exactly 0 at step 357. The decoders follow the same shape x K 8.
  - Eval every 25 steps. Exports at round(357 x k / 6) = **60, 119, 178, 238, 298**, all node-local. The final step 357 goes node-local and also to a consolidated `export-357` on /data.
- **Micro layout:** `row_per_rank` (cfg key `micro_layout`). Each of the 64 ranks forwards one row per step, so each rank gets exactly one micro of at most 9,502 positions. This is 22317's layout.
  - 23051's LPT layout (bins of mean 6,000 positions) would leave 40% of all micros as dummies on these long items (18,058 empty bins over 357 steps x 2 rounds x 64). That is roughly 2x the compute for the same update.
  - The layout does not change the maths. The tokens_step normalisers and the joint decoder nanosteps are unchanged, the same as 24198's 4 -> 5 round change.
  - Only the last step has spare ranks: 23 dummy micros with zero weight.
- **Eval packing:** as in 23051, eval items are packed per swap into micros of at most 2,048 frames; longer items sit alone. The position bound used here holds under any prompt rule (P <= sp_cap), so the same micros serve the step-0 check under 23051's rule. This gives 529 micros.

## Prompt rule (the new standard)
Each item gets two independent seeded draws, keyed on `row_key|<B's data side>`, so the two tracks of a conversation draw separately. seed = 20261011.
- **Voice clip:** drawn iff `h64("aloe-pose-mono-8020-voice|seed|key") % 10 < 8`. The clip itself comes from the existing `data_ppm.voice_clips` rule: B's own clip from a later chunk in the second half of the same conversation.
- **Persona text:** drawn iff `h64("aloe-pose-mono-8020-persona|seed|key") % 10 < 2`. The text is B's own persona from the Aloe persona sidecar, built by the existing persona-text builder (`persona_sides` B: only B's block, since channel A is empty).

**Realised fractions** (pre-launch check over every item, built with the micro's own prefix builder; the trainer logs the same at launch):

| | voice drawn | persona drawn | clip present | persona present | both | clip only | persona only | empty |
|---|---|---|---|---|---|---|---|---|
| train (27,517) | 0.8007 | 0.2009 | 0.2818 | 0.2009 | 0.0584 | 0.2234 | 0.1425 | 0.5757 |
| eval (531) | 0.8192 | 0.1959 | 0.2938 | 0.1959 | 0.0640 | 0.2298 | 0.1318 | 0.5744 |
| 23051's rule, train | 0.333 | 1 | 0.1163 | 0.9999 | 0.1163 | 0 | 0.8836 | 0.0001 |

**Caveat: a clip is present on only 28% of items, not 80%.** The existing clip rule only takes a clip from a later chunk in the second half of the same conversation. Most Aloe tracks are the last or only chunk of their conversation, so only 35% of drawn items find a clip. On release-v2 (24198) the figure is 93%.

Reaching an 80% realised clip rate needs another clip source, such as an earlier chunk, the same chunk, or another conversation of the same speaker. That would be a recipe change and was not made without the user's OK.

## Resolved-config diff (`[cfgdiff]`, trainer log)
From the job 24272 log, 10-18 17:28 PT. Every other key is identical. The keys group as follows.

| | vs 23051 (30 keys) | vs 24198 clip80 (27 keys) |
|---|---|---|
| DATA source / rows | `stores` (aloe-v2pure-ft/stores -> aloe-pose-ft/stores), `data_src` aloe-mono, `micro_layout` row_per_rank | same |
| step count it gives | `steps` 200 -> 357; `warmup` / `sched.*` 7 / 200 -> 12 / 357; `dec_sched.*` 56 / 1600 -> 96 / 2856; `export_steps` [33 .. 167] -> [60, 119, 178, 238, 298] | same |
| PROMPT RULE | `data.persona_sides` both -> B, `persona_rule` p10, `persona_p10` 2, `persona_tag`, `draw_key` item, `voice_rule` p10, `voice_p10` 8, `voice_tag`, `voice_mod` 3 -> absent | `persona_sides` none -> B, `persona_rule` / `persona_p10` / `persona_tag` / `draw_key` added, `voice_tag` |
| names / paths | control_dir, export (export-357), local_export_root, log_root, plan_dir, run_tag, save_root | same |

`micro_layout` is the only non-obvious key. It is part of the data change: the micro layout follows the long Aloe items (see Data) and leaves the maths unchanged.

## Checks in the run
- **Init identity:** under 23051's prompt rule (persona on every item, voice clip 1/3), these eval items are exactly job 22317's eval with 22317's prompt rule. The one-off `eval_ref23051rule` at step 0 must reproduce **22317's step-0 eval_speech/word_ce_norm 1.9574**.
- **Step 0:** packed vs standalone loss check (`[pack] LOSS CHECK`, tolerance 2e-3). The rows digest is asserted against the pre-launch plan (`3681cc26ea97cad7`). The realised prompt cells are logged (`[prompt]`).

## Run
- **Job 24229** (17:26 PT) aborted in preflight. /scratch on slinky-9/12/16/19 was below the run script's 237 GB need.
  - Freed space by deleting only node-local export copies that are fully HF-verified and have a final export on /data: aloe-v2init-8869/exports (7 nodes), aloe-next-6242/exports and hfb/xet-cache (slinky-9).
  - Every node now has 249-316 GB free. List: `/data/home/amartyavarma/scratch-inspect/posemono-cleanup-1018/DELETED.log`.
- **Job 24272** launched 17:28 PT on slinky-9..14, 16, 19 (world 64). RUN_TAG stays `ppm-aloe-pose-mono-8020-j24229`.
- **Init** (step 11488, resharded 128 -> 64): `pose_head` is not loaded (not built, as in 23051).
- **Prompt cells** logged: they match the pre-launch table exactly.
- **Step-0 eval** (normalised word CE):
  - this run's prompt rule: **2.0112**;
  - under 23051's rule: **1.9574**, which is exactly 22317's step-0 value (1.9574), so the identity check passes.
- **Packing check:** packed vs standalone loss max rel err 6.0e-05 text, 7.8e-05 cb0 (tolerance 2e-3).
- **Speed:** ~12.0 s/step (step 1: 52 s with compile). Peak 45.5 GiB per GPU.
- **ETA:** ~18:55 PT for 357 steps, including evals, exports and saves.

## Nodes
- 8 nodes: slinky-9, 10, 11, 12, 13, 14, 16, 19 (world 64). These are the 14B Voxtral FT's nodes (job 24193).
- The run is submitted with `--dependency=afterany:24193`, and the sbatch gates on `READY`.
- The nodes are held through `/data/home/amartyavarma/data/si20k-tts/control/DRAIN_FOR_FT.posemono`.
- `release_nodes.sh` runs as job 24241 (afterany). It deletes ONLY that file:
  - at once if the run finished 357 steps;
  - otherwise after 60 min, unless a new posemono8020 job has been queued by then.
- The run script aborts on slinky-0/1/2/4/7/15/18.

## Stop, resume
- **Stop** (full save, clean exit): `touch /data/home/amartyavarma/ppm/aloe-pose-mono-8020/control/STOP_AFTER_SAVE`, or `scancel -b -s USR1 <job>`.
- **Resume:** `sbatch /data/home/amartyavarma/ppm/aloe-pose-mono-8020/aloe_mono8020.sbatch`. It needs the same 8 nodes. ALOE_RESUME=auto, and RUN_TAG.txt keeps one metrics.jsonl.
- **Saves:** `/scratch/amartyavarma/aloe-pose-mono-8020/saves/step-NNNNN/` on each node, hourly, keeping the last 2.
- **Exports:** `/scratch/amartyavarma/aloe-pose-mono-8020/exports/export-N`, plus the consolidated `export-357` on /data.
- **Backups:** private HF `InternalCan/ppm-aloe-pose-mono-8020` (never GCS), driven by the controller `setup-logs/hfb/aloe-bk/posemono/controller_posemono.sh` (job 24245). Log: `exports/BACKUP_LOG.txt`.
