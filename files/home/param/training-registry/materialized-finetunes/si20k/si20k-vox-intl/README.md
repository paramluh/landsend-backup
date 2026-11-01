# si20k-vox-intl: INTL 14B (next-arch) fine-tune on spoken-instruct 20k (TTS), user side = Voxtral latents, co-trained 1:1 with unseen podcast rows

User (10-18 PT): fine-tune on the spoken-instruct 20k tt-v3 TTS set, "ASR version" (the user side enters as Voxtral latents), INTL 14B,
8 nodes; co-train 1:1 with unseen pretraining podcast rows; init = the LATEST checkpoint (w-6242). Prepared, NOT launched (waits for the go).
Page label: **INTL · w-6242 + ASR w-480 · si20k + podcast 1:1** (aloe_ft_compare.html, dotted curve + podcast-eval chart).

## Reference recipe copied: aloe-nextvox-6242, job 24193 (14B next-arch w-6242 + 14B ASR adapter w-480)
- Trainer `code/train_si20k_vox.py` is GENERATED from 24193's `train_aloe_nextvox.py` by si20k-vox-ppm/tools/make_trainers.py (+ co_patch.py):
  exact text replacements only; diff in `tests/trainer_vs_ref.diff`. Env `si20k_vox_intl.env` = 24193's env verbatim + the data / podcast blocks.
- Unchanged: model (model_ilp.ILP + VOX edit: user_text slot = s x adapter(Voxtral(A)), emb_A masked there), arch block (= next.env), peaks
  (trunk 3e-6, spec 6e-5, proj 3e-5, audio 6e-5, cb0 head 1.5e-5, decoder 1.5e-5, vadapter 5e-4, vgain 5e-4), clips, warmup 8, linear decay
  to 0, loss weights, decoder K 1 on ~10 % of B frames, tokens_step, 64 rows per step on 64 ranks, PARITY + DATA checks, frozen inp.aud.0.
- Init: w-6242 (/data/home/amartyavarma/ppm/next-arch/ckpt-weka/ppm-next/w-6242), the latest 14B checkpoint = 24193's.
- Adapter: asr14b w-480 (trained on w-6242; sha b6ca8779f8d0b1bd), 24193's local copy adapter-asr14b-w480 (identical to
  asr-adapter-14b/ckpt-weka/asr14b/w-480; sha asserted at init). Step-0 eval logged with the latents and without any user input (base path,
  A = PAD) as eval_noA_speech.

## Data (qwen-1ch-data/si20k-vox)
- Source: the TTS run's PRIMARY renders only (queue/done markers; v148 extras never read). Stereo FLAC ch0 = user (A), ch1 = assistant (B).
- Builder `build_si20k_rows.py` (incremental: `--mode prep` caches every finished render, `--mode assemble` cuts + writes the stores).
  Checks per conversation (failing = dropped with a reason): lane words == manifest words, the trainer's lane rule rebuilds every generated
  piece, codes / FLAC shapes and byte sizes, B bank clip present.
- Filter: the filter agent's `filter/keep_spans.tsv` (id, end_time_s, end_turn): each conversation is CUT at end_time_s (both channels,
  words, codes, A audio -> latents), on a frame boundary, extended so no word straddles the cut and by <= 1 s for the last word's audio, never
  onto the next word; conversations without a usable span are dropped. keep.txt also supported.
- Rows: 7,500-frame rows (c00 + a c01 tail when the tail is >= 1,500 frames; ~1 % of conversations), 4 stores s20n0..3, s3-duplex-rows-v1
  + persona sidecar (both sides 'missing': spoken-instruct has no assistant persona) + voiceclip_B.npz + A_LEDGER.json.
- Eval: stratified ~1.5 % of conversations by length bucket (uncut) x domain x task (systematic sampling over the sorted strata).
- A = user: Voxtral latents of the RAW user channel (0 dB, no loudness normalisation = 23081's raw A), no text, no Mimi, no loss.
  B = assistant: words + Mimi + NEW_WORD timing targets. No A/B swap (B = assistant always).
- **Voxtral latents are computed inside the trainer** (frozen Voxtral Realtime encoder on each micro's A waveform), exactly as 23081 and
  24193 do (0.35 s of a ~6 s step). Precomputing them was NOT done: the references consume waveforms, not stored latents, and the encoder
  output for ~3,000 h is ~1.4 TB (50 Hz x 1280 x bf16), beyond /data's budget. The CPU check encodes a real A window: [1, 4n, 1280], finite.

## Prompt rule (new standard, user 10-19): independent seeded draws per row
- B voice clip p 0.8: h64('si20k-voice|20261011|row_key') / 2^64 < 0.8. B persona p 0.2: h64('si20k-persona|20261011|row_key') / 2^64 < 0.2.
- **Clip source (to confirm with the user):** the exact tt-v3 cloning reference that voiced B in that conversation (manifest voice.B ->
  bank Mimi codes, 64-190 frames, median 125 = 10 s), in the generator's own form (whole clip, blank lane, codes at the code delay, channel B).
- Persona: spoken-instruct carries NO assistant persona / system text (only a hidden user persona, generation-time only, not exposed), so
  the realised persona fraction is 0 (drawn ~0.2, logged). Provisional build: voice clip realised 0.802 (train) / 0.783 (eval).

## Co-training with unseen pretraining podcast rows (user 10-18 PT)
- `code/podcast_rows.py` + `plan/PODCAST_PLAN_intl.json` (`tools/podcast_plan_intl.py`): ppm-next's own speech-row stream reproduced per
  pretraining rank; rows AFTER step 6,242 (verified unseen by row key). Node-local on the 16 pretraining hosts; FT rank (host H, local j) reads the unseen
  continuation of the pretraining rank that ran on (H, j). No staging copy (rows read in place; preflight checks the slices + first rows).
- 1:1: every step = 64 si20k micros + 64 podcast micros (one each per GPU). Steps = one pass over the filtered si20k train rows.
  Normalisers = tokens_step pooled over all 128 micros (every target token of the step weighs the same; the alternative, per-stream means
  summed, would keep the si20k gradient scale of the reference).
- Podcast micro = the 14B pretraining's speech-micro maths exactly = NATIVE A (partner words in the user_text slot: model path 'base' =
  model_ilp.ILP.forward). NOT Voxtral: the alternative would be Voxtral-encoding the podcast A channel too. One knob follows the FT recipe:
  light_frac 0.1 in training (pretraining 1.0 with decoder K 8). Same loss terms / weights as pretraining.
- Eval: si20k held-out (eval_speech) + the pretraining eval_speech rows of the 64 pretraining ranks on these hosts (eval_pod_speech). Report
  only normalised word CE. train_speech = si20k micros (comparable to the reference curve), train_pod_speech = podcast micros.

### Decisions 10-18 PT (coordinator)
- **Same podcast rows for both runs** (unless the user overrides): the 1,163,344 rows NEITHER model trained on = ppm-v1 stream positions
  >= 25,658 (12,829 x 2) per rank. The 14B's per-rank streams are identical to PPM's; it consumed positions < 12,484, so its positions
  12,484..25,657 (PPM-trained) are skipped. Both envs point SI20K_PODCAST_PLAN at si20k-vox-ppm/plan/PODCAST_PLAN_ppm.json.
- **slinky-0/1/2**: their slices were deleted 10-18 (HF-verified); another agent restores them. podcast_rows uses the host's own slice when
  it is complete on that node at launch, else the staged shards (overlay PODCAST_STAGED.json: 10 shards / ~17 GB per host from slinky-5 / 6 /
  8, 3,360 rows each, positions >= 400 of the donor ranks, disjoint from everything); logged per host ("native" / "staged").
- **Volume**: channel A is NOT loudness-normalised (raw TTS level). The gain step is the switch SI20K_A_GAIN = raw (default) | fixed:<dB> |
  lufs:<LUFS> (per row, voxtral_front.lufs_gain_db = the -23 LUFS / peak-limited Mimi rule). No latents are stored, so a rule change needs no
  rebuild; 0 latents were ever built with -23 normalisation.

### Co-training numbers (measured on the provisional build + the plans)
- Frames / tokens per step: si20k rows avg 6,357 frames (provisional, synthetic spans), 710 B words / row; podcast FULL rows 7,500 frames
  minus the prefix (P 5 on most rows, up to 2,000 with persona text), ~906 B words / row. B text targets ~ frames: si20k : podcast ~ 46 : 54
  in frames, ~44 : 56 in B words. Close to 1:1, podcast slightly heavier.
- Cost: peak GPU memory unchanged (micros run one after another; the podcast micro is no longer than an si20k micro: T <= 7,502 vs <= 9,502
  budget); step time ~2x the reference's one-micro step (PPM 22420 layout 5.6 s -> ~11 s; 14B 24193 6.2 s -> ~12 s; the podcast micro has
  no Voxtral encode). CPU RAM: + 2 podcast DataLoader workers per rank (~1 GB each).
- Podcast rows: unseen and node-local on 13 of the 16 pretraining hosts (slinky-3,5,6,8,9,10,11,12,13,14,16,17,19); slinky-0/1/2 lost
  their slices (no copy anywhere), so whole shards of other hosts' slices are staged to their /scratch (plan/PODCAST_STAGED.json overlay +
  extra_plan_si20k_staged.json; PPM-unseen rows, also unseen by the 14B whose consumed rows are a prefix of PPM's). Staging size: ~12 GB per
  staged host; nothing else is copied (the other hosts read their own slices in place). Podcast eval covers the pretraining ranks of the
  slice hosts in the run (2 rows per rank); staged hosts evaluate zero-weight dummies.
- CPU check tests/pod_check-*.log (slinky-9, both trainers): 8 local ranks x podcast micro T 7,502, light ratio 0.095-0.106, A lane words +
  present (native A), INTL forward = base path, eval micros T 7,502, cfg diff 0 keys outside the allowed set.

## Launch (on the user's go)
- `SPANS=/data/home/amartyavarma/data/si20k-tts/filter/keep_spans.tsv bash tools/launch_both.sh` (DRY=1: checks only). It submits
  tools/assemble.sbatch (CPU, slinky-2: final stores + STORES.json + plan/PLAN.env), writes the drain files
  si20k-tts/control/DRAIN_FOR_FT.si20kppm / .si20kintl, submits both trainers afterok (PPM on slinky-0,1,3,5,6,8,10,17; INTL on slinky-2,9,11,
  12,13,14,16,19) and the two export watchers (CPU, slinky-15; HF uploads there too).
- Node-local exports (6 x ~3.6 GB per node); watcher copies each to /data, uploads to PRIVATE HF InternalCan/si20k-vox-intl, deletes the
  /data copy after HF verify; final consolidated DCP stays on /data (+ HF), then the drain file is deleted. Never GCS.
- Stop: `touch control/STOP_AFTER_SAVE` (save-and-exit). Never a bare scancel.

## /data budget (floor 300 GB)
- No Voxtral latent stores exist (latents are computed in the trainer). si20k stores (codes / words, shared by both runs): measured 10.8 GB
  for 11,067 rows (~0.98 MB per row: codes padded to 7,500 frames) -> ~20 GB for the full set, staged to node scratch at preflight.
- 10-18 20:28 PT: /data 460 GB free; TTS still ~117 GB -> ~343; stores 20 -> ~323 at launch. Final exports PPM ~78 GB + INTL ~30 GB do not
  fit above the floor, so the watchers free, only when needed: (1) after PPM step 1, /data local-12829 (83 GB; a conversion of the
  HF-verified node-local dcp-12829, re-creatable in ~2 min); (2) after both runs pass step 1, the /data si20k stores (20 GB; rebuilt from the
  prep cache in ~4 min). Then ~426 -> both finals -> ~318 GB. Intermediate exports pass through /data one at a time (deleted after HF verify).
  Nothing else is deleted (not w-3683, w-6242, local-12529, demo checkpoints, the aloe-nextvox-6242 exports).

## Checks done (CPU / tiny, no checkpoints on /data)
- tests/cpu_checks_prov.json (provisional build of 10,963 conversations, synthetic spans): store rows 160/160 (check_row, lanes exact,
  NEW_WORD rule, codes, PAD, persona, B clip == bank, ledger), cut rule 1,381/1,381 cut conversations, A audio length 40/40, soundfile ==
  ffmpeg ch0 decode bit-exact 4/4, user-word vs assistant-only ch0 level gap median 39 dB, micros 160/160 (T, A empty, vin, clip block ==
  bank codes 129/129), light ratio 0.1006, prompt fractions as above, Voxtral CPU encode [1, 800, 1280] finite, cfg diff 0 keys outside the
  allowed set (43 keys differ vs 24193: run dirs, schedule length, data / prompt keys; + podcast once co-training is on).

## Log
- 10-18 20:10 PT: slinky-9 /scratch: deleted the stale aside copy /scratch/amartyavarma/aloe-m1vox-ft/flac.aside-node6-1018 (67 GB of
  22420-era channel-A FLACs; source = the shared /data copy) -> 97 GB free (needed: si20k stores 20 GB + exports ~22 GB).

## FINAL launch recipe (10-18 21:30 PT, user GO + decisions; supersedes earlier sections where they differ)
- Data: all 14,212 final new-engine conversations done at 21:26 PT (the unbiased_order prefix was 1 conversation, so all final ones are
  used); STRICT filter (user: a user turn fails if Whisper misses > 23.3 % of its words after the corruption or it is > 15 dB below the
  opening turns; cut at the first failing turn; frozen copy qwen-1ch-data/si20k-vox/logs/keep_spans_strict_used.tsv, 99.6 % coverage) ->
  12,737 conversations, 12,625 train / 192 eval rows, 1,535 h, 198 steps (64 si20k + 64 podcast rows per step).
- A path, identical for si20k user audio and podcast partner audio, all IN THE COLLATOR / training loop (nothing precomputed):
  raw waveform (si20k: ch0 of the TTS FLAC; podcast: the mono FLAC of the manifest) -> qqa-corrupt-v2 (35 % clean; else 70 % realistic /
  30 % random, severity 40/40/20; drawn per row + epoch; si20k far-end = the assistant channel, podcast far-end = silence; code/acorrupt.py,
  corrupt_lib run in a voicebank-venv subprocess) -> mic gain (si20k only: SI20K_A_GAIN=gauss:6 = per conversation N(0, 6 dB) around its
  own level, clipped +-2.5 SD, peak-limited at -1 dBFS, rate logged) -> Voxtral in the loop -> adapter x s -> the user slot.
  The fitted-population mode (normal) stays in the code, off.
- Podcast micros: B exactly as pretraining (random side); A = Voxtral (A's prefix clip / persona dropped, A text / Mimi removed).
- Eval: si20k held-out clean at the conversation's own level (median gain) + a low-mic split at -12 dB (eval_lowmic) + podcast eval
  (pretraining eval_speech rows, Voxtral A from FLAC, clean) + the step-0 no-user-input eval. Only normalised word CE is reported.
- Decoder K 1 on ~10 % of the light audio for BOTH micro types (light_frac 0.1 train; eval light as the reference); logged per type at step 0.
- Nodes: PPM slinky-3,5,6,8,9,10,14,17 (5 CPUs / task: pose-rb jobs hold 12 CPUs per node); INTL slinky-0,1,2,11,12,13,16,19 (6 CPUs / task).
- Live page: klweb/si20k_ft.html (separate from aloe_ft_compare, which is back to Aloe runs only).
- 10-18 21:37 / 21:43 PT: first launch (PPM 24857 at 5 CPUs / rank, INTL 24903 at 3) ran ~5x slow: the collator's qqa-corrupt-v2 (CPU,
  ~20-35 s per realistic 10-min row) starved the ranks' CPUs (pausing the corruption servers gave 12-13 s / step at once). Stopped cleanly
  with STOP_AFTER_SAVE (PPM at step 21 -> export-198-stop21, INTL at step 3 -> export-198-stop3; both superseded) and relaunched from step 0
  at 7 CPUs / rank (pose rebuild paused), deeper prefetch (4). No recipe change. The trainer keeps no optimizer state, so a mid-run resume
  from a weights-only export would change the recipe; the restart from step 0 keeps it exact.
- 10-18 22:42 PT: relaunched 22:31 (main threads keep all 7 CPUs; collator + corruption on the rank's last 4 CPUs, nice 19) and set the collator / corruption processes to SCHED_IDLE live (now also in co_patch): INTL ~14.5 s / step, PPM ~20 s / step. No maths change.
- 10-18 23:08 PT: remaining slowdown = the collator's corruption threads sharing PHYSICAL cores (SMT siblings 2k / 2k+1) with the ranks' main / NCCL threads. Live re-pin (tools/repin_smt.py via srun --overlap, no restart, no maths change): each rank's main process on 2 whole cores, workers + corruption servers on the socket's remaining whole cores. PPM 11.1-11.6 s / step, INTL 11.9-12.5 s / step. A relaunch must re-run tools/repin_smt.py on every node after the trainers start.
- 10-18 23:34:16 PT: FINAL EXPORT (consolidated bf16 DCP, step 198, verified): /data/home/amartyavarma/ppm/si20k-vox-intl/export-198  -> FINAL_EXPORT_READY written; drain file deleted.
- 10-18 23:34 PT: DONE 198 steps (job 24921). Final export /data/home/amartyavarma/ppm/si20k-vox-intl/export-198 (consolidated bf16 DCP),
  HF-verified (private InternalCan/si20k-vox-intl/export-198-dcp). Normalised word CE, step 0 -> 198: si20k eval 2.450 -> 1.545 (no user
  input at step 0: 2.351), low-mic eval 2.453 -> 1.547, podcast eval (pretraining eval_speech rows) 2.144 -> 1.992.
  Intermediates: export-33 on HF; 66 / 99 / 132 / 165 backfilled to HF by tools/backfill_exports.sbatch (node-local shards kept).
- 10-19 01:09 PT: export-198 stores vgain.theta in bf16 (s off +0.9 %); loaders should PREFER the exact fp32 value in export-198/THETA_FP32.json (also on HF export-198-dcp/THETA_FP32.json); the DCP itself is unchanged.
