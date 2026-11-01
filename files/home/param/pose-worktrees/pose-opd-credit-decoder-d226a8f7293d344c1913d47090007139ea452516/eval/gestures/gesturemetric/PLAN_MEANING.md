# gesturemetric (2) MEANING — contrastive speech<->gesture judge, trained on REAL Aloe only (plan, 10-26; nothing launched)

## (a) Data on disk (no downloads)
- POSE_INDEX (aloe-ft-32-samsep/poseduplex/data/POSE_INDEX.json), counted 10-26:
  train 7,949 convs (6,039 with both sides ok) = 1,075 conv-h, 2,379 side-h of ok pose;
  eval 158 convs (123 both sides) = 22 conv-h, 49.7 side-h. Eval = held out from the pose models too -> judge eval / calibration set.
- Per side, data/aloe-handover-full/DATA/files/<bucket>/<session>/seg00/: w00.SPEAKER_0N.pose_tokens.npy [32, T] (12.5 Hz, 2 pose
  frames / token, aligned to the Mimi codes: same T, same cut), w00.SPEAKER_0N.codes.npy (Mimi stream-v5, 32 cb), w00.words.json
  (dataset WLT words, clip-relative, `_separated_speaker` per word). Spot check of 200 random ok sides: pose / words / Mimi 200/200/200.
- No FLAC on disk (NOTE_local.md) -> any loudness comes from Mimi codes (embed directly, or decode with local models/mimi on GPU).
- Train the judge on a subset: ~600 side-h of train convs is plenty for a ~10M-param model; all 49.7 eval side-h held out.

## (b) Inputs (4 s windows = 50 tokens = 100 pose frames)
- POSE — recommend DECODED KINEMATICS, not token embeddings. Decode through the same path as gm.py / run_stats.py (PoseTokenizer
  decode, person-01 frame0), keep 13 body joints + both wrists + 2 hand centroids + 2 hand-spread scalars, coordinates relative to the
  window-mean shoulder midpoint / frame-0 shoulder width, plus velocities and confidence masks (conf <= 0.3 -> masked).
  Why: (1) per-window mean removal kills static framing / posture / identity shortcuts that the 31 "light" codebooks carry;
  (2) it scores what the avatar actually shows (same decode as the stroke detector); (3) generated tokens go through the identical
  decode, so no train/gen representation mismatch. Cost: CPU decode measured 8 s per 10-min track on 4 CPUs (real_extract.py);
  600 side-h = 3,600 tracks -> ~8 CPU-h, or minutes on the GPU pair at start. Token embeddings stay an ablation only.
- SPEECH (own speaker only):
  - words: frozen LOCAL Qwen3-0.6B (models/Qwen3-0.6B; no sentence-encoder on disk, checked models/ + HF cache) — mid-layer hidden
    states of the window's words (+-0.5 s context), each word's states pooled and placed at its onset step (12.5 Hz) so timing survives.
  - prosody: own-channel Mimi cb0-7 codes -> learned embeddings (summed) at 12.5 Hz. Speaker identity in Mimi is useless here
    because every negative is the same speaker (c). Generated runs give Mimi codes + the model word lane; real gives Mimi + WLT words.
  - variants trained/reported: words-only (pure meaning), audio-only (prosody/timing), both (headline).

## (c) Model + training (target < 2 h on 2 GPUs incl. precompute)
- Two encoders, each 4-layer transformer d=256, 4 heads (pose: 100 frames patched x2 -> 50 steps; speech: 50 steps), attention pool
  -> 256-d, cosine, learned temperature. ~8-10M trainable params; Qwen3-0.6B frozen, used once for feature precompute.
- Windows: own-speaker speech windows (>= 3 own words in the 4 s) — co-speech gesture only, listener stretches excluded.
- Batch = 64 conversation-sides x 16 windows each (>= 4 s apart) from the SAME side; InfoNCE pose->speech and speech->pose over the
  16 same-conversation candidates (speaker identity, camera, lighting all constant -> only content/timing discriminates).
- Hard-negative variant: add the same window's speech shifted by 1-3 s as extra negatives (forces timing, not just topic).
- ~20k steps x 1,024 windows, bf16, 2 GPUs DDP: ~30-60 min; precompute (pose decode subset + Qwen text features) ~20-40 min.
- Gates before use: eval top-1@16 well above 6.25 % chance; shuffled pairs = chance; pose shifted by 2 s drops toward chance;
  words-only > chance (= real semantic signal). If words-only is at chance, report (2) as "timing only" and say so.

## (d) Score for generated clips
- Unit = 4 s window, hop 1 s, generated part only (prefill excluded: drop t < prefill_frames x 0.08 s + 0.5 s), >= 3 own words.
- Retrieval: true speech window vs 15 distractors from "other moments of the same conversation". A 20 s playground run has too few
  non-overlapping moments, so distractors = the run's own windows >= 4 s away + windows from other runs of the SAME model / person /
  voice (constant identity); live calls (minutes long) use their own call. Real ceiling computed by the identical protocol on 20 s
  cuts of eval convs (distractors = other moments of that conv).
- Report top-1@16 (chance 1/16 = 6.25 %), mean rank, and normalised meaning = (acc - 1/16) / (acc_real - 1/16): 0 = flailing,
  1 = real-like. Shuffled control = pose of run i vs speech of run j -> must sit at chance. A 20 s run gives ~10-14 windows -> per-run
  number is noisy (show with CI); headline = pooled per model / settings.

## (e) GPU need + start rule
- Need: 2 GPUs (one is enough; 40 GB plenty), ~2 h wall total, CPU -c 16, --mem=96G.
- Now (10-26, squeue/scontrol): every GPU except slinky-9 is allocated — ppm-mot (16 nodes x 8, training), motasr slinky-10+18
  (training), slinky-4: posedup-lb-S4 2 / genweb-avhybrid 2 / motlive-gpu 2 (live) + posefix-phys 2 (job 30749, side job).
  First eligible pair = posefix-phys's 2 GPUs on slinky-4 when 30749 ends (watch slinky-4 RAM: ~20 GB physically free now).
- START RULE: only when a free GPU pair appears; never live-demo or training GPUs; never slinky-9; never preempt another agent.
- CPU pre-stage allowed before then (non-training node, --mem set): window index from words.json, pose kinematics decode of eval +
  a train subset, Mimi cb0-7 slices.

## (f) Risks
- Webcam framing: hands leave frame / low conf -> masked features; windows with < 50 % confident wrist frames dropped.
- Listener gestures / backchannels: own-speech windows only; overlapping-talk windows kept but flagged.
- WLT word timing noise (~0.1 s): harmless at 4 s windows; hard negatives use >= 1 s shifts.
- Shortcut on speech rate/loudness alone (gesture energy ~ speech energy): audio-only vs words-only split shows how much is meaning.
- Generated pose is out of distribution (smoother / jitterier): calibrate with shuffled-generated as the true chance, not 1/16 alone.
- No leakage: judge trains on POSE_INDEX train convs only; all calibration on eval convs; playground prefills come from eval convs.
