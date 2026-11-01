# NAME: why does the Kate twin say "Lauren" ~90 % of the time? (posefix NAME-A, 10-27 ~00:55 UTC)

FIRST READOUT (NAME-A, 10-27 ~00:58 UTC): cause = absolute word min-p 0.05 collapses a flat name distribution to its argmax (Lauren); see A.

## A. Offline live-engine replay (speech twin vox17127 export-249, live_ppm_twin, call 20261027-003457)
Inputs rebuilt from pinned copies (posefix/name/pinroot; Kate's live package was being rebuilt): voice clip = openings-sam/06 mimi_local[:, :125]
(sha aadfaa2e... = the call's), 2 s conversation-start prefill (codes sha 3c734ab2..., lane b521c653... = the call's), A = the call's mic
(after the server's level stage), the call's params + seed. B's lane teacher-forced to the call ("Hey, how are you doing? Yeah, good. My name
is" -> frame 50 = " Lauren"); B's Mimi codes = the engine's own draws (prefill codes forced), identical across counterfactuals.

Model distribution at the name frame (word stage = within word pieces, before penalties / min-p): FLAT.
| condition | P(Lauren) | P(Rosie) | P(Kate) | top word pieces |
|---|---|---|---|---|
| ref (Kate clip, prefill on) | 0.026 | 0.012 | 0.002 | ".", Lauren, Sarah, Ella, Anna, Amy, Ashley, Sophie |
| no voice clip | 0.028 | 0.017 | 0.002 | Lauren, Sarah, Ella, Sh, Rosie, A, N, T, Anna |
| clip 09 (Lorenzo) | 0.046 | 0.015 | 0.003 | Sarah 0.050, Lauren, Ella, Laura, Liz |
| prefill off (A unmuted, B history same) | 0.025 | 0.012 | 0.003 | Lauren, ".", Sarah, Ella |

THE SAMPLER (t_text_draw exact path, the call's knobs: factorised, min_p 0.05 ABSOLUTE (min_p_rel 0), freq_pen 0.25, top-k 50, T 1):
| setting | word-stage survivors | 20 draws |
|---|---|---|
| call (abs min-p 0.05) | 1: Lauren 1.0 (one-hot) | Lauren 20/20 |
| relative min-p 0.05 | 50 (Lauren 0.078, Ella 0.051, Rosie 0.048, Sarah 0.045, Amy, Ellie) | 15 distinct (Lauren 3, Ellie 3, Rosie 2, ... Kate 1) |
| min-p off | 50 (same) | 13 distinct, no Lauren |
| freq_pen 0, abs min-p | 1: Lauren 1.0 | Lauren 20/20 |
P(WORD category) at that frame = 0.894 -> P(emit " Lauren") = 0.894 x 1.0 = 89 % (the user's "90 %").

Cause: absolute word min-p 0.05 keeps a word piece only if its within-word probability >= 5 % OR it is the argmax
(live_ppm_twin.t_word_min_p: `(lpw >= lgp) | (lpw == top)`). Names are a flat distribution (< 5 % each), so only the argmax survives and the
draw is deterministic; the model's argmax female name in this context is "Lauren" regardless of the voice clip (the voice/identity barely
moves P(Kate): 0.2-0.3 %). freq_pen is irrelevant. Data: posefix/name/name_minp.json, name_twin.log.

Kate-intro clip counterfactuals (sam-sep-handoff/test/06_kate_fdba6c86, clip = fdba6c86 0-10 s with "My name is Kate"; same 2 s prefill):
| condition | model word stage | engine draw at the call's knobs (abs min-p 0.05) | relative min-p / min-p off |
|---|---|---|---|
| (a) clip = kate_intro (blank lane) | Kay 0.066, Kate 0.051, Katie 0.035, Rosie 0.029, Lauren 0.024 | 2 survivors: Kay 0.56, Kate 0.44 (Lauren 0) | Kate 0.088, Lauren 0.042 |
| (b) clip = kate_intro + its words lane | Kate 0.855, Katie 0.016, Kay 0.013 | Kate 1.0 | Kate 1.0 / 0.91 |
| prefill off, B silent first 2 s (PAD lane + silence codes) | Lauren 0.031, Sarah 0.026, Amy 0.021 | Lauren 1.0 | Lauren 0.064 |
Reading: the name comes from the TEXT the model has seen. A voice clip whose audio says "Kate" (no words lane) moves Kate from 0.2 % to ~5 %
(Kay/Kate then share the min-p survivors). With the clip's words lane (as in training rows when the lane is not blanked) Kate is 86 %.
Prefill on/off and A silent/recorded do not move P(Lauren) (~2.5-3 %). Pose-model replay not run (GPU handed to NAME-C).

## SUMMARY (posefix, 10-27 ~01:25 UTC) — why "Lauren"
- CAUSE: the sampler. Factorised text with ABSOLUTE min-p 0.05 (user's NOPERSONA_DEFAULTS, minp_mode absolute) uses fast_engine.word_min_p's absolute rule
  (p_w >= 0.05, top word always kept; qwen-1ch/infer/s3fast/fast_engine.py:71-87 = live_ppm_pose.py t_word_min_p). The model's name distribution is flat
  (Lauren 2.6-4.6 %, Sarah, Ella, Amy, Anna, Rosie ...; Kate 0.2-0.5 %), no name reaches 5 %, so only the argmax (Lauren) survives:
  P(say Lauren) = P(WORD) 0.894 x 1.0 = 89 %; 20/20 draws. Relative min-p 0.05: 15 distinct names / 20; min-p off: 13; freq-pen irrelevant.
- NOT an inference-construction bug: for Kate's own training row (m1n0 #2937), the trainer-built and live-built system prompts are identical
  (ids / depth / present, P 134); shared libs md5-identical; only A during the 2 s prefill differs (zeros live vs partner Voxtral in training).
- Identity: Kate's training row had no voice clip; only 13 % of micros carried a B clip (always a later chunk), so the clip barely conditions
  the name. A clip in which she says "My name is Kate" lifts Kate to 5 % (audio) / 86 % (+ its words lane).
- Fix candidates (user's OK; live slider, no restart): min_p_rel = 1 (standard relative min-p) or min-p off.
- Not done: pose-model replay; NAME-C step 2 (teacher-forced trainer-vs-live P(Kate)) stopped by main to free GPUs.

## B. Training-data side (posefix NAME-B, 10-27 ~01:05 UTC; posefix/name/names_spk.py -> name/names_spk.json)
Data: the duplex FT's s2h-m1-samsep stores = 8,107 handover conversations, 16,214 sides (both lanes can be B), 1,589 actors.
No gender column exists in the manifest; "female" below = sides whose strict intro is a female first name (name-list proxy).
- Name stats (sides; actors):
  - Lauren: 18 strict ("my name is") / 103 loose ("I'm Lauren") sides, 13 actors; 238 raw word occurrences.
    Loose "I'm X" ranking: Alex 183, Sarah 164, Ashley 162, Sam 137, Joe 123, Anna 113, **Lauren 103**, Rosie 102 ...
    -> Lauren is the 4th most common female "I'm X" self-intro in the training data.
  - Rosie: 71 strict / 102 loose sides but only 6 actors (one actor, xyvR1rjH, has 59 sides); 233 words.
  - Laura: 46 / 71 sides, 8 actors. Kate: 1 strict / 26 loose sides, 4 actors; 21 of those sides are actor 06 herself ("I'm Kate").
  - Share of female-name strict intros (1,364 sides): Rosie 5.2 %, Laura 3.4 %, Lauren 1.3 %, Kate 0.07 %.
- Voice proximity (WeSpeaker voxceleb resnet221-LM on Mimi-decoded 10 s SAM-code windows, cosine to Kate's mean embedding from her live
  voice clip + 3 other Kate windows; Kate refs self-cos 0.85-0.89, Kate's own training window 0.80):
  - 118 random actors: p50 0.11, p90 0.27, max 0.59.
  - Lauren self-introducers: 0.17, 0.16, 0.14, 0.10, 0.02, -0.01 (ranks 35-122 of 135) = no closer than random.
  - Rosie: the main Rosie actor (59 sides) 0.31 (rank 9, ~random p93; same-speaker level is 0.65+); other Rosie actors 0.04-0.07.
  - Laura: -0.03.
  - 20 nearest embedded actors call themselves Nika, Danielle, -, Alinda, Kathy, Jane, Jacqueline, Liz, Rosie (#9), Alba, Emma, Beth, ...
    No Lauren among them. (Nearest-neighbour search covers the embedded subset only: all Lauren/Laura/Rosie actors + 118 random.)
- Facts: Lauren is NOT a voice-neighbour name of Kate; it is a high-frequency female self-intro in the training data (generic prior),
  while "Kate" is almost exclusively actor 06's own name (21 of 26 sides).
