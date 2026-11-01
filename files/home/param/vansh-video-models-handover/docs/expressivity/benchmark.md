# Expressivity benchmark v0: does the face follow the voice?

Mosaic run only (`aloe-h4-mosaic/run_1/step10440/model_ema.safetensors`), seen16 studio clips, first 60 s
(376 latent frames = 1501 px frames), seed 2026 + clip, 50 steps per frame; one 300 s run for drift. Code, notes and
every number: `/home/vansh/eval/expressivity/` (`NOTES.md`, `results/`, `results300/`). Review clips:
`/home/vansh/eval/expressivity/review/`. Score card: `bench.py NAME ...`.

## Result

Two phenomena, each scored relative to GT under the same audio, with controls.

1. **Lips: P/B/M closures.** At the bilabials of the audio, GT seals its lips 97.7% of the time; plain 78.3%,
   w2s10 85.1%, a face with the wrong audio 56%. Timing is right, depth is not, and the misses happen while the
   generated face holds a smile: the model can't superpose a lip closure on a smile, real people can. Guidance
   saturates at 89% (w = 3-4) while vowels open 1.24x wider than GT.
2. **Beyond the lips: smile and upper face.** The smile is the one non-mouth channel the model drives from the audio,
   at 65-75% of GT's coupling (r 0.35 / 0.40 vs 0.54), and its smiles start at the wrong times. Brows, lids, blinks
   and gaze are set by seed, pose and sink: with another clip's audio they stay the same (r 0.66-0.82) and follow the
   new audio at r 0.05 (GT 0.17, head motion removed). Guidance w = 4 lifts that to 0.14: the model has learned the
   weak voice -> upper-face link but at w = 1 doesn't use it.
3. **Over 300 s** lips and smile don't drift (closure 0.77 / 0.78 / 0.77 in 0-1 / 1-3 / 3-5 min, GT 0.98 / 0.98 /
   0.97). The upper face does: its audio coupling starts at GT's level and falls to 0.65x GT after 3 min.
4. **Three seeds (section 7)** confirm the main claims: plain < w2s10 on closure (+0.070 ± 0.013, every seed), the
   smile gap (GT - plain 0.16 [0.11, 0.21]), and the upper face being set by seed + pose + sink, not by the audio (the
   swap follows its new audio at 0.062 ± 0.012). Not robust: the own-audio upper-face gap to GT (0.026 [-0.003,
   +0.057]) and every per-clip closure value. The 300 s upper-face decay is still one seed.

SyncNet (LSE-C) rates w2s10 above GT (4.03 vs 3.79, 13/16 clips) and keeps rising with guidance (4.49 at w = 4)
while closures stay below GT. Blink timing leaks from the pose. Adaptive sampling changes nothing here.

## Samples (5d1)

| variant | how | notes |
|---|---|---|
| plain | `cached_universal.py`, guidance 1 | |
| w2s10 | `audio_guidance_scale 2`, `audio_guidance_steps 10` | |
| adaptive | r4's `adaptive_sampling.py --mode rewrite`, threshold 1.0, mosaic toml (`gen/adaptive_rewrite.py`, verbatim) | r0's checkout has the old script |
| adaptive_w2s10 | same + `_guided_step` on the first 10 steps (`gen/adaptive_guided.py`) | copy, not the repo |
| w1.5s10, w3s10, w4s10 | guidance dose-response (`gen/dose.sh`) | section 5 |
| plain_swap, w2s10_swap, w4s10_swap | clip n's sink, pose, seed; clip n+1's Mimi codes (`gen/swap.sh`, `gen/swap_w4.sh`) | model-side control |
| plain300 | plain over the full 300 s chunk (`gen/long.sh`); its first 60 s = plain_seed2, a second noise draw | section 6 |

GT = the VAE decode of the same latents (`samples_n.mp4`), audio = Mimi decode, the same audio the model gets.

## Tools

- Face: MediaPipe FaceLandmarker (478 landmarks, 52 blendshapes, head matrix) on a 384 px crop from GT's smoothed
  face box, the same box for every variant (pose is shared). 100% detection on every video. Channels: inner-lip
  aperture / outer-eye distance, mouth width, EAR, iris-in-eye gaze, brow-eye distance, blendshape groups, head angles.
- Audio events: `words.json` (actor's own word timing) + CMUdict: words with P/B/M, words with stressed open vowels,
  each timed at the wav2vec2 phoneme-CTC (`wav2vec2-lv-60-espeak-cv-ft`) posterior peak of that class inside the word.
  309 bilabial events in 16 x 60 s.
- Audio -> face regressor (the 5d4 "small model"): ridge from frozen wav2vec2 layers 8 + 16 (PCA 64 each) + energy, log
  F0, voicing, ±0.96 s context, to per-clip z-scored face channels. Trained on 144 studio actors not in seen16 (first
  120 s each, VAE-decoded so the face goes through the same VAE as gen, tracked identically); 120 train / 24 held-out
  actors. Held-out r: aperture 0.72, jaw 0.65, width 0.51, smile 0.45, pucker 0.36, squint 0.21, gaze_y 0.20,
  head pitch 0.20, brow_bs 0.16, EAR 0.16, look-down 0.13, blink 0.13, brow 0.11; gaze_x -0.04 and yaw -0.01 (not
  predictable from audio, not used).
- SyncNet v2 (syncnet_python weights), crop from GT landmarks, for reference.

## 1. Lips: P/B/M closure

**Measure.** For each bilabial event, c = min inner-lip aperture in ±2 frames around the event (lag -2 frames: GT's
lips close ~80 ms before the CTC peak, from GT's event-triggered average), divided by the clip's GT speech-open level
A (p90 of GT aperture over word frames). Closed if c < 0.25. Score = closure rate over the events. Same events for
every variant (shared audio).

**GT floor and controls** (`results/lips.txt`, `lips.png`):

| | closure rate |
|---|---|
| GT | 0.977 [0.958, 0.994] |
| GT shifted later +1 / +2 / +3 / +5 frames | 0.98 / 0.92 / 0.71 / 0.48 |
| GT shifted earlier -1 / -2 / -3 / -5 frames | 0.98 / 0.98 / 0.86 / 0.42 |
| GT smoothed 3 / 5 / 9 frames | 0.90 / 0.66 / 0.42 |
| GT with another clip's audio events | 0.560 |
| plain / plain_seed2 (second noise draw) | 0.783 [0.735, 0.828] / 0.780 [0.731, 0.825] |
| w2s10 | 0.851 [0.809, 0.890] |
| adaptive / adaptive_w2s10 | 0.783 / 0.851 |
| plain_swap / w2s10_swap / w4s10_swap at their own (swapped) audio | 0.761 / 0.861 / 0.909 |
| plain_swap / w2s10_swap / w4s10_swap at the pose's audio | 0.489 / 0.524 / 0.570 |

The order GT > w2s10 > plain > wrong audio holds for every threshold from 0.15 to 0.45 (`results/lips_tau.txt`).
The swap rows show the lips are driven by the audio and by nothing else (not the pose, not the sink).

**10 s blocks:** GT 1.00 0.98 0.98 0.98 1.00 0.94; plain 0.83 0.71 0.85 0.78 0.74 0.78; w2s10 0.88 0.79 0.82 0.89
0.91 0.86. No drift within 60 s (nor over 300 s, section 6). Lowest clips for plain: 07 0.44, 15 0.57, 06 / 13 0.67;
with the second noise draw 07 is 0.78 and 15 is 1.00, so per-clip values are seed noise (section 6).

**Timing vs depth.** The gen's closure minimum lands at GT's lag (-2 frames) in every 30 s block of the 300 s run
(`results300/drift.txt`); what differs is depth: median c GT 0.036, plain 0.067, w2s10 0.038. Open vowels: median
opening vs GT 1.00 (plain), 1.13 (w2s10), 1.24 (w4s10); share of vowels opened > 1.25x GT 0.27 / 0.36 / 0.49.

**Why: the smile blocks the closure** (`why_lips.py`, `results/why_lips.txt`). Plain's miss rate by its own smile
blendshape at the event: < 0.1: 0.10 (166 events), 0.1-0.3: 0.23, 0.3-0.6: 0.38, > 0.6: 0.60. Smile on missed
events 0.37, on hits 0.14, GT 0.12. GT closes through its smiles (miss 0.13 when GT smile > 0.3, 45 events). Across
clips, plain's excess smiling vs GT correlates with its closure gap (r -0.50). w2s10 rescues 25 of plain's misses
(loses 4), and on those it smiles less (0.27 vs 0.37). Clip 07 is the extreme: plain grins with teeth through the
"M"s and "P"s that GT seals.

**SyncNet** (`results/syncnet.txt`). LSE-C GT 3.79, plain 3.49, w2s10 4.03 (adaptive 3.49 / 4.03), wrong audio 0.23.
w2s10 > GT on 13/16 clips, > plain on 16/16. Per clip, SyncNet's gap to GT tracks the closure gap for plain (r 0.57)
but not under guidance (r 0.13): SyncNet rewards the wider mouth and stops seeing whether the lips seal.

## 2. Beyond the lips: smile and the upper face

**Measure.** r(audio prediction, face channel) per clip over 60 s, where the prediction comes from the regressor
trained on other actors; GT and every variant share the audio and hence the prediction. Groups: articulation
(aperture, jaw, width, pucker), expression (smile, brow, brow_bs, squint), eyes (EAR, blink, gaze_y, look-down),
head pitch (given by the pose: control).

| r (mean over 16 clips) | GT | plain | w2s10 | GT, wrong audio | GT shifted ±1 s | GT smoothed 1 s |
|---|---|---|---|---|---|---|
| articulation | 0.57 | 0.44 | 0.52 | -0.01 | 0.18-0.19 | 0.47 |
| smile | 0.54 | 0.35 | 0.40 | -0.02 | 0.31 | 0.51 |
| expression | 0.29 | 0.20 | 0.24 | -0.03 | 0.13-0.15 | 0.30 |
| eyes | 0.15 | 0.12 | 0.13 | -0.02 | 0.07-0.08 | 0.19 |
| head pitch | 0.27 | 0.23 | 0.26 | 0.03 | 0.07-0.09 | 0.27 |

Gap to GT, 95% CI over clips: smile plain -0.19 [-0.26, -0.12], w2s10 -0.13 [-0.19, -0.08]; expression plain -0.09
[-0.13, -0.05]; eyes plain -0.04 [-0.07, -0.00]; brows alone not significant (brow_bs -0.06 [-0.14, +0.02]). The
measure is about slow, prosody-scale timing (shifting GT by 1 s halves it, smoothing GT by 1 s doesn't hurt). 10 s
blocks are noisy and show no drift (`results/regress.txt`, `regress.png`).

**What is actually driven by the audio: the swap generation.** Same seed, pose and sink as clip n, the audio of clip
n+1 (`swap_analysis.py`, `results/swap.txt`):

| median over 16 clips | plain | w2s10 | w4s10 |
|---|---|---|---|
| r(own-audio face, swapped-audio face): aperture | 0.22 | 0.15 | 0.07 |
| ... smile | 0.40 | 0.31 | 0.22 |
| ... brow / brow_bs / squint | 0.82 / 0.73 / 0.69 | 0.76 / 0.57 / 0.62 | 0.58 / 0.42 / 0.52 |
| ... EAR / blink / gaze_y / look-down | 0.81 / 0.81 / 0.66 / 0.80 | 0.76 / 0.76 / 0.59 / 0.76 | 0.66 / 0.67 / 0.44 / 0.68 |
| ... head yaw / pitch | 0.98 / 0.95 | 0.98 / 0.93 | 0.97 / 0.88 |
| swapped face vs the audio it was given: smile | 0.44 | 0.49 | 0.49 |
| ... brow_bs / squint / gaze_y | 0.07 / 0.13 / 0.10 | 0.08 / 0.22 / 0.14 | 0.22 / 0.22 / 0.20 |
| ... look-down / pitch | -0.02 / 0.05 | 0.03 / 0.05 | 0.03 / 0.08 |
| swapped face vs the pose's (original) audio: look-down / pitch | 0.14 / 0.22 | 0.16 / 0.27 | 0.19 / 0.26 |

Change the audio and the mouth and smile change; at w = 1 the brows, lids, blinks and gaze barely move. Look-down and
pitch keep following the original audio, through the pose. With guidance the upper face starts to listen (w = 4:
own~swap brow_bs 0.42, gaze_y 0.44; follows the new audio at 0.20-0.22).

**Pose-independent coupling** (`partial.py`, `results/partial.txt`): regress head yaw / pitch / roll (+ 1 s means and
velocities) out of both prediction and face, per clip. Upper face (brow, brow_bs, squint, EAR, blink, gaze_y,
look-down):

| | upper-face r [95% CI over clips] |
|---|---|
| GT | 0.165 [0.137, 0.195] |
| plain / w2s10 / w4s10 | 0.117 [0.079, 0.152] / 0.132 [0.099, 0.164] / 0.171 [0.130, 0.211] |
| plain_seed2 (second noise draw) | 0.171 |
| plain_swap / w2s10_swap / w4s10_swap, vs the audio they were given | 0.052 [0.021, 0.083] / 0.089 [0.054, 0.121] / 0.140 [0.102, 0.178] |

Per channel, GT / plain / plain_swap: smile 0.55 / 0.36 / 0.40, squint 0.23 / 0.16 / 0.14, brow_bs 0.27 / 0.21 /
0.07, brow 0.12 / 0.08 / 0.01, gaze_y 0.19 / 0.15 / 0.09, look-down 0.11 / 0.06 / -0.03, EAR 0.13 / 0.09 / 0.05,
blink 0.11 / 0.08 / 0.04. GT's upper face follows the voice beyond its head motion. At w = 1 the model's upper face
follows seed, pose and sink, and what it takes from the audio is a third of GT's (swap 0.052 vs 0.165). Only smile and
squint (the eye part of a real smile) are audio-driven at w = 1. The own-audio gap (plain 0.117 vs GT 0.165) is not
a robust number: a second noise draw gives 0.171 (section 6).

**Why the smile decouples** (`why_smile.py`, `results/why_smile.txt`). The gen's smile is not frozen: level 0.29 vs
GT 0.27, std 0.24 vs 0.27, ACF half-life 0.94 s vs 1.10 s. It smiles as much as GT, at the wrong times: only 42%
(plain) / 49% (w2s10) of GT smile onsets are reproduced within ±1 s, chance 0.27-0.29.

## 3. Blinks come from the pose

Eyelids aren't pose joints, but gen blinks land within ±3 frames of a GT blink 47% of the time (chance 13%; blink
rate 25 vs GT 26 per minute), and still 41-43% with another clip's audio, at every guidance level (`blinks.py`,
`results/blinks.txt`). The pose raster draws the eye points with a confidence colour ramp; eye-point confidence drops
when the eyes close. So blink timing (and part of look-down) is copied from the real take: good for reconstruction,
but not something the model produces from audio, and a benchmark that rewards blinks on GT's beat would measure this
leak.

## 4. Adaptive sampling

At threshold 1.0 the rewrite detector fired on 2/16 clips (clips 10 and 12, ≤ 3 episodes of ~1 s) in both adaptive
variants. Every measure equals the non-adaptive variant to the third decimal (closure, smile r, LSE-C, blinks).

## 5. Guidance dose-response

`audio_guidance_scale` w on the first 10 steps of each frame (`dose_report.py`, `results/dose.txt`, `dose.png`):

| w | closure | vowel opening / GT | vowels > 1.25x GT | smile r | upper r (pose-partialled) | upper r, swap vs its audio | LSE-C |
|---|---|---|---|---|---|---|---|
| GT | 0.977 | 1.00 | 0 | 0.537 | 0.165 | - | 3.79 |
| 1 | 0.783 | 1.00 | 0.27 | 0.351 | 0.117 | 0.052 | 3.49 |
| 1.5 | 0.835 | 1.06 | 0.31 | 0.383 | 0.128 | | 3.80 |
| 2 | 0.851 | 1.13 | 0.36 | 0.404 | 0.132 | 0.089 | 4.03 |
| 3 | 0.890 | 1.20 | 0.44 | 0.446 | 0.154 | | 4.33 |
| 4 | 0.890 | 1.24 | 0.49 | 0.447 | 0.171 | 0.140 | 4.49 |

- Closures saturate at 0.89 from w = 3 and never reach GT: the smile conflict stays. The price is range: at w = 4 half
  the open vowels are > 1.25x GT's opening (the "believable but unphysical" mouth). SyncNet passes GT at w = 1.5 and
  keeps rising, i.e. it rewards the exaggeration.
- Smile coupling saturates at 0.45 (GT 0.54).
- Upper face: at w = 4 the coupling equals GT's, and the swap at w = 4 shows it is audio-driven (0.140 with another
  clip's audio vs 0.052 at w = 1; own~swap brow_bs drops 0.73 -> 0.42). The conditional model has the weak voice ->
  brow / lid / gaze link; at w = 1 it is swamped by seed, pose and sink.
- Review: `lips_DOSE-w4_c05_...` (vowels 1.87x GT, LSE-C 4.57 vs GT 2.67).

## 6. Over 300 s, and the seed floor

**Drift** (`long_report.py`, `long_upper.py`, `drift.py`; `results300/`). plain300 = plain over the full 300 s chunk,
15 clips (clip 10 excluded: its GT goes dark at 126 s), 30 s blocks:

| 30 s block from | 0 | 30 | 60 | 90 | 120 | 150 | 180 | 210 | 240 | 270 |
|---|---|---|---|---|---|---|---|---|---|---|
| closure GT | 0.99 | 0.98 | 0.99 | 0.98 | 0.98 | 0.96 | 0.99 | 0.98 | 0.97 | 0.94 |
| closure gen | 0.78 | 0.76 | 0.81 | 0.85 | 0.74 | 0.74 | 0.86 | 0.78 | 0.74 | 0.71 |
| articulation r GT | 0.56 | 0.59 | 0.56 | 0.54 | 0.53 | 0.59 | 0.55 | 0.58 | 0.52 | 0.49 |
| articulation r gen | 0.44 | 0.45 | 0.47 | 0.42 | 0.43 | 0.47 | 0.42 | 0.48 | 0.42 | 0.34 |
| smile r GT | 0.54 | 0.53 | 0.46 | 0.49 | 0.48 | 0.62 | 0.45 | 0.52 | 0.37 | 0.49 |
| smile r gen | 0.36 | 0.38 | 0.33 | 0.36 | 0.32 | 0.45 | 0.27 | 0.38 | 0.28 | 0.21 |
| upper r (pose-partialled) GT | 0.17 | 0.12 | 0.16 | 0.15 | 0.14 | 0.17 | 0.14 | 0.18 | 0.11 | 0.14 |
| upper r (pose-partialled) gen | 0.17 | 0.14 | 0.13 | 0.08 | 0.09 | 0.12 | 0.09 | 0.11 | 0.09 | 0.08 |

Change of the gap (gen - GT) from 0-60 s to 180-300 s, 95% CI over clips: articulation +0.01 [-0.04, +0.07], smile
-0.00 [-0.09, +0.09], upper face -0.065 [-0.116, -0.016] (raw, pose not removed: -0.04 [-0.10, +0.02]). Closure
0-60 / 60-180 / 180-300 s: GT 0.985 / 0.976 / 0.973, gen 0.769 / 0.783 / 0.773. The lip lag vs the audio is a
constant -1.5 frames for GT and gen in every block, gen-vs-GT lag 0.

So the mouth and the smile are as good at minute 5 as at minute 1. The upper face starts at GT's coupling (gen/GT
1.10 in the first minute) and settles at about 0.65x GT after 90 s. One seed: this seed's first minute is high (0.17,
the 60 s run's seed gives 0.12), so the size of the decay is uncertain; its direction holds, the blocks after 90 s
(0.08-0.12) sit at or below both 60 s seeds (0.117, 0.171) and below GT in every block.

**Seed floor** (`seed.py`, `results/seed.txt`). plain_seed2 = the first 60 s of plain300 (same model, inputs and
seed config, different noise because the noise is drawn for the whole chunk):

| | plain | plain_seed2 | w2s10 | per-clip r(plain, seed2) | per-clip mean abs diff plain-seed2 / plain-w2s10 |
|---|---|---|---|---|---|
| closure | 0.783 | 0.780 | 0.851 | 0.10 | 0.137 / 0.075 |
| smile r | 0.351 | 0.378 | 0.404 | 0.77 | 0.104 / 0.057 |
| upper r (pose-partialled) | 0.117 | 0.171 | 0.132 | 0.28 | 0.084 / 0.023 |

Pooled closure and smile coupling are stable; per-clip closure is not (the seed decides when the face smiles, and
that decides which closures fail). Smile coupling per clip is a property of the clip (r 0.77). The upper-face gap to
GT is within seed noise at one seed; same-seed contrasts are 4x tighter (0.023), which is why the swap and the dose
carry the upper-face conclusions.

## 7. Seed robustness: 3 seeds

`gen/seeds.sh`, `seed3.py`, `results/seed3.txt`. plain, w2s10 and plain_swap with config seed 2026 (original), 3026
and 4026 (seed + clip as before), 60 s, 16 clips. Values: per seed | mean ± SD over seeds | pooled over seeds
[95% CI, clips resampled].

| | seeds 2026 / 3026 / 4026 | mean ± SD | pooled [95% CI] |
|---|---|---|---|
| closure plain | 0.783 / 0.819 / 0.806 | 0.803 ± 0.018 | 0.803 [0.744, 0.854] |
| closure w2s10 | 0.851 / 0.877 / 0.890 | 0.873 ± 0.020 | 0.873 [0.835, 0.904] |
| closure w2s10 - plain | +0.068 / +0.058 / +0.084 | +0.070 ± 0.013 | +0.070 [+0.038, +0.106] |
| closure GT - plain (GT 0.977) | 0.194 / 0.159 / 0.172 | 0.175 ± 0.018 | 0.175 [0.126, 0.230] |
| closure plain_swap at its audio / at the pose's audio | 0.761 / 0.793 / 0.718; 0.489 / 0.511 / 0.453 | 0.757 ± 0.037; 0.484 ± 0.029 | |
| smile r plain / w2s10 (GT 0.537) | 0.351 / 0.362 / 0.412; 0.404 / 0.412 / 0.460 | 0.375 ± 0.033; 0.425 ± 0.030 | |
| smile gap GT - plain | 0.187 / 0.175 / 0.125 | 0.162 ± 0.033 | 0.162 [0.111, 0.213] |
| smile gap GT - w2s10 | 0.133 / 0.126 / 0.078 | 0.112 ± 0.030 | 0.112 [0.072, 0.151] |
| upper r plain (pose-partialled; GT 0.165) | 0.117 / 0.141 / 0.161 | 0.139 ± 0.022 | 0.139 [0.110, 0.168] |
| upper r plain_swap vs its audio | 0.052 / 0.076 / 0.060 | 0.062 ± 0.012 | 0.062 [0.041, 0.085] |
| upper r plain - plain_swap | 0.065 / 0.065 / 0.101 | 0.077 ± 0.020 | 0.077 [0.047, 0.107] |
| upper r GT - plain | 0.048 / 0.024 / 0.004 | 0.026 ± 0.022 | 0.026 [-0.003, +0.057] |

What sets each part of the face: r(face, face), mean over clips; every pair shares pose and sink.

| pair | upper face | smile | aperture |
|---|---|---|---|
| same seed, other audio (3 pairs) | 0.72 | 0.36 | 0.24 |
| other seed, same audio (3 pairs) | 0.41 | 0.36 | 0.58 |
| other seed, other audio = pose + sink only (6 pairs) | 0.35 | 0.11 | 0.11 |
| GT vs plain (same audio, pose, sink) | 0.38 | | |

**Solid (3 seeds):** guidance improves closures (+0.07, every seed) and both stay well below GT (0.80 / 0.87 vs
0.98). The smile gap holds (CI excludes 0 for plain and w2s10). The upper face is set by pose + sink (0.35) plus the
seed (0.72 with the same seed); the audio adds 0.06 (0.41 vs 0.35). With another clip's audio, the upper face follows
that audio at 0.06 vs GT 0.17 in every seed. The mouth is audio-set (0.58 same audio vs 0.24 same seed). The smile is
half audio, half seed.
**Tentative:** the own-audio upper-face gap to GT (one seed said 0.05, three say 0.026, CI includes 0); the own-audio
coupling is mostly the pose, so it doesn't measure listening; the swap does. Per-clip closures and smiles are seed
noise (pooled values are not). The 300 s upper-face decay (section 6) and the w = 4 results (section 5) are one seed.

## Score card

`bench.py gt plain w2s10 ...` (`results/bench.txt`). Closure and the two r columns with 95% CI over events / clips;
drift = (30-60 s) - (0-30 s); opening = median vowel opening / GT's; blink@GT = gen blinks within ±3 frames of a GT
blink (chance = GT blinks circularly shifted).

| | closure | closure drift | opening | smile r | expr+eyes r | expr drift | LSE-C | blink@GT |
|---|---|---|---|---|---|---|---|---|
| gt | 0.977 [0.96, 0.99] | -0.023 | 1.000 | 0.537 [0.45, 0.61] | 0.245 [0.20, 0.29] | -0.009 | 3.79 | - |
| plain | 0.783 [0.74, 0.83] | -0.023 | 0.999 | 0.351 [0.27, 0.44] | 0.176 [0.13, 0.22] | -0.015 | 3.49 | 0.47 (0.13) |
| plain_seed2 | 0.780 [0.73, 0.83] | -0.017 | 0.970 | 0.378 [0.31, 0.45] | 0.222 [0.20, 0.25] | -0.011 | - | 0.48 (0.13) |
| w1.5s10 | 0.835 [0.79, 0.87] | +0.023 | 1.064 | 0.383 [0.29, 0.47] | 0.194 [0.15, 0.24] | -0.010 | 3.80 | 0.46 (0.13) |
| w2s10 | 0.851 [0.81, 0.89] | +0.058 | 1.125 | 0.404 [0.32, 0.49] | 0.203 [0.16, 0.24] | -0.027 | 4.03 | 0.46 (0.13) |
| w3s10 | 0.890 [0.85, 0.93] | +0.024 | 1.195 | 0.446 [0.39, 0.52] | 0.221 [0.18, 0.27] | -0.031 | 4.33 | 0.48 (0.13) |
| w4s10 | 0.890 [0.85, 0.92] | +0.037 | 1.237 | 0.447 [0.38, 0.52] | 0.239 [0.19, 0.28] | -0.035 | 4.49 | 0.48 (0.13) |
| adaptive | 0.783 [0.74, 0.83] | -0.023 | 1.008 | 0.351 [0.27, 0.43] | 0.176 [0.13, 0.22] | -0.015 | 3.49 | 0.46 (0.13) |
| adaptive_w2s10 | 0.851 [0.81, 0.89] | +0.058 | 1.125 | 0.404 [0.32, 0.49] | 0.203 [0.16, 0.25] | -0.027 | 4.03 | 0.46 (0.13) |
| GT, wrong audio | 0.560 [0.50, 0.61] | +0.044 | - | -0.018 [-0.11, 0.07] | -0.025 [-0.05, 0.00] | +0.042 | 0.23 | - |

## What didn't pan out

- Prior (a), "eyes stare": dropped. Blink rate is normal; blink timing even matches GT, through the pose. Horizontal
  gaze is not predictable from audio (held-out r -0.04), so there is nothing to score it against.
- Prior (b), "guidance opens the mouth but misses p/b/m": the opposite. Guidance improves closures (0.78 -> 0.89)
  while opening vowels wider than GT (1.24x at w = 4). The unphysical part is the range, not the closures.
- Prior (c), "face goes mushy": not visible at 4x zoom in 60 s studio clips; not measured.
- An "A/V drift" over 300 s (lips lagging the audio by +7 frames at minute 5, GT and gen alike) was my bug: wav2vec2
  on 20 s pieces returns 999 frames per piece (400-sample receptive field), so the audio timeline lost 20 ms per
  piece. Fixed (+80 samples per piece), every feature recomputed and the regressor refit; all numbers here are after
  the fix (60 s numbers moved by < 0.02; old outputs in `*_drift/`).
- Brows look flatter than GT, but their audio-coupling gap is not significant at one seed; only the swap shows that
  they don't listen.
- Mouth state in silence differs from GT in both directions: not dictated by the audio, dropped.
- SyncNet vs opening ratio across clips: weak at n = 16, not claimed.
- Hand over the face (clip 09): tracks are meaningless there. GT and gen share the hand pose, so the comparison stays
  fair, but those stretches add noise; review windows require both traces to move.

## Review clips

`/home/vansh/eval/expressivity/review/index.md`, one line per clip. GT | gen side by side, each with a face zoom and a
mouth (lips) or eye + mouth (smile) zoom, the audio, the current word, the score in the filename and burned in.
Lips clips: the mouth box flashes green (closed) / red (not closed) at every P/B/M of the audio. Smile clips: a
scrolling strip per side, grey = smile / squint / look-down predicted from the audio, white / yellow = that face, with
r over the last 10 s. Controls: GT's face with another clip's audio next to the model given that audio; the same
seed and pose with two audios. Per-clip LOW picks show the mechanism (the smile blocking the closure), not a property
of the clip: another seed scores clip 07 at 0.78.

- `lips_LOW_c07_plain_closure0.44_gt0.94_t7-19.mp4`: clip 07 plain closure 0.44 vs GT 0.94; window: plain 2/7, GT 7/7
- `lips_SAME-WINDOW_c07_w2s10_closure0.89_gt0.94_t7-19.mp4`: the same window with w2s10 (0.89)
- `lips_LOW_c06_plain_closure0.67_gt0.90_t6-18.mp4`: window: plain 7/10, GT 10/10
- `lips_LOW_c13_plain_closure0.67_gt1.00_t17-29.mp4`: window: plain 10/16, GT 16/16
- `lips_HIGH_c08_w2s10_closure0.96_gt1.00_t16-28.mp4`: window: w2s10 8/8, GT 8/8
- `lips_HIGH_c00_w2s10_closure0.95_gt1.00_t30-42.mp4`: window: w2s10 8/8, GT 8/8
- `smile_LOW_c07_plain_r0.22_gt0.65_t20-35.mp4`: window: plain -0.16, GT 0.63
- `smile_LOW_c09_plain_r0.15_gt0.55_t30-45.mp4`: window: plain 0.16, GT 0.72
- `smile_HIGH_c08_w2s10_r0.74_gt0.78_t33-48.mp4`: window: w2s10 0.92, GT 0.93
- `smile_HIGH_c01_w2s10_r0.68_gt0.75_t38-53.mp4`: window: w2s10 0.90, GT 0.86
- `lips_CTRL-swap_c08_audio09_gt0.14_plain_swap0.91_t23-35.mp4`: GT control: real face 08 under clip 09's audio closes
  0.14 (chance) vs the model given that audio 0.91
- `smile_CTRL-swap_c07_audio08_gt0.07_plain_swap0.66_t41-56.mp4`: GT control, smile: real face 07 vs clip 08's audio
  r 0.07; the model given that audio 0.66
- `pair_SAME-SEED-POSE_c15_plain_own-vs-audio00_upper-r0.95_mouth-r-0.36_t3-18.mp4`: same seed and pose, two audios:
  mouth r -0.36, upper face r 0.95
- `lips_SYNCNET-TRAP_c11_w2s10_lsec3.79_gt3.26_closure0.79_gt1.00_t1-13.mp4`: LSE-C above GT, closure 0.79 vs 1.00,
  vowels 1.16x GT
- `lips_DOSE-w4_c05_w4s10_opening1.87_closure0.82_gt1.00_lsec4.57_gt2.67_t8-20.mp4`: w = 4: vowels 1.87x GT, closure
  0.82 vs 1.00, LSE-C 4.57 vs 2.67

## Next

- Make the swap generation part of the benchmark: one extra sample per clip with another clip's audio. It is the only
  measure that separates "follows the audio" from "copied from the pose / sink", and it costs one sampling run.
- More seeds for the 300 s upper-face decay and for w = 4 (60 s claims for plain / w2s10 / swap now have 3 seeds).
- The upper-face link exists in the conditional model (w = 4 swap) but is unused at w = 1: candidates are guidance
  restricted to the upper face or to later steps (keeps the mouth range physical), a larger loss weight on the eye
  region, and removing the eye-point confidence from the pose raster (an easier path for lids and look-down).
- The smile/closure conflict suggests the model treats expression and articulation as competing states rather than
  superposing them; a targeted check is closure rate on GT-smiling events after fine-tuning or with mouth-only guidance.
- Hand-occlusion masking (e.g. MediaPipe hands on GT) before scoring the upper face.
