# GAZE causal render tests (render agent, 10-23 03:25-04:05 UTC)

## Harness
- code/prep_edits.py: a step-for-step reimplementation of demo-v3 pose.tokens_to_latents. Between offsets_to_positions and raw_to_recon there is an edit hook on the decoded px keypoints. Only pose frames >= 50 (after the 2 s prefill) are edited, and confidences are never edited.
  - Modes for generated tokens: E0 none, E1 smooth, E2 headlock, E3 shape.
  - Modes for GT tokens: E7 headlock, E8 half motion.
  - PP mode: E10, a 20 s ping-pong of real keypoints.
- code/make_swaps.py: builds the 12 s pose/audio swaps (76 latents, 150 Mimi codes). E4 = gen pose + GT audio, E5 = GT pose + gen audio, E6 = GT pose + GT audio, E9 = gen pose + gen audio at 12 s.
- Renders go through the hot pose-server video workers' s20 vrefs queue (code/jobs.py, atomic add/rm of GAZE-* keys). That is the same vrefs_lib Renderer: seed 2026 (equal to the samples' request video_seed), 40/10 steps, the package's 9 refs.
- Scratch vdata (latents) has been deleted. Kept: render/vdata/kp*.npz (edited head and body keypoints per variant) and prep_*report.json.
- Outputs are in out/:
  - <key>.mp4
  - <key>-strip.png: top = reference, bottom = render, 160 px wide.
  - <key>-face.png: 2x head crops, which is what the gaze judgements below are based on.
  - <row>-swap12-{strip,face}.png: every 12 s variant of a row in one grid.
  - psnr.json
- code/strips.py and code/pair_psnr.py rebuild the strips and PSNR.

## Harness check (E0) and renderer determinism
- E0 PSNR vs the original video.mp4: S-05 42.4 dB, E-03 43.0, E-18 41.6. S-20 is only 27.3 dB, with the same faces and gaze by eye.
- R0-S-20 (the sample's OWN vdata re-rendered on another GPU) vs the original: 43.7 dB. So the renderer is reproducible. The 27 dB comes from my re-encode, whose pose latents differ from the sample's by about 0.2% relative (bf16 / cudnn noise, every row). On some clips the renderer amplifies that tiny difference into a visibly different trajectory.
- E6 (GT vdata through the queue) vs the existing GT render: 42.4 / 52.0 / 41.6 / 40.9 dB.
- Render LENGTH changes the noise path. E9 (identical inputs to the original's first 12 s, rendered as a 12 s clip) vs the original's first 12 s is 22-27 dB, even inside the identical 2 s prefill.
- Consequence: single gaze events (one glance at one second) are partly a property of the noise realisation, not of the inputs. Example: S-05's original has a strong side glance at 8 s, but E9 (same pose and same audio, 12 s render) looks at camera at 8 s. Edits must be compared on robust trends across rows, not on single frames.

## Per row (gaze judged from the head crops at 1/4/8/12/16/19 s, or 1/4/8/11 s for 12 s renders)
S-05 (10_27286bdd, clip is in the video model's train split; the strongest drift example):
- original/E0: camera at 1-4 s; 8 s strong side glance (eyes to her left); 12 s squint, eyes half shut and sideways; 16 s eyes up and averted; 19 s head tilted back, eyes rolled up. Clear drift, worst after 12 s.
- E1 (head jitter smoothed): still drifts (8 s glance, 16-19 s looking up and away).
- E2 (head locked to the torso): still drifts (4 s up and right, 8 s sideways, 12 s and 16 s sideways, 19 s down and sideways).
- E3 (head shape locked, translation kept): still drifts (8-19 s sideways, down-left).
- 12 s swaps: E9 (gen pose + gen audio), E4 (gen pose + GT audio), E5 (GT pose + gen audio), E6 (GT) are all mostly on camera. GT has one wide-eyed head turn at 4 s.
- So the drift in this row does not come from the head keypoints, and in the first 12 s nothing separates gen from GT.

E-03 (03_b437950e):
- original/E0: camera at 4-12 s; 16 s eyes up and to the side; 19 s eyes half closed and down. Drift only after 12 s.
- E1, E2 and E3 all keep the eyes on camera at 16 s and 19 s. Even E1, which moves the head points by only 0.7 px on average, removes the late excursions. This points to chaotic sensitivity rather than a causal head-geometry effect.
- 12 s swaps: gen variants (original, E9, E4) are on camera at 4-11 s. GT pose variants (E5, E6) look down at 4 s, a real look-down. Generated looks at camera MORE than GT in the first 12 s.

S-20 (19_47473a46, unseen by the video model):
- original/E0/E2: eyes on camera at every sampled time, 1-19 s. No drift in this generated sample at all.
- GT pose (E5, E6, GT render): 4 s eyes down and left, 11 s eyes to screen right. These are real look-aways that follow the real head.
- E7 (GT pose, head frozen to the torso, GT audio): camera at 4, 8 and 11 s. Freezing the head REMOVES the look-aways; it does not create drift.
- E8 (GT pose, all motion x0.5): 4 s looking up-left (averted), 8 s camera, 11 s slightly right. Similar to GT.
- E4 (gen pose + GT audio): camera. Audio does not move gaze.

E-18 (18_1ffa9083, unseen; glasses make gaze hard to read):
- GT (E6/E5): eyes down or closed at 8-11 s.
- E7 (head frozen): eyes forward at 8-11 s.
- Generated (original/E9/E4): down or closed at 8-11 s, similar to GT.
- 20 s original vs E2: both eyes half closed and looking down at 12-19 s.

## Conclusions so far
1. Rendered gaze follows the head keypoints. Real head motion produces real look-aways (S-20 and E-18 GT). Freezing the head on GT pose (E7) puts the eyes on camera. Stillness or a frozen head does NOT cause drift: E7 drifts less than GT.
2. The audio stream does not drive gaze. E4 vs E9 and E5 vs E6 show no systematic change in any row.
3. Within 0-12 s (the GT length, which is also the video model's training chunk length of 76 latents), generated pose renders do NOT drift more than GT renders. In 3 of 4 rows they look at camera more.
4. The visible drift in generated samples sits mostly in 12-20 s (S-05 at 12/16/19 s, E-03 at 16/19 s), beyond the length of any GT render. Head edits E1-E3 do not reliably remove it in S-05. Note that E2 locks the head to the prefill MEAN, and S-05's prefill already has a turned head, so E2 holds a turned head.

## E10h horizon test (keys GAZE-E10-S-05/E-03/S-20): REAL pose for 20 s
Setup: GT keypoints 0..299, then played backwards 299..100 (no jump), plus the generated sample's 20 s Mimi. Strips: out/GAZE-E10-<row>-face.png and the "E10h" row of out/{S-05,E-03}-face1fps*.png.
- S-05: camera at 8-19 s, apart from a few real glances (14 s, 17 s) that the real head makes. No sustained drift, where the generated pose drifts badly.
- E-03: mostly camera, with a side glance at 17 s and 19 s.
- S-20: camera.
=> Render length on its own does not cause the drift. The late generated pose does.

## E11-E13 hybrids, 20 s (S-05, E-03; strips out/{S-05,E-03}-face1fps{,-a,-b}.png)
All three use the generated Mimi and edit frames >= 2 s; R = the E10h real pose.
- E11 = real body + generated head offsets from the shoulder midpoint.
- E12 = generated body + real head.
- E13 = generated head and torso with both REAL arm chains (elbow, wrist, hand), each attached to its generated shoulder.
Results:
- S-05: E11 drifts like the original (13-17 s sideways, 18-19 s head back with eyes up). E12 behaves like the real pose (mostly camera, a few glances). E13 removes most of the 13-17 s glances but keeps the 18-19 s head-back eyes-up. => In S-05 the generated HEAD drives it; the arms contribute part.
- E-03: original/E0 have the hand at the mouth at 13-15 s and 18 s, eyes closed at 15 s, eyes up or sideways at 16-17 s, half-closed at 19 s. E13 (real arms) and E11 (real body) are on camera at 16-19 s. E12 (generated body including the hand-to-face) keeps the 16-17 s excursions. => In E-03 the generated hands near the face co-occur with the drift and removing them fixes it. One render each, though, and E-03 is perturbation-sensitive.

## Final round: does the HELD head orientation drive eyes-down / eyes-up? (20 s; out/<row>-face1fps{,-a,-b}.png, 1 fps, 20 tiles)
- E3 = head shape (nose/eyes/ears relative to the eye midpoint, divided by eye distance) locked to the prefill mean.
- E10 (in these rows) = the SLOW orientation removed: shape minus (2 s moving average minus prefill mean). Translation and fast motion are kept.
- E-15 fs (3819f107b056): original/E0 have eyes down or closed at almost every second from 5 s on. E3: eyes open and on camera from 5 s on. E10: eyes open, mostly camera, a few slightly-down frames (8, 13, 15 s).
- E-15 l500 (4248a3501e6d): original/E0 have eyes down or closed from 4 s on. E3: open and on camera from 4 s on. E10: mostly open and on camera; down or half-closed at 4-5, 9 and 16-18 s.
- S-13 d3 (a61e16a0f067): original/E0 roll the eyes up or up-left from 6 s to 19 s, with the head tilted back. E3: wide-eyed stare at camera throughout. E10: a roll-up at 6 s and side glances at 7-8 s, then camera from 9 s to 19 s.
=> YES. The held, slowly-drifting head ORIENTATION (pitch) in the generated nose/eye/ear geometry drives the eyes-down and eyes-up. Removing just its slow component (E10) fixes most of it, and locking the shape (E3) fixes all of it.

## Bottom line
The video model renders the eyes to match the head keypoint geometry. The gaze drift in generated videos comes from the generated pose's head orientation. It slowly settles into a held pitch or yaw away from the prefill (down: E-15; up: S-13, S-05), and in some clips hands are raised to the face (E-03).
- Not causes: audio, render length (E10h), stillness or jitter (E7/E1), or the video model itself (GT pose and real-pose renders are fine).
Fix directions:
- In the LLM pose sampling, penalise or anchor the slow head-orientation drift relative to the prefill.
- Or post-process before the video model: subtract the slow orientation component as E10 does, which keeps the fast head motion.
