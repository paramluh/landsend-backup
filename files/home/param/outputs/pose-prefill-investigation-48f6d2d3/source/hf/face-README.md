# encode_v1 — aloe face/hands-filtered talking-head clips, hires 512x960 @ 25 fps (PRIVATE)

Generated 2026-10-22 21:47Z. **35972 clips / 1962.3 h / 2.00 TB** (401 black-cut pieces).

| kind | clips | hours |
|---|---|---|
| app | 22089 | 1225.4 |
| booth | 10019 | 518.4 |
| studio | 3783 | 214.3 |
| unknown | 81 | 4.2 |

- Source keep list: hands-filter rule `site3_crop20` (`out_v3/encode_manifest_launch.jsonl`, 37,358 rows; `too_close_unfixable` rows excluded).
- Crop policy v3: `c = clamp(max(c_hands_min, c_face), 0, 0.25)` (c_hands_min = smallest level where the hand rule holds; c_face from a median
  face-height target of 0.26 of the output frame), closeness bound (face width <= 0.555, headroom >= 0.20), face-safety, never-upsample.
  Window: autorotate -> fps=25 -> [booth / HF-original: centre 256:480 pre-crop] -> crop=iw(1-c):ih(1-c):iw*x0:0 -> centre 8:15 -> lanczos to 512x960.
- Black-cut: ffmpeg blackdetect (d=0.08, pix_th=0.10) in the encode pass; clips with black inserts were re-cut into pieces >= 120 s
  (window minus [black-1 s, black+1 s]) with the parent's crop; piece rows have `parent_clip_id`, `piece_index`, `black_runs`, `dropped_seconds`,
  `black_check=pass`; parents without a >= 120 s piece are in `dropped.jsonl` (`black_no_piece_ge_120s`).
- Format: H.264 libx264 crf 16 preset slow yuv420p, 512x960, 25 fps, +faststart, mono AAC 128k (own audio). Verified per clip: duration +-0.2 s,
  resolution, fps, sha256, MediaPipe face detection on 5 frames (`face_check`).
- Files: `encode_v1/<kind>/<clip_id>.mp4`; `manifest.jsonl` (one row per clip: provenance, crop decision, output ffprobe, sha256, face_check, hf_path);
  `dropped.jsonl` (split/dropped/failed parents with reasons: {('split', None): 419, ('dropped', 'black_no_piece_ge_120s'): 295, ('failed', 'exception'): 590, ('failed', 'verify_failed'): 163, ('failed', 'face_check_failed'): 74, ('failed', 'ffmpeg_failed'): 2}); `manifests/shard_XXXX.jsonl` (raw per-shard uploads).
- Encoder + fleet scripts: aloe-hands-filter/encode/, fleet/ (encode_v1).

## Not encoded (1364 clips / 74.7 h) — see `not_encoded.jsonl`
- failed/exception: 554
- dropped/black_no_piece_ge_120s: 295
- unfinished_shard: 276
- failed/verify_failed: 163
- failed/face_check_failed: 74
- failed/ffmpeg_failed: 2
Run stopped by user decision at ~99 % (unfinished shards listed as `unfinished_shard`; re-run them with encode_clips.py on a manifest filtered to these ids).
`unmanifested.jsonl` lists 4 mp4s that were uploaded by shards whose per-shard manifest was never written (in flight at shutdown); their
sidecar rows are missing — re-encode or keep as-is without provenance.
