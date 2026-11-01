# aloe-v2pure-ft stores (built 2026-10-17)

**Source.** Only the Aloe pose-speech-vad release v2: the local cache `/data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure`, which was built from `gs://aloe-processed-data/pose-speech-vad/releases/v2` only. This is the same data the 8B "SAM pose Aloe v2pure" model was fine-tuned on. The cache shards are train-0000..0959 plus eval.

**Builder.** `../code/build_v2pure_rows.py`, with these steps: `plan` writes PLAN.json, `build --store K` builds one store, `stores` writes STORES.json. Rows are written by Model 1's `s2h_to_s3rows.write_node`, unchanged. Validation is in `../tests/validate_v2pure_stores.py`.

**Format.** The same as `aloe-pose-ft/stores`: s3-duplex-rows-v1 plus the pose add-on (`pose_{user,assistant}.npy` int16 [R,32,7500], `pose_index.npz` b_ok/split, `POSE.json`, `BUILD_DONE`). The trainer reads it with `code/data_aloe_pose.py`.

## Row design
- **One row per cache item.** Each item is one active-speaker clip, with `n_frames` = clip length (1..7494). The trainer needs `S3_MIN_FRAMES=1`.
- **Keys and indices.** `row_key = ALOE/aloe-v2pure/<rid>/c<k:05d>`, where k is the clip's ordinal in its recording. Clips are ordered by start time, with ties broken by speaker and then segment_id, over both speakers' clips.
  - `chunk_index` = k, `frame_start` = 7500k, `frame_end` = 7500k + n.
  - `episode_id` = rid, `podcast_id` = `aloe-v2pure`, `dataset` = `ALOE`.
  - `sysprompt.rec_key(row_key)` = `ALOE/aloe-v2pure/<rid>`.
  - All clips of a recording are in one store.
- **Sides.** speaker_1 is data `assistant` and speaker_2 is data `user`, on every row. Channel B is the clip's active speaker: that side carries the clip's Mimi codes, pose codes and words. `b_ok` is 1 on that side only.
- **Partner side.** The partner is absent from this data. Its side holds:
  - Mimi codes of digital silence: `setup-logs/s3hestia/ourrows/silence_codes.json` `seq`, tiled from frame 0. These are the same codes used by the PPM trainer's xdep 'silent' and by `infer_ppm.sil_seq`.
  - no words;
  - pose -1;
  - provenance `a_present` false, `partner` 'absent'.
- **Persona.** Both sides are 'missing' on every row.
- **Words.** The cache words are clip-local. A start in [-0.5, 0) is clamped to 0, then Model 1's `side_words` is applied unchanged. Pieces come from `speech_q.word_pieces` with the Qwen3-32B tokenizer.

## Split
A recording is eval iff blake2b8('aloe-pose-v1|' + rid), read little-endian, % 10000 < 100. This is the cache builder's own rule; it was checked on every recording and matches the cache's eval shard exactly.

## Counts
| | Rows | Frames | Recordings |
|---|---|---|---|
| Train | 367,496 | 92,132,615 | 12,384 |
| Eval | 3,482 | 868,932 | 113 |

- Train frames per store run from 11,516,528 to 11,516,628, and eval frames from 108,194 to 108,865.
- Rows per store run from 45,898 to 46,915.
- Total size is 713.8 GB (88.3–90.3 GB per store).

## Deviations and notes
- **CJK words dropped.** 6,158 of 21,670,181 cache words were dropped by Model 1's CJK rule. They sit in 129 rows, mostly in p1 and p2. The 8B saw these words.
- **Clamping.** 371 word starts were clamped to 0, and 44,285 word ends were clamped to the clip end.
- **Two clips have n = 1 and fail in the trainer.** They are train (store 7, row 45554, swap 1) and eval (store 2, row 29188, swap 1). `data_s3q.row_arrays` asserts nc >= 2, so `Stores.conv` fails on them. The trainer must skip items with `n_frames < 2`.
