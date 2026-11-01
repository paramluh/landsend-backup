# pose6 / B2: G0COND frame-0 inputs for every Aloe monoplex pose item

Built 10-27 12:12 PT (job 31117, slinky-18, 4 CPUs, 23 s). Script: `build_g0aloe.py`. Log: `logs/build-31117.log`.

## Result
- **Coverage: 28,048 of 28,048 items have every field and g0_ok = True.** That is 27,517 train and 531 eval, which match `aloe-pose-mono-8020/tests/plan_mono_check.json` exactly in both items and frames.
- frame0_ok false: **0**. Missing source: **0**. The missing list is empty.
- Anchor check: the stored anchor and the recompute from `frame0_pose_yxc_pixels` agree with **max |diff| = 0.0**. This holds over the c0 items, and also over all 18,093 B tracks.
- Token tie-in: on all 28,048 items, the store's B pose (`pose_<side>.npy[r][:, :n]`) equals `tokens[:, 7500c : 7500c + n]` of the track's `pose_tokens.npy`. The sha256 equals `pose.json`'s value, and the store is -1 beyond n. This proves the sidecar comes from the same files the store was tokenised from.
- f digest (blake2b16 of the f [28048, 8] float32 array, in the npz order): `d64a00c20f166fbb81c217ead187467b`.

## Files
- `g0aloe.parquet`: one row per item, train items first (idx = position in `splits_v2['train']`), then eval. The columns are:
  - item key: `split`, `idx`, `store`, `row`, `swap`, `row_key`, `b_side`, and `item_key` = `row_key|b_side` (= the draw key).
  - item layout: `n_frames`, `chunk_index`, `frame_start`, `anchor_is_item_frame0`.
  - provenance: `conversation`, `speaker`, `pose_json`.
  - **the G0 fields under the exact clip-record names**: `orig_h`, `orig_w`, `frame0_root` [y, x], `frame0_width` and `frame0_ok`. `{k: row[k] for k in data_ppm.G0_FIELDS}` can go straight into `g0_feat`.
  - `f` (8 x float32, = `g0_feat`) and `g0_ok`.
  - audit columns: `rotation`, `anchor_stored`, `anchor_recompute_maxdiff`, `problem` (empty when the item is OK).
- `g0aloe_f.npz`: the same f [N, 8] float32 with `g0_ok`, `split` (0 train / 1 eval), `idx`, `store`, `row`, `swap` and `item_key`, in the same order.
- `COVERAGE.json`: counts per split and per split x chunk (c0 / c>0), the diff maxima, the rotation and frame-size histograms, the f min and max, the missing and invalid lists (both empty), the md5 of every source file, and the md5 of the exec'd fresh_pose block.
- The build re-reads both files and asserts `f == g0_feat(fields)` byte-exactly on every row.

## Definitions (= training; executed from the training sources' own text, not retyped)
- **f**: `data_ppm.g0_feat`, `pose5/code-811-s2p/data_ppm.py:257-275`, with `G0_DIM` / `G0_FIELDS` / `g0_problem` at `:243-254`. These are AST-extracted from that file and exec'd.
  - f = [r_y, r_x, log w, -r_y/w, (1-r_y)/w, -r_x/w, (1-r_x)/w, log(H/W)], computed in float64 and stored as float32.
  - An invalid frame 0 gives f = 0. Training then warns per clip in `pose_micro_ppm`, `:288-299`.
- **frame0_root / frame0_width / frame0_ok / orig_h / orig_w**: the frame-0 block of `pose5/code-811-s2p/fresh/fresh_pose.py:131-141` (decode_new). It is identical to `qwen-1ch-data/pose-s3-samdata/scripts/build_pose_rows_sam.py:136-146`. The block is exec'd verbatim (block md5 `0b5147caed518a567359b17d2e43debf`).
  - j55 = frame0_pose[JOINTS] / [H, W, 1], with JOINTS = COCO-WholeBody 0..12 + 91..132 (`build_pose_rows_sam.py:42`).
  - root = (j55[5] + j55[6]) / 2 in (y, x), i.e. the shoulder midpoint, image-normalised.
  - w = |x5 - x6|, the horizontal shoulder width / orig_w.
  - frame0_ok = all finite, H, W > 0, w > 0, and `allclose(global_state[0], [root, w], atol=1e-4)`.
  - Aloe's `global_state` = `[[root0_yx_normalized, shoulder_width0_normalized]]` from the same pose.json. This is the tokenizer's stored frame-0 anchor, the same quantity as fresh's constant per-frame global_state.
- **Source**: the B track's `DATA/<src_dir>/w00.<SPEAKER>.pose.json`, field `normalization`. Recipe `frame0-root-frame0-horizontal-shoulder-width-v1 (Xabi preprocess_frame0)`, tokenizer `pose-tokenizer-32cb-200k-noshift@1d6ff4f0` (= 811's). This is the file `aloe-pose-ft/code/build_poseB_rows.py:115-122` asserted and tokenised from.
  - The build re-checks the same contract per track: status ok, speaker, tokenizer, recipe, 12.5 Hz, shape [32, T].
  - B side: swap 0 is data 'assistant' = SPEAKER_00; swap 1 is 'user' = SPEAKER_01.
- **Items**: `aloe-pose-mono-8020/code/data_aloe_pose.py:402-421` (`splits_v2`, MONO), re-implemented line for line. One item per (store k, row r, swap s) with `b_ok[r, 1-s]` and n >= MIN_CLIP = 2 (`:368`).
  - Item frame f = store row frame f = conversation frame 7500·chunk_index + f (`build_poseB_rows.py:135-168`).
  - The pose goes at position P + pose_delay + f, with no crop or offset (`data_aloe_pose.py:164-183`).

## Caveat: c>0 items (decision (a), pose6 10-27)
- 9,775 train and 180 eval items are 10-min chunks with chunk_index >= 1 (`anchor_is_item_frame0 = False`).
- Their tokens come from ONE tokenizer call over the whole cut (`tokenizer_call: "one call over the whole cut"`). So they are normalised to the **cut's frame-0 anchor**, not to the item's own first frame. Raw keypoints at frame 7500·c were never fetched (`DATA/NOTE_local.md`: no raw 25 fps keypoints).
- These items therefore get the cut-frame-0 anchor as f. **That anchor is NOT the item's first frame, and the item's frame-0 pose is coded relative to it.**
- This is what training defines f as, the anchor the tokens are relative to: in 811, `frame0_ok` requires gs == anchor.
- c0 items (17,742 train / 351 eval): the anchor IS the item's frame 0.
- If a consumer ever wants the other choice (f = 0 for c>0), the filter is `anchor_is_item_frame0 == False`.

## Other facts
- `rotation = -90` on 298 items (282 tracks). Their orig_h x orig_w = 3840 x 2160, which is display-oriented. The keypoints are upright in that frame on 40 of 40 sampled items, and the anchor recompute is exact. So these items need no special handling.
- All items are portrait 16:9: 1920x1080 on 12,796 items, 3840x2160 on 12,318, 2560x1440 on 2,919, and 1280x720 on 15. So f[7] = log(16/9) = 0.575 is constant.
- f percentiles over the items (p5 / p50 / p95):
  - r_y: 0.53 / 0.58 / 0.66
  - log w: -0.43 / -0.27 / -0.10
  - (1 - r_y)/w: 0.39 / 0.55 / 0.70
- These sit inside the range of the 23 clips in `pose5/g0-test-micros/pose.pkl` (log w -0.70 .. -0.25). That is a tiny sample and only an indication.

## Rebuild
`sbatch /data/home/amartyavarma/ppm/pose6/g0aloe/build_g0aloe.sbatch` (CPU, 4 cores, 16 GB, ~25 s; reads only; writes only this dir; exits 3 and lists items if any source is missing).
