# aloe-pose-ft-sam: the Aloe pose stores with SAM-separated Mimi codes (built 10-26 05:15 PT)

**What it is:** a copy of job 22091's stores (`ppm/aloe-pose-ft/stores` p0..p7) where ONLY the two Mimi codes arrays are swapped. Every other file is byte-identical, so rows, items, order, prompts, persona, pose tokens and words are unchanged. The stores feed the SAM re-run of the latest monoplex, `ppm/aloe-pose-mono-8020-sam`.

- **Source of the codes:** HF `InternalCan/stage2-sam-sep-v1-mimi` @ `ecb72b2d`, the local copy in `data/aloe-sam-sep/hf` (no downloads). It uses the per-speaker isolated, crossbleed-removed stems and the same mimi-stream-v5 recipe, keys and T as ours.
- **Mapping:** `codes_assistant` ← `SPEAKER_00_isolated_npy` and `codes_user` ← `SPEAKER_01_isolated_npy`. This is build_poseB_rows.py's role rule, asserted per row.
- **Window:** each row takes its own window `[frame_start, frame_start + n_frames)` of the conversation, with PAD -1 after `n_frames`, exactly as built.
- **Voice clips:** the trainer cuts the voice clip from B's own codes (a later chunk of the same conversation, `data_ppm.voice_clips`), so **the voice clips are SAM codes too**.

## Build (`code/build_sam_stores.py`, job 30621 on slinky-4, 2 min 16 s)
1. `cache`: 32 SAM parquet shards are decoded to `_samcache/<session>__<seg>.npy` int16 [2, 32, T].
2. `check`: **17,872 rows over 11,443 conversations; 0 missing, 0 T mismatches.** Every row has its SAM pair with `equal_T`, and SAM T equals the row's conversation T.
3. `build`: p0..p7 run at once. The original row layout is asserted, the SAM rows are written and read back, and every copied file's sha256 equals the original's. Results are in `stores/pK/SAM.json` and `sam_rowstats.npz`.
4. `report`: `SAM_BUILD.json`.

**Independent check** (`code/verify_sam_stores.py`, which does not use the build code or its cache):
- all 12 non-codes files per store are byte-equal (filecmp), and so are STORES.json and PLAN.json;
- the codes npy header and size equal the original's;
- every row is in range with the pad region intact;
- 64 random rows per store, decoded straight from the HF parquet, match exactly. None equals the original row.
Results are in `verify/pK.json`. All 8 stores PASS.

## How much the codes change (fraction of frames whose code differs, per codebook)
| | frames | all cb | cb0 | cb1 | cb2 | cb3 | cb4..31 |
|---|---|---|---|---|---|---|---|
| B tracks, train (what the run trains on) | 136,497,507 | 0.789 | 0.205 | 0.341 | 0.578 | 0.639 | 0.708-0.892 |
| B tracks, eval | 2,639,831 | 0.786 | 0.199 | 0.335 | 0.573 | 0.634 | 0.703-0.890 |
| all rows, both lanes | 178,430,280 | 0.791 | 0.222 | 0.351 | 0.580 | 0.642 | 0.713-0.893 |

The full 32-codebook lists are in `SAM_BUILD.json`.

**Speaker sanity (cb0 agreement with the original lanes):** in all 22,886 conversation sides, the SAM stream agrees more with its own lane than with the other lane, so no side is flipped. At row level, 2 short rows fall the other way.

**SAM QC flags in the used conversations (FYI, nothing dropped):**
- 646 have sam_sep or mimi flags;
- 197 have at least one chunk flagged "swapped";
- 499 have an "ambiguous" chunk.

Of the trained B frames, only **0.017 %** (23,067 frames in 17 tracks) lie in a chunk where B's own stem is flagged swapped; eval has 0. See `SWAPPED_FRAC.json`. The rows are kept as they are, as specified.

`_samcache/` (11 GB) is a build intermediate. It is safe to delete once main OKs it; it is not a checkpoint.
