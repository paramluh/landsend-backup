# Downloading the "both speakers have pose, at least one lane clean" subset

`tools/download_stage2_pose_mimi_sam.py` is a standalone script (Python >= 3.9; `huggingface_hub`, `pyarrow`, `numpy`) that pulls
the colleague subset of this dataset into a plain folder tree: one directory per conversation with `.npy` token arrays, the
per-speaker words, the black-band intervals and (optionally) the raw FLAC lanes and the v1 pose files from the companion bucket folder
`hf://buckets/InternalCan/aloe-conversationa-data/stage2-handover-pose-mimi-sam-v1/<session>/<seg>/`.

```bash
pip install -U "huggingface_hub[hf_xet]>=2.2" pyarrow numpy       # >= 2.2 + hf_xet is needed for --audio / --poses (bucket API)
export HF_TOKEN=hf_...                                            # or `hf auth login`; the token must read the InternalCan org
hf download InternalCan/stage2-handover-pose-mimi-sam-v1 tools/download_stage2_pose_mimi_sam.py --repo-type dataset --local-dir .
python tools/download_stage2_pose_mimi_sam.py --out ./s2 --dry-run                   # hours table + plan, downloads only the two small index files
python tools/download_stage2_pose_mimi_sam.py --out ./s2                             # tokens of the without_bands conversations (~25 GB)
python tools/download_stage2_pose_mimi_sam.py --out ./s2 --with-bands --what all     # everything: + with_bands, + FLAC (291 GB), + poses (483 GB)
```

Both the dataset repo and the bucket are **private**; nothing in the script writes to Hugging Face.

## Selection rule

A conversation (one parquet row) is selected when

* `pose_both_ok` is true — both speakers have v1 pose status `ok` (with `--include-flagged`: `pose_both_ok_or_flagged`, i.e. `ok` or `flagged`), and
* `crossbleed_side in {none, SPEAKER_00, SPEAKER_01}` — at least one lane is clean. `both` and `unknown` are excluded.

The **named crossbleed side is the one that bleeds**: `crossbleed_side == "SPEAKER_01"` means speaker_00's lane is clean and the
partner's voice is audible in speaker_01's lane. (Rule from the handover downloader: a side is confirmed when its acoustic
`audible_bleed` is true or its transcript tier is `bleed`; see the dataset README.)

## Folder layout

```
<out>/
  both_clean/                          crossbleed_side == none
  speaker_00_clean_speaker_01_bleeds/  crossbleed_side == SPEAKER_01
  speaker_01_clean_speaker_00_bleeds/  crossbleed_side == SPEAKER_00
    without_bands/                     no black chunk >= --band-min-seconds in either speaker
    with_bands/                        at least one black chunk >= --band-min-seconds   (downloaded only with --with-bands)
      <session_id>/<seg>/
        conversation.json                        every scalar column of the row (both speakers) + _downloader + _files
        SPEAKER_00.pose_tokens.npy               int16 [32, T]   v1 pose tokens (InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0)
        SPEAKER_00.mimi_tokens.npy               int16 [32, T]   mimi-stream-v5 of the raw lane
        SPEAKER_00.mimi_sam_isolated_tokens.npy  int16 [32, T]   mimi-stream-v5 of the SAM-isolated stem (partner bleed removed)
        SPEAKER_00.frame0_pose.npy               float32 [133, 3] first pose tick (y, x, conf in source pixels)
        SPEAKER_00.global_state0.npy             float32 [3]
        SPEAKER_00.words.json                    [{"word", "start", "end"}, ...]  clip-relative seconds
        SPEAKER_01.*                             same for lane 1
        black_bands.json                         only when either speaker has ANY no-detection interval (see below)
        SPEAKER_00.flac, SPEAKER_01.flac         --audio / --what tokens+audio / --what all : 48 kHz PCM16 original lanes, sha256-verified
        SPEAKER_0N/poses_25fps.npy               --poses / --what all : float32 [2T, 133, 3], sha256-verified
        SPEAKER_0N/metadata.json, timeline.npz, job.json          v1 pose run outputs (checks, lip-sync table, hashes)
  manifest.csv                         one line per SELECTED conversation (folder, bands, session, seg, T, hours, planned, status, dir, bytes, shard, errors)
  summary.json                         flags, the hours table, counts, failures
  .cache/                              index files + parquet shards in flight (shards are deleted after use unless --keep-cache)
```

`T` is the number of 12.5 Hz frames; it is identical for the pose tokens, both Mimi streams and both SAM-isolated streams of a
conversation, and `poses_25fps.npy` has exactly `2T` ticks. **Token index `j` covers `[0.08 j, 0.08 j + 0.08)` seconds of the clip**
(clip-relative; time 0 of every file is `clip_start_sec` of the source recording); pose tick `k` covers `[0.04 k, 0.04 k + 0.04)` s.
Seconds -> token: `floor(t / 0.08)`; token -> seconds: `0.08 j`. Word timestamps in `SPEAKER_0N.words.json` are on the same clock.

## Black bands (`black_bands.json`, `with_bands` / `without_bands`)

Definition (dataset column `speaker_0N_no_detection_intervals`): 25 fps ticks of the v1 `poses_25fps.npy` where **all 133 joints have
confidence < 0.3**, merged into `[start_s, end_s)` intervals. These are the ~6 s chunks of the source video that the app never uploaded
(the aligned render paints them black); the detector sees no person there. v1 `pose_tokens` **hold the last confident pose through
these gaps** (One-Euro confidence hold), so the token stream is continuous but carries no information inside the intervals — mask
those tokens or drop the conversation for training.

* A conversation goes to `with_bands/` when either speaker has an interval of at least `--band-min-seconds` (default **1.0 s**; the
  real black chunks are 6 s or multiples, sub-second intervals are edge blips at the start/end of the clip). `with_bands/`
  conversations are **skipped and counted** unless `--with-bands` is given.
* `black_bands.json` is written whenever either speaker has any interval at all (including the sub-second blips of `without_bands`
  conversations). Per speaker: `intervals` (`[[start_s, end_s], ...]`, tick-aligned), `seconds`, `n`, `max_seconds`,
  `pose_tick_intervals` (`[start * 25, end * 25)`) and `token_intervals` (`[floor(start / 0.08), ceil(end / 0.08))`, clipped to `T`);
  plus `conf_threshold` (0.3), `band_min_seconds_used`, `with_bands`, `T`, `pose_frames`.
* Keep `--band-min-seconds` constant for one output directory (the script warns if `summary.json` was written with another value);
  the manifest column `max_no_detection_sec` lets you re-split differently without re-downloading.

## Hours table (dry run of 2026-10-29, default flags: `pose_both_ok`, band threshold 1.0 s)

```
folder                              bands          convs    hours tokens_GB  audio_GB  poses_GB
----------------------------------------------------------------------------------------------
both_clean                          without_bands   5938   977.03     16.88    168.49    280.70
both_clean                          with_bands       710   131.90      2.28     22.05     37.90
speaker_00_clean_speaker_01_bleeds  without_bands   1201   253.52      4.38     44.37     72.84
speaker_00_clean_speaker_01_bleeds  with_bands       155    32.56      0.56      5.75      9.36
speaker_01_clean_speaker_00_bleeds  without_bands   1245   252.04      4.36     44.25     72.41
speaker_01_clean_speaker_00_bleeds  with_bands       155    34.78      0.60      6.01      9.99
----------------------------------------------------------------------------------------------
TOTAL selected                                      9404  1681.83
  with_bands (>= 1 s black chunk in either speaker): 1020 conversations / 199.25 h
  default plan (tokens, without_bands only): 8384 conversations / 1482.58 h, ~25.6 GB on disk
```

Hours are the sum of `clip_duration_sec / 3600` (conversation hours; both speakers are in every conversation). `tokens_GB` is the
size of the six int16 `[32, T]` arrays on disk; `audio_GB` / `poses_GB` are the FLAC and `poses_25fps.npy` bytes from the index
(`metadata.json`, `timeline.npz`, `job.json` add ~1 MB per speaker). `both_clean` = the 6,648 conversations / 1,108.9 h of the
previous `pose_both_ok & crossbleed_side == "none"` subset. Run `--dry-run` yourself (3 s) to get the table for `--include-flagged`
or another `--band-min-seconds`.

## Options

| flag | meaning |
|---|---|
| `--out DIR` | output root (required) |
| `--dry-run` | fetch only `index/conversations.parquet` + `tools/shard_index.parquet`, print the table and the plan, write `manifest.csv` / `summary.json` |
| `--with-bands` | also download the `with_bands/` conversations |
| `--band-min-seconds S` | band threshold (default 1.0) |
| `--include-flagged` | select `pose_both_ok_or_flagged` instead of `pose_both_ok` (adds conversations whose pose did not pass the lip-sync alignment verdict) |
| `--what tokens\|tokens+audio\|all` | shortcut for `--audio` / `--audio --poses` |
| `--audio`, `--poses` | add the FLAC lanes / the v1 pose files from the bucket |
| `--limit N` | first N conversations per output leaf (`<folder>/<bands>`), for testing |
| `--workers N` | parallel shard downloads (default 8) |
| `--cache DIR`, `--keep-cache` | where shards land (default `<out>/.cache`); keep them instead of deleting after extraction |
| `--verify-existing` | re-hash FLAC / poses that are already on disk (default resume trusts a size match) |
| `--rescan` | rebuild the shard index by scanning the shards (only if `tools/shard_index.parquet` is stale; ~5 min) |
| `--token`, `--revision`, `--repo`, `--bucket`, `--progress` | plumbing |

## How it works / resume / verification

1. Downloads `index/conversations.parquet` (all scalar columns, 12,279 rows) and `tools/shard_index.parquet` (session, seg, T,
   `no_detection_intervals` of both speakers, parquet shard + row index for every row) — about 11 MB in total.
2. Applies the selection rule and the band rule, prints the table, applies `--with-bands` / `--limit`.
3. Downloads only the parquet shards that hold planned conversations (`--workers` in parallel), extracts the rows with pyarrow, writes
   the `.npy` / `.json` files, deletes the shard (unless `--keep-cache`). The whole subset touches 690 of the 720 shards (22.8 GB in transit).
4. Fetches the bucket files in batches with `HfApi.download_bucket_files` (`--audio` / `--poses`); each file is written to `*.part`,
   its size and sha256 are checked against the row (`speaker_0N_audio_sha256`, `speaker_0N_poses_sha256`), then renamed.
5. Verifies every planned conversation: six `int16 [32, T]` streams with the same `T` as the index, `frame0_pose` `[133, 3]`,
   `poses_25fps.npy` frames `== 2T`, all requested files present. Failures are listed at the end and in `summary.json`; exit code 1 if any.

**Resume**: a conversation is skipped when its `conversation.json` says the token stage completed and every file it lists is present
with the recorded size; bucket files are skipped when present with the expected size (`--verify-existing` re-hashes them). Adding
`--audio` / `--poses` / `--with-bands` to a finished run only fetches what is missing; re-running a finished run takes seconds.

## `conversation.json`

All scalar columns of the row (see "Row schema" in the dataset README for the full list): identity (`session_id`, `seg`,
`recording_id`, `account_uid_0/1`), the clip window (`clip_start_sec`, `clip_end_sec`, `clip_duration_sec`, `T`, `pose_frames`, cut
rules), the selection facts (`crossbleed_side`, `pose_both_ok`, `pose_both_ok_or_flagged`, `pose_sides`, `sam_available`, `sam_flags`),
and per speaker `speaker_0N_*`: pose status/reason/alignment and lip-sync metrics (`pose_status`, `pose_aligned`, `lip_sync_*`,
`face_visible_frac`, `pose_resolution`, `pose_rotation`, `pose_recipe`, `pose_tokenizer`), `no_detection_seconds`, `no_detection_n`,
SAM QC (`sam_qc` struct: SI-SDR own/cross, bleed reduction, solo energies, LUFS gain), crossbleed evidence (`crossbleed_basis`,
`acoustic_audible_bleed`, `acoustic_severity`, `transcript_bleed_tier`), `n_words`, and the provenance / integrity fields
(`audio_hf_path`, `audio_gcs_uri`, `audio_bytes`, `audio_sha256`, `poses_hf_path`, `poses_sha256`, `pose_dir_hf_path`, `mimi_sha256`,
`sam_isolated_sha256`, ...). The array columns are not repeated there; they live in the `.npy` / `.json` files next to it.
Two extra keys are added by the script: `_downloader` (script version, repo revision, source shard, output folder, band decision,
`tokens_complete`, errors) and `_files` (every written file with its byte size — used for resume).

## Test record (2026-10-29)

`--limit 2 --with-bands --what all` on 12 conversations (2 per leaf, 2.24 h, 1.1 GB) took 19 s; an independent check re-read the
source shards and confirmed every `.npy` equals the parquet array, FLAC and `poses_25fps.npy` sha256 equal the row columns,
`poses_25fps.npy` frames `== 2T`, every listed black band has all joints below 0.3 confidence, and the folder / band placement matches
the row. Re-running the same command resumed with nothing to download.
