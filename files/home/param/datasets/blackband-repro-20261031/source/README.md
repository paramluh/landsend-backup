---
license: other
pretty_name: stage2-handover — v1 pose + mimi-stream-v5 + SAM-isolated tokens, one row per conversation
language: [en]
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*.parquet
- config_name: index
  data_files:
  - split: train
    path: index/conversations.parquet
---
# stage2-handover-20261008 — pose v1 + Mimi + SAM-isolated tokens (private)

**12,279 two-person conversations, 2139.4 h, ONE ROW PER CONVERSATION** (both speakers in the row as `speaker_00_*` /
`speaker_01_*` column groups). Pushed 2026-10-29 10:22 UTC by `aloe-audio-pipeline/runs/2026-10-29-stage2-pose-mimi-sam-hf/`.

| where | what |
|---|---|
| this repo, `data/*.parquet` (720 shards, 22.8 GB) | one row per conversation: scalars + `pose_tokens` / `mimi_tokens` / `mimi_sam_isolated_tokens` (`int16 [32, T]` as `list<list<int16>>`), `frame0_pose`, `global_state0`, `no_detection_intervals`, `words`, SAM QC, links |
| this repo, `index/conversations.parquet` | the same rows **without** the array/word columns (fast filtering), + `speaker_0N_audio_verified`, `complete` |
| this repo, `index/files.parquet` | every large file in the bucket folder: `path, bytes, sha256, crc32c, gcs_uri` (sha256 of the bytes as downloaded from GCS and uploaded) |
| this repo, `index/verification.json`, `index/failures.jsonl`, `index/stats.json`, `index/index_summary.json` | what was checked, what failed, counts |
| bucket folder `hf://buckets/InternalCan/aloe-conversationa-data/stage2-handover-pose-mimi-sam-v1/<session>/<seg>/` (117,434 files, 966.1 GB) | `SPEAKER_00.flac`, `SPEAKER_01.flac` (the raw 48 kHz stereo PCM16 lanes of the cut window, byte-identical to GCS `wlt/`), and per speaker with pose `SPEAKER_0N/poses_25fps.npy` (`float32 [2T, 133, 3]` = y, x, conf in original pixels), `SPEAKER_0N/metadata.json`, `SPEAKER_0N/timeline.npz`, `SPEAKER_0N/job.json` — the v1 pose outputs 1:1 |

## Which conversations
Source population: the 2026-10-08 stage-two handover dataset `gs://aloe-processed-datasets/stage2-handover-20261008/` (12,685 manifest
conversations; cut v3, head-trim/duration/misalignment/no-face purges already applied). **A conversation is a row when at least one speaker has
v1 pose status `ok` or `flagged`**; the other speaker may be `error` (its `speaker_0N_pose_*` columns are null, `pose_status = "error"` +
`pose_reason`). Conversations where both sides are pose `error` (402, 57.58 h) and the 4 rows
without Mimi tokens are not included.

| category | rows | hours |
|---|---|---|
| both speakers pose `ok` (`pose_both_ok`) | 9,645 | 1735.5 |
| both speakers have pose, at least one `flagged` | 1,295 | 180.8 |
| one-sided pose (other side `error`) | 1,339 | 223.1 |
| **the colleague's previous subset: `pose_both_ok and crossbleed_side == "none"`** | 6,648 | 1108.9 |

Every selected conversation completed (0 failures).

## Row schema (per conversation; `N` = `00` / `01` = lane 0 / lane 1 of the recording)
Scalars: `session_id`, `seg`, `recording_id` (`<session>:<segment>`), `language`, `clip_start_sec`, `clip_end_sec`, `clip_duration_sec` (window in the source
recording; everything below is clip-relative, time 0 = `clip_start_sec`), `T` (Mimi frames at 12.5 Hz; **equal for both lanes, for the SAM-isolated
streams and for the pose tokens; `poses_25fps.npy` has exactly `2T` ticks**), `pose_frames` (= 2T), `cut_version`, `start_rule`, `end_rule`, `cut_family`,
`cut_metric`, `cut_confidence`, `cut_at_sec`, `account_uid_0/1` (app account ids; no names anywhere), `crossbleed_side`, `crossbleed_label_status`,
`pose_sides` (`both|SPEAKER_00|SPEAKER_01`), `pose_both_ok`, `pose_both_ok_or_flagged`, `sam_available`, `sam_flags`, `sam_n_chunks`, `sam_n_ambiguous_chunks`,
`sam_n_swapped_chunks`, `sam_mix_identity_swapped`, `sam_mix_identity_straight_db`, `sam_mix_identity_crossed_db`, `sam_true_len_samples_24k`,
`sam_tail_pad_samples_48k`, `hf_folder`, `gcs_wlt_dir`.

Per speaker `speaker_0N_…`:
- `account_uid`
- **pose** (null when `pose_status == "error"`): `pose_status` (`ok|flagged|error`), `pose_reason`, `pose_tokens` (`int16 [32, T]`, Xabi's
  `InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0`, one tokenizer call over the whole cut), `frame0_pose` (`[133, 3]`, first tick, pixels),
  `global_state0` (`[3]` = first row of `global_state.npy`), `pose_aligned`, `pose_mapping_offset_ms`, `pose_mapping_drift_ppm`,
  `pose_mapping_max_residual_ms`, `pose_tick_to_frame_start_ms_max`, `pose_edge_held_ticks_start/end`, `lip_sync_best_lag_ms`,
  `lip_sync_corr_at_best`, `lip_sync_corr_at_zero`, `lip_sync_conclusive`, `lip_sync_frames_used`, `face_visible_frac`
  (= `lip_sync.frames_used / 2T`), `pose_resolution`, `pose_rotation`, `pose_recipe`, `pose_tokenizer`,
  `no_detection_intervals`, `no_detection_seconds`, `no_detection_n` (see below), `poses_hf_path`, `poses_gcs_uri`, `poses_bytes`, `poses_sha256`, `pose_dir_hf_path`
- **audio / tokens**: `mimi_tokens` (`int16 [32, T]`, `mimi-stream-v5` of the raw lane), `mimi_gcs_uri`, `mimi_sha256`, `mimi_sam_isolated_tokens`
  (`int16 [32, T]`, `mimi-stream-v5` of the SAM-isolated stem of this lane; null if `sam_available` is false), `sam_isolated_gcs_uri`,
  `sam_isolated_sha256`, `sam_qc` (struct: `si_sdr_own_db`, `si_sdr_cross_db`, `bleed_reduction_db`, `other_solo_energy_raw_db`,
  `other_solo_energy_iso_db`, `own_solo_energy_raw_db`, `own_solo_energy_iso_db`, `own_solo_other_speaker_sec`, `lufs_gain_db`, `n_swapped_chunks`),
  `audio_hf_path`, `audio_gcs_uri`, `audio_bytes`, `audio_sha256`, `audio_samples_per_channel`
- **words**: `words` = `list<struct{word, start, end}>` of this speaker (clip-relative seconds, from `wlt/<s>/<seg>/words.json`, speaker = the lane the
  word was transcribed on), `n_words`
- **crossbleed**: `crossbleed_basis` (`none|acoustic|transcript|acoustic+transcript|unknown`), `acoustic_audible_bleed`, `acoustic_severity`, `transcript_bleed_tier`

### Pose status semantics (v1 = `gs://aloe-processed-datasets-eu/stage2-handover-20261008/pose-tiktok-recipe-v1/`)
- `ok`: v1 `done.json` marker — mapping gate passed, every tick covered, pose frames == 2 × Mimi T, lip-sync conclusive within tolerance.
- `flagged`: v1 `flagged.json` — all files exist and are shipped, but the track did not pass the alignment verdict (`pose_aligned = false`; typical
  reason `lip_sync_best_lag_ms` beyond ±1 pose tick, see `pose_reason` and `metadata.json["checks"]`). Use at your own risk / filter with `pose_both_ok`.
- `error`: v1 `error.json` — no pose; `pose_reason` carries the v1 error (2,079 × "video ends before the audio cut", 56 × "video starts after the
  audio cut", 8 × download errors, 3 × shoulder width below minimum) or "v1 marker done but objects missing" (5 tracks with incomplete v1 uploads).

### `no_detection_intervals` (black bands)
Ticks (25 fps, clip-relative) where **all 133 joints in v1 `poses_25fps.npy` have conf < 0.3**, as `[start_s, end_s]` pairs (end exclusive), plus
`no_detection_seconds` and `no_detection_n`. These are the ~6 s chunks of the source video that were never uploaded by the app (the aligned
render paints them black); the detector returns no person there. **In `pose_tokens` v1 holds the last confident pose through these
gaps** (the One-Euro smoother's confidence hold, `min_conf 0.3`), so the token stream is continuous but not informative inside the intervals —
mask them (or drop the conversation) for training.

### Recipes
- **Pose v1** (`tiktok-stage3-rtmw-v2frame0-onepass-v1`, run 2026-10-16, `~/generation_speaking/runs/2026-10-16-handover-pose-pretrain-recipe-pilot/`):
  audio clock mapped onto the source-video clock (cross-correlation probes, linear fit); pose tick k = clip_start + k/25 s, 2 ticks per Mimi frame,
  source frame on screen at that moment; RTMW (TensorRT FP16, full-frame box, `stream_nvdec_decode` every 2nd frame at 0.5 scale, ×2 scale-back,
  `interpolate_poses`, `smooth_dense_poses(1.0, 0.05)`); tokens = Xabi's `preprocess_frame0` + confidence features, tokenizer
  `InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0`, TF32 off. `metadata.json` carries the checks, lip-sync table and sha256 of the arrays.
- **mimi-stream-v5** (`CONVENTIONS.md` §5a): `kyutai/mimi@89091b3e`, 32 codebooks, 12.5 Hz, `T = ceil(samples_24k / 1920)`; 24 kHz mono,
  pure-gain -23 LUFS (BS.1770-4, peak 0.995), zero-pad to a frame multiple, fixed frame-aligned 30 s stateful streaming calls; L4-made.
  `mimi_tokens` are byte-identical to GCS `mimi-stream-v5/<s>/<seg>/w00.SPEAKER_0N.codes.npy` (sha256 checked against the Oct-26 verified mirror
  `hf://buckets/InternalCan/aloe-conversationa-data/stage2-handover-20261008-conversations-v1/`).
- **SAM isolated stems** (run 2026-10-23, `aloe-audio-pipeline/runs/2026-10-23-stage2-sam-separation/`): each lane alone through the SAM
  separation stack (`facebook/sam-audio-large` + `InternalCan/sam-distilled-gate-frozen9000-baseline`, bf16, euler-8, target_residual pairing,
  words.json-derived reference diarization) → 24 kHz stem with the partner's cross-bleed removed → `mimi-stream-v5` on H100 (hardware pin waived).
  Only the `*.isolated` streams are here (no `mix.*`; those are in `InternalCan/stage2-sam-sep-v1-mimi`). QC per stem in `sam_qc`; pair flags in `sam_flags`.
- **Crossbleed side** (`crossbleed_side ∈ none|SPEAKER_00|SPEAKER_01|both|unknown`): the handover downloader's rule — a side is confirmed when its
  acoustic `audible_bleed` is true (`_meta/track_bleed_metadata_v2.jsonl`) OR its transcript tier is `bleed` (`_meta/transcript_bleed_events_v3.jsonl`);
  missing/invalid labels → `unknown`.

## Reading
```python
import numpy as np, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfApi
repo = "InternalCan/stage2-handover-pose-mimi-sam-v1"
idx = pq.read_table(hf_hub_download(repo, "index/conversations.parquet", repo_type="dataset")).to_pandas()
sub = idx[idx.pose_both_ok & (idx.crossbleed_side == "none")]          # the previous colleague subset
# full rows (tokens, words) live in data/*.parquet; datasets.load_dataset(repo, split="train", streaming=True) yields them too
t = pq.read_table(hf_hub_download(repo, "data/" + sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith("data/"))[0], repo_type="dataset"))
row = t.slice(0, 1).to_pylist()[0]
pose = np.array(row["speaker_00_pose_tokens"], dtype=np.int16)         # [32, T]
mimi = np.array(row["speaker_00_mimi_tokens"], dtype=np.int16)         # [32, T]
assert pose.shape == mimi.shape == (32, row["T"])
# large files: hf buckets  (pip install -U "huggingface_hub>=2.2")
api = HfApi(); api.download_bucket_files("InternalCan/aloe-conversationa-data",
    [(row["speaker_00_poses_hf_path"], "poses.npy"), (row["speaker_00_audio_hf_path"], "lane0.flac")])
poses = np.load("poses.npy")                                            # float32 [2T, 133, 3] (y, x, conf)
```
`hf buckets sync hf://buckets/InternalCan/aloe-conversationa-data/stage2-handover-pose-mimi-sam-v1/<session>/<seg> ./x` fetches one conversation's files.

## Verification (details in `index/verification.json`)
- every file: crc32c checked against the GCS listing on download; sha256 recorded; Mimi codes sha256 == the Oct-26 verified mirror; `poses_25fps.npy`
  and `pose_tokens.npy` sha256 == v1 `metadata.json["hashes"]`; FLAC sha256 == Oct-26 mirror, STREAMINFO (48 kHz / 2 ch / 16 bit / pinned sample count).
- every row: `T` equal across both lanes' Mimi codes, both SAM-isolated streams, pose tokens; `poses_25fps` frames == 2T; `global_state` frames == 2T;
  metadata `mimi_T == T`; codes in [0, 2048).
- HF bucket: size verified after every batch; final listing == plan: 117,434 files / 966.1 GB, missing 0,
  size mismatch 0, extra 0; read-back of 120 random files: 120/120 sha256-identical.
- dataset repo: every parquet shard was re-downloaded after its commit on the VM that made it (sha256 equal, row invariants re-checked:
  all 720 ok); 12,279 rows, 0 duplicate keys, keys == pose ledger: True.

## Provenance
- manifest `gs://aloe-processed-datasets/stage2-handover-20261008/_meta/manifest.jsonl`; FLAC/words `.../wlt/`; Mimi `.../mimi-stream-v5/`;
  SAM `.../mimi-stream-v5-samsep/` + `.../sam-sep-v1/<s>/<seg>/sam_sep.json`; pose `gs://aloe-processed-datasets-eu/stage2-handover-20261008/pose-tiktok-recipe-v1/`.
- run folder `aloe-audio-pipeline/runs/2026-10-29-stage2-pose-mimi-sam-hf/` (`spec.md`, `status.md`, `index/`, `tools/`); fleet: 60 × n2-standard-32
  europe-west2 (pose + rows) + 30 × n2-standard-32 us-east4 (FLAC).
- Nothing in GCS was modified. This README is written last (completion sentinel).

## Download (ready-made script)
`tools/download_stage2_pose_mimi_sam.py` pulls the "both speakers have pose + at least one clean lane" subset
(`pose_both_ok` and `crossbleed_side in {none, SPEAKER_00, SPEAKER_01}`: 9,404 conversations / 1,681.8 h) into a plain folder tree —
`both_clean/`, `speaker_00_clean_speaker_01_bleeds/`, `speaker_01_clean_speaker_00_bleeds/`, each split into `without_bands/` / `with_bands/`
(black missing-upload chunks >= 1 s), one `<session>/<seg>/` directory per conversation with `.npy` token arrays, words, `black_bands.json`,
and optionally the FLAC lanes and v1 pose files from the bucket (sha256-verified). Resumable; `--dry-run` prints the hours table.
```bash
pip install -U "huggingface_hub[hf_xet]>=2.2" pyarrow numpy && export HF_TOKEN=hf_...
hf download InternalCan/stage2-handover-pose-mimi-sam-v1 tools/download_stage2_pose_mimi_sam.py --repo-type dataset --local-dir .
python tools/download_stage2_pose_mimi_sam.py --out ./s2 --dry-run            # table only
python tools/download_stage2_pose_mimi_sam.py --out ./s2 --with-bands --what all
```
Details (layout, selection rule, band definition, hours table, `conversation.json` fields, token index -> seconds): [`tools/README_download.md`](tools/README_download.md).
`tools/shard_index.parquet` (session, seg, T, `no_detection_intervals`, shard, row index per row) is what the script uses for the exact band split and shard lookup.
