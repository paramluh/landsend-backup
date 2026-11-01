---
pretty_name: Aloe Pose Speech VAD — corrected release v2
configs:
- config_name: default
  data_files:
  - split: train
    path: data-compact/train-*.parquet
---

# Aloe Pose Speech VAD — corrected release v2

One row per **recording, speaker and speech segment**, with both `pose_tokens`
and `mimi_tokens` in the same row. This is a lossless repack of
`gs://aloe-processed-data/pose-speech-vad/releases/v2/`, not another encoding run.
Do not consume as a complete delivery until `RUN_STATUS.json` says `completed`.
Expected: 762,028 rows, 25,350 speaker tracks, 12,675 recordings, 26 shards.
Both speakers are retained; the partner row is the synchronized listening track.
The `train` split is the entire release, not a preselected clean training subset.

```python
from datasets import load_dataset
rows = load_dataset("InternalCan/aloe-pose-speech-vad-v2", split="train",
                    streaming=True, token=True)
# Optional: select historically eligible floor-holder segments.
rows = rows.filter(lambda row: row["eligible"] and row["is_active_speaker"])
```

## Columns and clocks

- Key: `recording_id`, `speaker`, `segment_id`. `segment_index` is preserved.
- `pose_tokens` and `mimi_tokens`: nested integer arrays `[32, T]`, int16,
  codebook first, 12.5 steps/second. They are independently encoded per segment.
- `start`, `end`, `requested_start`, `requested_end`: original recording seconds.
  `start_frame`, `num_frames`: 25 fps pose clock. `sample_start`, `sample_end`:
  24 kHz audio clock. Token step 0 belongs to the segment start; no new offset,
  truncation, resampling, quantization, padding or normalization is introduced.
- `frame0_pose`: full 133-joint frame-zero pose; `global_state`: original
  normalized root/scale state per frame. The separate per-frame `confidence`
  column has been removed; confidence already represented in pose tokens and
  the confidence channel of `frame0_pose` remain unchanged.
  Preserve these when reconstructing poses; tokens alone do not retain the absolute
  image coordinate transform. Source pose metadata specifies dimensions/order.
- `words`: word/timestamp structs including source and local times and boundary
  flags. `words_json`, `words_metadata_json` retain the original detailed records.
- `active_speaker`, `is_active_speaker`: floor-holder attribution. A row may be
  the listening speaker. Never assume every row contains active speech.
- `eligible`, `exclusion_reasons`, `partner_eligible`, `transcript_status`,
  `audio_crossbleed_status`, `audio_crossbleed_result_uri`: inherited audit labels.
  This release is NOT a certification that every row is clean. Unknown screening
  is not a pass. Audio crossbleed remains an optional separate selection criterion.
- `audio_metadata_json`, `pose_metadata_json`, and modality-prefixed
  `*_segment_metadata_json`: original encoding metadata without merging conflicting
  fields. Original artifact SHA256 columns remain unchanged.
- `pose_source_uri`, `audio_source_uri`, and `*_source_generation`: pinned GCS inputs.

## Encoding versions

**Mimi:** `mimi-local-attention-v1`, checkpoint_sliding_window_v1: corrected causal
sliding-window mask, 250 transformer steps = 10 seconds per layer. The whole
segment is encoded in one call with state reset at its start. These are not
independent ten-second windows. Stacked layers may carry older information.

**Pose:** `confidence-hold-03-frame0-init-v2`. Confidence threshold 0.3; initialize
all joint XY positions from segment frame 0, update when reliable, otherwise hold
the last reliable XY. Confidence is preserved. **No hold duration cap**, no
One Euro filter, no interpolation. State resets at each original speech segment.
Frame-zero root and horizontal shoulder-width normalization; causal whole-segment
encoding using `InternalCan/pose-tokenizer-32cb-200k-noshift` revision
`1d6ff4f0de89408cb054da618e309d4334596252`.

## Validation and provenance

Each join validates unique segment IDs, all shared scalar clock/identity columns,
and exact membership in the release quality manifest. All retained columns are read back and checked for exact Arrow equality against
the previous HF revision; only the standalone `confidence` column is removed.
The original token repack also compared numeric value hashes; 32-codebook shapes and frame counts are checked.
`provenance/shards.jsonl` records shard counts, sizes and hashes.
`provenance/source-manifest.parquet` preserves the source release's audit mapping.
The upstream quality audit is `gs://aloe-processed-data/pose-speech-vad/manifests/clean-20261005-parquet-v1/`.
Rows sort by recording and speaker, retaining source segment order. No random
train/test split is imposed; split by recording/identity to avoid leakage.

Sharded Parquet configuration follows the [HF data files guidance](https://huggingface.co/docs/hub/datasets-data-files-configuration).

## Compact revision

The current 26 data shards replace the original 254 shards. Every column except
`confidence` is preserved, including int16 token values, frame0_pose, global_state,
words, metadata, ordering and audit labels. The GCS release remains unchanged.
`provenance/compaction-source.json` identifies the previous immutable HF revision.
