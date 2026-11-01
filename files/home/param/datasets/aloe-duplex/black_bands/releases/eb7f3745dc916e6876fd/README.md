# Stage2 black-band annotations

These annotations use the supplied InternalCan/stage2-handover-pose-mimi-sam-v1
shard index: all 133 joints below confidence 0.3, at least 25 consecutive 25 Hz
ticks (1 second). They are inferred from pose confidence, not a new pixel scan.

`conversations.parquet` contains every source conversation, the original raw
intervals, source `shard` and `row_index`, and the new annotations.
`banded_conversations.parquet` is its positive subset.
`banded` reproduces the supplied script (null source lanes count as empty).

Each local store has `rows.black_bands.parquet`, an enriched copy of ALL its
original row columns in exactly the same order. Read it instead of rows.parquet
when you need the new metadata. Existing training loaders are not changed.
`pose/rows.black_bands.parquet` contains the rows selected by POSE_INDEX.json.

- `has_black_band`: true if either lane has a qualifying band overlapping the
  row; false only when both lane annotations are available and empty; null when
  the answer cannot be established. Never coerce null to false when filtering.
- `black_band_status`: band_detected, no_band_detected,
  pose_annotation_unavailable, or source_conversation_missing.
- `black_band_annotation_complete`: both source lane annotations are available.
  A positive row can still be incomplete for its other speaker.
- `speaker_00_black_band_*` / `speaker_01_black_band_*`: availability, flag,
  intervals_seconds, pose_tick_intervals, token_intervals, seconds and n.
- `assistant_has_black_band`, `user_has_black_band` and their interval columns
  follow the local store's explicit speaker mapping.

All new intervals are half-open [start,end), relative to the represented row.
For source conversations, zero is clip_start_sec in the recording. Local store
rows subtract frame_start / 12.5 seconds and clip at n_frames / 12.5. A source
band qualifies BEFORE clipping, so a row can contain less than one second of a
qualifying band at its boundary. Token indices are bounded by row length.
The 25 Hz grid is recovered before floor/ceil conversion to avoid float32
off-by-one errors. Pose ticks are NOT model token indices.

Missing source conversations and null pose annotations remain explicitly
unknown. `unmatched_rows.json` lists every local row absent from the source
index. Xabi V2 uses unrelated recording IDs and is outside this source index.

All original metadata, tokens, manifests, and active training behavior are
preserved. MANIFEST.json records source revision, SHA256 hashes, coverage and
per-store counts. BLACK_BANDS.json in the dataset root locates this release.

Example:

```python
import pyarrow.parquet as pq
rows = pq.read_table('/home/param/datasets/aloe-duplex/s2h-m1-samsep/stores/m1n0/rows.black_bands.parquet')
for row in rows.select(['row_key', 'has_black_band', 'black_band_status',
                        'assistant_black_band_token_intervals',
                        'user_black_band_token_intervals']).to_pylist():
    if row['has_black_band'] is True:
        print(row)
```
