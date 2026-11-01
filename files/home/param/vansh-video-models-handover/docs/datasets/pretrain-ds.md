# Pretrain clip source and dense dataset

On-disk layout the loader owns. Run keys (`root_dir`, `clipnames_root`,
`buckets`, `pose`) live in the experiment toml.

`dataloading/sources.py` owns manifest filtering, path construction,
cross-manifest deduplication, deterministic clip ordering, whole-clip
I/O/validation, and the one-entry worker-local source cache.
`dataloading/ds_dense.py` owns H/P window scheduling and train/eval batch
assembly. `BaseLatentDataset` and `PipelineDataLoader` contain neither concern.

```
<root_dir>/<bucket>/<subbucket>/<video_id[-2:]>/<clip_id>/
    <clip_id>.safetensors              latents        [48, T, H, W]
    <clip_id>_pose_latents.safetensors pose_latents   [48, T, 30, 16]
    <clip_id>_mimi.npy                 codes          [32, T_mimi] int32, 12.5 Hz
```

Manifest: `<clipnames_root>/<bucket>/<subbucket>.csv`. The usable length
`clip_len = latent_t - 1` determines the dense window stream. Video and pose
are curtailed to that length; files that cannot cover it fail when loaded.
Clips with fewer than 19 usable frames or exceeding the one-quantum audio
limit are omitted. Eligible unique clips are initially ordered with
`shuffle_seed=43`.

`-orig` / `-unorig` = original audio or not.

## Columns

| column | behavior |
| --- | --- |
| `paths` | clip dir relative to `root_dir` |
| `duration` | unread |
| `has_pose` / `pose_v2` | gate when `pose=True`; `pose_column` selects the manifest column. Missing column = all-pose |
| `has_mimi` | gate when `audio=True`; missing column = all-mimi |
| `latent_t` | source latent length; subtract one before constructing the deterministic window stream |

`pose_version` (toml `pose`) selects the file: unset → `_pose_latents.safetensors`, `'v2'` → `_pose_latents_v2.safetensors`. v1/v2 are the same latents, different processing rounds.

For source length `V = latent_t`, Mimi files hold `T_mimi = 2V−1` codes
(2× upsample, `drop=2`), or `2V−2` for ~4.5% of clips. Both cover `V−1`
complete latent frames, matching inference's `T_mimi // 2` cutoff. The source
drops the final video and pose latent before window scheduling and never
repeats audio codes. Fewer than `2·clip_len` real codes raises. The fixed
750-code table still has structural zero padding beyond the usable audio;
supervised frames never consume it.

## Dense samples

Each source pass draws `startup_window` uniformly per clip from `1, ..., P`:
the present-frame count of the clip's first sample. Frames
`1, ..., startup_window` form that sample, right-aligned with no history
(a full sample when `startup_window = P`). Starting at `1 + startup_window`,
consecutive full groups of `P` form the remaining samples. The final incomplete
tail is dropped. Present targets do not overlap within a pass; the random
`startup_window` changes the slot assignment across passes. There is no
draining sample.

Each sample is `[sink | history | present]`. History is the latest available
frames before the present group, right-aligned with zero padding. Mimi codes
remain per source clip.

Each loader epoch is one seeded source pass. Epoch and clip identity determine
the offset statelessly. Every clip contributes its minimum
`floor((clip_len−1)/P)` windows; when an offset would create one extra full
window, the last one is omitted. Each rank item is exactly
`micro_batch_size_per_gpu` windows. Complete DP and gradient-accumulation
groups define the epoch; only the final incomplete global group is omitted.
Epoch and microbatch position reconstruct the same samples after resume.

Each microbatch has a fixed B-row source-code table and `window_source`
mapping. Shared clips use one occupied row within a microbatch, unused rows
are padded, and clips crossing a boundary are duplicated into the next table.
The code tensor is always `[B, codebooks, 750]`; rows and time are never
silently extended to a second quantum.

Loss is normalized by each window's valid element count before the fixed B
windows are averaged, so startup and full windows have equal sample weight.
Full-window frames receive weight `1/P`; frames in an `r`-frame startup receive
weight `1/r`. Clips otherwise contribute in proportion to their number of
windows, hence approximately their length. Present slot statistics remain
separate.

In eval mode the loader returns each trimmed source clip without constructing
training windows. This lets evaluation construct multiple states from the same
source frames and reuse a consistent source-noise path.

## Buckets / subbuckets

`buckets` is bucket → list of subbuckets. Known buckets: `hands`,
`hands_good_framing`, `perfect_framing`, `face`. Subbuckets:

- orig: `both-orig`, `caption-orig`, `clean-orig`, `object-orig`, `parked-orig`
- unorig: `both-unorig`, `caption-unorig`, `clean-unorig`, `object-unorig`, `parked-unorig`

## Stages

Disjoint subbucket sets, applied across every bucket.

- stage 1: parked-orig, parked-unorig
- stage 2: both-orig, object-orig, both-unorig, object-unorig, clean-unorig, caption-unorig
- stage 3: clean-orig, caption-orig
