---
license: other
pretty_name: stage2-handover SAM separation v1 — mimi-stream-v5 tokens of the separated stems
language: [en]
configs:
- config_name: default
  data_files:
  - split: train
    path: tokens/*.parquet
- config_name: index
  data_files:
  - split: train
    path: index/pairs.parquet
---
# stage2-handover SAM separation v1 — Mimi token streams (private, tokens only)

**12,681 (session, seg) two-person conversations, 2196.9 h — 32 parquet files, 14.4 GB stored (25.5 GB of `.npy`).**
Token-only mirror of `gs://aloe-processed-datasets/stage2-handover-20261008/mimi-stream-v5-samsep/<session>/<seg>/` made by run
**`aloe-audio-pipeline/runs/2026-10-23-stage2-sam-separation`** (fleet run 2026-10-23, raw-lane token swap 2026-10-24; this mirror pushed
2026-10-24). Each conversation is two time-aligned lanes `SPEAKER_00` / `SPEAKER_01` (one person per lane, separate devices) that were run
through the SAM separation stack three times — lane 0 alone, lane 1 alone, and the summed mixture — and the resulting four 24 kHz stems were
tokenised with `mimi-stream-v5`. **No audio is in this repo**; the FLAC stems stay in GCS (`.../sam-sep-v1/<session>/<seg>/*.flac`).

## The four streams (one row per pair)
| parquet column | source file in GCS | what it is |
|---|---|---|
| `SPEAKER_00_isolated_npy` | `w00.SPEAKER_00.isolated.codes.npy` | the lane-0 person: lane 0 processed on its own with the other person's cross-bleed removed |
| `SPEAKER_01_isolated_npy` | `w00.SPEAKER_01.isolated.codes.npy` | the lane-1 person: lane 1 processed on its own with the cross-bleed removed |
| `mix_SPEAKER_00_npy` | `w00.mix.SPEAKER_00.codes.npy` | the summed mixture (lane 0 + lane 1, -23 LUFS) separated back into its two speakers — the stem labelled lane 0 |
| `mix_SPEAKER_01_npy` | `w00.mix.SPEAKER_01.codes.npy` | the mixture-separated stem labelled lane 1 |

Lane labels come from the dataset's per-lane word timestamps (`words.json`-derived reference diarization), so `SPEAKER_00` is the lane-0
person by construction; the mixture stems are checked against the lanes (`mix_identity_*` in the index: 0 track-level swaps in the final audit).

**NOT included: `w00.SPEAKER_00.raw.codes.npy` / `w00.SPEAKER_01.raw.codes.npy`** (the untouched raw lanes). Those files are
byte-identical copies of the dataset's original `mimi-stream-v5` tokens (`stage2-handover-20261008/mimi-stream-v5/<session>/<seg>/w00.SPEAKER_0x.codes.npy`,
L4-made), which already exist elsewhere — take the raw-lane tokens from there. They have the same `T` as the four streams here.

## Token format
`mimi-stream-v5` (locked recipe, `CONVENTIONS.md` §5a): `kyutai/mimi` revision `89091b3e466eb6a9d11e537bf26b144f194978f7`
(Transformers 5.17.0, torch 2.5.1+cu124), all 32 codebooks, **`int16 [32, T]`**, **12.5 frames/s** (1,920 samples per frame at 24 kHz),
**`T = ceil(samples / 1920)`**. Audio is decoded to 24 kHz mono, pure-gain normalised to -23 LUFS (BS.1770-4, peak 0.995), zero-padded to a
frame multiple after the gain, and encoded with fixed frame-aligned streaming calls of at most 30 s / 375 frames (`use_streaming=True`,
encoder KV cache + padding cache carried across calls). These four streams were **encoded on NVIDIA H100 with the recipe's L4 hardware pin
waived** (recorded per pair in `mimi_json["encoder"]["hardware_pin"]`; measured H100-vs-L4 agreement on identical audio: 90-93 % of all
codes, 99.8 % of codebook 0). All four streams of a pair have the same `T` (`equal_T`), equal to the raw-lane tokens' `T`.

## Files
| path | what |
|---|---|
| `tokens/mimi-stream-v5-samsep-NNNN.parquet` (32 files, 14.4 GB, zstd) | one row per pair, 400 pairs per file in (session, seg) order, row groups of 64. Columns: `session` (str), `seg` (str), `clip_duration_sec` (f64), `T` (i32), `equal_T` (bool), `SPEAKER_00_isolated_npy`, `SPEAKER_01_isolated_npy`, `mix_SPEAKER_00_npy`, `mix_SPEAKER_01_npy` (binary — **the exact bytes of the GCS `.npy` file**, `np.load(io.BytesIO(b))`), `mimi_json` (str — the pair's `w00.mimi.json` sentinel: per-stream T / LUFS / chunk schedule, full encoder metadata, flags) |
| `index/pairs.parquet` | one row per pair: `session, seg, clip_duration_sec, T, equal_T, token_shard, token_row` (where the pair's row lives), `true_len_samples_24k`, `tail_pad_samples_48k`, per-stem QC from `sam_sep.json` (isolated stems: `*_si_sdr_own_db`, `*_si_sdr_cross_db`, `*_bleed_reduction_db`; mixture stems: `*_si_sdr_vs_lane_db`, `*_si_sdr_vs_other_lane_db`, `*_si_sdr_vs_isolated_db`; `mix_identity_straight_db`, `mix_identity_crossed_db`, `mix_identity_swapped`), `n_chunks`, `n_swapped_chunks`, `n_ambiguous_chunks`, `sam_sep_flags`, `mimi_flags` (JSON lists), `chunk_consistency` and `qc_json` (full JSON), `bytes_*_npy` |
| `index/members.parquet` | one row per source file: `path` (object path below the dataset root), `shard`, `size`, `crc32c` (base64, as GCS reports it) |
| `index/shards.jsonl` | per parquet file: pairs, files, source bytes, stored bytes, sha256 |

QC in this set: 716 pairs with a non-empty `sam_sep_flags`, 716 with non-empty `mimi_flags`, 556 with >= 1
`ambiguous` stitcher chunk (left as stitched), 217 with >= 1 corrected (swapped) chunk. Separation is unseeded bf16 diffusion
(not bit-reproducible run to run); the tokens are what they are — re-encoding the same FLAC on an L4 would differ slightly (see above).

## Reading
```python
import io, json, numpy as np, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
STREAMS = ("SPEAKER_00.isolated", "SPEAKER_01.isolated", "mix.SPEAKER_00", "mix.SPEAKER_01")
idx = pq.read_table(hf_hub_download("InternalCan/stage2-sam-sep-v1-mimi", "index/pairs.parquet", repo_type="dataset")).to_pandas()
r = idx.iloc[0]                                                     # pick a pair (filter on the QC columns first if you like)
tbl = pq.read_table(hf_hub_download("InternalCan/stage2-sam-sep-v1-mimi", r.token_shard, repo_type="dataset"))
row = tbl.slice(int(r.token_row), 1).to_pylist()[0]
assert (row["session"], row["seg"]) == (r.session, r.seg)
codes = {s: np.load(io.BytesIO(row[s.replace(".", "_") + "_npy"])) for s in STREAMS}   # each int16 [32, T]
meta = json.loads(row["mimi_json"])                                 # == the pair's w00.mimi.json
```
`datasets.load_dataset("InternalCan/stage2-sam-sep-v1-mimi", split="train", streaming=True)` yields the same rows (binary columns as `bytes`).

## Provenance
* source of truth: `gs://aloe-processed-datasets/stage2-handover-20261008/mimi-stream-v5-samsep/<session>/<seg>/w00.<stream>.codes.npy`
  (+ `w00.mimi.json`), QC from `gs://aloe-processed-datasets/stage2-handover-20261008/sam-sep-v1/<session>/<seg>/sam_sep.json`,
  pair list `gs://aloe-processed-datasets/stage2-handover-20261008/_meta/manifest.jsonl` (complete pairs = those with a `w00.mimi.json` sentinel).
* every `.npy` was crc32c-checked against the GCS listing when read and stored byte-for-byte; `index/members.parquet` carries the GCS size/crc32c.
* run folder: `aloe-audio-pipeline/runs/2026-10-23-stage2-sam-separation/` (`status.md`, `evidence/audit.json`, `evidence/raw_lane_token_swap_report.json`);
  dataset-level notes in `gs://aloe-processed-datasets/stage2-handover-20261008/_meta/sam_sep_v1_README.md`.
* produced by `aloe-audio-pipeline/tools/push_sam_sep_v1_to_hf.py --tokens-only`; verified with `--verify` (random rows byte-compared with GCS).
