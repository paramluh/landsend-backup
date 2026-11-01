---
license: other
pretty_name: Rosie Stamp conversations v1 (Mimi tokens, word timestamps, crossbleed labels, FLAC lanes)
language: [en]
configs:
- config_name: default
  data_files:
  - split: train
    path: tokens/*.parquet
- config_name: index
  data_files:
  - split: train
    path: index/conversations.parquet
---
# Rosie Stamp conversations v1 (private)

**216 two-person conversations, 35.0 h, both speakers.** Everything processed for Rosie Stamp's recordings, mirrored from GCS so it can be
fetched from Hugging Face only. In every conversation **Rosie Stamp is the ASSISTANT** and the person she talks to is the **USER**:
files / JSON labels for her lane are named `Rosie Stamp` (`Rosie_Stamp.*`), the partner's lane is `user` (`user.*`). The dataset's own lane ids
(`SPEAKER_00` / `SPEAKER_01`) are kept in every row (`rosie_lane`, `user_lane`) and inside the JSON documents (`lane` fields).

| set | source dataset (GCS) | conversations | hours | SAM-separated streams |
|---|---|---|---|---|
| A | `stage2-handover-20261008` | 123 | 21.7 | yes (SAM separation run 2026-10-23: `sam_mix` + `sam_isolated` tokens and stems) |
| B | `stage2-handover-20261029-rosie` | 93 | 13.3 | **no** (original tokens + words/vad + crossbleed labels only) |

Status: phase 1 (tokens + labels) committed 2026-10-30T05:59:16Z (5 parquet files, 292.2 MB). Phase 2 (audio tars): **complete**: 8 tars, 11.52 GB, 924 FLAC members, committed 2026-10-30T06:02:12Z.

## Quick start (HF only)
```bash
pip install -U "huggingface_hub[hf_xet]" pyarrow numpy
hf auth login                      # a READ token of a member of InternalCan, or export HF_TOKEN=hf_...
hf download InternalCan/rosie-stamp-conversations-v1 --repo-type dataset --local-dir ./rosie
python3 rosie/tools/download_rosie_hf.py --out ./rosie-data               # per-conversation folders from the parquet rows
python3 rosie/tools/download_rosie_hf.py --out ./rosie-data --with-audio  # + the FLAC lanes from audio/*.tar
```
`download_rosie_hf.py` writes `Rosie_Stamp/<session_id>/<seg>/` with exactly the file names of `tools/download_actor_tokens.py` (the GCS downloader):
`Rosie_Stamp.original.codes.npy`, `user.original.codes.npy`, `Rosie_Stamp.sam_mix.codes.npy`, `user.sam_mix.codes.npy`, `Rosie_Stamp.sam_isolated.codes.npy`,
`user.sam_isolated.codes.npy` (set A only), `words.json`, `vad.json`, `crossbleed.json`, `conversation.json`, `_gcs/` (untouched source JSONs:
`w00.mimi.json`, `words.json`, `vad.json`, + `w00.samsep.mimi.json`, `sam_sep.json` for set A) and, with
`--with-audio`, `Rosie_Stamp.original.flac`, `user.original.flac` (+ `*.sam_mix.flac`, `*.sam_isolated.flac` for set A).

## Files
| path | what |
|---|---|
| `tokens/rosie-stamp-NNNN.parquet` (5 files, ~50 conversations each, (session_id, seg) order, zstd) | one row per conversation. Scalars: `session_id, seg, source_dataset (A|B), gcs_root, recorded_date, clip_start_sec, clip_end_sec, clip_duration_sec, mimi_T, rosie_lane, user_lane, partner_name, has_sam`. Token streams as **the exact bytes of the GCS `.npy` file** (`np.load(io.BytesIO(b))` -> `int16 [32, T]`): `rosie_original_codes, user_original_codes, rosie_sam_mix_codes, user_sam_mix_codes, rosie_sam_isolated_codes, user_sam_isolated_codes` (the four SAM columns are null for set B). JSON strings: `words_json, vad_json, crossbleed_json, conversation_json` (relabelled by role) and the untouched source documents `original_mimi_json` (mimi-stream-v5 sentinel), `samsep_mimi_json`, `sam_sep_json` (null for set B), `source_words_json`, `source_vad_json` (the dataset's own `wlt/.../words.json`, `vad.json` with SPEAKER_0x labels) |
| `index/conversations.parquet` | all scalar columns + `token_shard`/`token_row` (where the row lives), `audio_tar`, crossbleed status/class/tier per role, pose status, `samsep_flags`, `n_words`, `source_files_json` (GCS object, size, crc32c of every source file) |
| `audio/rosie-stamp-NNNNN.tar` | phase 2: uncompressed PAX tars (~1.5 GB, whole conversations). Members `Rosie_Stamp/<session_id>/<seg>/Rosie_Stamp.original.flac`, `user.original.flac` (48 kHz lanes, clip-cut) and for set A `Rosie_Stamp.sam_mix.flac`, `user.sam_mix.flac`, `Rosie_Stamp.sam_isolated.flac`, `user.sam_isolated.flac` (24 kHz SAM stems) |
| `index/audio_members.parquet` | one row per FLAC: `member, tar, offset, size, crc32c` (offset = byte offset of the member data inside the tar -> HTTP range reads) |
| `tools/download_rosie_hf.py` | unpacker (see quick start) |

## Reading a row
```python
import io, json, numpy as np, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
idx = pq.read_table(hf_hub_download("InternalCan/rosie-stamp-conversations-v1", "index/conversations.parquet", repo_type="dataset")).to_pandas()
r = idx.iloc[0]
tbl = pq.read_table(hf_hub_download("InternalCan/rosie-stamp-conversations-v1", r.token_shard, repo_type="dataset"))
row = tbl.slice(int(r.token_row), 1).to_pylist()[0]
rosie = np.load(io.BytesIO(row["rosie_original_codes"]))     # int16 [32, T], 12.5 tokens/s (1920 samples @ 24 kHz per token)
user = np.load(io.BytesIO(row["user_original_codes"]))       # same T, same clock (token 0 = clip start)
words = json.loads(row["words_json"])["words"]               # each: text, start, end (s from clip start), speaker "Rosie Stamp"|"user", role, lane
if row["has_sam"]:
    rosie_iso = np.load(io.BytesIO(row["rosie_sam_isolated_codes"]))
```
`datasets.load_dataset("InternalCan/rosie-stamp-conversations-v1", split="train")` yields the same rows (binary columns as `bytes`).

## Token / label conventions
* Mimi: `mimi-stream-v5` (kyutai/mimi@89091b3e, 32 codebooks, 12.5 frames/s, `T = ceil(samples_24k / 1920)`), `int16 [32, T]`; all streams of a
  conversation share the same `T` (= `mimi_T`, checked against every `.npy` header). Original lanes were encoded on L4; set A's `sam_*` streams on H100
  (2026-10-23). `sam_mix` = lanes summed then SAM-separated into the two speakers; `sam_isolated` = SAM run on each lane alone (partner's bleed removed).
* `words_json`: word-level timestamps of both speakers, seconds from the clip start (token k covers `[k/12.5, (k+1)/12.5)` s); `vad_json`: voice-activity
  intervals per speaker. `crossbleed_json.summary[role]`: `acoustic_crossbleed` / `transcript_crossbleed` (true|false|null), `status`
  (clean | diagnostic_only | crossbleed | unknown), `acoustic_class`, `transcript_tier`, plus the full acoustic-bleed-v2.3 and transcript-bleed-v1.3 rows.
* `conversation_json`: roles + lanes, partner name, clip window on the source recording, T, file map, SAM QC (set A), pose status, provenance.
* Audio FLACs (phase 2) are the published lanes: `*.original.flac` = `wlt/<s>/<seg>/SPEAKER_0x.flac` (48 kHz, clip-cut); set A stems = `sam-sep-v1/<s>/<seg>/*.flac` (24 kHz).

## Provenance
Source of truth: `gs://aloe-processed-datasets/stage2-handover-20261008/` (set A) and `gs://aloe-processed-datasets/stage2-handover-20261029-rosie/` (set B);
row selection `aloe-audio-pipeline/tools/actor_tokens_manifest_20261029.csv` (actor == Rosie Stamp). Every source object was crc32c-verified against the GCS
listing when read and stored byte-for-byte (`index/conversations.parquet.source_files_json`). Produced by `aloe-audio-pipeline/tools/push_rosie_to_hf.py`
(run `aloe-audio-pipeline/runs/2026-10-29-rosie-filtered/hf/`), verified with `--verify` (random rows byte-compared with GCS).
