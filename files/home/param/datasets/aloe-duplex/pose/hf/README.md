---
pretty_name: Exact Aloe pose-duplex v5 training pose tracks
---

# Aloe pose-duplex v5 training pose data

Private, byte-preserving copy of the valid pose tracks referenced by job 30074's
original POSE_INDEX.json (the fine-tune producing export-498).

- 8,107 conversations: 7,949 train and 158 held out.
- 13,857 valid pose tracks: 13,578 train and 279 evaluation.
- The original index also retains 2,357 sides without valid pose. Their pose
  inputs/loss were absent in training; no replacement pose is invented here.
- Tokens are int16 [32, T] at 12.5 Hz, using
  InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0 and the original frame-0
  root/shoulder-width normalization. Whole source tracks are preserved; the
  original fine-tune uses first chunks (at most 7,500 frames).

`shards/*.tar` contain the original `files/none/<session>/<segment>/` paths,
pose-token files, and their pose JSON sidecars. Extract all shards into one
data directory. `indexes/FILES.jsonl` records every extracted file's SHA256 and
size; `indexes/SHARDS.json` records archive checksums. All pose-token SHA256s
were checked against the original training index before packaging.

`indexes/POSE_INDEX.original.json` and `provenance/` preserve original training
metadata. Their old absolute paths are provenance, not paths to execute. For
a local training index, update `data_root` to the extracted directory and
`stores` to the four local SAM-separated stores; preserve row IDs and digests.
The copied global manifest CSVs describe the broader source corpus; the exact
files shipped here are defined by FILES.jsonl and TRACKS.jsonl.

Companion HF datasets: `InternalCan/aloe-v2vox-speech-export253` (original row
stores, transcripts, and raw audio) and `InternalCan/stage2-sam-sep-v1-mimi`
at revision ecb72b2de2347bfbbc3b71b0236e2986b0c44572 (SAM-isolated audio tokens).
Raw audio and SAM audio tokens are not duplicated in this pose package.

All original-owner access was read-only copying. Verification, archive creation,
and upload ran exclusively against the copy under the requesting user's folder.
