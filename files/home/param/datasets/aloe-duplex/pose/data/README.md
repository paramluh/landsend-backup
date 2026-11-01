# stage2-handover-20261008 colleague download

Generated: 2026-10-16T11:10:13Z

Command:

    download_handover_crossbleed_dataset.py --out /data/home/amartyavarma/data/aloe-handover-full/DATA --workers 64

Canonical source: `gs://aloe-processed-datasets/stage2-handover-20261008/_meta/manifest.jsonl`
Canonical manifest generation: `1792142093602330`
Canonical manifest SHA-256: `95b1a72feb680693c36d4d5f04b1d68e1934ce8022c5836f22698305cfb0b57d`

This folder contains one physical copy per downloaded conversation under
`files/{none,SPEAKER_00,SPEAKER_01,both,unknown}/<session>/<seg>/`.
The strict per-side flag is acoustic `audible_bleed == true` OR transcript
`tier == bleed`. Acoustic `detected`, acoustic `mirror`, and transcript
`suspect` remain diagnostics and do not set that strict flag. Missing or
invalid labels always route to `unknown`; they are never treated as clean.

Downloaded records: 12,681
Folders: {"SPEAKER_00": 1768, "SPEAKER_01": 1712, "both": 301, "none": 8885, "unknown": 15}
Planned payload for this invocation: 17.29 GiB
Audio format: flac (`--audio-format`; default is lossless FLAC).

`SPEAKER_00` is the lexicographically smaller actor id. `w00.pair.json`,
`manifest.jsonl`, and `manifest.csv` preserve both actor ids and account UIDs.
Mimi files are the locked `mimi-stream-v5` representation. Accepted-incomplete
rows are excluded unless `--include-incomplete`; no placeholder files are made.

Portable indexes live in `indexes/`; fail-closed omissions live in
`exceptions/`; `SHA256SUMS` covers every materialized file except itself.
