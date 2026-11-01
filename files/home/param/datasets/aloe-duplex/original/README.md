---
pretty_name: Aloe V2Vox speech export253 training data
configs: []
---

Private staging copy of the exact export253 first-chunk, crossbleed-none Aloe speech subset used for Landsend training: 8,067 training and 162 evaluation conversations, 16,458 FLAC tracks, and the four full original row-code stores (42 files).

`manifests/landsend-*.json` records disjoint six-node audio assignments, original paths, byte sizes, original MD5 checksums, GCS generations where available, training/evaluation splits, and row references. `manifests/STORES-INVENTORY.json` contains SHA256 checksums of every store file. Full stores preserve original row indices and are replicated to every node; only assigned manifest rows are selected for training. `provenance/` retains the source training index and raw-gain restoration metadata.

Audio paths and bytes are unchanged. Includes the existing 244 Mimi-decoded fallback tracks for source conversations purged after row construction. Transfers use existing authorized organization credentials. This repository is private and contains no credentials.
