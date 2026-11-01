Local copy of gs://aloe-processed-datasets/stage2-handover-20261008 (+ pose add-on from -eu bucket), fetched 2026-10-16 ~04:00 PT.
Layout/pair.json/manifests/SHA256SUMS written by Elias's download_handover_crossbleed_dataset.py, pose by download_handover_pose.py --include-flagged
(both run unchanged in logic; see /data/home/amartyavarma/data/aloe-redl/scripts). Differences from his default:
- NO FLAC audio (skipped by request; pair.json "audio_format" says flac but no audio files are present).
- ADDED per-speaker system prompts: files/<side>/<session>/<seg>/system_prompts/SPEAKER_0N/{chunk_KK,episode}.json
- _meta/ = canonical GCS _meta (manifest, crossbleed label jsonl, summary, verification).
- Not fetched: turn_taking*/, vad-only extras, poses_native.npz / timeline.npz / raw 25fps keypoints.
