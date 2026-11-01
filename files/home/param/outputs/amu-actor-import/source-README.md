# Aloe conversational five-minute Muon / Neutrino encoding

Source corpus: 4,293.53 speaker-hours. After dropping every incomplete five-minute tail: **3,418.17 planned speaker-hours**, 10,188 recordings, 41,018 speaker chunks. The folder name identifies the original 4.3k-hour corpus; it is not a claim that 4.3k hours survive chunking.

**Production launched with user approval on October 1, 2026.** Only recordings with `done.json` are complete. The root `manifest.jsonl` lists completed recordings and is refreshed by the run controller. `manifests/planned.jsonl` is the work plan, not a completion manifest. `manifests/sources.jsonl` pins exact GCS object generations and checksums for reproducibility.

## Layout

```
manifest.jsonl
manifests/planned.jsonl
manifests/sources.jsonl
recordings/<recording_id>/
  recipe.json
  sources.json
  chunks.json
  speaker_1/chunks/000000/
    video_latents.safetensors
    pose_latents.safetensors
    poses.safetensors
    pose_tokens.safetensors
    mimi.safetensors
    words.json
    chunk.json
  speaker_2/chunks/000000/...
  done.json
```

Muon is the original undistilled Wan 2.2 video encoder. Neutrino is the distilled Wan 2.2 pose encoder. Both encode 256×480 canvases at 25 fps. Videos retain the existing full-frame crop geometry. Video and pose latents are BF16 `[1876,48,30,16]` in T,C,H,W order; raw input is 7,500 frames, with one endpoint frame repeated only inside the VAE. Pose and Mimi codes are separate int16 `[32,3750]` arrays from complete 300-second windows. Pose normalization resets at segment frame zero and preserves confidence. Raw 133-point poses and normalization metadata are retained. Words preserve source absolute timestamps plus chunk-local offsets; ASR is not rerun.

Source archives remain in GCS; this delivery contains encodings and provenance, not another copy of the full source videos. Read model revisions, hashes, source generations, numerical options and exact segment coverage in each recording's metadata. Completion is uploaded last. No consumer should treat an incomplete directory or the planned manifest as completed training data.


## Verified canary and launch state

The full work manifest is being published to the shared queue. The approved
fleet target is 64 Spot L4s in europe-west4, alongside the source archives. Preview files
under `previews/0076c046f94a72009f78b66941d4468aceb72a61/` reconstruct Muon and
Neutrino latents and Mimi audio from the saved artifacts. They are review aids,
not additional training chunks.

Measured warm cloud chunk turnaround is 3.525× realtime (85.10 s per 300 s).
This excludes startup and HF publication; the >5× target has not been achieved.
The full cold canary, including compilation and all-artifact readback, took
743 s for 1,200 retained speaker-seconds. Fleet throughput is not yet measured.
