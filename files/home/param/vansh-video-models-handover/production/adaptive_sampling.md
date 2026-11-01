# Adaptive sampling

`adaptive_sampling.py` is `cached_universal.py`'s sampler with a background-colour drift detector that runs while sampling. While the detector reports drift, the history is corrupted with the training run's `HistoryCorruption` block. A model trained with it has learned not to copy a history that disagrees with the sink, so it falls back on the sink and pose instead of copying the drift forward. The generated latents are never corrupted.

There are two modes, `--mode rewrite` and `--mode commit`. They differ only in where the corruption goes.

## Windows and cache

- **Window *k*** is `[sink | history k−H..k−1 | present k..k+P−1]`, with present frame `k+j` at band `j`. Slot shapes are written oldest → newest, 1 = corrupted.
- **Each AR step samples one frame:** step `k` takes frame `k` from noise through bands `P−1 … 0` (10 Euler substeps each). At band `b` the frame is in window `k−b`, so window `k` is read by steps `k … k+P−1`.
- **Committing a frame:** a finished frame is committed with one pass at `t = 0` that attends only to the sink. This writes its K/V once for each of the `H` history slots it will occupy, in windows `k+1..k+H`. Each (frame, slot) entry is read by exactly one window.

## Trigger

1. **Decode:** each new frame is decoded with the Wan2.2 VAE one latent frame at a time with the decoder's causal cache (4 pixel frames per latent frame).
2. **Check:** when a frame's pixel frames cross a multiple of 25 (once per generated second), the newest pixel frame is compared with the sink.
   - **Person mask:** `ZhengPeng7/BiRefNet-portrait`, fp16, input resized to 1024², sigmoid > 0.5.
   - **Background:** pixels with no person pixel within a 7 px disk, in both the sink and the frame.
   - **Measure:** CIE76 ΔE between the region's mean Lab in the frame and in the sink (AutoBench M1, `color_bg`). Person ΔE and background texture are logged, not used.
3. **Rule:** active while ΔE > `--threshold` (default 1.0), with no hysteresis. A decision holds from the next step until the next check, so steps are active or not in runs of 6–7.

## Modes

**rewrite** (`adaptive_ab.py --arm rewrite4`; used on no-bb and mosaic):
- At the start of every active step `k`, one fresh `HistoryCorruption` draw on the clean history `k−H..k−1` corrupts all of its real slots.
- Only the entries for window `k` are overwritten (`H` single-frame commits); the frames' other slots are written to scratch.
- Windows are 1111 while active and 0000 right after. The cache for later windows stays clean, so switching off needs no cleanup.

**commit** (corrupt-and-cache, `adaptive_ab.py --arm cache`; used on cc):
- At an onset step, one draw `d` is made and held for the whole episode. Its scale is the per-channel std of the onset window's clean history.
- Every active step `k` commits frame `k−1` once more as `x + d` into all `H` of its entries (1 commit). None of them has been read yet, so this is the same as committing the frame corrupted. It then stays corrupted as it slides through history.
- Nothing already read is rewritten. Windows go 0001 → 0011 → 0111 → 1111 and after the episode 1110 → 1100 → 1000.
- `d`'s wall mask (colour) or mosaic envelope is computed from each frame's own clean latent and pose. The draw itself (direction, sign, amplitudes, mosaic field and envelope noise) is shared by every frame of the episode.

**Cost** (H100, compile and CUDA graph, 480×256): sampling is ~368 ms per frame, streaming decode 50 ms, and a check ~50 ms once per second. A corrupted step adds ~33 ms in rewrite (4 single-frame commits) and ~7–10 ms in commit (1 commit).

## Corruption block at inference

The block is `[history_corruption]` of `--toml`. The default is the final config, `experiments/universal/aloe-h4-complete-corruption.toml`. Pass the toml of the run the checkpoint was trained with. Amplitudes are in per-channel stds of the window's real history.

| key | at inference |
|---|---|
| `color_p`, `mosaic_p` | Rescaled to sum to 1, so every active step (rewrite) or episode (commit) gets exactly one of colour or mosaic. |
| `person_color`, `background_color` | Log-uniform colour amplitude over the whole frame, plus extra on the wall mask. |
| `directions` | Probabilities of the DARK, DRIFT or random colour direction, with a random sign. |
| `mosaic`, `mosaic_dc`, `mosaic_coverage` | Mosaic non-DC amplitude (log-uniform), DC as a fraction of it, and the covered top fraction (uniform) of the pose-hull background. |
| `mask_tau`, `mask_tau_pose` | Wall-mask thresholds: distance from the top-corner wall colour, and the pose skeleton. |
| `slots` | Dropped. It trains the partial windows that commit produces by placement; rewrite always corrupts every real slot. |
| `per_slot_mask` | rewrite: as trained (true: each slot's mask from its own frame; false: from the newest). commit: always per frame, which is what `true` trains. |

Older checkerboard blocks (with `grid`) also work: `p` is set to 1, and `grid` / `grid_p` apply as trained.

Draws use the global torch RNG, seeded with `seed + clip`, never the noise generator.

## Running

```bash
cd production
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python adaptive_sampling.py CONFIG.yaml --mode rewrite --clip 7 --out /abs/out
CUDA_VISIBLE_DEVICES=1 ../.venv/bin/python adaptive_sampling.py CONFIG.yaml --mode commit --clip 7 --out /abs/out2
```

**Config:** `CONFIG.yaml` is a yaml like `cached_universal.yaml`.
- **Audio guidance** must be off: `audio_guidance_scale: 1.0` or `audio_guidance_steps: 0`.
- **Paths:** `checkpoint_path` and the VAE resolve relative to the yaml's directory. `samples_dir` is used as is, so make it absolute.
- **Clip files:** it needs `samples_<n>_low_res.safetensors` (frame 0 is the sink), `samples_<n>_pose.safetensors` and `samples_<n>_mimi.npy`. `--manifest CSV` builds missing ones from the dataset root hard-coded in the script.

**Dependencies:** `kornia` and BiRefNet's dependencies come from `$ARBENCH_PKGS` (default `/tmp/arbench_pkgs`) if they're not in the venv. `$REPO` sets the repo it imports from.

**Flags:**
- `--mode rewrite|commit`, `--clip N`, `--out DIR`: required.
- `--toml PATH`: the corruption block.
- `--threshold T`: background ΔE, default 1.0.
- `--no-corrupt`: detect and log only.
- `--stop-frame F`: stop after `F` latent frames, writing latents and log but no mp4.

**Outputs in `--out`:**
- `sample_<n>_latents.safetensors`.
- `sample_<n>.mp4`, with audio.
- `sample_<n>_log.json`, which holds:
  - one row per check: `t, frame, bg_dE, person_dE, bg_tex, active`;
  - `onsets` / `offsets` as `[frame, seconds, check time]`;
  - `corrupted_steps` and `corrupted_count`;
  - commit only: `episodes` as `[onset frame, draw]`;
  - per-frame `timing` (`corrupt_ms, sample_ms, decode_ms, detect_ms`).

## Controls and reproducibility

- Noise for all frames is drawn up front from `seed + clip`, so it doesn't depend on corruption.
- `cached_universal.py` uses the first Triton config instead of the fastest, so sampling is bit-identical across processes and nodes. A plain `cached_universal.py` run (or `--no-corrupt`) with the same seed is the matched control. The cc commit run matched plain until the first onset on 16 of 16 clips.
