# aloe-handheld-expressive-v1

Every row of the Lounge and Talk handover run `aloe-lt-expressive-20261028`, the facially expressive subset
(`gs://aloe-processed-data/conversational-data/_runs/aloe-lt-expressive-20261028/_meta/handover.csv`): 228 clips, 37.08 h of usable
video, 64 people, all duplex (booth 144 clips / 21.58 h, home 84 / 15.50 h). One clip = one speaker's usable-video window:
**assistant** = that speaker (video + audio), **user** = the partner's audio over exactly the same time range. When both speakers of a
call had usable video, the same minutes are two clips (one per video side; 157 unique call windows, 71 with both sides). Clips are
20 s - 26.7 min. Same format as `aloe-duplex-handheld-v1` and `aloe-cropped-v1`, read by `dataloading/ds_stream.py`.

**Overlap with `aloe-duplex-handheld-v1`:** 138 of the 228 clips (22.09 h) are the same clips (same clip_id, call, speaker and window)
as in `aloe-duplex-handheld-v1`; the handover moved them out of `aloe-lt-20261028` (now 2,078 rows) into this run. The other 90
clips (14.99 h) are new. 58 of the 64 people are in `aloe-duplex-handheld-v1`, 52 in `aloe-cropped-v1` (5 of them have
`aloe-cropped-v1` eval clips). Training on both releases sees those 138 clips twice.

- `latents/<clip_id>.safetensors`: `latents` bf16 [T, 48, 30, 16] time-first from byte 232 (`latents_offset`), after the int64 scalars
  `n_frames`, `valid_frames`; Wan2.2 VAE at 256x480, 25 fps; T = 1 + frames / 4, latent 0 = frame 0, latent t = frames 4t-3 .. 4t; the last
  latent holds one padded frame.
- `assistant/{k:05d}.parquet`, `user/{k:05d}.parquet`, k = int(sha256(clip_id)[:8], 16) % 32, rows sorted by sha256, row groups of 64:
  `clip_id`, `codes` list<fixed_size_list<int16, 32>> time-first [n_codes, 32], 12.5 Hz (mimi-stream-v5, same length both sides,
  n_codes = frames / 2), `words` JSON `{"words": [{w, start, end}]}`, clip-relative seconds.
- `raw_audio/assistant/<clip_id>.flac`, `raw_audio/user/<clip_id>.flac`: FLAC mono 16-bit 48 kHz, the clip window, sample-aligned
  (1 Mimi frame = 3,840 samples = 2 video frames).
- `manifest.parquet`: one row per clip, sorted by sha256(clip_id): clip_id, location (booth / home), booth, person_uuid, person_name
  ("unknown" where unsure), session_id (= call id), duration, T, n_frames, valid_frames, n_codes, codes_shard, bytes, latents_offset,
  lufs_in, lufs_gain_db, crossbleed / crossbleed_human (assistant side: clean | trace; human clean | confirmed), partner_* (the
  user side; partner_crossbleed audible = the assistant is audible in the user audio), conv_unusable, src_* (call, window, start / end s).
  `bad_audio` = crossbleed audible or crossbleed_human confirmed (27 clips). `eval` False everywhere.
- `manifest_monoplex.parquet`, `manifest_duplex.parquet` (all clips), `shards.parquet` (sizes, sha256 of the code shards).
- manifest.parquet is uploaded last: its presence means the upload is complete.

## Hi-res latents (512x960)

`latents_512x960/<clip_id>.safetensors`: the same window's video at 512x960, from the data team's run aloe-lt-hires-20261029
(handover_hires_main.csv + handover_hires_expressive.csv), copied as they are: int64 scalars `n_frames`, `valid_frames`, then `latents`
bf16 [T, 48, 60, 32] time-first from byte 232; same Wan 2.2 VAE as the 256x480 `latents/`, encoded from the same 512x960 frames without the
downscale; same T and clock as `latents/` (latent 0 = frame 0, latent t = frames 4t-3 .. 4t). Every file was checked: bytes and sha256 =
the run's done.json, header shape [T, 48, 60, 32] with T = this release's T. manifest.parquet: `latents_512x960_bytes`, `latents_512x960_sha256`.
