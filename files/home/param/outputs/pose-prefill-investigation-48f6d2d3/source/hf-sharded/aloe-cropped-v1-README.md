# aloe-cropped-v1

**2026-10-27: `audio/` was renamed `raw_audio/` (Vansh); the manifests' audio path columns say `raw_audio/...`.**

**2026-10-27: `duplex_hiddens/`, `manifest_duplex_hiddens.parquet` and `shards_duplex_hiddens.parquet` were deleted (Vansh: wrong and old); duplex hidden states will be rebuilt.**

**2026-10-27: `hiddens/`, `manifest_hiddens.parquet` and `shards_hiddens.parquet` (the Orpheus hidden states, 360 tars, 694 GB) were deleted (Vansh): they were computed from the OLD codes and will be rebuilt from the new `assistant/` codes. The manifest columns `has_hiddens` / `orpheus_cb0_ce` refer to the deleted set. `duplex_hiddens/` is unaffected.**

34,671 clips, 1,888.3 h: the aloe face clips of `InternalCan/aloe-face-clips-hires-v1/encode_v1` (512x960 mp4, 25 fps,
120-300 s, face/hands-filtered crops; their README / LABELS_README / BLEED_README) with audio and bleed tier clean or
pending (88 clips without audio and the bleed / suspect tiers dropped: 1,301 clips, 74 h), Wan2.2 video latents at
256x480, Mimi codes, word transcripts and Orpheus hidden states. Per-clip release in the loader's format
(`dataloading/ds_stream.py`, like aloe-conv-crop-v0). Every selected clip encoded (0 failures).

- `latents/<clip_id>.safetensors`: `latents` bf16 [T, 48, 30, 16] time-first, from `latents_offset` (manifest), 46,080 B
  per latent frame. Muon = Wan2.2 VAE (Wan-AI/Wan2.2-TI2V-5B `Wan2.2_VAE.pth`), wan-encode fastenc backend, whole clip with
  carried causal caches. Pixels: wan-encode's graph `fps=25, crop 8:15 (identity here), scale=256:480:flags=bilinear`
  (CPU ffmpeg 6.1.1), the last native frame dropped, then the first 4k+1 frames: T = (n_frames - 1) / 4 + 1; latent 0 =
  frame 0, latent t >= 1 = frames 4t-3 .. 4t.
- `codes/{k:05d}.parquet`, k = int(sha256(clip_id)[:8], 16) % 32, rows sorted by sha256, row groups of 64: `clip_id`,
  `mimi_codes` list<fixed_size_list<int16, 32>> time-first [n_codes, 32], 12.5 Hz, row j = [0.08 j, 0.08 (j + 1)) s.
  Recipe aloe-mimi-local-v1 (= the conversation release's `mimi-local-attention-v1`, what Orpheus was trained on):
  kyutai/mimi @89091b3e, the clip's AAC decoded to 24 kHz mono (ffmpeg), integrated -23 LUFS gain (pyloudnorm, peak
  ceiling 0.995), ONE cold encode per clip with the checkpoint's 250-step attention window (transformers 5.15.1),
  n_codes = min(Mimi frames, 2T).
- `hiddens/{s:05d}.tar` (s = node * 10000 + gpu * 1000 + chunk of the encode run; PAX tar), one member
  `<clip_id>.safetensors` per clip: `hiddens` bf16 [n_codes + 2, 4096], the final-norm hidden states of Orpheus
  (vanshjjw/orpheus-cd2-long-mimi-local-step4900, code delay 2; `orpheus/README.md`) run teacher-forced over the clip's
  words (`start` = clip-local word start) and all 32 codebooks; row m + 1 produces Mimi frame m. `manifest_hiddens.parquet`:
  clip_id, hiddens_shard, hiddens_offset (byte offset of the tensor data in the tar), hiddens_rows; row r at
  hiddens_offset + 8192 r. Clips without a transcript (128 pending) have none, and the region / token+region booth tiers
  (3,496 clips) are left out of manifest_hiddens (their members stay in the tars): their words are early by ~80-240 ms
  (Orpheus CE minimal at word shift +1..+3 frames: token+region 1.618 at 0 vs 1.358 at +3, region 1.400 vs 1.239 at +1;
  the token tier is aligned, 1.175 at 0), so they train with no_hidden_tokens until their offsets are fixed.
- Words: the source bucket's `encode_v1/transcripts/<clip_id>.words.json` (clip-relative `w`, `start`, `end`), not copied.
- `manifest.parquet`: one row per clip, sorted by sha256(clip_id), the index into everything; `shards.parquet`: set (codes),
  shard, bytes, n_records, sha256. Columns: clip_id, tier, kind (app / booth / studio / unknown), location (home / booth: the source's location v3.1, 2026-10-24, from session metadata; location_v1 = the earlier label,
  location_evidence = how v3 decided), session_id, person, duration, bleed_tier, transcript_source, transcript_status, eval, eval_clip, T, n_frames,
  n_codes, mimi_n, codes_shard, bytes, latents_offset, lufs_in, lufs_gain_db, orpheus_cb0_ce (codebook-0 CE of Orpheus on
  the clip: codes + words sanity), has_hiddens, qa_* (below), qa_keep (v3 locations; qa_keep_v1 = on location_v1), src_* (provenance from the source manifest).
- Split: `eval` = all clips of 40 home + 40 booth held-out persons (booth tiers: the speaker's user uuid; else session +
  speaker), 274 clips / 15.6 h; `eval_clip` = one clip per held-out person (80,
  `dataloading/splits/aloe-cropped-eval-80.csv`).
- QA (1 sample per second at 256x480, no models): `qa_overexposed` / `qa_overexposed_p95` = mean / p95 over seconds of the
  fraction of pixels with a channel >= 250; `qa_cast` = chroma of the clip's mean CIE a*, b*; `qa_L_mean`; `qa_L_range` /
  `qa_drift_dE` = L* range / max dE76 from the first 5 s, on 5-s running medians of the per-second Lab means.
  `qa_keep`: booth (studio, grey background) clips always kept; home clips dropped above qa_overexposed_p95 0.0745 (home p97), qa_cast 33.58 (p99.5), qa_drift_dE 10.76 (p98), qa_L_range 14.91 (p99): 1,088 clips / 59.8 h dropped (overexposed 34.3 h, cast 6.3 h, drift 23.0 h, L_range 11.6 h, overlapping); train = eval False and qa_keep: 33,313 clips / 1,813.0 h.
- Orpheus check: codebook-0 CE flat over time (the codes are sliding-window) and minimal at word shift 0 for handover,
  hf_conv and token (booth) words; mean CE per source: booth_wlt_words 1.241 (9891 clips); handover_wlt_words 1.108 (18534 clips); hf_conv_chunk_words 1.019 (6118 clips).
- manifest.parquet is uploaded last: its presence means the upload is complete.

- 2026-10-24 relabel: location from the source's location v3.1 (8,601 clips home -> booth); qa_keep recomputed with the same
  cutoffs on the v3 home clips (1,048 clips / 57.2 h dropped). Train (eval False, qa_keep): 11,207 home clips / 621.9 h,
  22,146 booth clips / 1,193.7 h. location_v1 and qa_keep_v1 keep the earlier values.

- Note: `hiddens/` (+ manifest_hiddens / shards_hiddens) are the Orpheus (monoplex) hidden states, computed from the OLD codes (the removed
  `codes/`, aloe-mimi-local-v1 of the mp4 AAC) and the transcripts, not from the new assistant/ codes below. The PPM hidden states coming next
  will be called duplex hidden states.
- 2026-10-25: the old `codes/` (Mimi codes of the mp4's AAC track) removed and replaced by the audio below; manifest.parquet's codes
  columns (codes_shard, n_codes, mimi_n) describe the removed files and are unused.

## Audio (2026-10-25): assistant = the person on camera, user = the other side of the conversation

From the source bucket's `audio_v2` (`InternalCan/aloe-face-clips-hires-v1/audio_v2/README.md`; its `user_*` = our assistant, its `partner_*` =
our user), every clip `train_safe=yes` (the av-drift booth tiers token+region / region are not in it), lossless source cuts on the video clock.

- `raw_audio/assistant/<clip_id>.flac`, `raw_audio/user/<clip_id>.flac`: FLAC mono 16-bit at the source rate (booth 44.1 kHz, phone 48 kHz), the clip window.
- `assistant/{k:05d}.parquet`, `user/{k:05d}.parquet`, k = int(sha256(clip_id)[:8], 16) % 32, rows sorted by sha256, row groups of 64: `clip_id`,
  `codes` list<fixed_size_list<int16, 32>> time-first [n_codes, 32], 12.5 Hz, row j = [0.08 j, 0.08 (j + 1)) s (mimi-stream-v5: kyutai/mimi,
  -23 LUFS, stateful 30 s streaming calls; both speakers the same way, same length), `words` JSON `{"words": [{w, start, end[, conf]}]}`,
  clip-relative seconds.
- `manifest_monoplex.parquet` (27,269 clips, 1,493 h): every clip with assistant audio: clip_id, shard, n_codes, n_words, audio, sr. Enough for the
  video model (video + assistant codes + words).
- `manifest_duplex.parquet` (24,636 clips): the clips that also have a verified user track: clip_id, shard, n_codes, n_words_assistant,
  n_words_user, assistant_audio, user_audio, assistant_sr, user_sr. Not in it: 1,980 booth clips with no user track, 653 booth clips whose
  user-track offset is unverified.
- 7,402 of the 34,671 clips have no audio here: not in audio_v2 (train_safe=hold or failed its QC), or duplicate footage (1,949 phone
  clips whose window is also in this release as an hf_conv or booth clip; that copy is kept).
- `shards.parquet`: sizes and sha256 of the assistant / user shards (sets assistant, user). manifest_monoplex.parquet is uploaded last.

## Duplex hidden states (added 2026-10-25)

`duplex_hiddens/{s:05d}.tar` (100 PAX tars, 556 GB; s = node * 1000 + worker * 100 + chunk of the extraction run), one member
`<clip_id>.safetensors` per clip: `hiddens` bf16 [n_codes + 2, 5120], the raw final head.norm output of the PPM duplex LLM
vox17127 (HF InternalCan/aloe-ft-vox17127-nopersona, Qwen3-32B backbone) run teacher-forced over the clip as in its aloe
training: channel B (the model's own voice) = the person on camera (assistant/ words + mimi-stream-v5 codes, code delay 2),
channel A = the other side's raw audio (raw_audio/user FLAC -> 24 kHz -> Voxtral front -> vadapter x s), empty system prompt.
Row m + 1 conditions Mimi frame m (the Orpheus hiddens' layout). `manifest_duplex_hiddens.parquet`: clip_id, hiddens_shard,
hiddens_offset (byte offset of the tensor data in the tar), hiddens_rows; `shards_duplex_hiddens.parquet`: set, shard, bytes,
n_records, sha256. Clips: the 21,897 manifest_duplex clips in train-clean (2026-10-25 09:58) + 10 home eval clips
(21,907, 1,205.6 h). Check: the extraction reproduces the vox17127 per-clip text / cb0 CE of Leia's teacher-forced test.
Read with `[model.hidden] source = 'duplex_hiddens'`, `in_dim = 5120` (dataloading/ds_stream.py HIDDEN_DIMS).

## Hi-res latents: `latents_512x960/` (2026-10-29)

Wan 2.2 VAE latents at 512x960 for 16,675 clips (922 h), in long-v2's hi-res layout:
- **Clips:** the 900 h training selection plus the eval clips.
  - The training selection comes from `aloe-duplex-train.csv` (1,198 h). Per location, one percentile cutoff is applied to all four QA metrics (`qa_overexposed_p95`, `qa_cast`, `qa_drift_dE`, `qa_L_range`), with no per-person cap: booth 400 h (7,274 clips), home 500 h (9,007 clips). The list is `aloe-cropped-v1-hires-900h.csv` in the train lists folder.
  - The eval clips are 394 clips (22 h): the eval split (`eval`, 274 clips) plus the aloe eval lists (duplex-eval, cropped-eval-8/80, idling-40, duplex-handheld-eval, switch-1-eval) that are not in the selection.
- **Encoding:** the clip's source mp4 (`InternalCan/aloe-face-clips-hires-v1` `encode_v1/<dir>/<clip_id>.mp4`, 512x960, 25 fps), frames 0 .. n_frames-1 (the same frames as `latents/`). Decoded with ffmpeg to rgb24, scaled as x / 255 * 2 - 1, then encoded with the exact Wan 2.2 VAE (`Wan2.2_VAE.pth`) through wan-vae-collection's `FastMuonEncoder` (bf16, compiled, chunk 32). Latent mean / std are folded in, as in `latents/`.
- **Check:** for every clip, the same frames scaled to 256x480 (ffmpeg bilinear) and encoded the same way were compared with the clip's stored `latents/`:
  - first 9 latents for every clip, all T latents for every 8th clip (2,086 clips);
  - relative L2 median 0.0040, max 0.0052, worst single latent frame 0.0085 (bf16 rounding);
  - so the hi-res latent t is frame-aligned with `latents/` latent t for t = 0 .. T-1.
- `latents_512x960/{s:05d}.tar`: 608 shards (0-508: training selection, 32 clips each; 509-607: eval, 4 clips each), 3.83 TB. Each USTAR member (GNU long name if needed) is `<clip_id>.latents` = T frames x 184,320 B, frame = bf16 [48, 60, 32], time-first. These shards are not one-to-one with `latents/`, which stays per clip.
- `manifest_512x960.parquet`: the manifest rows of these clips in shard order, plus `latents_512x960_shard`, `latents_512x960_offset` (the member's data offset in its tar) and `crc_start_512x960`.
- `checksums_512x960.bin`: uint32 LE crc32 per 184,320-B frame, manifest_512x960 order.
- `shards_512x960.parquet`: set, shard, bytes, n_records, sha256.

Per-shard build records (offsets, crcs, per-clip check numbers) are in `_build/hires512/`.
