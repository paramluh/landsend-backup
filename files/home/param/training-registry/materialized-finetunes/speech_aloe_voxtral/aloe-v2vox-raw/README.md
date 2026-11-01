# aloe-v2vox-raw (10-17)

Re-run of aloe-v2vox-ft (job 22420) with ONE change: no loudness normalisation of the channel-A audio going into Voxtral
(`VOX_A_GAIN=raw`: WLT FLAC at 0 dB, mimidec tracks x 10^(-recorded Mimi LUFS gain / 20); gains in `plan/RAWGAIN.json`).

- Code: `code/train_aloe_v2vox_raw.py`, `code/data_aloe_v2vox.py` (switch + `AccDS`); env `aloe_v2vox_raw.env`; sbatch `aloe_v2vox_raw.sbatch`.
- 22420's 64 ranks run as 64 logical ranks on 24 GPUs (3 micros per GPU per step, same rows per step); `ALOE_LNODES` from the node count.
- Config diff vs 22420: 15 keys, all run dirs / A gain / accumulation (checked at start, aborts otherwise).
- Channel-A tracks: shared copy `flac/` (238 GB, all 8 logical nodes); each node's `/scratch/amartyavarma/aloe-m1vox-ft/flac` is a symlink to it.
- Tests (`tests/`): raw waveform == raw FLAC samples (bit-exact, train + eval), mimidec x inverse gain, AccDS == 22420's per-rank micros, CPU config diff, loudness report.

## Node notes

- **slinky-4 GPUs 6 and 7 are thermally throttled** (10-17: 87-88 C, SM clock 345-525 MHz vs 1980 MHz, throttle reason 0x20 thermal slowdown).
  They gate every step of a synchronous multi-node job (job 23060 ran ~50 s/step vs ~6 s expected; the pose FT 23051 was slowed the same way).
  **Avoid slinky-4 for synchronous multi-node jobs** until it is fixed.
- Job history: 23060 (slinky-4,7,15) stopped at step 6 via STOP_AFTER_SAVE, its stop export `export-253-stop6` deleted, run dir archived as
  `runs/ABANDONED-slinky4-throttle-ppm-aloe-v2vox-raw-j23060`; 23079 (coordinator's 2-node restart) stopped in init via STOP_AFTER_SAVE;
  its stop export `export-253-stop1` deleted, run dir archived as `runs/ABANDONED-2node-ppm-aloe-v2vox-raw-j23079`; the run restarted from step 0 as 23081 on slinky-7,15,18 (--cpus-per-task 5: slinky-18's klweb page jobs hold 24 CPUs).
