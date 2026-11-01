# aloe-v2pure-ft: PPM fine-tune on the Aloe pose release v2 (the 8B SAM pose v2pure data)

PPM 32B (init PPM main step 11488, `ppm/ckpt-infer/ppm-v1/local-11488`), monoplex Aloe pose: channel A empty, channel B generates
speech + pose. Recipe = job 22317 (`ppm/aloe-uncond-ft`) verbatim; only the data changes (see `aloe_v2pure.env`, lines marked [V2PURE];
the trainer logs the resolved-config diff vs 22317 as `[cfgdiff]`).

- Data: `stores/` (built by `code/build_v2pure_rows.py` from `aloe-pose/sam-cache-active-v2pure`, the cache the 8B v2pure model trained on,
  itself built only from `gs://aloe-processed-data/pose-speech-vad/releases/v2/`). One store row per active-speaker clip; held-out split =
  the 8B's rule `blake2b8('aloe-pose-v1|' + recording_id) % 10000 < 100`. See `stores/README.md`.
- Rows per step: 64 rows of <= 7,500 frames (as 22317). Release clips are short (median 12.6 s), so a row packs several clips; every clip is
  its own attention segment with its own B system block. A step's clips are spread over 24 GPUs x R packed micros (rounds); the loss is
  normalised by the step's token totals and the 0.6B decoders take their K = 8 nanostep updates per step over all the step's frames, so the
  step's update is 22317's maths. Step 0 checks packed vs standalone losses (`[pack] LOSS CHECK`).
- Nodes: slinky-4, slinky-7, slinky-15 (world 24). Code: `code/` (copy of 22317's; changes in `data_aloe_pose.py` (v2pure section) and
  `train_aloe_pose.py` (`train_step_multi`, `nanosteps_multi`, `save_full` / `load_full`)).

## Run (job 23051, launched 10-17 10:28 PT)
- Data: 367,495 train clips (92.13M frames, 2,047 h, 12,384 recordings) -> 12,791 rows -> 200 steps x 64 rows (mean 460k frames/step;
  22317: 317k, its 10-min rows averaged 4,960 frames); 4 rounds of ~5.4k positions per GPU per step. Eval: 3,481 held-out clips (868,931
  frames, 113 recordings = the 8B's held-out set) in 486 packed micros. Skipped: the 2 one-frame clips (1 train, 1 eval).
- Schedule: warmup 7 (= round(10 x 200 / 306), 22317's rule), linear decay to 0 at step 200, eval every 25 + exports at 33, 67, 100, 133,
  167 (node-local) and 200 (node-local + /data). `[cfgdiff]` vs 22317: 26 keys, all paths / schedule length / the new data + save keys.
- Checks in the run: step-0 packed vs standalone loss max rel err 5.2e-4 (< 2e-3); pass A vs pass B forwards bit-identical (hv diff 0);
  stop/resume tested at step 3 (save 91 s, clean exit, in-job resume, state digests equal on all 24 ranks, LRs restored).
- Logging note: steps 1-3's train/loss, uloss_word, wloss_word, loss_unweighted came from the no-grad pass, whose text loss omits the
  normalised-word term (heads_s2q.ce_head); fixed from step 4 (values = 22317's definition). Gradients were 22317's maths throughout.
- Speed: ~99 s/step, peak 60.4 GiB/GPU.

## Stop, resume
- Stop with a full resumable save (node-local), clean exit: `touch /data/home/amartyavarma/ppm/aloe-v2pure-ft/control/STOP_AFTER_SAVE`
  (or `scancel -b -s USR1 <job>`). The trainer saves at the next step boundary, writes `control/STOPPED`, exits 0.
- Resume: `sbatch /data/home/amartyavarma/ppm/aloe-v2pure-ft/aloe_v2pure.sbatch` (ALOE_RESUME=auto: newest save complete on every node;
  same 3 nodes, world, plan asserted; state digests compared after load, `[resume]` line in the log). Same run tag (`RUN_TAG.txt`) so
  metrics continue in one `metrics.jsonl`.
- In-job stop/resume test: `touch control/RESTART_AFTER_STOP control/STOP_AFTER_SAVE`: the job saves, the trainer exits, and the job
  relaunches it from the save while keeping its nodes.
- Saves: `/scratch/amartyavarma/aloe-v2pure-ft/saves/step-NNNNN/` on each node (rank<R>.pt, node.pt, NODE_COMPLETE; meta.json on
  slinky-4), hourly, keep last 2, plus on stop and at the final step. Node-local weight exports at 5 points + final:
  `/scratch/amartyavarma/aloe-v2pure-ft/exports/export-<step>` (ppm-local-bf16-shards-v1). Final consolidated bf16 export on /data:
  `/data/home/amartyavarma/ppm/aloe-v2pure-ft/export-<steps>`.
- `RUNNING_JOB.txt` = job id + nodes (the chained Aloe Voxtral re-run depends on it).

## Services stopped to free the nodes (user-authorised, 10-17)
- slinky-7: Aloe Voxtral live demos `m1vox-live` (job 22356) and `m1vox-live2` (job 22561), both serving
  `ppm/aloe-v2vox-ft/export-253` (LOADED.json). Restart (from `ppm/aloe-m1vox-live/RESTART_CMD.txt`; the quick-tunnel URLs change on restart,
  see URL.txt / URL2.txt; any node with 2 free GPUs works: change `-w`):
  - instance 1: `sbatch /data/home/amartyavarma/ppm/aloe-m1vox-live/demo.sbatch` (header pins `-w slinky-7`; on another node:
    `sbatch -w <node> /data/home/amartyavarma/ppm/aloe-m1vox-live/demo.sbatch`); it hot-swaps to SWAP_TO.txt = export-253.
  - instance 2: `D=/data/home/amartyavarma/ppm/aloe-m1vox-live; E=/data/home/amartyavarma/ppm/aloe-v2vox-ft/export-253; sbatch -J m1vox-live2 -w slinky-7 -o $D/logs/demo2-%j.out --export=ALL,FRONT=8795,BACK=8796,PPMLIVE_STATE_DIR=$D/inst2,URLF=$D/URL2.txt,CK=$E,PPMLIVE_SWAP_TO=$E $D/demo.sbatch`
- slinky-18: not used (s4live-demo 22232, t2v-server 21918 and the klweb pages keep running).
- slinky-4: the probe watcher (22958) keeps running; its 4-GPU probe waits while this run holds slinky-4's GPUs.
- slinky-7 /scratch was cleaned to make room for the saves: list + evidence in
  `/data/home/amartyavarma/scratch-inspect/slinky7-cleanup-1017/DELETED.log`.
