# ppm qqa-ft STATUS — run C's speech-duplex quick-QA fine-tune on the latest PPM checkpoint
Started 10-15 22:59 PT. Updated 10-15 23:25 PT.

## Stage
- Training job 21817 submitted 23:20 PT on slinky-7/15/18 (world 24). Init = PPM step 11263.
- QA eval engine: being built by a sub-agent (qqa-ft/qa_eval/, PROGRESS.md there); init eval (tag ppmqqa-init) runs on slinky-4 once it passes parity.

## Stopped inference servers (user OK; relaunch from their sbatch)
- t2v-server job 21772 (slinky-18 GPUs 0-1) stopped 10-15 22:59 PT -> relaunch: sbatch /data/home/amartyavarma/video/t2v_serve.sbatch
- s4live-demo job 21152 (slinky-4 GPUs 0-1) stopped 10-15 23:29 PT for the PPM QA init eval -> relaunch: sbatch /data/home/amartyavarma/qwen-1ch/infer/s4live/demo.sbatch
- s4live-asr job 21792 (slinky-4 GPUs 2-3) stopped 10-15 23:29 PT for the PPM QA init eval -> relaunch: sbatch /data/home/amartyavarma/qwen-1ch/infer/s4live-asr/demo.sbatch

## Init
- Newest PPM save at copy time: node-local dcp-11263 (job 21765, saved 22:42 PT). Copied 23:01-23:03 PT with ckpt_local_export.py as a niced
  (19) + ionice-idle CPU-only overlap step (qqa-ft/copy_init.sh) -> /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11263 (bf16, 86 GB).
  No GCS backup of PPM saves exists. PPM step time 28.4 s before; 39.0 / 32.3 s for the 2 steps during the copy (~15 s lost); 28.4 s after.
  Job 21765 never signalled / modified.
- Fresh AdamW (run C: fresh Adam from its init export). Weights resharded 128 -> 24 FSDP shards (train_qqa.load_init).

## Data (qqa-ft/make_plan.py, code/data_qqa.py)
- Same store, same 95,192 conversations, same order: run C's node sequence re-derived with run C's own code + cfg-21169.json and asserted equal
  to its offline plan (PACKS-<host>.json sequence digests) on all 3 hosts; same packs (conversations per micro) per local rank and step; same
  96 held-out eval conversations (4 / rank).
- Each conversation = PPM's speech row (data_ppm.speech_micro): A = user lane + Mimi codes, B = assistant lane (NEW_WORD) + Mimi codes,
  loss on B only; swap 0 (B = assistant, as run C S3_SWAP none). PPM system-prompt rule -> EMPTY system block (P = 5) on every conversation
  checked (quick-qa has no persona sidecar, no later-chunk siblings). Run C: no prefix (P = 0) on 2/3 of conversations, B's CC-BY bank voice
  clip on 1/3.
- PACKED with per-conversation attention segments (FA3 varlen block-diagonal, PPM's Ctx), positions 0..T_k-1 restarting per conversation,
  tower shifts restart per segment. Segment = P + n + CD (no 7,500 padding: crop_conv False). ~6.7k positions / micro (run C ~7.45k: its
  clips + CD 3).

## Hyperparameter mapping (qqa.env)
| knob | run C (21169) | PPM prod (21765) | qqa-ft |
|---|---|---|---|
| steps / epochs | 223 packed steps, 1 epoch | 17,400 horizon | 223, 1 epoch, same conversations / step (17.8) |
| schedule | warmup 7, linear decay to 0 at 223 (wsd floor 0) | warmup 100, constant | = run C |
| backbone (trunk incl. emb_A/B, lm_head, norm) | LR_TRUNK 3e-6 (+ LoRA r256 at 1.5e-4) | 5e-5 | 3e-6 (no LoRA: full backbone at 3e-6) |
| spec rows (PAD/NW/EOB) | 6e-5 | 6e-5 | 6e-5 |
| head projector | 3e-5 | 3e-5 | 3e-5 |
| audio code tables (inp.aud) | LR_AUDIO 6e-5 | 2e-4 | 6e-5 |
| speech tower (4B) | LR_VOICE 1.5e-5 | 3e-4 | 1.5e-5 |
| 0.6B speech decoder | LR_LIGHT 1.5e-5 (1 update / step) | 4e-4 peak, 8 nanostep updates / micro | 1.5e-5 peak, 8 nanosteps / step, warmup 56 + decay to 0 at 1,784 updates |
| clips | trunk 0.5, towers (voice+light) 1.0, spec/proj/audio 1.0 | all 1.0, decoder 0.5 | trunk 0.5, tower 1.0, decoder 1.0, spec/proj/audio 1.0 |
| Adam | b2 0.95, eps 1e-8, wd 0, fresh | same | same, fresh |
| loss weights | text x5 word, cat/tim/nw/eob 1, cb0 1, light 1 | same | same |
| loss normalisation | tokens_step (step token totals / 24) | per-row mean | tokens_step (text, cb0, and per decoder update) |
| light frames | 10 % of frames | 100 % (8 disjoint chunks) | 100 % (PPM's) |
| code delay | 3 | 2 | 2 (PPM native) |
| positions | stage-3 word positions (mix-RoPE) | plain RoPE frame index | plain RoPE, frame index per conversation |
| system prompt | none 2/3, B bank clip 1/3 | PPM rule | PPM rule -> empty block |
| frozen | Voxtral adapter | – | pose tower / decoder / pose head / decp_bb / pose emb |
| eval | 96 held-out, every 25 | – | same 96, every 25 |
- No counterpart: run C's LoRA (1.5e-4), router / inscale / new / vadapter groups; PPM's pose groups (frozen), PPM_DEC_FLOOR (decay to 0 instead).

## Progress
- 23:20 PT launched job 21817.
- 23:22 PT init loaded (86.1 GB, 144 s). Step-0 eval (96 held-out, PPM empty-prefix rows): B normalised word CE 1.9862 (run C at its init: 2.2205),
  word CE 2.1342, PAD CE 0.0877, cb0 CE 0.896, light CE 2.066.
- 23:22 PT PACKED-LOSS GATE PASS on 24 ranks: max rel err text 4.5e-4, cb0 1.8e-4, light 3.0e-5 (< 2e-3). Step-0 totals: 160,223 text targets,
  160,635 cb0 labels / light frames.
- First steps: step 1 116 s (decoder compile), then ~6-7 s/step, peak 58 GiB. Train word CE norm 1.98 -> 1.93 by step 3. ETA ~23:50 PT + export.
- 23:33 PT job 21817 CRASHED at step 82 on my own data guard: one packed micro was 7,533 positions > the 7,503 budget assert (PPM's empty system
  block P = 5 on every conversation vs run C's P = 0 on 2/3, = +4 positions each; same conversations per micro). Whole plan scanned: max
  7,560, 2 of 5,352 micros over 7,503. Guard raised to 7,600 (packs unchanged = run C's). Evals before the crash: B norm word CE 1.9862 (0) ->
  1.5836 (25) -> 1.4985 (50) (run C 2.2205 -> 1.5211 -> 1.4773). No optimizer checkpoint -> relaunched from the init (fresh).
- 23:34 PT relaunched as job 21823 (identical config, guard 7,600). Step-0 eval and loss-check identical to 21817 (1.9862; 4.5e-4). 6.4 s/step.
- 10-16 00:00 PT job 21823 COMPLETED (exit 0, 223 steps, mean 6.41 s/step). Pose parts bit-identical on every rank.
  Eval B normalised word CE (96 held-out): 0: 1.9862 | 25: 1.5841 | 50: 1.5004 | 75: 1.4717 | 100: 1.4542 | 125: 1.4407 | 150: 1.4342 | 175: 1.4309 | 200: 1.4275 | 223: 1.4268
  (run C: 2.2205 | 1.5211 | 1.4773 | 1.4527 | 1.4271 | 1.4061 | 1.3946 | 1.3801 | 1.3735 | 1.3688). Final light CE 1.963, cb0 0.857.
- Export: /data/home/amartyavarma/ppm/qqa-ft/export-223 (consolidated bf16 DCP, 88.4 GB, COMPLETE; infer_ppm 'dcp' kind). GCS backup job 21827
  -> gs://podcast-raw-audio-data-mass/amartyavarma/ckpt-backup/qqa-ft/export-223/ (log qqa-ft/logs/gcs_backup/).
- 00:01 PT QA eval: ppmqqa-w223 jobs 21824 (slinky-7) + 21825 (slinky-15); ppmqqa-init job 21819 (slinky-4) + 21826 (slinky-18).
- QA eval (416 scorable of 427, run C's items / seeds rule / scorer, word-greedy only, PPM empty-prefix prompts, A text + Mimi forced; engine
  parity vs the trainer forward: prefill logits bit-identical, incremental bf16 noise; qqa-ft/qa_eval/PROGRESS.md):
  | model | acc_greedy | 1st-sentence greedy | TF answer nword |
  |---|---|---|---|
  | PPM init (step 11263) ppmqqa-init | 0.0962 | 0.0673 | 2.048 |
  | PPM fine-tune w223 ppmqqa-w223 | 0.2861 | 0.2476 | 1.438 |
  | run C qqaS3text21169w223 | 0.3846 | 0.3389 | 1.339 |
  w223 by source: triviaqa 0.378, nq 0.193; A clean 0.278 / corrupted 0.294. All gates PASS. Done 10-16 ~00:35 PT.
- Inference servers still DOWN (relaunch commands above): t2v-server, s4live-demo, s4live-asr. slinky-4/7/15/18 now free.
- 10-16 00:31 PT stopped t2v-server job 21828 (slinky-18) again for the scaling rerun -> relaunch: sbatch /data/home/amartyavarma/video/t2v_serve.sbatch

## SCALING rerun (user 10-16, job 21832) — DONE
- = job 21823 exactly + weights-only exports at 14 / 28 / 59 / 117 / 223 (cfg diff logs/cfg-diff-21832-vs-21823.txt: export, export_steps,
  export_wait_min, run_tag only). Exports GCS-verified (gs://podcast-raw-audio-data-mass/amartyavarma/ckpt-backup/scale/export-N), intermediate
  ones moved to eval-node /scratch/amartyavarma/ppm-qqa-scale (slinky-4: 14, 59; slinky-18: 14, 117; slinky-7: 28, 59; slinky-15: 59) and
  deleted from /data (one 88 GB export fits above the 300 GB floor); export-223 kept at qqa-ft/scale/export-223. Exports waited ~15 min in
  total for /data space.
- Eval B norm word CE 21832 vs 21823: 0 1.9862/1.9862 | 25 1.5856/1.5841 | 50 1.4973/1.5004 | 75 1.4693/1.4717 | 100 1.4584/1.4542 |
  125 1.4460/1.4407 | 150 1.4392/1.4342 | 175 1.4358/1.4309 | 200 1.4331/1.4275 | 223 1.4318/1.4268 (bf16 run-to-run noise <= 0.006).
- QA (greedy only, 416 scorable, same items / scorer), vs run C's stage-3-mode scaling curve (scale-stage3-*):
  | conversations | PPM acc_greedy | run-C-mode acc_greedy | PPM TF nword |
  |---|---|---|---|
  | 5,913 | 0.2452 | 0.3101 | 1.733 |
  | 11,925 | 0.2476 | 0.3125 | 1.593 |
  | 25,126 | 0.2668 | 0.3510 | 1.505 |
  | 49,865 | 0.2716 | 0.3750 | 1.459 |
  | 95,192 | 0.2981 | 0.3894 | 1.437 |
  (PPM init 0.0962; first w223 run 0.2861.) All gates PASS. Tags ppmqqa-scale-* in results/summary.json.
- t2v-server relaunched on slinky-18 after the evals. s4live-demo / s4live-asr (stopped 23:29 PT) left to the coordinator.
