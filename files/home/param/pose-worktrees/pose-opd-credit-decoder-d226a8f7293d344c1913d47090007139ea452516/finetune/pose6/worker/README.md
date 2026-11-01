# pose6/worker: playground worker for s2p cb0 + G0COND (workstream A3)

**Status (10-27 13:03 PT):** written, CPU-tested (9/9 PASS, job 31119), GPU-tested with a test-only overlay on export-357 (8/8 PASS, job 31142), and NOT deployed. Nothing was started, restarted or written under genweb.

The worker serves a monoplex pose checkpoint that carries both mods: `tower_p.s_emb` (s2p cb0) and `inp.g0` (G0COND). It is meant for the pose6 monoplex FT export (deliverable B). The playground prompt is the FT layout, so the SLOT is P - 1 (D2). A raw 31101 base export would run in the FT layout, which is not how the base model trained; see the open issues.

## Files

| file | what it is |
|---|---|
| `worker_s2pg0.py` | the variant (md5 c5afe6e5…) |
| `worker.diff` | `diff -u genweb/worker.py worker_s2pg0.py`. Deployed worker.py md5 is 88197f01… (`ORIG_MD5.txt`, mtime 10-27 11:29 UTC). |
| `s2pg0_unit.sh` | the unit script for a later hot-swap. NOT launched. It is `genweb/rosie_unit.sh` line for line except the worker file, `GENWEB_CK=$S2PG0_CK`, `GENWEB_WEIGHTS=s2pg0` and worker id `s2pgw<job>`. |
| `tests/test_worker_cpu.py` (+ `run_cpu_test.sh`, `test_worker_cpu.sbatch`) | CPU-only test, no model load. Results are in `tests/out/test_worker_cpu.json` and `tests/out/real_g0src_cases.json`. |
| `tests/test_worker_gpu.py` (+ `test_worker_gpu.sbatch`, `run_gpu_test.sh`) | end-to-end GPU test. Ready, NOT run. The sbatch runs export-357 with the test-only overlay (see the GPU section). |

## Differences from the deployed worker.py

Every changed line is marked `pose6` (`worker.diff`, 182 lines):

1. **Engine.** `MI = /data/home/amartyavarma/ppm/pose6/infer`, by absolute path. That is A1's engine plus A2's `g0src.py`.
2. **Playground dir.** The file lives in pose6/worker, so `G` is the absolute genweb path. genweb also goes on sys.path in the deployed order (MI/ppm, MI, genweb), so `import stick` still resolves to `genweb/stick.py`.
3. **g0src import.** The worker imports `g0src as G0S` and asserts at import that its KINDS contain `openings`, `openings-sam` and `clip`.
4. **Checkpoint.** `GENWEB_CK` has no default and Worker asserts it is set. At start, after capture, `g0_warm` warms g0src's caches over every source a run can pick; its first call scans the v2 items in about 1.4 s, so this stays out of requests. A source that fails here is only logged, and the run that picks it still fails loudly. `check_mods` refuses any checkpoint that is not s2p cb0 with `inp.g0`, or that has any `*zero_init` flag in binfo (a new parameter zero-filled instead of loaded).
5. **Weights id.** `GENWEB_WEIGHTS` defaults to `s2pg0`, so by default the worker claims ONLY runs with `mono_weights: "s2pg0"`, never base or Rosie runs. `WEIGHTS_LABEL[WEIGHTS]` names the checkpoint. `options()` uses that label for `model_label`; it only writes SCHEMA / OPTIONS when the worker runs as `base`, exactly as deployed.
6. **run().**
   - After the deployed source pick (start / clip / pose-only / no-avatar), `g0f, g0s = run_g0f(...)` runs. The engine is only started after it.
   - `eng.start([dict(x, a_mode="empty", g0f=g0f)], ...)`: g0f is a per-instance key (CONTRACT.md). A1's engine puts it at SLOT = P - 1 and refuses an instance without it.
   - summary.json gains `g0f` and `g0_source`.

Everything else is byte-identical to the deployed worker, and T1 confirms this at runtime. That covers:
- DEF params and seeds;
- `FastPPM(n_inst=1, cfg_rows=True, max_s=133.68, mimi_stream=False, silence=…)` and `capture()`;
- the claim loop: drain flag, hidden runs last, the duplex skip and the `mono_weights` filter;
- CANCEL, the live feed, render queues, on-demand GT, pose-only and the Rosie stick figure.

The worker itself has no changelog hook. The changelog line belongs to the deploy step (below).

## g0f per run: which source

The source is the frame 0 that the run's pose is decoded or rendered against. That is the deployed worker's own f0gs / sgs choice, so the model's f and the avatar / stick decode always describe the same frame. `g0_source()` decides; A2's `g0_from_opening(dir, kind)` computes f.

| run | source (dir, kind) |
|---|---|
| avatar person, prefill src start, pose prefill > 0, opening meta has frame-0 `fields` (00, 01, 03, 08, 17) | `genweb/openings/<p>`, openings |
| avatar person, src start, but the opening is the package itself (`frame0.path`: 09, 10, 12, 13, 14, 15, 19), or no opening on disk (falls back to the clip) | package `demo-data/prompts/<p>`, clip |
| src clip, prefill 0 s, or pose prefill off | package, clip |
| pose only (src ignored, as deployed) | `openings-sam/<p>` if present (06, 09, 10, 12, 19), else `openings/<p>`, else the package |
| Rosie (no avatar), src start | `openings/rosie_xyvR1rjH` (its meta has fields), openings |
| Rosie, src clip | `people_extra/rosie_xyvR1rjH`, clip |

**Override.** A request.json may carry `"g0f": [8 finite numbers]` at top level. It is optional and meant for tests or what-ifs. It bypasses g0src, and the summary records `g0_source.kind = "request override"`. A wrong length, NaN or non-number fails the run. The page and server.py do not pass this key today; only a directly written request.json carries it (agent runs must also carry `"hidden": true`).

**Fail loud.** If g0src raises, or returns anything other than a finite float32[8], the run ends with an error in status.json before the engine starts, for example `ValueError: G0COND: no frame-0 features for clip 00_14b594aa: KeyError: 'orig_h'`. The page shows it as the run's error. f = 0 only ever comes from g0src itself, where training gives 0 (frame0_ok false), per D6.

## Tests

### CPU: job 31119 on slinky-18 (4 CPUs, 32 GB), 10-27 19:16-19:17 UTC, 19 s. 9/9 PASS

The first run, job 31118 (19:13 UTC, before g0_warm and T6 existed), passed 8/8. Job 31119 tested these md5s:
- worker_s2pg0.py c5afe6e5;
- deployed worker.py 88197f01;
- pose6 fast_ppm.py 11f9da9b;
- g0src.py 23e2c548.

| test | result |
|---|---|
| T1: the deployed worker (imported read-only, `-B`) vs the variant | DEF, LABEL, GROUP, PEOPLE, MAX_*, CD/PAD/DEPTH/SPF, OFF, persons() (21) and schema_rows (52 rows) are equal; FP.PARAMS is identical |
| T2: imports | FP, MIF and IO come from pose6/infer and G0S from pose6/infer/g0src.py; 0 modules load from aloe-pose-monoplex-infer; stick resolves to genweb/stick.py; WEIGHTS defaults to s2pg0 |
| T3a: 126 cases (21 persons x 6 source configs), fake engine plus a recording g0src stub | each case calls g0_from_opening(dir, kind) exactly once, with the expected decode frame (independent expectation table); g0f float32[8] lands in eng.start's instance dict; all 3 kinds are exercised |
| T3b: request override | reaches eng.start; g0src is not called |
| T3c: bad overrides (7 values, NaN, strings, scalar, nested) | ValueError before eng.start, every time |
| T3d: g0src raising | ValueError naming the kind and dir; the engine is never started |
| T4: REAL g0src (A2's final file, 23e2c548) | 126/126 cases give a finite f[8]; 0 errors |
| T5: check_mods | accepts only s2p cb0 with inp.g0 loaded; refuses s2p full, s2p off, a missing inp.g0, g0cond False, and s2p or g0cond zero-init |
| T6: g0_warm at start | 39 sources warm, 0 failing, 1.67 s (cold) |

To re-run (about 20 s), on a compute node: `sbatch /data/home/amartyavarma/ppm/pose6/worker/tests/test_worker_cpu.sbatch`. The output JSON now records the md5 of the worker, deployed worker, fast_ppm and g0src it tested.

### GPU: job 31142 on slinky-4, physical GPUs 4,5, 10-27 13:01:21-13:03:30 PT. 8/8 PASS in 92.8 s

The run used export-357 with the TEST-ONLY overlay (below); worker_s2pg0.py md5 was c5afe6e5… (unchanged). Main submitted it after A1's job 31140.
- **Guard:** OK. GPU 4: SM clock 1380 / 1980 MHz, 683 TFLOPS bf16. GPU 5: 1410 / 1980 MHz, 690 TFLOPS.
- **Overlay:** sigma 0.0600, ||g0(F_A)|| 6.77 vs median pose-frame input norm 6.79; s_emb [2048, 2560], g0 [5120, 8].
- **W0:** the worker comes up in 48.0 s at step 357; the engine has s2p cb0 and inp.g0, both non-zero.
- **R1-R4:** done; each summary g0f equals g0src(dir, kind) exactly, for kinds openings, clip, openings-sam and openings (Rosie, stick figure).
- **R5:** override = R1's own f gives pose, mimi and lane bit-identical to R1.
- **R6:** override f = 0 changes the generated pose on 49 of 50 frames. g0f reaches the model.
- **E1:** the engine refuses a start without g0f ("G0COND model: every instance needs g0f …").
- **ms/frame:** 77-80 over each 76-frame run, prefill included. These are short 4 s runs, not a latency measurement; A1's (iv) has the latency numbers.
- **Results:** `tests/out/gpu/test_worker_gpu.json`; log `tests/logs/gpu-31142.log`.

`tests/test_worker_gpu.py` runs the real Worker: build, check_mods and capture, in the playground layout (2 GPUs, one process). Every output path is redirected to `tests/out/gpu`; nothing goes to genweb runs/, render/jobs or the live feed. The cases:
- **R1:** start opening, kind openings.
- **R2:** clip, kind clip.
- **R3:** pose only on a SAM opening, kind openings-sam.
- **R4:** Rosie, start opening, kind openings.
- **R5:** R1 with the override set to R1's own f. Expected: tokens bit-identical to R1.
- **R6:** R1 with the override f = 0. Expected: generated pose differs from R1.
- **E1:** the engine refuses a start without g0f.

- **W0:** the worker comes up; the engine follows the model's s_emb (cb0) and inp.g0, both non-zero.

**TEST-ONLY overlay** (`POSE6_TEST_OVERLAY=1`, the sbatch default, with `GENWEB_CK` = export-357). No checkpoint with both keys exists yet, so the test, not the worker, wraps the engine's `build_model`. After the real export-357 load, the wrapper:
- attaches `tower_p.s_emb` (cb0, N(0, 0.02/sqrt 32)) and `inp.g0` (Linear 8 -> H, no bias, N(0, sigma^2)) on the pose GPU;
- seeds them exactly as A1's `test_engine6.set_new_params` does: one CPU generator, seed 20261027, s_emb first;
- sets sigma by A1's rule: median ||inp.pose(Person 01 opening frame)|| / (||F_A|| sqrt(H)), with A1's F_A;
- marks binfo `s2p=cb0` and `g0cond=True` with `zero_init` False, so the worker's own `check_mods` runs unmodified;
- asserts that the checkpoint had neither key before, that both weights are non-zero, and that `worker_s2pg0.py` md5 = c5afe6e5… (unchanged).

**Job spec:**
- `-J pose6-a3-gpu -p all -x slinky-9 -c 8 --mem=120G --gres=gpu:2 -t 0:25:00`;
- Triton cache `/scratch/amartyavarma/pose6/triton-cache`;
- A1's GPU guard (`pose6/infer/tests/gpu_guard.py`): exit 3 on slinky-4 physical GPU 6/7, or when the SM clock under load is below 75 % of max.

Without the overlay, `tests/run_gpu_test.sh` runs the same test on a real checkpoint with both keys (`GENWEB_CK=<ck>`).

## How to hot-swap later (NOT done; only on main's word)

**Recommended path: add it as its own weights choice, the Rosie pattern.** No existing process restarts, and base export-357 and Rosie are untouched.

0. **Preconditions.**
   - The checkpoint has `tower_p.s_emb` (cb0) and `inp.g0`. This is the pose6 FT export; local bf16 shards or a COMPLETE DCP both load.
   - A1's proofs (i) to (iv) PASS.
   - The CPU test has been re-run on the frozen A1/A2 files.
   - The GPU test PASSES on that checkpoint.
   - The user's queue is empty, nothing is generating or rendering, and no call is open (playground-never-starve-queue).
1. **Get a free GPU pair from main.** Never slinky-9. Never the only Param render pair. No reshuffle while user runs are queued or rendering.
2. **Start the worker first.** This is make-before-break: it claims nothing until a run carries `mono_weights: "s2pg0"`.

   ```
   sbatch -J genweb-s2pg0 -w <node> -x slinky-9 --gres=gpu:2 -c 8 --mem=120G -t 7-0 \
     -o /data/home/amartyavarma/ppm/aloe-posedup-live/genweb/logs/%x-%j.out \
     --export=ALL,S2PG0_CK=<ckpt> /data/home/amartyavarma/ppm/pose6/worker/s2pg0_unit.sh
   ```
   - The process is `python worker_s2pg0.py s2pgw<job>`, and its log is `genweb/logs/worker-s2pg0-<job>.log`.
   - Wait for "model step … loaded + captured".
   - A check_mods assert in that log means the wrong checkpoint: scancel this new job only.
   - The id is `s2pgw<job>`, NOT `s2pg0<job>`: server.py's reaper takes the first 5-digit run in the worker id as the Slurm job.
   - Use `-t 7-0` (memory demo-jobs-long-limits).
3. **Verify with one hidden run.** Write `genweb/runs/<rid>/request.json` with `"hidden": true`, `"label": "agent test s2pg0"`, `"mono_weights": "s2pg0"` and default params, then `READY`. Do this only while the user is idle.
   - Check status done and summary `g0f` / `g0_source`.
   - Expect ms/frame of about 66 for s2p cb0 (pose-s2p TESTS (d)).
   - Check the render appears.
4. **Expose it (genweb owner, aloedemo, on main's word).**
   - server.py `gen()`: let `mono_weights == "s2pg0"` through. Today only `"rosie"` is copied into the request; anything else becomes base.
   - Deploy that with `touch genweb/DEPLOY` (zero downtime; NEVER pkill server.py).
   - index.html: in the `<!-- BEGIN mwts -->` block and at line 458 (`mono_weights:` / `label:`), add an `s2pg0` option to `#mwts` and `#hwts`, the localStorage whitelist, and the model chip text.
   - Run `genweb/err_check.py` against `http://127.0.0.1:9205` (srun inside the genweb-app job). Page edits need no deploy.
5. **Changelog** (one line, real deploy time):

   ```
   python3 /data/home/amartyavarma/ppm/aloe-posedup-live/tools/changelog_add.py genweb "<who>" \
     "Monoplex: new weights option 's2p cb0 + G0COND' (<ckpt>), own worker; base and Rosie unchanged" --server --page --at <ISO-UTC>
   ```
   Keep the `running.video_*` keys true.
6. **Rollback.**
   - Grey out or remove the option first (the `rosiepause` pattern), so no run queues for a missing worker.
   - Then scancel the genweb-s2pg0 job once it is idle.
   - Add a changelog line.

**Alternative: replace base export-357.** Only if the user asks. It is invasive:
- unit.sbatch, hot_unit.sh, mono_guest.sh and the flex/pool scripts all launch `genweb/worker.py`, so the pool would keep starting old base workers;
- during a pair-by-pair swap, base runs would be served by two different models.

It would need an empty queue, drains (`flex/drain/<wid>`) of every base worker, `GENWEB_WEIGHTS=base` variants started first, and the launch scripts repointed, all as one coordinated genweb change. Not prepared here.

## Open issues / decisions

1. **Re-run the CPU test if A1's fast_ppm changes after 11f9da9b.** It takes 20 s on 4 CPUs. A2 reports g0src 23e2c548 as final.
2. **GPU test** PASSED with the test-only overlay (job 31142). Re-run it without the overlay on the first real checkpoint with both keys: `GENWEB_CK=<ck> POSE6_TEST_OVERLAY=0 sbatch tests/test_worker_gpu.sbatch`.
3. **Which checkpoint the playground should serve.** The worker suits the pose6 monoplex FT export. A raw 31101 export in this FT-layout prompt would put f at P - 1, not at the 8,1,1 SLOT (pose_s3q gpos), and the base model never saw this prompt (D2). Main to decide.
4. **Page exposure** needs genweb server and page edits by the genweb owner (step 4). This worker never touches them.
