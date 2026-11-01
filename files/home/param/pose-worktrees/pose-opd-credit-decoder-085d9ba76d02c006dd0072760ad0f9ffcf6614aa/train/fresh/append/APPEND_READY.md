# Fresh pose APPEND: ready to ingest the new pose data and resume from 27114's final save (10-22 PT)

User (10-22): "I'm going to supply more distinct pose data soon, which will be in the same format, and you should get ready to tack it onto
the end of the data schedule." Job 27114 (8,1,1, env `prod_811_fresh16861_v4.env`) ends with a final save at **dcp-17352**, where the
plan-1 fresh pose rows (new-pose-llm-data) run out. Everything below is prepared. **Nothing has been downloaded or launched.**

## 0. One command, once the data location is known

```bash
sbatch -w $(cat /data/home/amartyavarma/ppm/code-811/nodes16_s7.txt) /data/home/amartyavarma/ppm/code-811/fresh/append/ingest_more.sbatch <SRC> npl2
```

- `<SRC>` is either `hf://buckets/<owner>/<bucket>/<prefix>` (an HF bucket path like `hf://buckets/InternalCan/video-pretraining-data/new-pose-llm-data`) or `/abs/dir`, the directory that holds `jobs/`.
- The job allocates the 16 training hosts, CPU only, with 48 of each host's 64 CPUs. The other 16 CPUs stay free for the HF watcher's uploads.
- It is queued behind `ppm811`: it refuses to run while a `ppm811` job is RUNNING or PENDING. To run it inside a live job instead, use `OVERLAP_JOB=<job> bash fresh/append/ingest_more.sh <SRC> npl2` (light mode: `-c 4`, nice).
- Output goes to `fresh/append/npl2/`: `SUMMARY.md`, `TIMES.tsv`, `logs/`, `dryrun/RESULT.json`, `dryrun/GUARD.json`, and `CFG_DIFF_vs_27114.txt`.
- When it finishes, it has written `PPM_MAX_STEPS` into `prod_811_fresh_append.env`, keeping a backup at `.bak-pre-fill`. It does **not** launch training.
- Re-running the same command skips every finished step and redoes only the dry run. Use this to re-check against dcp-17352 if the ingest ran before 17352 existed.

## 1. Trainer support (installed; backups `*.bak-pre-append-1022`)

Edited files:
- `code-811/train_ppm.py`
- `code-811/fresh/fresh_pose.py`

New knobs, present in the cfg only when set:
- `PPM_FRESH_POSE_ROOT2=<plan dir>[,<dir>..]`
- `PPM_FRESH_POSE_TAIL=1|0`. It is required whenever ROOT2 is set.

How the stream is built:
- Rank stream = its plan-1 rows `l, l+8, ..`, then plan 2's rows `l, l+8, ..`, then plan 3's, and so on (`fresh_pose.append_stream`).
- Plan k+1 row ids are tagged `k << 40`. Plan-1 ids are plain, the same ones 27114 used.
- Step t, micro m reads stream row `(t - 16861) x 8 + m`, as before. So every step before 17352 maps to exactly 27114's rows: resume-exact, asserted, and unit-tested.

What `TAIL` controls:
- **TAIL=1** (in the env): each rank first uses its own unused plan-1 rows, then the new plan.
  - The plan-1 end of 17352 was set by the slowest rank (slinky-2). That leaves 26,073 plan-1 rows (~4.2k h) that have never been used. Each is still used once.
  - This adds ~+25-30 steps.
- **TAIL=0**: drops those rows, and every rank starts the new plan at 17352.

Run end:
- `max_steps` = the end of the combined stream.
- The run ends there with the usual `[fresh] pose exhausted` final save. A second log line gives the end after each plan.

Resume guard (`fresh_append_guard`, called in `s811_init`):
- From 27114's saves (fresh block without appended plans): every other fresh key must be equal, and `pose_append_at` is set to the checkpoint step.
- `main()` asserts `pose_append_at <= the end of the plans before it`. This means plans can only be appended at or before 17352, so history is unchanged.
- Later checkpoints carry `pose_root2`, `pose_tail`, `pose_append_at` (one step per appended plan) and `pose_digests` (host -> [digest of each plan]).
  - On every resume these must match, and the digests are re-checked.
  - A further plan can be appended later. The checkpoint's list must be a prefix of the new one, and the new plan must be added at or before the end of the earlier plans.
  - Dropping appended plans, changing TAIL, or changing any other fresh key is refused.

Unset = byte-identical:
- The resolved cfg under the v4 env equals 27114's logged cfg. The only difference is `resume_from`, which was a launch override (`CFG_DIFF_v4_vs_27114.txt`).
- Plan-1 rows and micros from the new `fresh_pose.py` equal the old module's (`tools/eq_plan1.py` on slinky-19, rows 0 / 1 / mid / last). `FreshPoseMultiDS` on plan-1 ids equals `FreshPoseDS`.
- Only two existing trainer lines changed, and both are conditional on ROOT2.

Appended-plan build (`fresh_pose.py`, with `FRESH_TAG` and `FRESH_PRIOR_SCANS`): the same scan, plan and check steps as plan 1, plus deduplication against the earlier plans' scans:
- the same `segment_id`
- the same `clip_id`
- a plan-1 video whose clip source span (`<video>_MM-SS.ss_to_MM-SS.ss`, parsed for 100% of plan-1 clips) overlaps an earlier span of that video. If the span cannot be parsed, the segment is dropped.

The other video modes (`FRESH_PRIOR_VIDEO`):
- `video`: drops every plan-1 video.
- `off`: deduplicates on ids only.

Every other plan-1 rule is kept:
- release pins
- release state
- `segment_id` dedup
- PPM TikTok v2+v3 videos
- eval videos
- the SAM filter

Row keys are `npl2/<host>/..` and the pack seed is per tag.

Tests:
- `fresh/append/test_append_cpu.py all`, run on CPU:
  - **map** (5/5): real plan-1 host row counts plus synthetic plan 2 and plan 3, TAIL 1 and 0. Checks history == 27114, resume at 17352 and at 17400 == an uninterrupted run, no repeats per host over the whole fresh run, every plan-1 row used once with TAIL=1, and a plan 3 appended later leaving earlier steps unchanged.
  - **guard[unset]** (3/3)
  - **guard[append]** (10/10): uses a real 27114 `train_state.json`.
- End-to-end smoke test: `fresh/append/npltest/` (local source = plan-1's Weka copy, prior = one host's plan-1 scan). Results are in §2.

## 2. Ingestion pipeline (`fresh/append/`)

| step | script | what |
|---|---|---|
| 0 | ingest_more.sh | Checks that the env names the plan dir and that no ppm811 job is running. Measures each host's /scratch free space. |
| 1 | src_list.py | Lists the source (HF bucket listing API, or a local walk). Keeps `jobs/*/*/{part-*.parquet, release.json, complete.json}`. Skips releases already placed for plan 1 (`PLACEMENT_files.tsv`) or listed for an earlier tag. |
| 2 | src_place.py | Places whole jobs per host, balanced so that each host's plan-1 rows + new rows come out equal (TAIL=1). Never fills a host below 60 GB free. Warns for hosts left below the ~126 GB HF-hold floor. |
| 3 | src_fetch.py | Downloads on 16 hosts in parallel. HF: `/resolve/` -> CDN at `HF_RATE`=60 files/s/host (= the 10-21 assemble.py). Local: 32-thread copy. Resume-safe. |
| 4 | src_verify.py | Checks sizes, plus xet hash vs the listing (HF) or sha256 vs the source (local). The scan then re-checks every parquet's sha256 against its release pins. |
| 5 | fresh_pose.py all | Runs the scan, then the global dedup (plan 1 + PPM TikTok + eval videos), then the per-host PLAN to `/scratch/amartyavarma/ppm-fresh-pose-npl2`. |
| 6 | fresh_pose.py check | Compares raw parquet with the built rows and micros (3 rows per host). |
| 7 | append_dryrun.py host / reduce / guard | Loads plans 1+2 on every host. Checks digests against the audit copies and records sha256 of the PLAN files. Computes all 128 ranks' streams. Builds micros at the resume step, at the first new-plan row and at the last step. Computes the boundaries, then fills `PPM_MAX_STEPS`. Runs the trainer's `s811_init` guard on the real resume `train_state.json`. Also runs the unit test and the cfg diff vs 27114. |

**Wall time with training stopped, for a plan-1-sized drop** (~300k parquet + ~350k json, ~480 GB; measured on 10-21):

| step | time |
|---|---|
| listing | ~5 min |
| fetch | ~12-15 min (41k files/host at 60 files/s/host, i.e. ~45 MB/s/host and 0.7 GB/s total) |
| verify | ~1-2 min |
| scan + plan | ~4-5 min (scan 60-90 s, prior index ~1-2 min, plan ~80 s) |
| check + dry run + tests | ~4 min |
| **total** | **~30-40 min** |

Scaling:
- Fetch and scan scale with the data, at roughly +3 min per extra 100 GB.
- `HF_RATE=90` cuts the fetch to ~8 min. The hub's resolver quota is ~1.6k files/s total and the edge 429s at ~2-4k/s.

**Smoke test, end to end (10-22 10:25 PT, inside 27114 with `OVERLAP_JOB`, `-c 4`, nice; no download):**
- Source: the first 2,500 jobs of plan-1's Weka copy, `LIST_NO_EXCLUDE=1`, test tag `npltest`, test env `npltest/test.env`. Prior = slinky-0's plan-1 scan.
- What it did:
  - listed 465 releases (0.8 GB) and placed them evenly on 16 hosts
  - copied 1,395 files and verified them (0 mismatches)
  - scanned and planned: 32,386 segments -> 30,359 kept. Dropped: 1,896 already in the prior scan, 72 duplicates, 55 eval-video, 4 SAM-filter. Result: 777 rows, 125 h.
  - `check` OK on all 16 hosts (raw parquet == built rows / micros)
  - dry run: all 128 rank streams computed, 260 micros built (resume step / first new row / last step), boundaries [17352, 17353] -> `PPM_MAX_STEPS=17353` written into the test env
  - guard OK on the real 27114 `train_state.json` (dcp-17281 stand-in)
  - unit test OK; cfg diff = `pose_root2`, `pose_tail`, `max_steps`, `resume_from`
- Took 1 min 20 s in total. Node-local test data deleted afterwards; logs kept in `npltest/`.
- Two bugs found and fixed: a host with no files must still get `jobs/` and a typed empty scan, and a crashed task now kills the step (`--kill-on-bad-exit`) instead of the plan waiting forever.

## 3. The other streams past 17352 (`STREAM_LIMITS.json`; step index = logged step - 1)

| stream | unseen until | then | code limit |
|---|---|---|---|
| speech / podcast (all rows flipped since 16861) | **17,872** (min over ranks; 1,011-1,037 unseen rows/rank at 16861) | epoch 2 = the same rows again, flipped. The ~1k rows/rank seen flipped in 16861..17872 repeat exactly; the rest were seen unflipped by PPM. | none (epochs) |
| chat (corpus15m splits 2,3 + appended unused split 1) | **24,305** (≥ 3,485 unseen packs/rank after 17352) | 2nd pass, same order | none (wraps) |
| Gemma txtdup | **39,902** (≥ 11,303 unused packs/rank after 17352, ~1.45M total) | - | **39,902: the run's hard text limit** |
| pose | the new data | - | the combined stream end = `PPM_MAX_STEPS` |

**Binding limit:** pose, i.e. the size of the new data. A plan-1-sized drop gives ≈ 554k rows ≈ +540 steps, ending ~17,880-17,890.

The first other stream to run out of unseen data is speech, at 17,872, at about the same point. After that, speech serves flipped epoch-2 repeats, which the 10-22 flip rule already covers.

Text does not run out:
- Chat repeats only after 24,305.
- Gemma lasts to 39,902.

No fix is needed now. If chat ever binds:
- **Proposed fix (not applied):** append more chat splits through the same `PPM_CHAT_FRESH_SPLITS` mechanism (per-rank prefixes stay unchanged). It needs the same "append allowed" guard as the pose append.
- On disk there are only klpre splits 1-3. corpus15 splits 4-7 are not local and would need a download. `e28-split0` was seen by the stage-1 init.

Gemma has ~1.45M unused packs, so no action is needed there.

## 4. Launch (after the ingest prints READY, 27114 has finished, and dcp-17352 is COMPLETE on all 16 hosts)

```bash
C=/data/home/amartyavarma/ppm/code-811; CTL=/data/home/amartyavarma/ppm/control/ppm-v1
# preconditions
grep -a "done 17352 steps" /data/home/amartyavarma/ppm/logs/ppm-27114.log && ! grep -aq "STOPPED at step" /data/home/amartyavarma/ppm/logs/ppm-27114.log
squeue -h -j 27114,27115,27116            # all gone: watchdog 27116 exits FINISHED and eval logger 27115 stops at 17352
                                          # (if 27116 is somehow still up: scancel 27116 - it must not adopt the new job with TARGET 17352)
cat $C/fresh/append/npl2/dryrun/RESULT.json   # resume_is_final_save: true, flags; GUARD.json: ok (re-run the ingest command if it ran before 17352)
M=$(grep '^PPM_MAX_STEPS=' $C/prod_811_fresh_append.env | tail -1 | cut -d= -f2); echo $M      # a number, not FILL_ME
# 1. trainer (the first launch pins the resume source, as 27114 did; watchdog relaunches use auto = the newest COMPLETE save)
J=$(PPM_ENV=$C/prod_811_fresh_append.env PPM_RESUME_FROM=/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17352 sbatch --parsable -N 16 -w $(cat $C/nodes16_s7.txt) $C/ppm811.sbatch)
echo $J > $CTL/CURRENT_JOB_811
# 2. watchdog (own state in $CTL/watchdog811f_state; relaunches only from saves >= 17352; exits when "done $M steps")
ENVF=$C/prod_811_fresh_append.env TARGET=$M MIN_STEP=17352 sbatch $C/ppm811f_watchdog.sbatch
# 3. eval logger -> fresh/EVALS_append.md
MIN_JOB=$J END=$M sbatch $C/fresh/append/eval_log_append.sbatch
```

What to check in the new log (`ppm/logs/ppm-$J.log`):
- `[resume] FRESH POSE APPEND ... added at step [17352]`
- `[fresh] POSE APPEND: ... ends after each plan [17352, M]`, with digests recorded
- `resumed .../dcp-17352 at step 17352`
- the start eval vs 27114's last eval

HF watcher (`setup-logs/hfb`, job 27117):
- No action is needed. It adopts the new `ppm811` job and snapshots its code.
- No done records exist past dcp-17229, so the new saves back up as plain `dcp-<step>/`. A colliding step would be re-tagged `dcp-<step>-j<job>` automatically by `lineage_check`.
- dcp-17352 itself is 27114's final save and uploads as usual.

Live page: `ppm_doc.py` follows the newest 8,1,1 leg back through its resume (new job -> 27114 -> ...) on its own.

## 5. Disk (/scratch per host)

Free space measured at 10-22 10:27 PT (dcp-16861 protected, plus 17229 and 17281 live, plus the 17281 HF hold):

| host | free GB |
|---|---|
| slinky-7 | 136 |
| slinky-6 | 147 |
| slinky-8 | 150 |
| slinky-2 | 162 |
| slinky-9 | 164 |
| slinky-5 | 171 |
| slinky-12 | 177 |
| slinky-14 | 174 |
| slinky-3 | 176 |
| slinky-19 | 182 |
| slinky-13 | 190 |
| slinky-17 | 193 |
| slinky-11 | 194 |
| slinky-16 | 197 |
| slinky-0 | 210 |
| slinky-1 | 215 |

It stays about the same at 17352: the final save is added and 17229 rotates out.

New data footprint:
- ≈ (new parquet GB / 16) x 1.03 + ~1 GB plan dir per host.
- For a plan-1-sized drop (~475 GB parquet) that is **~31 GB/host**. `inputs.json` and `runtime.json` are not fetched.
- Scan and plan outputs go to Weka.

Training saves (~53 GB needed per save) still fit everywhere.

**Shortfall: HF backups.** The watcher holds a save only if `free - saves already held - 46 GB >= 80 GB`, i.e. ~126 GB free. After a plan-1-sized ingest:
- blocked: slinky-7 (~105), slinky-6 (~116), slinky-8 (~119)
- borderline: slinky-2 and slinky-9 (~131-133)

So every hold, and with it every HF backup of the append run's saves, would stall.
- Fix: free ~25-50 GB on slinky-7, 6 and 8 before or at launch. Not done here.
- Candidates on slinky-7: `ppm-qqa-scale` 165G, `qwen-s3-ckpt` 120G, `qwen-s4-ckpt` 76G, `asrppm` 73G, `aloe-v2vox-asr-w15973` 73G.
- Candidates on slinky-6 and slinky-8: `qwen-s4-ckpt` 103G, `qwen-s3-ckpt` 93G, `aloe-v2init-10351` 54G.
- Only after confirming exports or backups exist (never-delete-unexported rule).
- A bigger drop scales linearly. `src_place.py` aborts if the data does not fit above a 60 GB floor on every host, and lists the hosts left below the HF-hold floor.

## 6. Choices made (change before running if wanted)

- `PPM_FRESH_POSE_TAIL=1`: uses the 26,073 never-used plan-1 rows. 0 drops them.
- `FRESH_PRIOR_VIDEO=overlap`: plan-1 videos are allowed only for new, non-overlapping source spans. `video` drops them entirely.
- Tag `npl2`, with plan dir `/scratch/amartyavarma/ppm-fresh-pose-npl2` and data `/scratch/amartyavarma/npl-append/npl2`.
- Recipe otherwise = v4 / 27114 exactly. The resolved-cfg diff vs 27114 is only `fresh.pose_root2`, `fresh.pose_tail`, `max_steps` and `resume_from` (`CFG_DIFF_append_vs_27114.txt`).
