# pose6/mono17850: the mono8020 recipe with init = 8,1,1 dcp-17850 (kit, NOT launched; nothing paused)

Main asked for this at 10-27 14:29 PT; written at 14:37 PT. The playground model export-357 was fine-tuned from local-11488. This kit runs the aloe-pose-mono-8020 recipe (job 24272) EXACTLY, with one change: the init is the 8,1,1 run's dcp-17850, in the old architecture (`PPM_S2P` and `PPM_G0COND` both OFF).
- **Base:** C1 converts dcp-17850 to `/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17850` (ETA ~16:30 PT). It has the same format and world (ppm-local-bf16-shards-v1, 128) as mono8020's local-11488, and is read in place the same way.
- **Code:** pose6/code with both flags off. B1 certified it identical to mono8020's trainer. md5s are 212b1ee9 / 47a55ed5 / df679691.
- **Flagged knob:** the decp_bb line is explicit, `QQA_DECP_BB_OFF=1`. dcp-17850 has been link-off since 10-20. The default is OFF; `--link-on` gives mono8020's value.
- **Separation:** the kit reuses pose6/kit's machinery (tools, overrides, run script logic) without changing the s2pg0 kit. Its node-local tree is `/scratch/amartyavarma/pose6-mono17850/`.

## GO (main 14:41 PT): option (a') with the link ON. Not submitted by me

Preconditions: C1 has verified local-17850, and pose5 has paused 31101 and its hosts are free (~3 min after STOP_AFTER_SAVE).

1. **Launch.** The preflight in this command runs live `df` on every host, the base key check on local-17850, and checks that every host is idle:
   `bash /data/home/amartyavarma/ppm/pose6/mono17850/launch.sh --saves none --final-export-only --link-on --hosts slinky-15,slinky-14,slinky-5,slinky-7,slinky-1,slinky-2,slinky-17,slinky-3 --go`
2. **Verify the final export** (exit 0/1):
   `bash /data/home/amartyavarma/ppm/pose6/mono17850/verify_final.sh /data/home/amartyavarma/ppm/pose6/mono17850/export-357`
   - **Trainer log lines:** `[export] ...: consolidated bf16 weights`, `[export-local] ... export-357 on every node`, `done 357 steps`, `== exit 0`.
   - **Checkpoint checks** (read-only srun with ≤4 CPUs): COMPLETE = 357; keys and shapes equal export-357's (2,150); all bf16; all values finite.
   - **Config checks:** init = local-17850, link ON (`decp_bb_off` absent), no s2p / g0cond, `save_every_min` 100000, `export_steps` [].
3. **HF push**, after 31101 resumes, on a small allocation:
   `sbatch -w <node with 4 free CPUs, never slinky-9> /data/home/amartyavarma/ppm/pose6/mono17850/push_hf.sh /data/home/amartyavarma/ppm/pose6/mono17850/export-357`
   - Destination: the private repo `InternalCan/ppm-pose6-mono17850-final`, path `export-357-dcp/`.
   - Method: the 27757 chain's (`hfb_up.py`). Its verify checks that the repo is private, that every file has the same size, that the LFS sha256 of every file matches, and streams 3 files back.
   - Never GCS; the token is never printed. The farm and caches go under `mono17850/hf` on /data.
4. **Per-host write log:** on (`logs/disk-<job>.txt`).
5. **Resources:** the FT takes 32 CPUs + 8 GPUs + 500G per host and is not exclusive. That leaves 32 CPUs and ~92 GiB per host for hfb's uploads of the pause save.

## CFG_DIFF vs 24272 (`CFG_DIFF.txt`)

The GO config is **M6, 10 keys** (job 31175): `init`, 7 names/paths, `save_every_min`, `export_steps`. `decp_bb_off` is **absent**, and the decp_bb group is trained, as in mono8020. The earlier variants (job 31167/31171) are below.

Default (M1): **9 keys** = `init`, 7 names/paths and the flagged `decp_bb_off`. The rows digest is 3681cc26ea97cad7 and the plan digest a5d995bcec45dc6f, both equal to 24272's.

| variant | keys | extra key(s) vs M1 |
|---|---|---|
| M2: placement (a), `--saves none` | 10 | `save_every_min` |
| M3: placement (b), fallback | 11 | `micro_layout` lpt, `save_every_min` |
| M4: `--link-on` | 8 | none: exactly init + names/paths |

The base is a STAND-IN for now: the path string of local-17850, which does not exist yet. The preflight (`--mods none`: the key set must equal the FT model's exactly, 2,150 keys, all bf16) passes on the 8,1,1 local-17641 and on export-357. It runs again on local-17850 at launch.

## (a) 64 ranks on 8 of 31101's 16 hosts, during a STOP_AFTER_SAVE pause of 31101

`bash /data/home/amartyavarma/ppm/pose6/mono17850/launch.sh --saves none [--go]`
- Hosts: slinky-15 (rank 0), 14, 5, 7, 1, 2, 17, 3. Spares: slinky-8 (97.1), slinky-0 (94.6), slinky-11 (93.9).

**/scratch free per host during a pause.** Source: pose5's hold-chain records (`pose5/kit/state/hold_chain/free-<host>.tsv`, GB, read 14:31 PT; no df on 31101's hosts). Figures are in GiB, shown as *now* / *while the previous save is still held*.
- The pause save adds 46.09 GB per host.
- The previous save stays hard-link-held until the pause save is HF-verified. That took 47-96 min for dcp-17956 and dcp-17903.

| slinky-15 | 14 | 5 | 7 | 1 | 2 | 17 | 3 |
|---|---|---|---|---|---|---|---|
| 569.9 / 527 | 214.9 / 172 | 166.2 / 123 | 136.4 / 93 | 115.8 / 73 | 111.9 / 69 | 106.7 / 64 | 102.1 / 59 |

**FT need per host** (GiB). Floor rule = measured write + 30; recipe rule = mono8020's.

| `--saves` | floor rule (rank 0) | recipe rule | hosts that fit after the hold release |
|---|---|---|---|
| hourly (the recipe) | 238 (251) | 237 | slinky-15 only |
| final | 165 (178) | 175 | slinky-15, 14, 5 |
| **none** | **92 (105)** | 113 | **all 8** |

**Minimal flagged deviation: `--saves none`.** No FULL save; the six weights-only exports stay. That changes the cfg key `save_every_min` and the env-only `ALOE_SKIP_FINAL_FULL`. Save cadence only, maths unchanged. The cost: no resume point, so a crash means a rerun of ~80 min.
- **It cannot start until the pause save is HF-verified.** While the previous save is held, only slinky-15, 14, 5 and 7 have 92 GiB.

**Before 31101 resumes:**
- 31101's next save (46.09 GB) plus the held pause save need ≥ ~71 GiB free per host.
- After the FT, slinky-3 would have 102 − 62 = 40 GiB.
- So the FT's node-local files (~62 GiB per host, 75 on rank 0, under `pose6-mono17850/`) must be backed up to the private HF repo `InternalCan/ppm-pose6-mono17850` (never GCS), then removed on main's OK.
- The final export-357 also lands on /data (`mono17850/export-357`, 88.4 GB, if ≥ 300 GB stays free). `launch.sh` section 7 prints this list.

**31101 resume-back:**
- **Who:** main, via pose5's relaunch path.
- **From:** the pause save (`PPM_RESUME_FROM=auto` = dcp-S, the same 16 hosts, the same aliases).
- **Restart time:** measured 11:35:08 submit → 11:40:40 first step, so ~6 min.
- **npl5 roll supervisor:** 31105 on slinky-4 keeps running and is ~100 steps ahead, so no refetch.

**ETA and pose-run delay:**

| step | duration |
|---|---|
| STOP_AFTER_SAVE (≤1 step of 69 s + 42 s save) | ~2 min |
| wait for the pause save's HF verify | 47-96 min |
| FT to its FINAL export (mono8020 job 24272: 80.2 min wall incl. evals / exports; base read 17 s) | ~82 min |
| **ETA to final export** | **≈ 2.2-3.0 h after the pause trigger** |
| HF backup of the 6 node-local exports (530 GB, at the hfb rate of 8-15 GB/min) + main's OK | 35-65 min |
| 31101 restart | 6 min |
| **Total 31101 delay** | **≈ 2.9-4.2 h**, ~200 steps of 31101 |

## (a') = (a) + `--final-export-only`: only the final export (flagged user call)

`bash /data/home/amartyavarma/ppm/pose6/mono17850/launch.sh --saves none --final-export-only [--go]`
- Same hosts and code as (a).
- Config diff vs 24272 (M5): **11 keys** = (a)'s 10 + `export_steps` [60, 119, 178, 238, 298] -> []. That is the export schedule only; maths unchanged.
- The final export is written twice: the node-local `export-357` slice, and the consolidated `/data/home/amartyavarma/ppm/pose6/mono17850/export-357`.

**Per-host need:** 41 GiB, or 44 on rank 0 (floor rule: write 11 = one 10.3 GiB export slice, +3 GiB rep.pt on rank 0, + 30). mono8020's style of rule gives 83/8 + 50 = 60. Against the 14:31 records, now / while the previous save is held:

| host | slinky-15 | 14 | 5 | 7 | 1 | 2 | 17 | 3 |
|---|---|---|---|---|---|---|---|---|
| free (GiB) | 570 / 527 | 215 / 172 | 166 / 123 | 136 / 93 | 116 / 73 | 112 / 69 | 107 / 64 | 102 / 59 |

Every host is above the need even while held (lowest: slinky-3 at 59).

**Start: right after the pause save.** No wait for its HF verify is needed.

**Before 31101 resumes, nothing has to be backed up or removed.**
- The FT leaves ~11 GiB per host (14 on rank 0) plus its compile caches.
- 31101's first new save comes ~60 min after its resume and needs ~71 GiB (46.09 GB save + 30 GB floor).
- By then the pause save has been HF-verified (47-96 min after the pause; FT + resume ≥ 88 min), so the previous save's hold is released. slinky-3 then has ≈ 102 - 11 = 91 GiB.
- The only risk is an HF verify slower than ~140 min. Then slinky-3 would have 59 - 11 = 48 GiB.
- Removing the node-local export slices can wait for main's OK; the /data export-357 is the copy.

**ETA to the final export:** ≈ 1.4 h after the pause trigger. That is ~2 min stop + ~3 min launch/preflight + ~77 min FT. The FT time is 24272's 80.2 min minus the 5 intermediate exports (5 × 12 s), the 5 evals tied to export steps (~20 s each), the hourly save (36 s) and the final FULL save (29 s).

**Total 31101 delay:** ≈ 1.5 h (stop 2 + FT ~82 + restart 6), about 78 steps.

**/data headroom** (floor 300 GB; consolidated export-357 = 88.4 GB):
- /data has 1,143 GB free (14:38 PT).
- After local-17850 (88.4 GB): ~1,055. After 31101's ~17:40 export: ~966. After the FT export: **~878 GB ≥ 300**.

## (b) One spare 8-GPU node, the fallback

`bash /data/home/amartyavarma/ppm/pose6/mono17850/launch.sh --nodes 1 --hosts slinky-10 [--final-export-only] [--go]`
- **Node:** slinky-10. **It displaces motasr2** (31088/31090 hold its 8 GPUs; main's call). The non-GPU services there can stay: the kit needs 32 CPUs and 530G.
- **Not slinky-4:** demos, pose5's companions (incl. 31105, the npl5 roller 31101 needs), and GPUs 6-7 throttled (the run script aborts on thermal slowdown).
- **Not slinky-18:** motasr2 plus demos, and /scratch at 20 GiB.
- **/scratch:** need 528 GiB (6 exports × 83 + 30; no FULL saves). Current slinky-10 free is not readable (0 free CPUs). It had 461 GiB at 05:40 PT, before motasr2 started, so it likely needs `--final-export-only` (113 GiB; user call: export schedule only, maths unchanged).
- **Wall to final export:** ~14 h (12-16), based on Rosie 30847's world-8 timings.
- **Maths:** world 8 ⇒ `micro_layout` lpt, same objective, NOT bit-identical (see pose6/kit/README.md "Fallback"). The user's OK is needed.

## Files

| file | what |
|---|---|
| `mono17850.env` | `aloe_mono8020.env` verbatim + `[M17850]` lines |
| `mono17850_run.sh`, `mono17850.sbatch` | run script and job header (from pose6/kit) |
| `launch.sh` | one command, dry run by default |
| `CFG_DIFF.txt` | resolved-config diff vs 24272 |
| `tests/dry_all.sbatch` | the dry-run job; results in `tests/out/` |

The shared tools are in `pose6/kit/tools`. `preflight.py` gained `--mods none`; the s2pg0 default is unchanged.
