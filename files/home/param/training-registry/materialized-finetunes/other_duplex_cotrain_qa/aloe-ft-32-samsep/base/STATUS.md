# ALOEFT32 base: PPM 32B init for the Aloe FT (BASE sub-agent)

## READY (10-23 22:33 PT): QQA_INIT=/scratch/amartyavarma/aloe-samsep-16861-init on slinky-18 AND slinky-4
Format ppm-local-bf16-shards-v1, PPM step 16861 (dcp-16861, job 26325), world 128, a full copy per host: 128 rank<r>.pt + rep.pt +
meta.json + COMPLETE (= 16861), 88.41 GB each. Any rank-to-host layout works.
Job 28238 COMPLETED in 9:20 (HF 530 GB raw per host: slinky-18 430 s, slinky-4 556 s).
- Checks: all 129 raw files' sha256 == HF lfs. The converted files are byte-identical across the two independently built hosts
  (129/129 sha256 equal; ledgers in base/ctl/ledger-16861-<host>.json). rep.pt is 2301002223 bytes (sha 53a6e621...), the same size as
  the monoft-made 17850 rep.pt, which confirms the naming fix.
- /scratch free after: slinky-18 151 GB, slinky-4 106 GB (the partial 17850 dirs, 26 GB per host, are still there and untouched).
- Not run: an explicit CPU load_init (tools/verify_17850_full.py exists, per the coordinator's "launch ASAP"). The real run's init load
  is the load check. Raw downloads were deleted after conversion (our own transient duplicates).
- Cosmetic: the stage log's DONE line prints a literal '{sum(...)}' (an f-string slip). All 129 files were converted.

## BASE = dcp-16861 (10-23 22:24 PT, coordinator: the pure PPM run's last save, prod_ppm.env job 26325). Was STAGING, job 28238
- Source: HF InternalCan/ppm-v1-b/dcp-16861. Complete and verified per hfb watch.log 10-21 07:00: "finalize: bundle rc 0 PASS True,
  hosts 16, HF host files 320, 737.4 GB, bad []; release dcp-16861 (uploaded + verified on HF): 16/16 hosts".
  The MoT hosts are not read.
- Target: /scratch/amartyavarma/aloe-samsep-16861-init on slinky-18 (24 streams) and slinky-4 (10 streams). Format:
  ppm-local-bf16-shards-v1, world 128, a full copy (128 rank files + rep.pt + meta.json + COMPLETE, ~88.4 GB). QQA_INIT = that path.
- Checks: every raw file's sha256 == HF lfs sha256 (end to end). There is no monoft manifest for 16861, so the converted files' sha256
  go to base/ctl/ledger-16861-<host>.json. Finalize writes COMPLETE = 16861. The load check is the real run's load_init.
- Same maths as w-15973 (checked in code): safety_ppm.export_weights = get_model_state_dict(model) -> .to(bfloat16) -> consolidated DCP.
  Ours = the node save's 'model' section (fp32 named_parameters shards, safety_ppm.local_state) -> .to(bfloat16) per tensor. Same
  key set: w-15973's .metadata has exactly the 2150 keys of the node-save layout (1522 sharded + 628 replicated incl. dec_s/dec_p),
  0 extra / 0 missing, all bf16. It is the same fp32 -> bf16 cast of the same tensors, and the loader skips the same pose prefixes in
  both paths (tower_p, dec_p, decp_bb, inp.pose, pose_head). Only the container differs (per-rank shards vs consolidated DCP).
- ASR adapter: none matched to 16861. No ASR training has run since asrppm-w15973 (job 26064, 10-20) -> use w-1441.
- The rep.pt naming fix (tmp dir, same basename) is kept. The partial 17850 dirs are left in place and nothing is deleted.

## (superseded) HOLD on 17850 (10-24 ~22:20 PT, coordinator per user via main: "17850 may be the wrong base")
Staging stopped. My own job 28235 was cancelled; it had run 4 s and already died on slinky-4's /scratch-floor assert, so it did nothing.
No aloeft32-base-* job is running. Nothing was deleted; partial files stay where they are:
- slinky-18: /scratch/amartyavarma/aloe-samsep-17850-init has 16 rank files + rep.pt (13 GB); -dl has 13 GB of raw downloads; /scratch free 234 GB
- slinky-4:  /scratch/amartyavarma/aloe-samsep-17850-init has 17 rank files + rep.pt (13 GB); -dl has 13 GB of raw downloads; /scratch free 163 GB
- There is no COMPLETE marker on either host, so the trainer will refuse these dirs (load_init asserts COMPLETE). They are NOT usable as QQA_INIT.

## What ran (job 28227, 10-23 22:16 PT, 1:44 min, FAILED)
- Every converted rank file was byte-identical to the 17850 monoplex FT init manifest (sha256 + size, 33 files across the two hosts).
  The raw HF downloads also matched HF's lfs sha256. Throughput: 4.04 GB raw per rank file in 16-28 s per stream at 8 streams,
  about 1-2 GB/s per host.
- rep.pt sha256 differed: 8c9a2b0c... vs manifest 58ce016f... (2301037917 vs 2301002223 bytes).
  Tensor level: the same 628 keys as the manifest's rep_keys, all bf16, the same order source (raw sd["model"] insertion order).
  The raw rep.pt has fp32 'model' (628) plus opt / opt_dec_s / opt_dec_p sections.
  Cause: torch.save names the zip archive after the target file's basename. ckpt_node_export.py wrote rank files via rank<r>.pt.tmp
  (which I reproduced, so they matched) but wrote rep.pt DIRECTLY (archive 'rep'). I wrote it via rep.pt.tmp (archive 'rep.pt'),
  so every record name is 3 bytes longer and the 64-byte record padding shifts: about 35 KB, and the weights are unaffected.
  Fixed in tools/stage_17850.py (rep.pt is now saved via a tmp dir under the same basename). The fix was not run because of the HOLD.
  Per-tensor equality was not run either. The manifest has no per-tensor digests (only rows per tensor), so a re-run of the fixed
  converter matching 58ce016f... byte-for-byte is the check.
- Job 28235 (resubmit with 24 / 16 streams) failed on slinky-4: 174 GB free - 77 GB still to write - 16 x 4.1 GB - 14 GB rep reserve
  is below the 30 GB floor. The rep reserve double-counts the rep raw that is already downloaded. Use 12 streams on slinky-4, or drop
  that reserve, when resuming.

## To resume (only if main confirms 17850)
`sbatch /data/home/amartyavarma/ppm/aloe-ft-32-samsep/base/tools/stage.sbatch`. It is resumable: converted files that match the
manifest sha are kept, and raw files already downloaded are reused. Lower slinky-4 to W=12 in tools/stage_host.sh first.
The post-stage step is tools/verify_17850.py = finalize only (stage marker + sizes + meta -> COMPLETE), per the coordinator's
"launch ASAP" instruction. The full CPU load_init / structure check is tools/verify_17850_full.py (not run).
ETA from now is about 6-10 min per host at the measured rate.
For a DIFFERENT step: the same tools work for any dcp-N on HF that has a matching monoft-style manifest. Without a manifest, drop the
manifest sha check and rely on the HF lfs sha plus the trainer's load.

## Source of 17850 found (in the order asked)
1. 17850 monoplex FT (aloe-v2pure-main-17850). Init job 27756 converted dcp-17850 to node-local `ppm-local-bf16-shards-v1`
   (world 128, bf16, rank<r>.pt + rep.pt + meta.json + COMPLETE). Each FT host held only 20 of the 130 files, and all 8 copies were
   deleted 10-22 23:36 PT (rm_saves-27760.log). Only the manifests survive: aloe-v2pure-main-17850/initctl/manifest-*.json, with
   size + sha256 + rows per tensor of every converted file (130 files, 88.41 GB).
2. HF: dcp-17850 (final save of job 27653, 737.4 GB full training state) is COMPLETE and VERIFIED. The 16 host folders are in
   InternalCan/ppm-v1-c/dcp-17850/<host>/ and the bundle is in InternalCan/ppm-v1-d (hfb watch.log 10-23: finalize PASS, 16/16 hosts).
   No other 17850 copy exists: ckpt-infer/ppm-v1 has local-11488/12529/12829/17641 only, and ckpt-weka/ppm-v1 has w-15973 only.
   The MoT hosts were never read.
3. Converter: monoft ckpt_node_export.py (= ckpt_local_export_slot.py maths). w-15973 = consolidated bf16 DCP (safety_ppm.export_weights).

## Format the reference trainer consumes
train_aloe_v2vox_raw.py load_init(QQA_INIT) accepts two formats:
- If meta.json is present: ppm-local-bf16-shards-v1. Any source world works; each rank mmaps only the source rank files covering its
  FSDP2 rows, plus rank0.pt and rep.pt; COMPLETE is required.
- Otherwise: a consolidated DCP (.metadata + COMPLETE).
Pose parts are skipped (asserted). Chosen: ppm-local-bf16-shards-v1, world 128, a FULL copy per host (88.4 GB each), so any
rank-to-host layout works.

## ASR adapter
No Voxtral ASR adapter newer than asrppm-w15973 w-1441 exists for the 32B. asr-adapter/ckpt-weka/asrppm-w15973 has
w-1200/1320/1440/1441 (job 26064, backbone w-15973), and no ASR training job has run since 10-20 (sacct).
asr-adapter-14b is the 14B adapter (10-18).
