# ppm-v1 dcp-18098: restore notes

Kind `dcp`, step 18098, written by ppm job 31101. This bundle is in `InternalCan/ppm-v1-d`; the per-host shard folders `dcp-<step>/<host>/` of a
full save are in `InternalCan/ppm-v1-d` (checkpoints are split across a chain of repos `InternalCan/ppm-v1`, `-b`, `-c`, ... because HF caps a
repo at 20,000 files; `_backup/MANIFEST.dcp-18098.tsv` lists every file with its source, `hf://<repo>/<path>` = kept in that repo).
- `w-<step>/` = the trainer's 6-hourly consolidated **bf16 weights export** (torch DCP: `.metadata` + `*.distcp`, `COMPLETE`;
  model only, no optimizer). Load with `torch.distributed.checkpoint` (`dcp.load(state_dict, checkpoint_id="w-<step>")`, keys = the
  PPM model's `get_model_state_dict`), or convert with `run_state/code/ppm-code/ckpt_local_export.py` / `infer_ppm.py`.
- `dcp-<step>/<host>/` = the **full resumable save** of every training host (per-rank fp32 shards `rank<R>.pt` with weights + 3
  optimizers + scheduler + RNG, `digest-*.txt`, `COMPLETE`, train state), exactly as written node-locally to
  `/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-<step>/` on each host. Rank R = 8 x (index of the host in the SORTED host list) + local id.

## What is in `run_state/`
- `code/ppm-code/` : snapshot of /data/home/amartyavarma/ppm/code (incl. lib/, vendor/, tests/) as job 31101 ran it;
  `code/ext/qwen-1ch/` : the qwen-1ch modules imported at runtime (chat renderer: data_q, render_q, stage2/*); `code/ext/env/` :
  env.sh sourced by task_ppm.sh, `pip freeze` of the training venv (/data/home/amartyavarma/venvs/qwen-s2), python version.
  `SHA256SUMS` of the snapshot is in the meta repo (`InternalCan/ppm-v1-meta`, codesnap/<job>/).
- `config/` : `cfg.json` (the resolved trainer config the job logged), `prod_ppm.env`, `ppm.sbatch`, `task_ppm.sh`, `slurm_job.txt`
  (submission line, hosts), `job_log_excerpt.txt` (data/resume/init/eval/save/export lines of the job log).
- `data_manifest/` : `stores_and_repos.json` (every store's node-local path, its HF repo, data knobs, eval holdouts),
  `s3-rows-keys/` (census key list per slice), `podcast_full_rows*.parquet`, `extra_plan_d2.json` (+ native counts; the D2 rule:
  PPM_EXTRA_FROM_STEP=170), `extra_slice_listings/` (file sizes of the slinky-4/7/15/18 slices), and `rank_plan/` (extract `meta.tar` there first: `tar -xf meta.tar`; regen_rank_plan.py +
  rank_plan_digest.json + DATA_PLAN.md: regenerates every rank's row plan/order from the seeds and checks it against the stored digest;
  DATA_PLAN.md maps a step to each stream's data position).
- `POSITION.json` : the step and data position of this checkpoint.

## Rebuild the node-local stores (each node)
Use /data/home/amartyavarma/venvs/duplex/bin/hf (huggingface_hub with hf_xet). On every training host H:

    X="--repo-type dataset --exclude RESTORE.md _backup/* .gitattributes --max-workers 32"
    hf download InternalCan/ppm-s3-rows-$H        $X --local-dir /scratch/amartyavarma/s3-rows            # H's own slice
    hf download InternalCan/ppm-pose-s3-samdata-rows $X --local-dir /scratch/amartyavarma/pose-s3-samdata/rows
    hf download InternalCan/ppm-txtdup-full       $X --local-dir /scratch/amartyavarma/ppm/txtdup-full
    # D2 extra slices (from step 170): for each slot of H in data_manifest/extra_plan_d2.json (hosts.H.slots[]):
    #   hf download InternalCan/ppm-s3-rows-<slot.node> $X --local-dir /scratch/amartyavarma/s3-rows-extra/<slot.node> --include <slot.files...>
    rm -rf /scratch/amartyavarma/s3-rows/.cache /scratch/amartyavarma/s3-rows-extra/*/.cache   # hf download metadata

Slices are keyed by the node that built them (NODE.json `node`); `stores_and_repos.json` has the slice -> repo map. Shared (Weka) inputs:
chat render tables `InternalCan/ppm-chat-corpus15m-render`, gemma eval store `InternalCan/ppm-gemma-txtdup-v1`, eval sets and the
census in `InternalCan/ppm-v1-meta`; the base model is /data/home/amartyavarma/models/Qwen3-32B (Qwen/Qwen3-32B).
Pose node swap (PPM_NODE_SWAP in prod_ppm.env) only changes which host's pose store a host reads; every host reads its own s3-rows slice.

## Resume exactly
1. Restore the stores (above) on the same 16 hosts (`config/slurm_job.txt`; hosts sorted = rank order) and put the code snapshot at
   /data/home/amartyavarma/ppm/code (paths in the code are absolute) and the qwen-1ch modules at /data/home/amartyavarma/qwen-1ch.
2. Full resume (dcp): on each host H: `hf download InternalCan/ppm-v1-d --include "dcp-18098/H/*" --local-dir /tmp/x` and move
   `/tmp/x/dcp-18098/H/*` to `/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-18098/`. Then
   `PPM_ENV=/data/home/amartyavarma/ppm/code/prod_ppm.env sbatch -N 16 -w <hosts> ppm.sbatch` (PPM_RESUME_FROM=auto picks the newest step
   COMPLETE on every node; the trainer checks per-rank digests on load). Keep PPM_EXTRA_FROM_STEP=170.
3. From a `w-` export only (no optimizer state): weights for eval/inference or a warm start; it is not an exact resume.
4. Check the data plan first: `python run_state/data_manifest/rank_plan/regen_rank_plan.py --check` (see DATA_PLAN.md).
