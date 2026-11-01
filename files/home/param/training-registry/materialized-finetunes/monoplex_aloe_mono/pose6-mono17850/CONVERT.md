# C1: dcp-17850 -> ckpt-infer/ppm-v1/local-17850 (ppm-local-bf16-shards-v1)

Workstream C1 of pose6, 10-27. Output: `/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17850`. It has the same layout as `local-17641`: `meta.json`, `rank0.pt` .. `rank127.pt`, `rep.pt`, `COMPLETE` (131 files, 83 GB).

STATUS 10-27 16:30 PT: **READY and verified.** `COMPLETE` = `17850`. Every one of the 129 files is byte-identical (sha256) to the 10-22 conversion of the same save, and both verifies PASS. Details in "Result" at the end.

## Source
- HF `InternalCan/ppm-v1-c`, pinned revision `c9b2213751cb3f2ede1dce2c5b9ed275b5cc3ca2`, path `dcp-17850/<physical host dir>/`.
- The run records LOGICAL hosts in `train_state.json` `hosts`. HF dirs are physical: logical `slinky-10` = dir `slinky-7`, and `slinky-9` = dir `slinky-9`.
- The converter checks that every `rank<R>.pt` exists in exactly the dir `alias(hosts[R])`, and that every dir's `COMPLETE` == `17850`. Either failing = fail loud.
- Read-only. Nothing was touched on 31101's hosts. The Weka copy is gone.

## Method (`convert/hf_local_export.py`)
1. **Range reads only.** Per file, the converter fetches the zip central directory, then `data.pkl`, then only the storage records of the `model` section, by HTTP range requests to the file's signed CDN URL (resolved once per file).
   - Per rank: 1.34 GB of 4.04 GB. The AdamW states are never downloaded.
   - `rep.pt`: 4.60 GB of 13.81 GB.
   - Total: about 177 GB downloaded, nothing staged on disk.
2. **Integrity.** Every record's CRC32 == the zip central directory's, and its length == numel x itemsize. Any mismatch fails loud.
3. **Tensor rebuild.** `data.pkl` is unpickled with a stub unpickler that records each tensor's storage key, dtype, offset, size and stride without reading any tensor data. Each tensor is then rebuilt as `torch._utils._rebuild_tensor` does (`set_(storage, offset, size, stride)`) on a 64-byte-aligned buffer.
4. **Conversion = `ckpt_local_export.py` (md5 87c639ee) verbatim:**
   - `out = {n: t.to(torch.bfloat16).contiguous().clone() for n, t in sd["model"].items()}`
   - `torch.save` to `rank<R>.pt.tmp`, then `os.replace` (the archive name inside the zip = `rank<R>.pt`, as the stock tool)
   - `rep.pt` saved directly
   - same `meta.json` dict / keys / order. `src` = `hf://InternalCan/ppm-v1-c@c9b2213751cb3f2ede1dce2c5b9ed275b5cc3ca2/dcp-17850`.
   - `COMPLETE` = the stock script's own `--finalize`.
5. **Inline checks.** Every output tensor is finite before it is saved. Each file's sha256 is compared with the 10-22 monoft manifests of the same dcp-17850 conversion (`aloe-v2pure-main-17850/initctl/manifest-*.json`: sha256 + per-tensor dim-0 rows).
6. **Write and gate.**
   - Output is written as `local-17850.partial`.
   - The rename to `local-17850` happens only if all 129 files' sha256 / size / rows == the manifests.
   - Then the stock `--finalize` writes `COMPLETE`, then verify (2) runs. If verify (2) fails, `COMPLETE` is moved aside.

## Verify (1): the converter on dcp-17641 vs the existing local-17641
Job 31173 (slinky-18, -c 2, 24G, 141 s). Source: HF `InternalCan/ppm-v1-c@c9b22137 dcp-17641`, the same save as local-17641's `/scratch/.../ckpt-keep-ft/dcp-17641`.

| file | tensors | torch.equal + dtype + shape + key order | file sha256 == local-17641's |
|---|---|---|---|
| rank0.pt | 1522 | all equal | yes (38475fe2...) |
| rank64.pt (logical slinky-10 = HF dir slinky-7) | 1522 | all equal | yes (1ceb71f4...) |
| rank127.pt | 1522 | all equal | yes (87fe71b0...) |
| rep.pt | 628 | all equal | yes (1862de22...) |
| meta.json | - | same keys, same order; differs only in `src`, `made` | - |

`torch.save` output is deterministic, so whole files are byte-identical. All 1522 + 628 CRC32 checks were OK. Stats: `convert/test17641_stats.json`. The test output was removed after the comparison.

## Commands
```
sbatch /data/home/amartyavarma/ppm/pose6/mono17850/convert/probe.sbatch        # 31169: HF listing + range-read probe
sbatch /data/home/amartyavarma/ppm/pose6/mono17850/convert/test17641.sbatch    # 31173: verify (1)
sbatch /data/home/amartyavarma/ppm/pose6/mono17850/convert/conv17850.sbatch    # 31176: convert + gate + finalize + verify (2)
# verify (2) as steps inside 31176 (see Result):
srun --jobid 31176 --overlap -n1 -c1 --mem=0 --cpu-bind=none python -I convert/verify_local.py --mmap 0 --new <local-17850> --ref <local-17641> --json convert/verify_local17850_fast.json
srun --jobid 31176 --overlap -n1 -c1 --mem=0 --cpu-bind=none python -I pose6/kit/tools/preflight.py --base <local-17850> --mods none --code pose6/code --json convert/preflight-local-17850.json
```
Job 31172 failed at start: the HF `run_state/` dir was picked as a host dir. This was fixed by detecting host dirs from their rank files.

## Speed
- Range reads: 187 MB/s for one file when uncontended; about 58 MB/s per file inside the 2-CPU job, which is CPU-bound (TLS + CRC32 + cast + save + sha256 on 2 cores).
- Only 2 CPUs were free on slinky-18, and every other node was full.

## Result (job 31176, slinky-18, -c 2, 24G)

**Convert**
- 14:46 to 15:58 PT, 4,302 s.
- 129 files. Downloaded 176.7 GB by range reads (avg 41 MB/s; 58 to 318 MB/s per file). Wrote 88.4 GB (83 GiB).
- All 195,444 CRC32 checks OK.
- Per-rank time was about 26 s at first and about 40 s later on. The bottleneck was cast + /data write + sha256 read-back on 2 CPUs.

**Gate**
- All 129 files (rank0..127 + rep.pt): sha256 and size == the 10-22 monoft manifests of dcp-17850 (`aloe-v2pure-main-17850/initctl/manifest-*.json`). Per-tensor dim-0 rows also equal.
- The output is therefore byte-identical to the 10-22 conversion of the same save.
- Then: renamed `.partial` -> `local-17850`, and the stock `ckpt_local_export.py --finalize` wrote `COMPLETE` (`17850`): "128 rank files + rep.pt".

**Verify (2a)** `verify_local.py --mmap 0` vs local-17641 (`convert/verify_local17850_fast.json`, 802 s): **PASS**.
- 129 files, 195,444 tensors, 44,175,900,672 elements, read back from /data.
- Every tensor is bf16 and finite.
- Per file, the key set, key order and shapes == local-17641's.
- The file listing equals local-17641's.
- meta.json has the same keys in the same order, the same `format` / `world` / `hosts` / `run_base`. It differs only in `src`, `step`, `run_tag` (ppm-v1-j27653), `cfg`, `save_time` (10-22 22:28:51 PT) and `made`.

**Verify (2b)** kit preflight `--mods none` (`convert/preflight-local-17850.json`, 68 s): **PASS**.
- 2,150 keys vs 2,150 wanted (= the mono8020 FT model, export-357's table).
- 0 missing, 0 extra, 0 shape mismatches, 0 non-bf16.

**Why job 31176 shows CANCELLED**
- The batch's own verify (2a) ran with `mmap=True`. On Weka, page faults read at only about 25 MB/s, so it would have run until about 17:05 PT on the only 2 free CPUs.
- The identical check with `--mmap 0` (110 MB/s) and the preflight were therefore run as overlap steps inside 31176, and both PASSED.
- Then 31176 was scancelled (my own job), with its slow duplicate check at rank ~50. Cancelling the job also kills its batch script, so the `COMPLETE` move-aside step never ran: `COMPLETE` stayed in place.

**Disk:** /data has 1,045 GB free after (+83 GB).

**Hosts semantics:** `meta.hosts` == train_state `hosts` (LOGICAL, slots slinky-9 / slinky-10, no slinky-7), exactly as in local-17641.

**Files**
- Code in `pose6/mono17850/convert/`: `hf_local_export.py`, `verify_local.py`, `conv17850.sbatch`, `test17641.sbatch`, `probe*.{py,sbatch}`.
- Stats: `conv17850_stats.json` (per file: fetch, CRC, sha256, manifest match), `test17641_stats.json`.
- Logs: `convert/logs/`.
