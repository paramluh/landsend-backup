# Core pretraining trainers

Code, run configs (envs, sbatch, launch kits) and docs of every pretraining trainer under `/data/home/amartyavarma/ppm`, copied
read-only on 2026-10-28 (PT) by `_copy_tool/copy_core.py`. Each subdirectory has a `README.md` (what it is, runs / job ids, init, data,
schedule, checkpoints + HF repos, state). Where the source already had a `README.md`, it is kept as `README_SOURCE.md`.

## Lineage
```
Qwen3-32B ──> ppm32b-main (ppm-v1, jobs 18873..26325, steps 0..16861)
                 ├─> variants/link-exp (26303, 300 pose-only steps from 16762)       ├─> variants/code-frozenbb (speed tests only)
                 ├─> ppm811 (8,1,1: 26391..27653, 16861 -> 17352 -> 17641 -> 17850)
                 │     ├─> ppm811-decay (27810, LR -> 0 branch, 17850..17928)
                 │     └─> ppm811-s2p-pose5 (31101, s2p + G0COND + delivery 5, 17850..18098 PAUSED, target 19382)
                 └─> ppmvox (Voxtral audio-in extension: from 16861 (v1-v3) and from 811 dcp-17850 (v5, v5b))
Mistral-128B ──> ppm-mistral128b (io run 26678/26776, 2 nodes, stopped at 216)
interleave3 (prototype, not launched) ; interleave-ppm (32B interleaved, 22813, stopped at 84)
Qwen3-14B ──> intl14b-next-arch (ppm-next, 22904..28069, PAUSED at 6982)
          ├─> ppm14b (PPM arch on 14B, 26513..28012, stopped at 4502)
          └─> mot14b (MoT 14B, 28071..30289, PAUSED at 14839) ──> motvox (31626, audio-in, LIVE 10-28)
megaport: Megatron-Core port + same-maths speed work of the 811 trainer (tests only, never a production run)
```

## Index
| dir | source | runs |
|---|---|---|
| `ppm32b-main/` | `ppm/code` (+ `ppm/DESIGN.md`, `RESUME_PPM_12829.txt`, `configs/`, ops docs) | 18873 .. 26325 |
| `ppm811/` | `ppm/code-811` (incl. `fresh/` ingest) | 26391, 27030, 27100-27114, 27204, 27653 |
| `ppm811-decay/` | `ppm/code-811-decay` (+ `ppm/decay/PROMPT.md`) | 27810 |
| `ppm811-s2p-pose5/` | `ppm/pose5/code-811-s2p` (code) + `ppm/pose5` (run kit, STATUS, audits, npl5 scripts) | 31101 |
| `intl14b-next-arch/` | `ppm/next-arch` | 22875 .. 23497, 28025, 28069 |
| `ppm14b/` | `ppm/ppm14b` | 26513 .. 26571, 28012 |
| `megaport/` | `ppm/megatron-port` | tests / benches only |
| `mot14b/` | `ppm/mot` | 28071, 28155, 29514, 30289 |
| `motvox/` | `ppm/motvox` | 31626 (live) |
| `ppmvox/` | `ppm/ppmvox` | 28734, 29005, 29813, 29863, 30104 |
| `ppm-mistral128b/` | `ppm/ppm-mistral128b` | 26678, 26764, 26776 |
| `interleave-ppm/` | `ppm/interleave-ppm` | 22792, 22813 |
| `interleave3/` | `ppm/interleave3` | none |
| `variants/link-exp/` | `ppm/link-exp` | 26303 |
| `variants/code-frozenbb/` | `ppm/code-frozenbb` | speed tests only |

## Not included (surveyed `ppm/*` for `train_*.py`)
- Fine-tune trainers (not pretraining; other registry sections): `aloe-*`, `si20k-*`, `xfer-*`, `qqa-ft`, `ttconvo-ft`, `samdata-ft`,
  `pose-s2p`, `pose6`, `asr-adapter*`, `mot-asr`.
- Old snapshots / dev copies superseded by `ppm/code`: `code-bak-pre-d32-1012-0147`, `code-bak-pre-d33-1012-0458`, `code-bak-pre-t25-1012-0427`,
  `code-d2` (10-12 D2 dev copy), `code-speed` (10-12 speed dev copy), `pose5/code-811-g0` (G0COND dev copy; its patch is in `ppm811-s2p-pose5/run/`).
- Analysis / diagnostics, not trainers: `gradtest`, `nodediag`, `mot-bench` (MoT-vs-dense benchmark, README only relevant), `steptime`.

## Copy rules
Text code/config/doc files only (`.py .sh .sbatch .env .md .txt .json<=256K .patch .diff .html ...`), each <= 2 MB; no checkpoints, data,
caches, logs, metrics (`runs/`), state backups, `__pycache__`, `.bak*` files; symlinks not followed. Secret scan (hf_, ghp_/gh*_,
github_pat_, sk-, AKIA, xox*, AIza, private-key blocks) run on every copied file: see `REDACTIONS.md`.
