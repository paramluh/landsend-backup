# Xabi FT scaling series: run scripts per base

One dir per base checkpoint of the Aloe v2pure (Xabi data) monoplex FT series: `7932`, `13533`, `16610` (PPM pretraining steps), `17352`,
`17641` (8,1,1 steps), and `clip80` (job 24198, 11488 base, 80% voice-clip prompt rule). The 17850 base is `../../xabi-main-17850/`.
Sources: `/data/home/amartyavarma/ppm/aloe-v2pure-<base>/` (see `MANIFEST.tsv`).

Each dir keeps its own `run.env` / `run.sbatch` / `run.sh` (+ HF export / save cleanup sbatches, tools, tests) as they were.

**De-duplicated code:** each source dir also had a full `code/` trainer copy. Here `<base>/code/` holds ONLY the files that differ from
`finetune/xabi-main-17850/code/` (the canonical copy); every other file of that run's `code/` was byte-identical to it (checked with `cmp`
at copy time):
- 7932, 13533, 16610: `task_ppm.sh`, `train_aloe_pose.py` differ.
- clip80: `data_aloe_pose.py`, `task_ppm.sh`, `train_aloe_pose.py` differ.
- 17352, 17641: no differences (their `code/` == `xabi-main-17850/code`).
The run scripts still point at their own live `code/` dirs by absolute path.
