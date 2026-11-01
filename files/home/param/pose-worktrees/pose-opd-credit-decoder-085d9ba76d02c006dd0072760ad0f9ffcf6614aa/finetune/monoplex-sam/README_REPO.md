# monoplex-sam in this repo

`code/` here holds ONLY the files that differ from `finetune/monoplex/code/` (job 24272's code, the canonical copy): `data_aloe_pose.py`
and `train_aloe_pose.py`. Every other file of `/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/code/` was byte-identical to
`/data/home/amartyavarma/ppm/aloe-pose-mono-8020/code/` at copy time (checked with `cmp`). The run scripts still point at the live
`code/` dir by absolute path.
