#!/bin/bash
# pdsamples post: skeleton videos for every finished run (render_pose.py, CPU) + the page data (build_page.py). Run on a compute node.
S=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/data/home/amartyavarma/.cache/huggingface
R=$(for d in $S/runs/B-*/DONE; do dd=$(dirname $d); [ -f $dd/POSE_DONE ] || echo $dd; done)
[ -n "$R" ] && $PY -I $S/code/render_pose.py $R
$PY $S/page/build_page.py $S/runs 8
