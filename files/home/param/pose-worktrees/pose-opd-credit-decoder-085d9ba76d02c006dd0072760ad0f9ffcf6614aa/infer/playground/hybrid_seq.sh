#!/bin/bash
# 10-26 (videort, user/main order): the playground's live avatar stream backend on Vansh's SEQUENTIAL sampler. = videort's
# demo/start_hybrid.sh (copy at start_hybrid.sh.ref-1026) with only its asked changes: ONE backend over both GPUs (GPUS_PER_BACKEND=2:
# sampler + decode + pose together on one GPU run at only ~0.85x real time), VRT_ENGINE/ENGINE = rt.seq_engine:SeqEngine, RS_SPECS=SPECS_seq.txt.
#   hybrid_seq.sh <comma GPU list of 2>
P=/data/home/amartyavarma/ppm/aloe-posedup-live/posefix
G=/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/render
export VRT_RENDER="01_ab63d745 $P/clips 9999 pg:$G/jobs:$G/out own:$P/out/fullD:fullD own:$P/out/fullM:fullM gt=10_27286bdd:gt $P/out/mono:mono $P/out/dupM:dupM $P/out/gt:gtc"
[ "$(hostname)" = slinky-10 ] || export BIND=0.0.0.0   # the front proxy runs on slinky-10: other nodes listen cluster-wide
export CUDA_VISIBLE_DEVICES=$1 SLURM_JOB_GPUS=$1 GPUS_PER_BACKEND=2 ENGINE=rt.seq_engine:SeqEngine VRT_ENGINE=rt.seq_engine:SeqEngine RS_SPECS=SPECS_seq.txt
export PATH=/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static:$PATH
export REGISTER=/data/home/amartyavarma/video/dmd-rt/demo/backends-P.d   # 10-26 videort: register ONLY with front P (the playground's own avatar pool; never the demo's gpu1)
exec bash /data/home/amartyavarma/video/dmd-rt/demo/start_backend.sbatch
