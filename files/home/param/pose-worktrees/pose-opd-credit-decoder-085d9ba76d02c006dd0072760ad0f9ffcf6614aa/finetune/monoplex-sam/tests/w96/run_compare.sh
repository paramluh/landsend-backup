#!/bin/bash
# compare.py on the test node (the fingerprints live on its /scratch): srun -w slinky-4 -n1 -c2 --mem=16G --gres=none bash run_compare.sh
T=/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/tests/w96
export OMP_NUM_THREADS=2
/data/home/amartyavarma/venvs/qwen-s2/bin/python $T/compare.py /scratch/amartyavarma/w96test W3 W4
# self-test of the launch-time checker: 24272 against itself (must fail ONLY on the world / dummies line: 64 / 23, the check wants 96 / 11447)
R=/data/home/amartyavarma/ppm/aloe-pose-mono-8020
echo "== check_launch_identity self-test (24272 vs itself)"
/data/home/amartyavarma/venvs/qwen-s2/bin/python $T/check_launch_identity.py $R/runs/ppm-aloe-pose-mono-8020-j24229 $R/logs/posemono8020-24272.log | tail -6
