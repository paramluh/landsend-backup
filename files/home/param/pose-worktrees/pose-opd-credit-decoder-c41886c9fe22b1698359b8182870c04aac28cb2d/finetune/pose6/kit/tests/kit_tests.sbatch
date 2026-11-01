#!/bin/bash
#SBATCH -J pose6-kit-test -p all -N1 -n1 -c2 --mem=32G -t 0-00:40:00 -x slinky-9
#SBATCH -o /data/home/amartyavarma/ppm/pose6/kit/tests/logs/kit_tests-%j.log
# pose6 kit (B3) CPU checks, submitted with -w slinky-4 (NEVER slinky-18: its /scratch is full, motasr2; never slinky-9). Nothing on /data except tests/out JSONs; test copies on node /scratch only.
#  D0  dry mode faithful: ORIGINAL mono8020 code + env at world 64 -> resolved CFG vs job 24272's (export-357 meta cfg): must be 0 keys
#  D1  pose6/code + kit env, 8 nodes (world 64)  -> CFG diff vs 24272 + plan      D2  same, 1 node (world 8) -> diff + plan (lpt)
#  D3  pose6/code + kit env with BOTH flags OFF, world 64 -> diff vs 24272 = names / base only (the flags are the only recipe change)
#  R   model key table of the FT model (export-357 rank / rep files)           P1/P2 preflight on w-15973 (DCP) / export-357 (local): both
#      must FAIL with exactly the 2 new keys missing (no extra, no shape mismatch); P1 also runs the sidecar coverage (27,517 + 531)
#  S   tests/test_stage.py (stage_base bit-exactness + load_init loop emulation; preflight logic)
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; K=/data/home/amartyavarma/ppm/pose6/kit; C=/data/home/amartyavarma/ppm/pose6/code
M8=/data/home/amartyavarma/ppm/aloe-pose-mono-8020; REF=$K/ref/ref_cfg_24272.json; O=$K/tests/out; T=/scratch/amartyavarma/pose6-kit-test
BASE_EX=/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-STEP
case "$(hostname -s)" in slinky-18|slinky-9) echo "ABORT: no test data on $(hostname -s) /scratch (slinky-18 full + motasr2; slinky-9 never): submit with -w slinky-4"; exit 1;; esac
mkdir -p $O $T
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=8
CLR='unset $(env | grep -E "^(QQA_|ALOE_|PPM_|S3_MIN|GEMM|FT_)" | cut -d= -f1)'
echo "== D0 $(TZ=America/Los_Angeles date +%T) on $(hostname -s)"
( eval "$CLR"; set -a; source $M8/aloe_mono8020.env; set +a
  export QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 WORLD_SIZE=64 QQA_RUN=ppm-aloe-pose-mono-8020-j24229 FT_ENV_FILE=$M8/aloe_mono8020.env
  $PY -u $K/tools/dry_cfg.py $M8/code $O/dry_D0_mono8020_w64.json --ref $REF ) 2>&1 | grep -v Warning
for V in D1:8:64 D2:1:8 D3:8:64; do
  N=${V%%:*}; R=${V#*:}; NN=${R%%:*}; W=${R#*:}
  echo "== $N nodes $NN world $W $(TZ=America/Los_Angeles date +%T)"
  ( eval "$CLR"; export FT_NODES=$NN; set -a; source $K/ft_s2pg0.env; set +a
    [ $N = D3 ] && { export PPM_S2P=0 PPM_G0COND=0; unset ALOE_G0_SIDECAR; }
    export QQA_INIT=$BASE_EX WORLD_SIZE=$W QQA_RUN=ppm-pose6-ft-s2pg0-jNNNNN FT_ENV_FILE=$K/ft_s2pg0.env
    $PY -u $K/tools/dry_cfg.py $C $O/dry_${N}_w$W.json plan --ref $REF ) 2>&1 | grep -v Warning
done
echo "== R $(TZ=America/Los_Angeles date +%T)"
$PY -u $K/tools/preflight.py --make-ref $M8/exports/export-357
for V in P1:/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-15973: P2:$M8/exports/export-357:--skip-sidecar; do
  N=${V%%:*}; R=${V#*:}; B=${R%%:*}; X=${R#*:}
  echo "== $N preflight $B (expect FAIL: exactly the 2 new keys missing) $(TZ=America/Los_Angeles date +%T)"
  ( eval "$CLR"; export FT_NODES=1; set -a; source $K/ft_s2pg0.env; set +a
    $PY -u $K/tools/preflight.py --base $B --code $C --json $O/preflight_$N.json $X ) 2>&1 | grep -v Warning | tail -40
done
echo "== S $(TZ=America/Los_Angeles date +%T)"
$PY -u $K/tests/test_stage.py $T $O/test_stage.json 2>&1 | grep -v Warning | tail -40
echo "== done $(TZ=America/Los_Angeles date +%T)"
