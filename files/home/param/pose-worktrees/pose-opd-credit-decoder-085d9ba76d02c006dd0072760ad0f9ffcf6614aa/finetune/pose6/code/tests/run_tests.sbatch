#!/bin/bash
#SBATCH -J pose6-b1-tests -p all -w slinky-4 -x slinky-9 -N1 -n1 -c 2 --mem=16G -t 0-02:00:00 --nice=100
#SBATCH -o /data/home/amartyavarma/ppm/pose6/code/tests/logs/run_tests-%j.log
# pose6 B1 proofs, CPU only (no GPU requested): stages A (micros) -> B (trainer variants) -> compare -> C (811 parity, speech/text rows)
# -> D (init load, torchrun gloo world 3 save / world 2 load). Reads Weka stores / sidecar / w-15973 metadata; writes ONLY pose6/code/tests/
# {out,logs} and node scratch $P6_SCR (micros + tiny synthetic bases; never /data). STAGES env: subset, e.g. "C D".
set -u
T=/data/home/amartyavarma/ppm/pose6/code/tests
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; TR=/data/home/amartyavarma/venvs/qwen-s2/bin/torchrun
P6=/data/home/amartyavarma/ppm/pose6/code; MONO=/data/home/amartyavarma/ppm/aloe-pose-mono-8020/code
set -a; source /data/home/amartyavarma/ppm/aloe-pose-mono-8020/aloe_mono8020.env; set +a
export P6_SCR=/scratch/amartyavarma/pose6-b1-tests
mkdir -p "$P6_SCR"
export QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 LOCAL_WORLD_SIZE=1
export QQA_RUN=pose6-b1-test QQA_LOG_ROOT=$P6_SCR/runs QQA_PLAN_DIR=$P6_SCR/plan QQA_CONTROL_DIR=$P6_SCR/control QQA_EXPORT=$P6_SCR/export \
       ALOE_LOCAL_EXPORT_ROOT=$P6_SCR/exports ALOE_SAVE_ROOT=$P6_SCR/saves
export ALOE_G0_SIDECAR=/data/home/amartyavarma/ppm/pose6/g0aloe/g0aloe.parquet
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
ST=${STAGES:-"A B C D"}
echo "== host $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT') stages $ST"
run() { local nm=$1; shift; echo "== [$nm] start $(TZ=America/Los_Angeles date '+%T PT')"; nice -n 10 "$@"; local rc=$?; echo "== [$nm] rc $rc $(TZ=America/Los_Angeles date '+%T PT')"; return $rc; }
ON="env PPM_G0COND=1 PPM_S2P=cb0"
OFF="env PPM_G0COND=0 PPM_S2P=0"
case " $ST " in *" A "*)
  run A-on     $ON  $PY -u $T/t_micros.py --code $P6 --tag on --select
  run A-off    $OFF $PY -u $T/t_micros.py --code $P6 --tag off
  run A-mono   $OFF $PY -u $T/t_micros.py --code $MONO --tag mono ;;
esac
case " $ST " in *" B "*)
  run B-mono   $OFF $PY -u $T/t_variant.py --code $MONO --variant mono --micros mono --port 29931
  run B-off    $OFF $PY -u $T/t_variant.py --code $P6 --variant off --micros off --port 29932
  run B-on0    $ON  $PY -u $T/t_variant.py --code $P6 --variant on0 --micros on --port 29933
  run B-onnz   $ON  $PY -u $T/t_variant.py --code $P6 --variant onnz --micros on --new nz --port 29934
  run B-bboff  $ON QQA_DECP_BB_OFF=1 $PY -u $T/t_variant.py --code $P6 --variant on0bboff --micros on --port 29935
  run AB-cmp   $PY -u $T/t_compare.py ;;
esac
case " $ST " in *" C "*)
  run C-parity $ON  $PY -u $T/t_parity.py ;;
esac
case " $ST " in *" D "*)
  run D-save   $OFF $TR --standalone --nproc_per_node=3 $T/t_init.py --phase save
  run D-load   $ON  $TR --standalone --nproc_per_node=2 $T/t_init.py --phase load ;;
esac
echo "== done $(TZ=America/Los_Angeles date '+%T PT')"
