#!/bin/bash
#SBATCH -J posemonosam -p all -N12 -x slinky-4,slinky-9,slinky-10,slinky-18 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 --mem=500G -t 2-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/logs/posemonosam-%j.log
# [SAM] aloe-pose-mono-8020-sam (10-26) = aloe-pose-mono-8020/aloe_mono8020.sbatch (job 24272) except: job name / log / run dir, and 12 nodes
#   (world 96: ranks 0..63 get exactly 24272's 64 rows per step, ranks 64..95 zero-weight dummy micros; tests/w96/REPORT.md). GO.sh passes
#   the 12 hosts with -w (12 of MoT's 16 while MoT is paused). The original header follows.
# aloe-pose-mono-8020 (10-18): PPM 32B MONOPLEX pose FT (= job 23051's recipe, v2pure) on the Aloe handover pose rows (the qualifying tracks of
# job 22091's stores, one monoplex item per track) with the 10-18 prompt rule (voice clip p 0.8, persona text p 0.2, independent draws).
# 8 nodes x 8 H100 (world 64), the nodes of the 14B Voxtral FT (job 24193) after it ends (submitted with --dependency=afterany:24193).
# Gate: waits (<= 90 min) for READY (written once the code / env / plan checks pass) so the job holds its nodes from the moment they free.
# Stop: touch control/STOP_AFTER_SAVE (or scancel -b -s USR1 <job>). Resume: sbatch this file again (same nodes).
Q=/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam
for i in $(seq 1 1080); do [ -f $Q/READY ] && break; [ -f $Q/ABORT ] && { echo "ABORT file present"; exit 1; }; sleep 5; done
[ -f $Q/READY ] || { echo "READY never written (90 min): exit"; exit 1; }
echo "== READY seen $(TZ=America/Los_Angeles date '+%F %T PT')"
# [SAM] launch-time identity check (12 nodes, only if GO.sh wrote IDCHECK): 1 step at world 96 on 24272's ORIGINAL stores (idcheck.env) vs
# job 24272 (tests/w96/check_launch_identity.py, tests/w96/REPORT.md section 6). PASS -> the real run below; FAIL -> exit 1 (GO resumes MoT).
if [ -f $Q/IDCHECK ] && [ "$SLURM_NNODES" = 12 ]; then
  mkdir -p $Q/idcheck; IL=$Q/logs/idcheck96-$SLURM_JOB_ID.log
  echo "== IDCHECK start $(TZ=America/Los_Angeles date '+%F %T PT') -> $IL"
  QQA_ENV=$Q/idcheck.env RUN_TAG_FILE=$Q/idcheck/RUN_TAG.txt RUNNING_FILE=$Q/idcheck/RUNNING_JOB.txt bash $Q/aloe_mono8020_run.sh > $IL 2>&1; irc=$?
  /data/home/amartyavarma/venvs/qwen-s2/bin/python -I $Q/tests/w96/check_launch_identity.py $Q/idcheck/runs/$(cat $Q/idcheck/RUN_TAG.txt) $IL > $IL.check 2>&1; crc=$?
  tail -3 $IL.check
  if [ $irc = 0 ] && [ $crc = 0 ]; then echo "PASS $(TZ=America/Los_Angeles date '+%T PT')" > $Q/IDCHECK_RESULT; echo "== IDCHECK PASS"
  else echo "FAIL trainer rc $irc checker rc $crc $(TZ=America/Los_Angeles date '+%T PT')" > $Q/IDCHECK_RESULT; echo "== IDCHECK FAIL (trainer rc $irc, checker rc $crc): no real run"; exit 1; fi
fi
exec bash $Q/aloe_mono8020_run.sh
