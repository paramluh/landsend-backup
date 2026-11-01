#!/bin/bash
#SBATCH -J hflx-dry -p all -N1 -n1 -c 2 --mem=2G --gres=none -t 00:15:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2vox-raw-w14503/tests/hf_local_dry-%j.log
# DRY test of ../hf_local_exports.sbatch: uploads nothing (hfb_up_stub.py, no network), writes only tiny fake files under
# /scratch/amartyavarma/hflx-dry-<job>-{a,b} on this node (deleted at the end via scratch_guard), never touches /data checkpoints or the training job.
#  case a (node 0): verify FAIL on export-84 -> kept; child upload fails once on export-126 -> retried, deleted; export-169 has 7 rank files -> kept;
#                   42 / 211 deleted; export-253 kept; consolidated fake weka export-253 -> HF_BACKUP.json; rc 1 (84 failed)
#  case b (node 1): rank8-15 only; export-211 on a test guard list (= the real list + that path) -> guard refuses, kept, rc 1; 42-169 deleted; 253 kept
#  case c: dry root outside /scratch/amartyavarma/hflx-dry-* -> ABORT before anything
Q=/data/home/amartyavarma/ppm/aloe-v2vox-raw-w14503; S=$Q/hf_local_exports.sbatch; PY=/data/home/amartyavarma/venvs/duplex/bin/python
H=$(hostname -s); A=/scratch/amartyavarma/hflx-dry-$SLURM_JOB_ID-a; B=/scratch/amartyavarma/hflx-dry-$SLURM_JOB_ID-b; nfail=0
chk() { if eval "$2"; then echo "  ok   $1"; else echo "  FAIL $1"; nfail=$((nfail+1)); fi; }
bash -n $S && echo "bash -n ok" || { echo "bash -n FAIL"; exit 1; }
$PY -m py_compile $Q/tests/hfb_up_stub.py && echo "stub compiles"
mk() {  # mk <export dir> <first rank> <n ranks> <rank0 node 0|1>
  mkdir -p $1; for r in $(seq $2 $(($2 + $3 - 1))); do head -c 4096 /dev/urandom > $1/rank$r.pt; done
  [ $4 = 1 ] && { head -c 2048 /dev/urandom > $1/rep.pt; echo '{"format": "ppm-local-bf16-shards-v1"}' > $1/meta.json; basename $1 | tr -dc 0-9 > $1/COMPLETE; }
  return 0
}
for s in 42 84 126 211 253; do mk $A/exports/export-$s 0 8 1; done; mk $A/exports/export-169 0 7 1
mkdir -p $A/weka/export-253; head -c 4096 /dev/urandom > $A/weka/export-253/__0_0.distcp; head -c 4096 /dev/urandom > $A/weka/export-253/__1_0.distcp
echo meta > $A/weka/export-253/.metadata; echo 253 > $A/weka/export-253/COMPLETE
for s in 42 84 126 169 211 253; do mk $B/exports/export-$s 8 8 0; done
{ cat /data/home/amartyavarma/ppm/PROTECTED_SCRATCH.txt; echo; echo $B/exports/export-211; } > $B/guard_list.txt

echo "== case a (node 0 of $H,fake-b)"
env HFLX_DRY=1 HFLX_ROOT=$A HFLX_UP=$Q/tests/hfb_up_stub.py HFLX_HOSTS=$H,fake-b HFLX_STUB_VFAIL=export-84 HFLX_STUB_CHILD_FAIL_ONCE=export-126 \
  bash $S; ra=$?
chk "a rc 1 (export-84 verify failed)" "[ $ra = 1 ]"
for s in 42 126 211; do chk "a export-$s deleted" "[ ! -e $A/exports/export-$s ]"; done
for s in 84 169 253; do chk "a export-$s kept" "[ -d $A/exports/export-$s ]"; done
chk "a export-126 retried after a child failure" "grep -q 'export-126 attempt 1: upload rc 1' $A/logs/hf_local-$SLURM_JOB_ID-$H.log"
chk "a HF_BACKUP.json written into the consolidated export" "[ -f $A/weka/export-253/HF_BACKUP.json ]"
chk "a weka export untouched otherwise" "[ \$(ls -A $A/weka/export-253 | wc -l) = 5 ]"
chk "a 6 upload lines + 1 dcp line in BACKUP_LOG" "[ \$(grep -c 'HF upload rc' $A/logs/BACKUP_LOG.txt) = 7 ]"
chk "a per-host manifest names" "[ -f $A/hfb/aloe-v2vox-raw-w14503-export-42-$H/_backup/MANIFEST.export-42.$H.tsv ]"
echo "-- BACKUP_LOG (a)"; cat $A/logs/BACKUP_LOG.txt

echo "== case b (node 1 of fake-a,$H)"
env HFLX_DRY=1 HFLX_ROOT=$B HFLX_UP=$Q/tests/hfb_up_stub.py HFLX_HOSTS=fake-a,$H HFLX_GUARD_LIST=$B/guard_list.txt bash $S; rb=$?
chk "b rc 1 (guard refused export-211)" "[ $rb = 1 ]"
for s in 42 84 126 169; do chk "b export-$s deleted" "[ ! -e $B/exports/export-$s ]"; done
for s in 211 253; do chk "b export-$s kept" "[ -d $B/exports/export-$s ]"; done
chk "b no consolidated upload on node 1" "! grep -q export-253-dcp $B/logs/BACKUP_LOG.txt"
echo "-- BACKUP_LOG (b)"; cat $B/logs/BACKUP_LOG.txt

echo "== case c (bad dry root)"
env HFLX_DRY=1 HFLX_ROOT=/scratch/amartyavarma/aloe-v2vox-raw-w14503 HFLX_UP=$Q/tests/hfb_up_stub.py bash $S; rc_=$?
chk "c refused (rc 1, nothing created)" "[ $rc_ = 1 ] && [ ! -e /scratch/amartyavarma/aloe-v2vox-raw-w14503/logs ]"

unset SCRATCH_GUARD_LIST; source /data/home/amartyavarma/ppm/scratch_guard.sh
guard_rm $A $B && echo "cleaned $A $B"
[ ! -e $A ] && [ ! -e $B ] || { echo "cleanup incomplete"; nfail=$((nfail+1)); }
echo "== DRY TEST $([ $nfail = 0 ] && echo PASS || echo "FAIL ($nfail)") on $H job $SLURM_JOB_ID"
