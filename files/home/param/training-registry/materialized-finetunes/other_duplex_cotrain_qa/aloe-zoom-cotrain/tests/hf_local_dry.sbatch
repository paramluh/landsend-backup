#!/bin/bash
#SBATCH -J hflx-dry-co -p all -N1 -n1 -c 2 --mem=2G --gres=none -t 00:15:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-zoom-cotrain/tests/logs/hf_local_dry-%j.log
# DRY test of ../hf_local_exports.sbatch (cotrain copy): uploads nothing (the template's hfb_up_stub.py: no network, refuses real repos), tiny fake
# files under /scratch/amartyavarma/hflx-dry-<job>-{a,b} on this node only, deleted at the end.
#  case a (node 1 of 16 fake hosts, default HFLX_DELETE_INTER=0): every export 42..253 + the consolidated weka export-253 "uploaded" (rc 0),
#         NOTHING deleted, HF_BACKUP.json written into the weka export
#  case b (node 2, HFLX_DELETE_INTER=1): the template's guarded deletion of the 5 verified intermediates; 253 kept; no weka upload on node 2
Q=/data/home/amartyavarma/ppm/aloe-zoom-cotrain; S=$Q/hf_local_exports.sbatch; STUB=/data/home/amartyavarma/ppm/aloe-v2vox-raw-w14503/tests/hfb_up_stub.py
H=$(hostname -s); A=/scratch/amartyavarma/hflx-dry-$SLURM_JOB_ID-a; B=/scratch/amartyavarma/hflx-dry-$SLURM_JOB_ID-b; nfail=0
chk() { if eval "$2"; then echo "  ok   $1"; else echo "  FAIL $1"; nfail=$((nfail+1)); fi; }
bash -n $S && echo "bash -n ok" || { echo "bash -n FAIL"; exit 1; }
mk() { mkdir -p $1; for r in $(seq $2 $(($2 + 7))); do head -c 4096 /dev/urandom > $1/rank$r.pt; done
  [ $3 = 1 ] && { head -c 2048 /dev/urandom > $1/rep.pt; echo '{"format": "ppm-local-bf16-shards-v1"}' > $1/meta.json; basename $1 | tr -dc 0-9 > $1/COMPLETE; }; return 0; }
for s in 42 84 126 169 211 253; do mk $A/exports/export-$s 0 1; mk $B/exports/export-$s 8 0; done
mkdir -p $A/weka/export-253; head -c 4096 /dev/urandom > $A/weka/export-253/__0_0.distcp; echo meta > $A/weka/export-253/.metadata; echo 253 > $A/weka/export-253/COMPLETE
FAKE=$(seq 2 15 | sed 's/^/fake-/' | paste -sd,)
echo "== case a"
env HFLX_DRY=1 HFLX_ROOT=$A HFLX_UP=$STUB HFLX_HOSTS=$H,$FAKE bash $S; ra=$?
chk "a rc 0" "[ $ra = 0 ]"
for s in 42 84 126 169 211 253; do chk "a export-$s kept (no deletion by default)" "[ -d $A/exports/export-$s ]"; done
chk "a HF_BACKUP.json in the weka export" "[ -f $A/weka/export-253/HF_BACKUP.json ] && grep -q '\"path\": \"export-253-dcp/\"' $A/weka/export-253/HF_BACKUP.json"
chk "a 6 + 1 upload lines" "[ \$(grep -c 'HF upload rc 0' $A/logs/BACKUP_LOG.txt) = 7 ]"
echo "== case b"
env HFLX_DRY=1 HFLX_ROOT=$B HFLX_UP=$STUB HFLX_HOSTS=fake-1,$H HFLX_DELETE_INTER=1 bash $S; rb=$?
chk "b rc 0" "[ $rb = 0 ]"
for s in 42 84 126 169 211; do chk "b export-$s deleted (HFLX_DELETE_INTER=1, verified)" "[ ! -e $B/exports/export-$s ]"; done
chk "b export-253 kept" "[ -d $B/exports/export-253 ]"
chk "b no weka upload on node 2" "! grep -q export-253-dcp $B/logs/BACKUP_LOG.txt"
echo "-- BACKUP_LOG (a)"; cat $A/logs/BACKUP_LOG.txt; echo "-- BACKUP_LOG (b)"; cat $B/logs/BACKUP_LOG.txt
unset SCRATCH_GUARD_LIST; source /data/home/amartyavarma/ppm/scratch_guard.sh
guard_rm $A $B && echo "cleaned $A $B"
echo "== DRY TEST $([ $nfail = 0 ] && echo PASS || echo "FAIL ($nfail)") on $H job $SLURM_JOB_ID"
