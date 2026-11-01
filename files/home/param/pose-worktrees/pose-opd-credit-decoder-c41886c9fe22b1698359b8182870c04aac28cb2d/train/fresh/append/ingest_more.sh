#!/bin/bash
# 10-22 FRESH POSE APPEND: ingest the user's new pose data (same format as new-pose-llm-data) and get the resume from job 27114's final save
# dcp-17352 ready, in ONE command (fresh/append/APPEND_READY.md):
#   sbatch -N16 -w $(cat /data/home/amartyavarma/ppm/code-811/nodes16_s7.txt) /data/home/amartyavarma/ppm/code-811/fresh/append/ingest_more.sbatch <SRC> [TAG]
# SRC = hf://buckets/<owner>/<bucket>/<prefix> (like hf://buckets/InternalCan/video-pretraining-data/new-pose-llm-data) | /abs/local/dir (holds jobs/)
# TAG = npl2 (default; the env's PPM_FRESH_POSE_ROOT2 = /scratch/amartyavarma/ppm-fresh-pose-<TAG>).
# Steps (each logged to fresh/append/<TAG>/logs, timed in TIMES.tsv, skipped when already done; FORCE=1 redoes them):
#   0 preflight (env, no training on the hosts, /scratch free)  1 list (src_list.py)  2 place (src_place.py: whole jobs, balanced for the
#   combined stream, disk floor)  3 fetch (src_fetch.py, every host in parallel: HF resolve/CDN at HF_RATE files/s/host, or local copy)
#   4 verify (src_verify.py: sizes + xet / sha256)  5 scan + plan (fresh_pose.py all: release pins, decode filter, global dedup incl. plan 1 /
#   PPM TikTok / eval videos)  6 check (fresh_pose.py check: raw parquet vs built rows / micros)  7 dry run (append_dryrun.py host / reduce
#   --fill-env / guard) + unit test + resolved-cfg diff vs 27114  8 SUMMARY.md. Training is NOT launched (APPEND_READY.md has the commands).
# Inside an sbatch allocation of the 16 hosts (ingest_more.sbatch), or OVERLAP_JOB=<running job> to run inside it (light: -c 4, nice).
set -uo pipefail
SRC=${1:?usage: ingest_more.sh <SRC> [TAG]}; TAG=${2:-npl2}
C=/data/home/amartyavarma/ppm/code-811; A=$C/fresh/append; W=$A/$TAG
ENVF=${ENVF:-$C/prod_811_fresh_append.env}
HOSTS=$(cat $C/nodes16_s7.txt); FIRST=${HOSTS%%,*}
DEST=/scratch/amartyavarma/npl-append/$TAG; OUT=/scratch/amartyavarma/ppm-fresh-pose-$TAG
PYV=/data/home/amartyavarma/venvs/voicebank/bin/python; PYQ=/data/home/amartyavarma/venvs/qwen-s2/bin/python
PRIOR=${PRIOR:-$C/fresh/scan$(for d in $A/*/scan; do t=$(basename $(dirname $d)); [ "$t" != "$TAG" ] && [ -f $A/$t/state/plan.ok ] && printf ",%s" $d; done)}
mkdir -p $W/logs $W/state
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
say() { echo "== $(pt) $*" | tee -a $W/logs/ingest.log; }
die() { say "ABORT: $*"; exit 1; }
if [ -n "${OVERLAP_JOB:-}" ]; then
  # 10-23: OVL_CPT / OVL_MEM / OVL_NICE override the light defaults (delivery 3 runs in its own CPU-only holder job: OVERLAP_JOB=<holder>)
  CPT=${OVL_CPT:-4}; WORKERS=${WORKERS:-$([ "$CPT" -gt 6 ] && echo $((CPT - 4)) || echo 4)}; NICE=${OVL_NICE:-"nice -n19 ionice -c3"}
  SR="srun --jobid $OVERLAP_JOB --overlap --cpu-bind=none --mem=${OVL_MEM:-24G}"
else
  [ -n "${SLURM_JOB_ID:-}" ] || die "run inside the allocation (ingest_more.sbatch) or set OVERLAP_JOB=<job on the 16 hosts>"
  CPT=${SLURM_CPUS_PER_TASK:-48}; WORKERS=${WORKERS:-$((CPT - 4))}; NICE=""
  SR="srun --cpu-bind=none --mem=180G"
fi
all() { local step=$1; shift; $SR -N16 -n16 --ntasks-per-node=1 -w $HOSTS -c $CPT --kill-on-bad-exit=1 -o "$W/logs/$step-%N.log" $NICE "$@"; }
one() { local step=$1; shift; $SR -N1 -n1 -w $FIRST -c $CPT -o "$W/logs/$step.log" $NICE "$@"; }
done_() { [ -f $W/state/$1.ok ] && [ "${FORCE:-0}" != 1 ]; }
mark() { echo "$(pt) $2" > $W/state/$1.ok; printf "%s\t%s\t%s\n" "$1" "$2" "$(pt)" >> $W/TIMES.tsv; }
timed() { local s=$1 t0=$(date +%s); shift; "$@"; local rc=$?; [ $rc = 0 ] || die "step $s failed (rc $rc): logs $W/logs/$s*"; mark $s "$(( $(date +%s) - t0 ))s"; say "step $s done in $(( $(date +%s) - t0 )) s"; }

say "ingest $SRC tag $TAG -> hosts $HOSTS: data $DEST, plan $OUT, work $W; env $ENVF; prior scans $PRIOR; $([ -n "${OVERLAP_JOB:-}" ] && echo "overlap job $OVERLAP_JOB" || echo "allocation $SLURM_JOB_ID") (-c $CPT, scan workers $WORKERS)"
# ---- 0 preflight
[[ "$TAG" =~ ^[a-z][a-z0-9-]*$ ]] && [ "$TAG" != npl ] || die "TAG must be lowercase [a-z0-9-], not npl"
grep -q "^PPM_FRESH_POSE_ROOT2=.*$OUT" $ENVF || die "$ENVF: PPM_FRESH_POSE_ROOT2 does not name $OUT (edit the env first)"
if [ -z "${OVERLAP_JOB:-}" ]; then
  RUN=$(squeue -h -u amartyavarma -n ppm811 -t R,PD -o %i | head -1); [ -z "$RUN" ] || die "ppm811 job $RUN is running / pending: ingest after it ends, or OVERLAP_JOB=$RUN"
fi
all free bash -c 'echo "$(hostname -s)=$(df -BG --output=avail /scratch | tail -1 | tr -d " G")"' || die "free-space probe failed"
FREE=$(cat $W/logs/free-*.log | grep -E '^slinky-[0-9]+=[0-9]+$' | sort -V | tr '\n' ' '); say "/scratch free GB: $FREE"
[ $(echo $FREE | wc -w) = 16 ] || die "free-space probe: $FREE"
# ---- 1 list
done_ list || timed list one list $PYV -I $A/src_list.py "$SRC" $W
say "listing: $(python3 -c "import json;d=json.load(open('$W/LISTING.json'));print(d['kept_parquet'],'parquet /',d['kept_files'],'files,',round(d['kept_bytes']/1e9,1),'GB,',d['kept_releases'],'releases of',d['kept_jobs'],'jobs; skipped',d['stats'])")"
# ---- 2 place
BAL=${BALANCE:-$([ "$(grep -E '^PPM_FRESH_POSE_TAIL=' $ENVF | tail -1 | cut -d= -f2)" = 1 ] && echo tail || echo equal)}
done_ place || timed place one place env BALANCE=$BAL PRIOR_TAGS=${PRIOR_TAGS:-} python3 -I $A/src_place.py $W $FREE
say "placement: $(python3 -c "import json;d=json.load(open('$W/PLACEMENT.json'));print(d['total_GB'],'GB,',d['est_new_rows'],'est rows (~',d['est_new_steps_at_64_rows_per_host_step'],'steps); per host GB',{h:v['GB'] for h,v in d['hosts'].items()},'; below the HF-hold floor after:',d['below_hold_gb_after'])")"
# ---- 3 fetch (resume-safe; a re-run skips files already present)
done_ fetch || timed fetch all fetch $PYV -I $A/src_fetch.py $W $DEST
# ---- 4 verify
done_ verify || timed verify all verify $PYV -I $A/src_verify.py $W $DEST
# ---- 5 scan + plan (every host waits for all 16 scans, then computes the same global dedup)

PE="FRESH_SRC=$DEST/jobs FRESH_OUT=$OUT FRESH_SCAN=$W/scan FRESH_TAG=$TAG FRESH_PRIOR_SCANS=$PRIOR FRESH_PRIOR_VIDEO=${PRIOR_VIDEO:-overlap} OMP_NUM_THREADS=1"
done_ plan || timed plan all plan env $PE $PYQ -I $C/fresh/fresh_pose.py all --workers $WORKERS
say "plan: $(python3 -c "
import json,glob
P=[json.load(open(f)) for f in sorted(glob.glob('$W/scan/plans/PLAN-*.json'))]
g=P[0]['global']; print(len(P),'hosts,',sum(p['n_rows'] for p in P),'rows',round(sum(p['kept_frames'] for p in P)/12.5/3600),'h; dedup',{k:v for k,v in g.items() if not k.startswith(('kept_slinky','frames_kept_slinky','frames_'))})")"
# ---- 6 check (3 rows per host: raw parquet == built rows / micros)
done_ check || timed check all check env $PE $PYQ -I $C/fresh/fresh_pose.py check
# ---- 7 dry run (CPU, no GPUs): every host's plans + micros, then the boundaries / max_steps (written into the env), then the trainer's guard
timed dryrun_host all dryrun env RESUME_STEP=${RESUME_STEP:-17352} $PYQ -I $A/append_dryrun.py host $ENVF $W/dryrun
timed dryrun_reduce one reduce env FILL_MARKER="${FILL_MARKER:-10-22 FRESH POSE APPEND}" RESUME_STEP=${RESUME_STEP:-17352} $PYQ -I $A/append_dryrun.py reduce $ENVF $W/dryrun --fill-env
MAXS=$(python3 -c "import json;print(json.load(open('$W/dryrun/RESULT.json'))['max_steps'])")
timed guard one guard bash -c "for v in \$(env | grep -oE '^PPM_[A-Za-z0-9_]*'); do unset \$v; done; set -a; source $ENVF; set +a; export RANK=0 WORLD_SIZE=128 LOCAL_RANK=0 LOCAL_WORLD_SIZE=8 PPM_RUN=ppm-v1-jDRY PPM_RUN_BASE=ppm-v1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1; RESUME_STEP=${RESUME_STEP:-17352} $PYQ $A/append_dryrun.py guard $ENVF $W/dryrun"
# 10-23: REF_JOB = the run being extended (27114 for npl2, 27204 for npl3): unit test 'all3' with its resume train_state; cfg diff vs its log
REF_JOB=${REF_JOB:-27114}
if [ "$REF_JOB" = 27114 ]; then
  timed unittest one unittest env OMP_NUM_THREADS=1 $PYQ $A/test_append_cpu.py all $W/dryrun/ts-$FIRST.json $ENVF
else
  timed unittest one unittest env OMP_NUM_THREADS=1 $PYQ $A/test_append_cpu.py all3 $W/dryrun/ts-$FIRST.json $ENVF
fi
timed cfgdiff one cfgdiff bash $A/tools/cfgdiff.sh $ENVF $W/CFG_DIFF_vs_$REF_JOB.txt /data/home/amartyavarma/ppm/logs/ppm-$REF_JOB.log
# ---- 8 summary
python3 - <<EOF > $W/SUMMARY.md
import json, glob
W = "$W"
L, P, R, G = (json.load(open(f"{W}/{f}")) for f in ("LISTING.json", "PLACEMENT.json", "dryrun/RESULT.json", "dryrun/GUARD.json"))
V = [json.load(open(f)) for f in sorted(glob.glob(f"{W}/verify/*.json"))]
print(f"# fresh pose append {'$TAG'}: ingest summary ({'$(pt)'})\n")
print(f"- source: {L['src_arg']} ({L['src']['kind']}); new: {L['kept_parquet']:,} parquet / {L['kept_files']:,} files, {L['kept_bytes'] / 1e9:.1f} GB, "
      f"{L['kept_releases']:,} releases of {L['kept_jobs']:,} jobs; listing skipped {L['stats']}")
print(f"- placed: {P['total_GB']} GB, balance {P['balance']}; per host GB {dict((h, v['GB']) for h, v in P['hosts'].items())}")
print(f"- verified: {sum(v['size_ok'] for v in V):,} files, hash mismatches {sum(v['hash_mismatch'] for v in V)}, missing {sum(v['missing_or_wrong_size'] for v in V)}")
print(f"- plan: rows per plan {R['rows_per_plan']}, hours {R['hours_per_plan']}; dedup (newest plan) {R['global_dedup_plan2'][-1] if R['global_dedup_plan2'] else None}")
print(f"- combined stream ends {R['ends']} -> PPM_MAX_STEPS={R['max_steps']} (+{R['steps_after_resume']} steps after the resume step {R['resume_step']}), tail {R['tail']}, "
      f"unused rows {R['rows_unused_all_hosts']:,}; resume checkpoint {R['resume_ckpt']} (final save: {R['resume_is_final_save']})")
print(f"- guard: {G['guard']} (append at {G['pose_append_at']}); flags: {R['flags'] or 'none'}; /scratch free GB now {R['scratch_free_GB']}")
print(f"- times: " + "; ".join(l.strip().replace(chr(9), ' ') for l in open(f"{W}/TIMES.tsv")))
EOF
cat $W/SUMMARY.md | tee -a $W/logs/ingest.log
say "READY: env $ENVF has PPM_MAX_STEPS=$MAXS; launch per $A/APPEND_READY.md (not launched here)"
