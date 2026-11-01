#!/bin/bash
# si20k-vox LAUNCH of both fine-tunes (login node: file checks + sbatch only, no compute). Run ONLY on the user's go.
#   1 assemble.sbatch (CPU, slinky-2: final stores from the prep cache + keep spans, STORES.json, both PLAN.env)
#   2 drain files DRAIN_FOR_FT.si20kppm / .si20kintl (the TTS fleet keeper never refills these nodes while they exist)
#   3 both trainers, afterok:assemble, 8 nodes each (PPM on PPM_HOSTS, INTL on INTL_HOSTS: all 16 = the pretraining hosts holding the
#     unseen podcast slices; never slinky-4 / slinky-18 / slinky-7)
#   4 the two export watchers (after each trainer starts): node-local exports -> /data copy -> private HF -> /data copy deleted; final DCP
#     verified -> drain file deleted + FINAL_EXPORT_READY + HF.
# usage: SPANS=<keep_spans.tsv> [KEEP=<keep.txt>] bash launch_both.sh        (DRY=1: checks only)
set -euo pipefail
P=/data/home/amartyavarma/ppm; T=$P/si20k-vox-ppm/tools; CTL=/data/home/amartyavarma/data/si20k-tts/control; RUN=/data/home/amartyavarma/data/si20k-tts
SPANS=${SPANS:-/data/home/amartyavarma/data/si20k-tts/filter/keep_spans.tsv}; KEEP=${KEEP:-}
PPM_HOSTS=${PPM_HOSTS:-slinky-3,slinky-5,slinky-6,slinky-8,slinky-9,slinky-10,slinky-14,slinky-17}
INTL_HOSTS=${INTL_HOSTS:-slinky-0,slinky-1,slinky-2,slinky-11,slinky-12,slinky-13,slinky-16,slinky-19}
pt() { TZ=America/Los_Angeles date '+%F %T PT'; }
[ -f $P/si20k-vox-ppm/control/LAUNCH_DONE ] && { echo "ABORT: already launched ($(cat $P/si20k-vox-ppm/control/LAUNCH_DONE))"; exit 1; }
[ -f $P/si20k-vox-ppm/control/NO_AUTO_LAUNCH ] && { echo "ABORT: launch stopped by control/NO_AUTO_LAUNCH"; exit 1; }
case ",$PPM_HOSTS,$INTL_HOSTS," in *,slinky-4,*|*,slinky-18,*|*,slinky-7,*) echo "ABORT: slinky-4 / 18 (reserved) or 7 (demos) in the host lists"; exit 1;; esac
[ -f "$SPANS" ] || [ "${SI20K_ALLOW_NO_FILTER:-0}" = 1 ] || { echo "ABORT: keep spans $SPANS missing (SI20K_ALLOW_NO_FILTER=1 to build untrimmed)"; exit 1; }
[ -f $P/ckpt-infer/ppm-v1/local-12829/COMPLETE ] || { echo "ABORT: PPM init local-12829 not COMPLETE"; exit 1; }
python3 - "$PPM_HOSTS" "$INTL_HOSTS" <<'PYEOF' || { echo "ABORT: podcast plan check"; exit 1; }
import json, sys
for run, hosts in (("ppm", sys.argv[1]), ("intl", sys.argv[2])):
    j = json.load(open("/data/home/amartyavarma/ppm/si20k-vox-ppm/plan/PODCAST_PLAN_ppm.json"))   # both runs: the same unseen rows
    miss = [h for h in hosts.split(",") if h not in j["hosts"]]
    assert not miss, (run, "hosts not in the podcast plan", miss)
    print(run, "podcast plan ok:", j["totals"])
PYEOF
if [ "${SI20K_PARTIAL:-0}" = 1 ]; then              # user GO 10-18 PT: launch on the conversations done so far (final new-engine renders covered by
  head -1 "$SPANS" | grep -Eq "${SPANS_STRICT_RE:-0\.233}" || { echo "ABORT: keep spans header is not the strict rule: $(head -1 $SPANS | cut -c1-160)"; exit 1; }
  B=/data/home/amartyavarma/qwen-1ch-data/si20k-vox
  srun --jobid=$(squeue -h -u amartyavarma -n si20k-prep -o %i | head -1) --overlap -n1 -c2 --mem=4G --cpu-bind=none nice -n 5 \
    /data/home/amartyavarma/venvs/qwen-s2/bin/python -I $B/tools/select_ids.py --out $B/logs/select_launch --spans "$SPANS" || { echo "ABORT: selection"; exit 1; }
  python3 - "${SPANS_MIN_COVER:-0.98}" <<'PYEOF' || { echo "ABORT: selection / spans coverage"; exit 1; }
import json, sys
st = json.load(open("/data/home/amartyavarma/qwen-1ch-data/si20k-vox/logs/select_launch.json"))
print("selection:", st["used"], "final", st["final_new_engine"], "unbiased prefix", st["unbiased_prefix"], "covered by the strict spans", st["spans_covered"],
      f"({st['spans_covered_frac']:.3f})", "stale render", st["spans_stale_render"])
assert st["spans_covered_frac"] >= float(sys.argv[1]), ("the strict spans do not cover the final conversations yet", st["spans_covered_frac"])
PYEOF
  KEEP=/data/home/amartyavarma/qwen-1ch-data/si20k-vox/logs/select_launch_covered.txt; REQ=0
  scancel $(squeue -h -u amartyavarma -n si20k-prep -o %i) 2>/dev/null; sleep 2    # the prep loop holds slinky-2 (an INTL node); assemble catches up
else
LEFT=$(python3 -c "
import os
q='$RUN/queue'; o=[l.split()[0] for l in open(q+'/order.tsv') if l.strip()]
ls=lambda d: {x[:-5] for x in os.listdir(q+'/'+d) if x.endswith('.json')} if os.path.isdir(q+'/'+d) else set()
f=ls('done')|ls('fail')|ls('dropped'); rv=[x for x in ls('revoice') if x not in ls('done')]
print(len({c for c in o if c not in f} | set(rv)))")
[ "$LEFT" = 0 ] || { echo "ABORT: TTS still has $LEFT conversations to render / re-voice"; exit 1; }
REQ=1
fi
FD=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9); echo "/data free $FD GB"
[ "${DRY:-0}" = 1 ] && { echo "DRY: all checks passed $(pt)"; exit 0; }
JA=$(sbatch --parsable --export=ALL,SPANS=$SPANS,KEEP=$KEEP,REQUIRE_FINAL=$REQ $T/assemble.sbatch)
{ echo "# si20k-vox-ppm FT (written $(pt) by launch_both.sh); until the watcher verifies the final export"; echo $PPM_HOSTS | tr , '\n'; } > $CTL/DRAIN_FOR_FT.si20kppm
{ echo "# si20k-vox-intl FT (written $(pt) by launch_both.sh); until the watcher verifies the final export"; echo $INTL_HOSTS | tr , '\n'; } > $CTL/DRAIN_FOR_FT.si20kintl
JP=$(sbatch --parsable --dependency=afterok:$JA -w $PPM_HOSTS $P/si20k-vox-ppm/si20k_vox_ppm.sbatch)
JI=$(sbatch --parsable --dependency=afterok:$JA -w $INTL_HOSTS $P/si20k-vox-intl/si20k_vox_intl.sbatch)
WP=$(sbatch --parsable --dependency=after:$JP -o $P/si20k-vox-ppm/logs/watch-%j.out $T/watch.sbatch $P/si20k-vox-ppm $JP si20kppm 80 $P/ckpt-infer/ppm-v1/local-12829)
WI=$(sbatch --parsable --dependency=after:$JI -o $P/si20k-vox-intl/logs/watch-%j.out $T/watch.sbatch $P/si20k-vox-intl $JI si20kintl 32 - $P/si20k-vox-ppm $JP)
echo "$JP $(pt) PPM on $PPM_HOSTS (assemble $JA, watcher $WP)" >> $P/si20k-vox-ppm/LAUNCHED_JOB
echo "$JI $(pt) INTL on $INTL_HOSTS (assemble $JA, watcher $WI)" >> $P/si20k-vox-intl/LAUNCHED_JOB
echo "$(pt) assemble $JA PPM $JP INTL $JI" > $P/si20k-vox-ppm/control/LAUNCH_DONE
echo "launched $(pt): assemble $JA -> PPM $JP (watch $WP), INTL $JI (watch $WI)"
