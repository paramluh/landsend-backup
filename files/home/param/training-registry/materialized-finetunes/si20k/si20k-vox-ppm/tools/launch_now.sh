#!/bin/bash
# si20k-vox: submit BOTH trainers on the ALREADY-ASSEMBLED stores (no re-assembly: the podcast FLACs were fetched for this plan's 198 steps),
# + the two export watchers. Drain files must already be written. Login node: checks + sbatch only.
set -euo pipefail
P=/data/home/amartyavarma/ppm; T=$P/si20k-vox-ppm/tools; CTL=/data/home/amartyavarma/data/si20k-tts/control
PPM_HOSTS=${PPM_HOSTS:-slinky-0,slinky-1,slinky-3,slinky-5,slinky-8,slinky-9,slinky-16,slinky-17}
INTL_HOSTS=${INTL_HOSTS:-slinky-2,slinky-6,slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,slinky-19}
pt() { TZ=America/Los_Angeles date '+%F %T PT'; }
[ -f $P/si20k-vox-ppm/control/LAUNCH_DONE ] && { echo "ABORT: already launched ($(cat $P/si20k-vox-ppm/control/LAUNCH_DONE))"; exit 1; }
[ -f $CTL/DRAIN_FOR_FT.si20kppm ] && [ -f $CTL/DRAIN_FOR_FT.si20kintl ] || { echo "ABORT: drain files missing"; exit 1; }
[ -f /data/home/amartyavarma/ppm/si20k-vox-ppm/plan/podcast_flac_manifest.jsonl ] || { echo "ABORT: podcast FLAC manifest missing"; exit 1; }
python3 -c "
import json
sj = json.load(open('/data/home/amartyavarma/qwen-1ch-data/si20k-vox/stores/STORES.json'))
for r in ('ppm', 'intl'):
    e = dict(l.strip().split('=', 1) for l in open(f'/data/home/amartyavarma/ppm/si20k-vox-{r}/plan/PLAN.env') if '=' in l and not l.startswith('#'))
    assert e['QQA_REF_PLAN_DIGEST'] == sj['plan']['digest'] and int(e['QQA_STEPS']) == sj['plan']['steps'], (r, e, sj['plan']['digest'])
print('plan ok', sj['plan']['digest'], sj['plan']['steps'], 'steps')"
JP=$(sbatch --parsable -w $PPM_HOSTS $P/si20k-vox-ppm/si20k_vox_ppm.sbatch)
JI=$(sbatch --parsable -w $INTL_HOSTS $P/si20k-vox-intl/si20k_vox_intl.sbatch)
WP=$(sbatch --parsable --dependency=after:$JP -o $P/si20k-vox-ppm/logs/watch-%j.out $T/watch.sbatch $P/si20k-vox-ppm $JP si20kppm 80 $P/ckpt-infer/ppm-v1/local-12829)
WI=$(sbatch --parsable --dependency=after:$JI -o $P/si20k-vox-intl/logs/watch-%j.out $T/watch.sbatch $P/si20k-vox-intl $JI si20kintl 32 - $P/si20k-vox-ppm $JP)
echo "$JP $(pt) PPM on $PPM_HOSTS (watcher $WP)" >> $P/si20k-vox-ppm/LAUNCHED_JOB
echo "$JI $(pt) INTL on $INTL_HOSTS (watcher $WI)" >> $P/si20k-vox-intl/LAUNCHED_JOB
echo "$(pt) PPM $JP INTL $JI" > $P/si20k-vox-ppm/control/LAUNCH_DONE
echo "launched $(pt): PPM $JP (watch $WP), INTL $JI (watch $WI)"
