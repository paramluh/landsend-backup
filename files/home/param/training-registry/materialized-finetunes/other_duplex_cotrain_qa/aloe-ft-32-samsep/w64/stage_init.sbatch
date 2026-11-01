#!/bin/bash
#SBATCH -J aloew64-stage -p all --ntasks-per-node=1 -c 8 --mem=32G -t 00:40:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/w64-stage-%j.log
# w64 relaunch: per target host, pull ONLY the 16861 init files its 8 ranks read at world 64 (w64/init_subset.json, + rep.pt / meta.json / COMPLETE)
# from the verified slinky-18 (hosts 0-3) / slinky-4 (hosts 4-7) copies over the LAN into /scratch/amartyavarma/aloe-samsep-16861-init(.part),
# every file sha256-checked against base/ctl/ledger-16861-slinky-18.json. Never touches anything else on the host (MoT's dcp-3270 is protected).
W=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/w64
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 bash -c '
set -uo pipefail
W='"$W"'; H=$(hostname -s); HOSTS=slinky-11,slinky-12,slinky-13,slinky-14,slinky-15,slinky-16,slinky-17,slinky-19
h=$(( $(echo $HOSTS | tr , "\n" | grep -nx $H | cut -d: -f1) - 1 )); [ $h -ge 0 ] || { echo "ABORT $H not a target"; exit 1; }
S=$([ $h -lt 4 ] && cat $W/SERVER-slinky-18 || cat $W/SERVER-slinky-4)
DST=/scratch/amartyavarma/aloe-samsep-16861-init; P=$DST.part
[ -f $DST/COMPLETE ] && { echo "$H: $DST already COMPLETE"; exit 0; }
F=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); [ $F -ge 60 ] || { echo "ABORT $H: /scratch free ${F} GB < 60 (15 init + 10 exports + 30 floor + margin)"; exit 1; }
mkdir -p $P; cd $P
FILES=$(python3 -I -c "import json,sys; j=json.load(open(sys.argv[1])); print(\" \".join(f\"rank{s}.pt\" for s in j[\"hosts\"][sys.argv[2]]))" $W/init_subset.json $h)
FILES="$FILES rep.pt meta.json"
t0=$(date +%s)
echo $FILES | tr " " "\n" | xargs -P 8 -I{} curl -sS --fail --retry 3 -o {} http://$S/{} || { echo "ABORT $H: download failed"; exit 1; }
t1=$(date +%s)
bad=$(echo $FILES | tr " " "\n" | xargs -P 8 -I{} sh -c "echo {} \$(sha256sum {} | cut -c1-64)" | python3 -I -c "
import json,sys; L=json.load(open(sys.argv[1])); bad=[]
for l in sys.stdin:
    f,s=l.split()
    if f == \"meta.json\": continue
    if f not in L or L[f][\"sha256\"]!=s: bad.append(f)
print(\" \".join(bad))" '"/data/home/amartyavarma/ppm/aloe-ft-32-samsep/base/ctl/ledger-16861-slinky-18.json"')
[ -z "$bad" ] || { echo "ABORT $H: sha256 mismatch: $bad"; exit 1; }
curl -sS --fail -o COMPLETE http://$S/COMPLETE
printf "{\"subset_for\": \"world 64, host index %s (%s), ranks %s-%s\", \"files\": %s, \"source\": \"%s\", \"time\": \"%s\"}\n" $h $H $((8*h)) $((8*h+7)) $(echo $FILES | wc -w) $S "$(date)" > SUBSET_W64.json
cd /tmp; mv $P $DST
echo "$H (host $h) staged $(echo $FILES | wc -w) files $(du -sh $DST | cut -f1) from $S in $((t1-t0)) s, sha256 OK; COMPLETE $(cat $DST/COMPLETE); /scratch free $(df -BG --output=avail /scratch | tail -1 | tr -d " ")"
'
