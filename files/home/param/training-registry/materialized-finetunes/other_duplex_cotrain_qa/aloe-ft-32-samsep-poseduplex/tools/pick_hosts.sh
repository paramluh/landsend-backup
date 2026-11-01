#!/bin/bash
# pose-duplex host pick + room preflight (main 10-25). Run AFTER v5 (29863) has ended (never on its hosts while it runs; never slinky-9).
# usage: bash pick_hosts.sh [h1,h2,...]  (default: v5's 16 hosts). Prints /scratch free per host, picks the 8 with the most free, and checks each
# against NEED = init subset 17 + floor 30 + 2 exports x 12 + end full save 78 = 149 GB; for short hosts prints the top dirs of /scratch/amartyavarma.
H=${1:-slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-7,slinky-8,slinky-11,slinky-12,slinky-13,slinky-14,slinky-15,slinky-16,slinky-17,slinky-19}
case ",$H," in *,slinky-9,*) echo "ABORT: slinky-9 is off-limits"; exit 1;; esac
squeue -h -j 29863 2>/dev/null | grep -q . && { echo "ABORT: v5 (29863) still running"; exit 1; }
N=$(echo $H | tr , '\n' | wc -l); NEED=${NEED:-149}
timeout 300 srun -p all -w $H -N$N -n$N --ntasks-per-node=1 -c1 --mem=200M -t 5 bash -c 'echo "$(hostname -s) $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9) $(du -s --block-size=1G /scratch/amartyavarma/* 2>/dev/null | sort -rn | head -5 | awk "{printf \"%sG:%s \", \$1, \$2}")"' 2>/dev/null | sort -k2,2rn > /tmp/pick_hosts.$$
echo "host free_GB top_dirs (sorted by free)"; cat /tmp/pick_hosts.$$
PICK=$(head -8 /tmp/pick_hosts.$$ | awk '{print $1}' | paste -sd,)
SHORT=$(head -8 /tmp/pick_hosts.$$ | awk -v n=$NEED '$2 < n {printf "%s short by %d GB; ", $1, n - $2}')
echo "PICK=$PICK"; [ -z "$SHORT" ] && echo "PREFLIGHT OK: all 8 have >= $NEED GB" || { echo "PREFLIGHT SHORT: $SHORT"; rm -f /tmp/pick_hosts.$$; exit 2; }
rm -f /tmp/pick_hosts.$$
