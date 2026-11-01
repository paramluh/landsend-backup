#!/bin/bash
# D1 (user's explicit OK via main amartyavarma-5f, 10-25 03:03 PT; POSEDUPLEX_GO D1_DELETE_DCP17850_OK=1; HF InternalCan/ppm-v1-c holds all 320 files):
# delete EXACTLY /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17850 on THIS host (plain rm: the path is under the protected ppm/ tree). Nothing else.
H=$(hostname -s); d=/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17850
case $H in slinky-1|slinky-2|slinky-6|slinky-15|slinky-7|slinky-13|slinky-14|slinky-17) ;; *) echo "$H: not a pose-duplex host: refusing"; exit 1;; esac
grep -q '^D1_DELETE_DCP17850_OK=1' /data/home/amartyavarma/ppm/control/POSEDUPLEX_GO || { echo "$H: no D1 OK in POSEDUPLEX_GO: refusing"; exit 1; }
[ -d "$d" ] || { echo "$H: $d absent"; exit 0; }
b=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); sz=$(du -sh "$d" | cut -f1)
rm -rf -- "$d" && echo "[$(TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT')] ALOEFT32 D1 (USER's explicit path-specific OK via main amartyavarma-5f 10-25 03:03 PT, POSEDUPLEX_GO): deleted $H:$d ($sz, node-local copy of PPM dcp-17850; HF InternalCan/ppm-v1-c holds all 320 files) for the pose-duplex end-save room" | tee -a /data/home/amartyavarma/setup-logs/hfb/DELETED-1021-hfb-hold.log
echo "$H /scratch ${b}G -> $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)G"
