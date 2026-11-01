#!/bin/bash
#SBATCH -J aloew64-serve -p all -N1 -n1 -c 2 --mem=4G -t 01:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/w64-serve-%j-%N.log
# w64 relaunch: serve this host's verified 16861 init copy (read only) over the LAN for stage_init.sbatch; nice 19 + idle IO. Killed when staging is done.
H=$(hostname -s); PORT=18771; W=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/w64
cd /tmp; echo "$H:$PORT" > $W/SERVER-$H
exec nice -n 19 ionice -c3 python3 -m http.server $PORT --bind 0.0.0.0 --directory /scratch/amartyavarma/aloe-samsep-16861-init
