#!/bin/bash
#SBATCH -p all -N1 --gres=gpu:8 -c 40 --mem=400G -t 0-12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/samples/xabi-ft-17641-x76l/logs/serve-%j-%N.log
# Hot pose server with the PPM export pinned by env PIN_CK (workers_ck.sh); else = serve.sbatch.
# args: "<ppm GPU pairs>" "<video GPUs>"   Stop: touch ctl/STOP-<node> (or scancel).
exec /data/home/amartyavarma/ppm/samples/xabi-ft-17641-x76l/ctl/workers_ck.sh "$1" "$2"
