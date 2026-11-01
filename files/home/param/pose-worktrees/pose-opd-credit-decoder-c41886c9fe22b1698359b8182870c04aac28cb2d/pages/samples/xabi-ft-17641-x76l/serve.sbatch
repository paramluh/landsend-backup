#!/bin/bash
#SBATCH -p all -N1 --gres=gpu:8 -c 40 --mem=400G -t 0-12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/samples/xabi-ft-17641-x76l/logs/serve-%j-%N.log
# Hot 17,641 pose server, successor of the 2 h X76L holder on the same node (starts when it ends): runs workers.sh as the batch step.
# args: "<ppm GPU pairs>" "<video GPUs>"   Stop: touch ctl/STOP-<node> (or scancel).
exec /data/home/amartyavarma/ppm/samples/xabi-ft-17641-x76l/ctl/workers.sh "$1" "$2"
