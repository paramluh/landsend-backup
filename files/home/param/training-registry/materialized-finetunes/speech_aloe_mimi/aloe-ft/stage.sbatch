#!/bin/bash
#SBATCH -J aloeft-stage -p all -N8 -w slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,slinky-16,slinky-17,slinky-19 --ntasks-per-node=1 -c 16 --mem=32G -t 0:30:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft/logs/stage-%j.log
# all 4 Model-1 stores (m1n0..m1n3, s3-duplex-rows-v1) -> every training node's /scratch/amartyavarma/aloe-ft/stores/m1nK, digests verified.
srun bash -c '
S=/data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores; D=/scratch/amartyavarma/aloe-ft/stores; H=$(hostname -s)
mkdir -p $D; df -h /scratch | tail -1
for k in 0 1 2 3; do ( rm -rf $D/m1n$k.tmp; cp -r $S/m1n$k $D/m1n$k.tmp && rm -rf $D/m1n$k && mv $D/m1n$k.tmp $D/m1n$k ) & done; wait
/data/home/amartyavarma/venvs/qwen-s2/bin/python -I - <<PYEOF
import sys; sys.path.insert(0, "/data/home/amartyavarma/ppm/qqa-ft/code/lib")
import s3rows as F
for k in range(4):
    a, b = F.load_node("$S/m1n%d" % k), F.load_node("$D/m1n%d" % k)
    assert a["digest"] == b["digest"] and (a["length"] == b["length"]).all()
    for suf in ("codes_user.npy", "codes_assistant.npy", "words.arrow", "persona.arrow"):
        import os; f = "rows-00000-of-00001." + suf
        assert os.path.getsize("$S/m1n%d/" % k + f) == os.path.getsize("$D/m1n%d/" % k + f), f
    print("$H m1n%d ok rows %d digest %s" % (k, len(b["length"]), b["digest"]))
PYEOF
'
