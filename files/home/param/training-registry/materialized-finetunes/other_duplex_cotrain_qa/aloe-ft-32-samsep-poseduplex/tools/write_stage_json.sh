#!/bin/bash
# write STAGE.json for a host-to-host (PPMVOX) init copy: host index = position in the pose-FT rank order; assert every source rank this index's
# world-64 ranks read (data/init_subset_v5.json, the stager's +-1 margin) is present + rep.pt/meta.json/COMPLETE + SHA256SUMS ok was checked earlier.
H=$(hostname -s); ORDER=slinky-1,slinky-2,slinky-6,slinky-15,slinky-7,slinky-13,slinky-14,slinky-17; D=/scratch/amartyavarma/poseduplex-init/w-18100
I=$(( $(echo $ORDER | tr , '\n' | grep -nx $H | cut -d: -f1) - 1 ))
python3 -I - "$D" "$I" "$H" <<'PY'
import json, os, sys
D, I, H = sys.argv[1], sys.argv[2], sys.argv[3]
need = set(json.load(open("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/data/init_subset_v5.json"))["hosts"][I])
need |= {s + d for s in list(need) for d in (-1, 1) if 0 <= s + d < 128}
miss = sorted(s for s in need if not os.path.isfile(f"{D}/rank{s}.pt"))
assert not miss, (H, "missing ranks", miss)
for f in ("rep.pt", "meta.json", "COMPLETE", "SHA256SUMS"): assert os.path.isfile(f"{D}/{f}"), (H, f)
json.dump({"host_index": I, "host": H, "tag": "w-18100", "source": "PPMVOX host-to-host copy (b7 fetcher), sha256 == source SHA256SUMS (checked by tools/check_init_local.sh)",
           "ranks": sorted(need)}, open(f"{D}/STAGE.json", "w"))
print(H, "idx", I, "ranks", len(need), "OK")
PY
