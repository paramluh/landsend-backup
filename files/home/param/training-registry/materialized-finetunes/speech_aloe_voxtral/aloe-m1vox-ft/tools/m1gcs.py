import json, os
from google.cloud import storage
cli = storage.Client.from_service_account_json("/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json")
it = cli.list_blobs("aloe-processed-datasets", prefix="stage2-handover-20261008/", delimiter="/")
list(it); print("top:", sorted(it.prefixes))
miss = json.load(open("/data/home/amartyavarma/ppm/aloe-m1vox-ft/plan/m1_missing.json"))
for c in miss[:4]:
    ses = c.split("/")[0]
    for top in sorted(it.prefixes):
        bl = list(cli.list_blobs("aloe-processed-datasets", prefix=f"{top}{ses}", max_results=10))
        if bl: print(c, top, [(x.name, x.size) for x in bl][:6])
