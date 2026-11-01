import json
from google.cloud import storage
cli = storage.Client.from_service_account_json("/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json")
miss = json.load(open("/data/home/amartyavarma/ppm/aloe-m1vox-ft/plan/m1_missing.json"))
for c in miss[:3]:
    bl = list(cli.list_blobs("aloe-processed-datasets", prefix=f"stage2-handover-20261008/wlt/{c}/", versions=True, max_results=20))
    print(c, [(x.name.split("/")[-1], x.size, x.generation, x.time_deleted) for x in bl][:8])
    bl = list(cli.list_blobs("aloe-processed-datasets", prefix=f"stage2-handover-20261008/mimi-stream-v5/{c}/", versions=True, max_results=20))
    print("  mimi", [(x.name.split("/")[-1], x.size) for x in bl][:8])
for f in ("stage2-handover-20261008/_meta/",):
    print([x.name for x in cli.list_blobs("aloe-processed-datasets", prefix=f, max_results=50)])
