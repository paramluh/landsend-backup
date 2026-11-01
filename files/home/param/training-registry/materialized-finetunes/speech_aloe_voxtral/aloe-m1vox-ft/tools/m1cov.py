import csv, json, sys, os, collections
import pyarrow.parquet as pq
D="/data/home/amartyavarma/data/aloe-handover-full/DATA"
man={(r["session_id"],r["seg"]) for r in csv.DictReader(open(f"{D}/manifest.csv"))}
convs=collections.Counter(); miss=collections.Counter(); split=collections.Counter()
for k in range(4):
    t=pq.read_table(f"/data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores/m1n{k}/rows.parquet",columns=["src_conversation","split"]).to_pydict()
    for c,s in zip(t["src_conversation"],t["split"]):
        ses,seg=c.split("/"); convs[c]+=1
        if (ses,seg) not in man: miss[c]+=1; split[s]+=1
print("m1 convs",len(convs),"rows",sum(convs.values()),"missing convs",len(miss),"missing rows",sum(miss.values()),dict(split))
ex=sorted(miss)[:3]; print(ex)
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS","/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json")
from google.cloud import storage
cli=storage.Client.from_service_account_json(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]); b=cli.bucket("aloe-processed-datasets")
for c in ex:
    bl=list(cli.list_blobs("aloe-processed-datasets",prefix=f"stage2-handover-20261008/wlt/{c}/",max_results=20))
    print(c,[(x.name.split("/")[-1],x.size) for x in bl][:12])
json.dump(sorted(miss),open("/data/home/amartyavarma/ppm/aloe-m1vox-ft/plan/m1_missing.json","w"))
