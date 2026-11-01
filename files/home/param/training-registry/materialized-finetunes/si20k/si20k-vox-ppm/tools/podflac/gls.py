import sys, os
from google.cloud import storage
c = storage.Client.from_service_account_json("/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json")
for pre in sys.argv[1:]:
    n = 0
    for b in c.list_blobs("podcast-raw-audio-data-mass", prefix=pre):
        print(b.name, b.size, b.content_type); n += 1
        if n > 60: break
