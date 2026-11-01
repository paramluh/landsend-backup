"""Bounded production website check; reads telemetry/media, starts no training."""
import json
from pathlib import Path
import time
import urllib.request
import soundfile as sf

ROOT=Path('/home/param/outputs/pose-opd-credit-live')
PUBLIC='https://mph-daniel-containers-afternoon.trycloudflare.com/'
last=None
for attempt in range(120):
    progress=json.load(urllib.request.urlopen('http://127.0.0.1:9236/opd_prog.json',timeout=10))
    manifest=json.load(urllib.request.urlopen('http://127.0.0.1:9236/samples/manifest.json',timeout=10))
    assert manifest.get('synthetic') is False
    state=(progress.get('state'),progress.get('max'),progress.get('epochs'),progress.get('completed'),
           (progress.get('phase') or {}).get('name'),len(manifest.get('rollouts',[])))
    if state!=last:
        print(json.dumps({'time':time.time(),'state':state,'lr':progress.get('learning_rate_current')}),flush=True)
        last=state
    if manifest.get('rollouts'):
        rollout=manifest['rollouts'][-1]
        assert len(rollout['samples'])==4
        checks=[]
        for sample in rollout['samples']:
            check={'id':sample['id'],'duration_seconds':sample['duration_seconds']}
            assert sample['duration_seconds']==120.0
            for key in ('input_audio_url','output_audio_url'):
                path=ROOT/'samples'/sample[key]
                info=sf.info(path)
                assert info.samplerate==24000 and info.frames==2880000 and info.duration==120.0
                request=urllib.request.Request(PUBLIC+'samples/'+sample[key],method='HEAD')
                with urllib.request.urlopen(request,timeout=20) as response:assert response.status==200
                check[key]={'url':PUBLIC+'samples/'+sample[key],'frames':info.frames,'sample_rate':info.samplerate,'duration':info.duration,'subtype':info.subtype}
            pose=json.loads((ROOT/'samples'/sample['pose_url']).read_text())
            assert pose['fps']==25 and len(pose['frames'])==3000 and pose['dimensionality']==2
            with urllib.request.urlopen(urllib.request.Request(PUBLIC+'samples/'+sample['pose_url'],method='HEAD'),timeout=20) as response:assert response.status==200
            check['pose']={'url':PUBLIC+'samples/'+sample['pose_url'],'frames':len(pose['frames']),'fps':pose['fps'],'dimensionality':pose['dimensionality']}
            checks.append(check)
        result={'update':rollout['update'],'synthetic':False,'samples':checks,'progress':progress,'checked_at':time.time()}
        (ROOT/'qa'/'first-production-samples.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({'event':'FIRST_FOUR_PUBLIC_SAMPLES_VALID','update':rollout['update'],'samples':4}),flush=True)
        break
    time.sleep(5)
else:
    raise RuntimeError('No production samples published during ten-minute bounded watch')
