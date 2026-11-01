from pathlib import Path
import json,subprocess,time,urllib.request

ROOT=Path('/home/param/outputs/pose-opd-teacher-partner-b336-general-live-20261031')
OUT=Path('/home/param/outputs/pose-opd-video-dashboard-stage-20261031/REAL-VIDEOS.json')
BASE='https://mph-daniel-containers-afternoon.trycloudflare.com/samples/'
raw=json.loads((ROOT/'raw-rollouts/update-000001/manifest.json').read_text())
ids={s['id'] for s in raw['samples']}
first_seen={};last_count=-1;deadline=time.monotonic()+900
report={'update':1,'expected_ids':sorted(ids),'videos':{},'complete':False}
while time.monotonic()<deadline:
    candidates={}
    for p in (ROOT/'samples/video-results').glob('u1-r0-s*-*.json'):
        d=json.loads(p.read_text())
        if d.get('sample_id') in ids:candidates[d['sample_id']]=d
    if candidates:
        try:
            with urllib.request.urlopen(BASE+'manifest.json?verify='+str(time.time_ns()),timeout=15) as response:manifest=json.load(response)
            rollout=next((r for r in manifest['rollouts'] if r['update']==1),None)
            if rollout:
                events={}
                logs=subprocess.check_output(['journalctl','--user','-u','pose-opd-video-bridge.service','--since','30 minutes ago','--no-pager','-o','cat'],text=True)
                for line in logs.splitlines():
                    try:d=json.loads(line)
                    except ValueError:continue
                    if d.get('event')=='video_published' and d.get('sample_id') in ids:events[d['sample_id']]=d['time']
                for sample in rollout['samples']:
                    sid=sample['id']
                    if sid not in candidates or not sample.get('video_url') or sid in report['videos']:continue
                    now=time.time();r=candidates[sid];source=rollout['source_manifest_sha256']
                    assert r['source_manifest_sha256']==source
                    expected=f"assets/update-000001-{source[:16]}/{sid}-video-{r['sha256'][:16]}.mp4"
                    assert sample['video_url']==expected,(sid,sample['video_url'],expected)
                    with urllib.request.urlopen(urllib.request.Request(BASE+expected,method='HEAD'),timeout=20) as response:
                        assert response.status==200
                        assert int(response.headers['Content-Length'])==r['bytes']
                        assert 'immutable' in response.headers.get('Cache-Control','')
                        head=dict(response.headers)
                    with urllib.request.urlopen(urllib.request.Request(BASE+expected,headers={'Range':'bytes=0-1023'}),timeout=20) as response:
                        data=response.read();assert response.status==206 and len(data)==1024
                        content_range=response.headers['Content-Range']
                    result={'video_url':expected,'source_manifest_sha256':source,'sha256':r['sha256'],'bytes':r['bytes'],
                        'head_status':200,'range_status':206,'range':content_range,'cache_control':head['Cache-Control'],
                        'public_first_observed':now,'bridge_published_time':events.get(sid),
                        'observed_adoption_seconds':now-events[sid] if sid in events else None}
                    report['videos'][sid]=result;print(json.dumps({'verified_video':sid,**result}),flush=True)
        except Exception as error:
            print(json.dumps({'retry_error':repr(error)}),flush=True)
    count=len(report['videos'])
    if count!=last_count:print(json.dumps({'verified_count':count,'expected':4}),flush=True);last_count=count
    report['complete']=set(report['videos'])==ids
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    if report['complete']:break
    time.sleep(2 if candidates else 5)
if not report['complete']:raise SystemExit('Timed out waiting for four real videos')
