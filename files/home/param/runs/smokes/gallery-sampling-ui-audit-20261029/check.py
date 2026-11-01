import json
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright
ROOT=Path('/home/param/pose-demo-website/comparison')
OUT=Path('/home/param/pose-runs/gallery-sampling-ui-audit-20261029')
manifests={name:json.loads((Path('/home/param/pose-runs')/directory/'gallery/manifest.json').read_text()) for name,directory in [('controlled','controlled-comparison-20261028'),('native','native-comparison-20261028')]}
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path='/home/param/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome',headless=True,args=['--disable-gpu','--no-sandbox'])
 page=browser.new_page(viewport={'width':1280,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 def route(r):
  path=urlsplit(r.request.url).path
  if path.endswith('.mp4'):
   data=Path('/home/param/pose-runs/native-comparison-20261028/outputs/01/compact-all/video.mp4').read_bytes()
   start,end=0,len(data)-1;raw=r.request.headers.get('range')
   if raw:
    bounds=raw.removeprefix('bytes=').split('-');start=int(bounds[0] or 0);end=min(int(bounds[1]) if bounds[1] else end,end)
   headers={'Accept-Ranges':'bytes','Content-Length':str(end-start+1)}
   if raw:headers['Content-Range']=f'bytes {start}-{end}/{len(data)}'
   return r.fulfill(status=206 if raw else 200,content_type='video/mp4',headers=headers,body=data[start:end+1])
  if path.endswith('.js') or path.endswith('.css'):
   return r.fulfill(status=200,content_type='application/javascript' if path.endswith('.js') else 'text/css',body=(ROOT/path[1:]).read_bytes())
  if path in ['/comparison','/native-comparison']:
   return r.fulfill(status=200,content_type='text/html',body=(ROOT/'index.html').read_bytes())
  if path in ['/manifest.json','/native-manifest.json']:
   return r.fulfill(status=200,content_type='application/json',body=json.dumps(manifests['native' if path.startswith('/native-') else 'controlled']))
  return r.fulfill(status=404)
 page.route('http://sampling.test/**',route)
 report={'status':'pass','galleries':[],'edge_cases':{}}
 for name,path in [('controlled','/comparison'),('native','/native-comparison')]:
  page.goto('http://sampling.test'+path)
  count=sum(o['status']=='complete' for c in manifests[name]['clips'] for o in c['outputs'].values())
  page.wait_for_function(f'document.querySelectorAll(".sampling-panel").length=={count}')
  texts=page.locator('.sampling-panel').all_text_contents()
  assert all('0 (none)' in t and 'Generation seed:' in t for t in texts)
  assert all(('Text: forced' in t and 'Not sampled' in t) if name=='controlled' else 'Text: native · sampled' in t for t in texts)
  assert all('Not recorded' not in t for t in texts),texts
  for c in manifests[name]['clips']:
   for model,o in c['outputs'].items():
    card=page.locator(f'section[data-clip="{c["id"]}"] .cell').nth([m['id'] for m in manifests[name]['models']].index(model))
    settings=o['metadata']['sampling']
    for stream in ['audio','pose']+(['text'] if name=='native' else []):
     assert card.locator(f'[data-stream="{stream}"] [data-setting="temp"]').inner_text()==str(settings[f'{stream}_temperature']).removesuffix('.0')
     assert card.locator(f'[data-stream="{stream}"] [data-setting="topk"]').inner_text()=='0 (none)'
  page.locator('.pose-player').first.evaluate('(x)=>x.posePlayer.select("original")')
  before=page.locator('.sampling-panel').first.inner_text()
  page.evaluate('revision="";update()');page.wait_for_timeout(150)
  assert page.locator('.sampling-panel').first.inner_text()==before
  assert page.locator('.pose-player').first.get_attribute('data-view')=='original'
  page.locator('section').first.screenshot(path=str(OUT/(name+'-desktop.png')))
  page.set_viewport_size({'width':390,'height':844});page.locator('section').first.screenshot(path=str(OUT/(name+'-mobile.png')))
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
  page.set_viewport_size({'width':1280,'height':1000})
  report['galleries'].append({'name':name,'panels_checked':count,'settings_match_manifest':True,'poll_preserves_player_and_settings':True,'mobile_no_overflow':True})
 # Formatter edge cases are explicit and must never silently inherit defaults.
 unknown=page.evaluate('sampling({metadata:{}}).innerText')
 assert 'not recorded' in unknown.lower() and '0 (none)' not in unknown and 'Generation seed:' not in unknown
 greedy=page.evaluate('sampling({metadata:{text_mode:"native",seed:0,sampling:{audio_temperature:0,audio_top_k:30,pose_temperature:.8,pose_top_k:150,text_temperature:1.2,text_top_k:7,residual_codebooks_use_same_settings:false}}}).innerText')
 assert '0 (greedy)' in greedy and all(value in greedy for value in ['30','150','1.2','0.8','7','Generation seed: 0','settings differ'])
 partial=page.evaluate('sampling({metadata:{force_text:true,sampling:{audio_temperature:1,audio_top_k:null,pose_temperature:"1",pose_top_k:-1}}}).innerText')
 assert partial.count('Not recorded')==3 and 'Not sampled' in partial
 report['edge_cases']={'unknown_no_defaults':True,'temperature_zero_greedy':True,'nondefault_values':True,'seed_zero_retained':True,'partial_invalid_metadata_explicit':True}
 assert not errors,errors
 report['javascript_errors']=errors;report['gpu_disabled']=True;report['jobs_submitted']=0
 (OUT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));browser.close()
