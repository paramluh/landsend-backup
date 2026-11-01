import json,pathlib,time,math,urllib.request
from playwright.sync_api import sync_playwright
out=pathlib.Path('/home/param/outputs/pose-opd-overlay-20261031')
base='https://mph-daniel-containers-afternoon.trycloudflare.com'
for attempt in range(60):
 m=json.load(urllib.request.urlopen(base+'/samples/manifest.json'))
 r=next((r for r in m['rollouts'] if r['update']==1),None)
 if r and r['samples'][0].get('video_url'):break
 time.sleep(5)
else:raise RuntimeError('First current-run video has not arrived within five minutes')
s=r['samples'][0];assert s['conversation_mode']=='symmetric_streaming'
assert s['video_offset_seconds']==s['pose_offset_seconds']==.163
assert s.get('video_pose_normalization')
report={'url':base+'/samples/','fixture':'None: unmodified live manifest and current-run first generated video','update':1,'sample_id':s['id'],'video_offset_seconds':s['video_offset_seconds'],'pose_offset_seconds':s['pose_offset_seconds'],'video_pose_normalization':s['video_pose_normalization'],'errors':[]}
with sync_playwright() as pw:
 b=pw.chromium.launch(headless=True,args=['--disable-gpu','--autoplay-policy=no-user-gesture-required'])
 page=b.new_page(viewport={'width':1280,'height':1000});page.on('pageerror',lambda e:report['errors'].append(str(e)))
 response=page.goto(base+'/samples/',wait_until='domcontentloaded');report['http_status']=response.status
 page.wait_for_selector('#rollout:not([disabled])');page.locator('#rollout').select_option(label='Step 1')
 page.wait_for_selector('video:not([hidden])',timeout=60000)
 page.wait_for_function("!document.querySelector('.pose-overlay-toggle').disabled",timeout=60000)
 bounds=page.locator('.pose-wrap').bounding_box()
 page.locator('.pose-overlay-toggle').click()
 seek=page.locator('.transport input');seek.evaluate("e=>{e.value=10;e.dispatchEvent(new Event('input'))}")
 page.wait_for_function("!document.querySelector('video').seeking&&document.querySelector('canvas').dataset.frame==='245'",timeout=60000)
 report['seek_timeline_seconds']=float(seek.input_value());report['video_time_seconds']=page.locator('video').evaluate('v=>v.currentTime')
 report['pose_frame']=int(page.locator('canvas').get_attribute('data-frame'))
 assert abs(report['video_time_seconds']+.163-10)<.001
 assert report['pose_frame']==math.floor((10-.163)*25)
 report['transparent_corner_alpha']=page.locator('canvas').evaluate("c=>c.getContext('2d').getImageData(0,0,1,1).data[3]")
 report['geometry_unchanged']=bounds==page.locator('.pose-wrap').bounding_box()
 page.screenshot(path=str(out/'overlay-current-run-step1.png'))
 button=page.locator('.transport button');button.click();page.wait_for_timeout(700);button.click()
 paused=float(seek.input_value());page.wait_for_timeout(200);assert abs(float(seek.input_value())-paused)<.02
 button.click();page.wait_for_timeout(500);button.click();report['pause_resume_advances']=float(seek.input_value())>paused+.1
 report['single_sample_count']=page.locator('article.sample').count()
 b.close()
assert not report['errors'] and report['geometry_unchanged'] and report['pause_resume_advances'] and report['transparent_corner_alpha']==0
report['verified_at_epoch']=time.time();(out/'CURRENT-RUN-VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
