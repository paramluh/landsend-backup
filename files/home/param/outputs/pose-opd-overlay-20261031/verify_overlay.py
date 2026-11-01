import json,pathlib,sys,time,urllib.request
from playwright.sync_api import sync_playwright
root=pathlib.Path('/home/param/websites/pose-opd-credit-dashboard/pose-preview')
out=pathlib.Path('/home/param/outputs/pose-opd-overlay-20261031')
sys.path.insert(0,str(root))
from video_results import VIDEO_POSE_NORMALIZATIONS
base='https://mph-daniel-containers-afternoon.trycloudflare.com'
source=pathlib.Path('/home/param/outputs/pose-opd-teacher-partner-b336-general-live-20261031/samples')
m=json.loads((source/'manifest.json').read_text());rollout=m['rollouts'][0];m['rollouts']=[rollout]
for s in rollout['samples']:
 rel=pathlib.Path(s['pose_url']);folder=root/'dist/samples'/rel.parent
 indexes=[p for p in folder.glob(rel.stem+'.motion-v2*.json') if 'frame_count' in json.loads(p.read_text()) and 'chunks' in json.loads(p.read_text())]
 s['pose_index_url']=(rel.parent/indexes[0].name).as_posix()
 receipt=json.loads((source/'video-results'/f"{s['id']}-{rollout['source_manifest_sha256']}.json").read_text())
 s['video_url']=(rel.parent/f"{s['id']}-video-{receipt['sha256'][:16]}.mp4").as_posix()
 sha=receipt['model']['frame0_pose_sha256'];s['video_pose_normalization']=dict(VIDEO_POSE_NORMALIZATIONS[sha],frame0_pose_sha256=sha)
 for k in ['pose_url','pose_index_url','video_url','input_audio_url','output_audio_url']:s[k]=base+'/samples/'+s[k]
(out/'visual-test-fixture.json').write_text(json.dumps(m,indent=2))
report={'fixture':'Intercepted browser manifest only; previous real update13 render. Live manifest is unchanged.','public_url':base+'/samples/','errors':[]}
with sync_playwright() as pw:
 b=pw.chromium.launch(headless=True,args=['--disable-gpu','--autoplay-policy=no-user-gesture-required'])
 page=b.new_page(viewport={'width':1280,'height':1000});page.on('pageerror',lambda e:report['errors'].append(str(e)))
 page.route('**/samples/manifest.json',lambda r:r.fulfill(json=m))
 response=page.goto(base+'/samples/',wait_until='domcontentloaded');report['http_status']=response.status
 page.wait_for_selector('video:not([hidden])',timeout=60000)
 page.wait_for_function("!document.querySelector('.pose-overlay-toggle').disabled",timeout=60000)
 seek=page.locator('.transport input');seek.evaluate("e=>{e.value=8;e.dispatchEvent(new Event('input'))}")
 page.wait_for_function("!document.querySelector('video').seeking")
 bounds=page.locator('.pose-wrap').bounding_box();page.screenshot(path=str(out/'video-without-overlay.png'))
 page.locator('.pose-overlay-toggle').click()
 page.wait_for_function("document.querySelector('canvas').dataset.frame==='200'")
 page.screenshot(path=str(out/'overlay-real-render.png'))
 report['geometry_unchanged']=bounds==page.locator('.pose-wrap').bounding_box()
 report['overlay_frame_at_8s']=page.locator('canvas').get_attribute('data-frame')
 report['transparent_corner_alpha']=page.locator('canvas').evaluate("c=>c.getContext('2d').getImageData(0,0,1,1).data[3]")
 report['stage_bounds']=bounds
 button=page.locator('.transport button');button.click();page.wait_for_timeout(700);button.click()
 paused=float(seek.input_value());page.wait_for_timeout(200);assert abs(float(seek.input_value())-paused)<.02
 button.click();page.wait_for_timeout(500);button.click();report['pause_resume_advances']=float(seek.input_value())>paused+.1
 page.locator('#sample').select_option('1');page.wait_for_selector('canvas.pose-over-video:not([hidden])')
 report['choice_persists_across_sample']=page.locator('.pose-overlay-toggle').get_attribute('aria-pressed')=='true'
 report['single_sample_count']=page.locator('article.sample').count()
 b.close()
assert not report['errors'] and report['geometry_unchanged'] and report['pause_resume_advances']
(out/'BROWSER-VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
