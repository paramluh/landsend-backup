import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
S=Path('/home/param/runs/pose-runs/pose-jump-audit-suite-20261031');out=S/'hand-focused-review';url='https://morris-rack-republic-appear.trycloudflare.com/diagnostics/pose-jumps/hand-focused/index.html';errors=[];checks=[]
with sync_playwright() as p:
 b=p.chromium.launch(headless=True);page=b.new_page(viewport={'width':1400,'height':1000});page.on('pageerror',lambda e:errors.append(str(e)));response=page.goto(url,wait_until='domcontentloaded',timeout=30000);assert response.status==200
 page.wait_for_function("document.querySelectorAll('.card').length===4 && [...document.querySelectorAll('[data-event]')].every(x=>!x.disabled)",timeout=30000)
 for i in range(4):
  c=page.locator('.card').nth(i);v=c.locator('video');c.locator('[data-before]').click();page.wait_for_timeout(150);before=c.get_attribute('data-frame');c.locator('[data-step="1"]').click();page.wait_for_timeout(150);after=c.get_attribute('data-frame');assert int(after)==int(before)+1 and 'CANDIDATE FRAME' in c.locator('.clock').inner_text();tm=v.evaluate('(v)=>v.currentTime');assert abs(tm-2.004)<.01
  c.locator('[data-overlay]').uncheck();page.wait_for_function('(i)=>!document.querySelectorAll("[data-event]")[i].disabled',arg=i);page.wait_for_timeout(100);assert c.get_attribute('data-frame')==after and '/video.mp4' in v.get_attribute('src');c.locator('[data-overlay]').check();page.wait_for_function('(i)=>!document.querySelectorAll("[data-event]")[i].disabled',arg=i);page.wait_for_timeout(100);assert c.get_attribute('data-frame')==after and '/overlay.mp4' in v.get_attribute('src');c.locator('[data-before]').click();checks.append({'card':c.get_attribute('id'),'before_frame':int(before),'candidate_frame':int(after),'candidate_seek_seconds':tm,'overlay_toggle_preserved_frame':True,'video_dimensions':v.evaluate('(v)=>[v.videoWidth,v.videoHeight]')})
 c=page.locator('.card').first;c.locator('[data-play]').click();page.wait_for_timeout(1300);a=c.locator('video').evaluate('(v)=>({time:v.currentTime,rate:v.playbackRate,paused:v.paused})');assert a['rate']==.1 and not a['paused'] and 1.99<a['time']<2.20;c.locator('[data-play]').click();assert c.locator('video').evaluate('(v)=>v.paused')
 page.screenshot(path=str(out/'focused-page.png'),full_page=True);assert not errors
 for route in ['hand-review/index.html','index.html']:
  page.goto('https://morris-rack-republic-appear.trycloudflare.com/diagnostics/pose-jumps/'+route,wait_until='domcontentloaded');assert page.locator('a[href*="hand-focused/index.html"]').count()>=1
 receipt={'url':url,'http_status':response.status,'cards':4,'frame_and_overlay_checks':checks,'slow_playback':a,'main_and_hand_review_links_present':True,'browser_errors':errors,'passed':True};(out/'browser-validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));b.close()
