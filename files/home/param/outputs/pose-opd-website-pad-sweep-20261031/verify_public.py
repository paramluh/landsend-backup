from pathlib import Path
import json,time,urllib.request,subprocess
from playwright.sync_api import sync_playwright
root=Path(__file__).parent
base='https://mph-daniel-containers-afternoon.trycloudflare.com/diagnostics/website-pad-sweep-20261031/'
checks=[]
for leaf,count in [('initial/',56),('validation/',24),('combined/',48)]:
 with urllib.request.urlopen(base+leaf+'report.json',timeout=20) as response:r=json.load(response)
 assert r['partner_sampling']=='website_20261031' and r['eligible_for_current_training_selection']
 assert len(r['conversations'])==count and r['partner_sampling_baseline_params']['cb0_topk']==50 and r['partner_sampling_baseline_params']['light_topk']==100
 checks.append({'url':base+leaf,'conversations':count,'policy':r['partner_sampling']})
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--disable-gpu','--autoplay-policy=no-user-gesture-required'])
 page=browser.new_page(viewport={'width':1400,'height':1000});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(base,wait_until='domcontentloaded'); assert '61.7%' in page.locator('body').inner_text();page.screenshot(path=str(root/'PUBLIC-OVERVIEW.png'),full_page=True)
 for leaf,count in [('initial/',56),('validation/',24),('combined/',48)]:
  page.goto(base+leaf,wait_until='domcontentloaded');assert page.locator('audio').count()==count
  player=page.locator('audio').first
  player.evaluate('(a)=>{a.muted=true;a.preload="metadata";a.load()}')
  page.wait_for_function('Number.isFinite(document.querySelector("audio").duration)',timeout=30000)
  duration=player.evaluate('a=>a.duration');assert 119.8<duration<120.2
  player.evaluate('a=>a.play()');page.wait_for_function('document.querySelector("audio").currentTime>.25',timeout=30000)
  player.evaluate('a=>a.pause()');paused=player.evaluate('a=>a.currentTime');page.wait_for_timeout(300)
  assert abs(player.evaluate('a=>a.currentTime')-paused)<.05
  player.evaluate('a=>a.play()');page.wait_for_function('(t)=>document.querySelector("audio").currentTime>t+.2',arg=paused,timeout=15000)
  player.evaluate('a=>{a.pause();a.currentTime=60}')
  page.wait_for_function('document.querySelector("audio").readyState>=2&&Math.abs(document.querySelector("audio").currentTime-60)<.1',timeout=30000)
  source=player.get_attribute('src')
  with urllib.request.urlopen(urllib.request.Request(base+leaf+source,headers={'Range':'bytes=0-1023'}),timeout=20) as response:
   assert response.status==206; range_header=response.headers.get('Content-Range')
  checks.append({'page':leaf,'duration':duration,'audio_count':count,'pause_resume':True,'seek60s':True,'range':range_header})
 assert not errors,errors
 browser.close()
result={'verified_unix':time.time(),'main_url':base,'checks':checks,'page_errors':errors}
(root/'PUBLIC-VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
