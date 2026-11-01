"""Read-only public browser QA: selected native viewer and actual media transport."""
import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT=Path('/home/param/outputs/pose-opd-credit-live-attempt2/qa')
BASE='https://mph-daniel-containers-afternoon.trycloudflare.com/samples/index.html'
PROBE="""window.probeMedia=[];const OriginalAudio=window.Audio;window.Audio=function(...args){const a=new OriginalAudio(...args);probeMedia.push(a);return a;};"""
reports=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--disable-gpu'])
 for throttled in (False,True):
  context=browser.new_context(viewport={'width':1440,'height':900})
  page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.add_init_script(PROBE)
  if throttled:
   cdp=context.new_cdp_session(page);cdp.send('Network.enable');cdp.send('Network.emulateNetworkConditions',{'offline':False,'latency':80,'downloadThroughput':625000,'uploadThroughput':125000})
  start=time.monotonic();page.goto(BASE,wait_until='domcontentloaded')
  card=page.locator('article.sample');card.wait_for(timeout=60000);assert card.count()==1
  page.wait_for_function("document.querySelector('.pose-state')?.hidden === true",timeout=60000)
  first_pose=time.monotonic()-start
  assert page.evaluate('probeMedia.length')==2
  stage=card.locator('.pose-wrap').bounding_box();assert abs(stage['width']/stage['height']-8/15)<.002,stage
  button=card.locator('.transport button');seek=card.locator('.transport input[type=range]')
  button.click();page.wait_for_function("Number(document.querySelector('.transport input[type=range]').value)>.25",timeout=60000)
  playing=time.monotonic()-start
  cycles=[]
  for _ in range(4):
   button.click();t=float(seek.input_value());page.wait_for_timeout(160);assert abs(float(seek.input_value())-t)<.05
   button.click();page.wait_for_timeout(500);u=float(seek.input_value());assert u>t+.1,(t,u)
   cycles.append([t,u])
  button.click();seek.evaluate("el=>{el.value=60;el.dispatchEvent(new Event('input'));}")
  page.wait_for_function("document.querySelector('.pose-state')?.hidden === true",timeout=60000)
  button.click();page.wait_for_function("Number(document.querySelector('.transport input[type=range]').value)>60.25",timeout=60000)
  button.click()
  old_media=page.evaluate('probeMedia.length')
  page.select_option('#sample','3');page.wait_for_function("document.querySelector('.pose-state')?.hidden === true",timeout=60000)
  assert card.count()==1 and card.get_attribute('aria-label')=='Sample 4'
  assert page.evaluate('probeMedia.slice(0,-2).every(m=>m.paused)')
  assert page.evaluate('probeMedia.length')==old_media+2
  options=page.locator('#rollout option').evaluate_all('els=>els.map(e=>({text:e.text,value:e.value}))')
  known=next((o for o in options if 'update-8' in o['text'] or o['text']=='Step 8'),None)
  if known:
   page.select_option('#rollout',known['value']);page.wait_for_function("document.querySelector('.pose-state')?.hidden === true",timeout=60000)
   assert '48f6d2d3' in card.locator('.identity').inner_text()
  chosen=page.locator('#rollout').input_value();page.wait_for_timeout(10500)
  # A manifest refresh may reindex options as new steps arrive; identity must remain.
  if known: assert '48f6d2d3' in card.locator('.identity').inner_text()
  assert page.locator('#sample').input_value()=='3' and card.count()==1
  resources=page.evaluate("performance.getEntriesByType('resource').filter(e=>/motion|pose.json|\\.wav/.test(e.name)).map(e=>({name:e.name.split('/').pop(),duration:e.duration,transfer:e.transferSize,encoded:e.encodedBodySize}))")
  assert not any(r['name'].endswith('-pose.json') for r in resources),resources
  assert not errors,errors
  report={'throttled_5mbps':throttled,'seconds_to_first_pose':first_pose,'seconds_to_playing':playing,'pause_resume':cycles,'stage':stage,'selected':card.locator('.identity').inner_text(),'resources':resources,'errors':errors}
  page.screenshot(path=str(OUT/('samples-mono-throttled.png' if throttled else 'samples-mono-live.png')),full_page=True)
  if not throttled:
   page.set_viewport_size({'width':390,'height':844});page.wait_for_timeout(200)
   assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   report['mobile_stage']=card.locator('.pose-wrap').bounding_box()
   page.screenshot(path=str(OUT/'samples-mono-mobile.png'),full_page=True)
  reports.append(report);print(json.dumps({k:v for k,v in report.items() if k!='resources'}),flush=True)
  context.close()
 browser.close()
(OUT/'samples-mono-live.json').write_text(json.dumps(reports,indent=2))
