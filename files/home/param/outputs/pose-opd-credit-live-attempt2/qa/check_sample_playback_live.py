import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
out=Path('/home/param/outputs/pose-opd-credit-live-attempt2/qa')
base='https://mph-daniel-containers-afternoon.trycloudflare.com/samples/index.html'
probe="""window.probeMedia=[];window.probeEvents=[]; const OriginalAudio=window.Audio; window.Audio=function(...args){const a=new OriginalAudio(...args);let i=probeMedia.push(a)-1;for(const event of ['playing','waiting','pause','seeking','seeked','error','canplay'])a.addEventListener(event,()=>{probeEvents.push({i,event,time:performance.now(),position:a.currentTime,ready:a.readyState,paused:a.paused});if(probeEvents.length>600)probeEvents.shift();});return a;};"""
reports=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--disable-gpu'])
 for throttled in [False,True]:
  context=browser.new_context(viewport={'width':1280,'height':900})
  page=context.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.add_init_script(probe)
  if throttled:
   cdp=context.new_cdp_session(page);cdp.send('Network.enable');cdp.send('Network.emulateNetworkConditions',{'offline':False,'latency':80,'downloadThroughput':625000,'uploadThroughput':125000})
  t=time.monotonic();page.goto(base,wait_until='domcontentloaded')
  button=page.locator('article.sample').first.locator('.transport button');button.wait_for(timeout=60000)
  initial=page.evaluate('probeMedia.map(m=>({ready:m.readyState,network:m.networkState,current:m.currentTime}))')
  card=page.locator('article.sample').first;seek=card.locator('input[type=range]')
  button.click();page.wait_for_function("Number(document.querySelector('article.sample input[type=range]').value)>.25",timeout=60000)
  started=time.monotonic()-t
  trials=[]
  for k in range(4):
   button.click();paused=float(seek.input_value());page.wait_for_timeout(120)
   assert abs(float(seek.input_value())-paused)<.05
   button.click();page.wait_for_timeout(450)
   resumed=float(seek.input_value());assert resumed>paused+.1,(paused,resumed,card.locator('.sample-status').inner_text())
   trials.append({'paused':paused,'resumed':resumed})
  button.click();seek.evaluate("el=>{el.value=60;el.dispatchEvent(new Event('input'));}")
  button.click();page.wait_for_function("Number(document.querySelector('article.sample input[type=range]').value)>60.25",timeout=60000)
  report={'throttled_5mbps':throttled,'seconds_to_playing':started,'initial':initial,'trials':trials,'seek60_position':float(seek.input_value()),'status':card.locator('.sample-status').inner_text(),'errors':errors,'events':page.evaluate('probeEvents'),'resources':page.evaluate("performance.getEntriesByType('resource').filter(e=>/pose.json|\\.wav/.test(e.name)).map(e=>({name:e.name.split('/').pop(),duration:e.duration,transfer:e.transferSize,encoded:e.encodedBodySize,decoded:e.decodedBodySize}))")}
  assert not errors,errors
  page.screenshot(path=str(out/('samples-playback-throttled.png' if throttled else 'samples-playback-live.png')),full_page=True)
  reports.append(report);print(json.dumps({k:v for k,v in report.items() if k not in ['events','resources','initial']}),flush=True)
  context.close()
 browser.close()
(out/'samples-playback-live.json').write_text(json.dumps(reports,indent=2))
