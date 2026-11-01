import asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright
OUT=Path('/home/param/runs/pose-runs/pose-sites-20261030/browser'); OUT.mkdir(exist_ok=True)
async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream','--autoplay-policy=no-user-gesture-required'])
  context=await browser.new_context(permissions=['microphone'],viewport={'width':1440,'height':1100})
  page=await context.new_page(); errors=[]; failed=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.on('requestfailed',lambda r:failed.append({'url':r.url.split('?')[0],'error':r.failure}))
  await page.goto('https://morris-rack-republic-appear.trycloudflare.com/?model=livedup&preset=heldout_02&prefill_s=1.92&prefill_src=clip',wait_until='domcontentloaded',timeout=60000)
  await page.wait_for_function('window.livedup && !livedup.LD.down && !livedup.LD.busy',timeout=60000)
  print('Live browser ready; starting fake-device microphone integration test.',flush=True)
  await page.get_by_role('button',name='Start',exact=True).click()
  states=[]
  for _ in range(16):
   await page.wait_for_timeout(2000)
   state=await page.evaluate('''() => ({session:livedup.LD.sid,running:livedup.LD.running,connecting:livedup.LD.connecting,frames:livedup.LD.frames,error:livedup.LD.err,step:livedup.LD.step,status:document.querySelector('#ldst')?.textContent,iframe:document.querySelector('#ldav')?.src})''')
   states.append(state); print(json.dumps(state),flush=True)
   if state.get('frames',0)>=80: break
  await page.screenshot(path=str(OUT/'live-call-running.png'),full_page=True)
  await page.evaluate('livedup.stop()')
  await page.wait_for_timeout(3000)
  result={'states':states,'errors':errors,'failed_requests':failed,'success':any((x.get('frames') or 0)>=50 and x.get('step')==424 for x in states)}
  (OUT/'live-browser-result.json').write_text(json.dumps(result,indent=2))
  print('RESULT',json.dumps(result),flush=True)
  await browser.close()
asyncio.run(main())
