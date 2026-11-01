import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright
BASE='https://awarded-slope-decimal-easier.trycloudflare.com/'
RID='20261031-131233-50b3'
async def main():
 out=Path('/home/param/websites/pose-mono-dcp227-deploy/qa')
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/home/param/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',args=['--no-sandbox'])
  page=await browser.new_page(viewport={'width':1450,'height':1080});errors=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  for i in range(180):
   r=await page.request.get(BASE+'api/run/'+RID);j=await r.json()
   if (j.get('render') or {}).get('mp4'):break
   if i%10==0:print('WAITING_VIDEO',j.get('render'),flush=True)
   await asyncio.sleep(2)
  else:raise RuntimeError('Rendered video not ready')
  await page.goto(BASE+'?model=monoplex&v=v1',wait_until='domcontentloaded')
  await page.wait_for_function("rid=>[...document.querySelectorAll('video')].some(v=>(v.currentSrc||v.src).includes(rid)&&v.readyState>=2)",arg=RID,timeout=45000)
  playback=await page.evaluate('''async rid=>{const v=[...document.querySelectorAll('video')].find(v=>(v.currentSrc||v.src).includes(rid)&&v.readyState>=2); v.muted=true;v.currentTime=3;await v.play();return {src:v.currentSrc,duration:v.duration,width:v.videoWidth,height:v.videoHeight,readyState:v.readyState,error:v.error?.message||null};}''',RID)
  await page.wait_for_timeout(800)
  await page.screenshot(path=str(out/'sample-playback-desktop.png'),full_page=True)
  playback['errors']=errors;playback['render']=j.get('render');playback['run_id']=RID
  (out/'playback-report.json').write_text(json.dumps(playback,indent=2)+'\n')
  print('PLAYBACK_VERIFIED',json.dumps(playback),flush=True)
  await browser.close()
asyncio.run(main())
