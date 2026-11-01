import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright

ROOT=Path('/home/param/outputs/pose-opd-credit-live-attempt2/qa')
URL='https://mph-daniel-containers-afternoon.trycloudflare.com/diagnostics/48f6d2d3/'

async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--autoplay-policy=no-user-gesture-required'])
  page=await browser.new_page(viewport={'width':1280,'height':1000})
  errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  await page.goto(URL,wait_until='domcontentloaded')
  await page.evaluate('''() => { window.events=[]; for(const [i,v] of [...document.querySelectorAll('video')].entries())for(const n of ['play','playing','pause','waiting','seeking','seeked','error'])v.addEventListener(n,()=>events.push({i,n,t:v.currentTime,ready:v.readyState,wall:performance.now()})); }''')
  results=[]
  for i in [0,1]:
   v=page.locator('video').nth(i)
   await v.scroll_into_view_if_needed()
   await v.evaluate('(v)=>v.play()')
   await page.wait_for_timeout(1400)
   before=await v.evaluate('(v)=>({t:v.currentTime,paused:v.paused,ready:v.readyState})')
   await v.evaluate('(v)=>v.pause()')
   await page.wait_for_timeout(350)
   paused=await v.evaluate('(v)=>({t:v.currentTime,paused:v.paused,ready:v.readyState})')
   await v.evaluate('(v)=>v.play()')
   await page.wait_for_timeout(1400)
   resumed=await v.evaluate('(v)=>({t:v.currentTime,paused:v.paused,ready:v.readyState})')
   assert paused['paused'] and not resumed['paused'] and resumed['t']>paused['t']+.5,(i,before,paused,resumed)
   await v.evaluate('(v)=>v.pause()')
   results.append(dict(video=i,before=before,paused=paused,resumed=resumed))
  doc={'url':URL,'results':results,'errors':errors,'events':await page.evaluate('events')}
  (ROOT/'diagnostic-video-playback.json').write_text(json.dumps(doc,indent=2))
  print(json.dumps({'results':results,'errors':errors}))
  await browser.close()
asyncio.run(main())
