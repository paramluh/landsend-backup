import asyncio,json,sys
from pathlib import Path
from playwright.async_api import async_playwright
URL='https://awarded-slope-decimal-easier.trycloudflare.com/?model=monoplex&prefill_person=heldout_02&prefill_src=clip&voice_clip=1&voice=same'
async def main():
 out=Path('/home/param/websites/pose-mono-dcp227-deploy/qa');phase='before' if '--before' in sys.argv else 'after'
 async with async_playwright() as p:
  b=await p.chromium.launch(executable_path='/home/param/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',args=['--no-sandbox'])
  page=await b.new_page(viewport={'width':1450,'height':1080});failures=[];errors=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.on('response',lambda r:failures.append({'url':r.url,'status':r.status}) if r.status>=400 and '/render/' in r.url else None)
  if phase=='after':
   for attempt in range(30):
    resp=await page.request.get('https://awarded-slope-decimal-easier.trycloudflare.com/api/schema');d=await resp.json()
    if all(d['prefill_data'].get(f'heldout_{i:02d}',{}).get('thumb') for i in range(1,9)):break
    await asyncio.sleep(2)
  await page.goto(URL,wait_until='domcontentloaded');await page.wait_for_selector('#mt-act');await page.locator('#mt-act').click();await page.wait_for_timeout(1200)
  info={'tiles':await page.locator('#pkreel .pkt').evaluate_all('(els)=>els.map(e=>({id:e.dataset.v,images:[...e.querySelectorAll("img")].map(i=>({src:i.src,loaded:i.complete&&i.naturalWidth>0,width:i.naturalWidth,height:i.naturalHeight}))}))')}
  await page.screenshot(path=str(out/f'actor-faces-{phase}.png'),full_page=True)
  await page.locator('#pkreel .pkt[data-v="heldout_02"]').click();await page.wait_for_timeout(1200)
  info['modal']=await page.locator('#agplay').evaluate('(e)=>({text:e.innerText,video:[...e.querySelectorAll("video")].map(v=>({src:v.currentSrc,readyState:v.readyState,width:v.videoWidth,height:v.videoHeight,duration:v.duration,error:v.error?.message})),images:[...e.querySelectorAll("img")].map(i=>({src:i.src,loaded:i.complete&&i.naturalWidth>0}))})')
  if phase=='after':
   await page.wait_for_function("document.querySelector('#agplay video')?.readyState>=2")
   info['modal']=await page.locator('#agplay').evaluate('(e)=>({text:e.innerText,video:[...e.querySelectorAll("video")].map(v=>({src:v.currentSrc,readyState:v.readyState,width:v.videoWidth,height:v.videoHeight,duration:v.duration,error:v.error?.message}))})')
   assert len(info['tiles'])==8 and all(x['images'] and all(i['loaded'] for i in x['images']) for x in info['tiles'])
   assert info['modal']['video'][0]['width']>0
  await page.screenshot(path=str(out/f'actor-preview-{phase}.png'),full_page=True)
  info.update(errors=errors,failures=failures);(out/f'actor-faces-{phase}.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2));await b.close()
asyncio.run(main())
