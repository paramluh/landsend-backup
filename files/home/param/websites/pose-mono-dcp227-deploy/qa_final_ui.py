import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright
BASE='https://awarded-slope-decimal-easier.trycloudflare.com/'
RID='20261031-131233-50b3'
async def main():
 out=Path('/home/param/websites/pose-mono-dcp227-deploy/qa')
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/home/param/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',args=['--no-sandbox'])
  page=await browser.new_page(viewport={'width':1450,'height':1080}); errors=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  await page.goto(BASE+'?model=monoplex&v=v1',wait_until='domcontentloaded')
  await page.wait_for_function("rid=>[...document.querySelectorAll('video')].some(v=>(v.currentSrc||v.src).includes(rid)&&v.readyState>=2)",arg=RID,timeout=45000)
  await page.locator('#posectl').click()
  await page.wait_for_function("rid=>[...document.querySelectorAll('.pose-overlay-canvas')].some(c=>c.dataset.rid===rid&&c.dataset.ready==='true'&&!c.hidden)",arg=RID,timeout=30000)
  await page.evaluate("rid=>{const v=[...document.querySelectorAll('video')].find(v=>(v.currentSrc||v.src).includes(rid)&&v.readyState>=2);v.pause();v.currentTime=3;}",RID)
  await page.wait_for_function("rid=>[...document.querySelectorAll('.pose-overlay-canvas')].some(c=>c.dataset.rid===rid&&c.dataset.frame==='75'&&!c.hidden)",arg=RID,timeout=15000)
  result=await page.evaluate("rid=>({overlay:[...document.querySelectorAll('.pose-overlay-canvas')].filter(c=>c.dataset.rid===rid).map(c=>({frame:c.dataset.frame,ready:c.dataset.ready,hidden:c.hidden,width:c.width,height:c.height})),button:document.querySelector('#posectl').textContent})",RID)
  resp=await page.request.get(BASE+'render/'+RID+'.mp4',headers={'Range':'bytes=0-1023'})
  result['range']={'status':resp.status,'content_range':resp.headers.get('content-range'),'bytes':len(await resp.body())}; assert result['range']['status']==206 and result['range']['bytes']==1024
  await page.screenshot(path=str(out/'sample-pose-overlay-desktop.png'),full_page=True)
  url=BASE+'?model=monoplex&prefill_person=heldout_02&prefill_src=clip&voice_clip=1&voice=same'
  await page.goto(url,wait_until='domcontentloaded');await page.wait_for_function("document.querySelector('#k-prefill_person')!==null")
  result['configured_link']=await page.evaluate("()=>({person:document.querySelector('#k-prefill_person').value,src:document.querySelector('#k-prefill_src').value,voice_clip:document.querySelector('#k-voice_clip').checked,voice:document.querySelector('#k-voice').value,generate_enabled:!document.querySelector('#go').disabled,model:document.querySelector('#model').textContent})")
  assert result['configured_link']['person']=='heldout_02' and result['configured_link']['src']=='clip' and result['configured_link']['voice_clip'] and result['configured_link']['voice']=='same' and result['configured_link']['generate_enabled']
  result['browser_errors']=errors;assert not errors
  (out/'final-ui-report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
  await browser.close()
asyncio.run(main())
