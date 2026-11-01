import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright
async def main():
 out=Path('/home/param/websites/pose-mono-dcp227-deploy/qa');out.mkdir(exist_ok=True)
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/home/param/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',headless=True,args=['--no-sandbox'])
  page=await browser.new_page(viewport={'width':1450,'height':1080})
  errors=[]; failed=[]
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.on('response',lambda r:failed.append({'url':r.url,'status':r.status}) if r.status>=400 else None)
  await page.goto('https://awarded-slope-decimal-easier.trycloudflare.com/?model=monoplex',wait_until='domcontentloaded')
  await page.wait_for_function("document.querySelector('#model').textContent.includes('227')",timeout=45000)
  if await page.locator('#mt-set').is_visible():
   await page.locator('#mt-set').click()
  await page.wait_for_timeout(2500)
  info=await page.evaluate('''() => ({title:document.title, model:document.querySelector('#model').innerText, controls:[...document.querySelectorAll('#form input,#form select,#form textarea')].map(e=>({id:e.id,name:e.name,type:e.type,value:e.value,checked:e.checked,disabled:e.disabled})),generatable:!document.querySelector('#go').disabled,streamDisabled:document.querySelector('#streamck').disabled,videoChoices:[...document.querySelector('#vwsel').options].map(e=>({value:e.value,checked:e.checked,disabled:e.disabled})),visibleText:document.body.innerText.slice(0,7000)})''')
  await page.screenshot(path=str(out/'settings-desktop.png'),full_page=True)
  await page.set_viewport_size({'width':390,'height':844})
  await page.wait_for_timeout(500)
  info['mobile']=await page.evaluate('''()=>({scrollWidth:document.documentElement.scrollWidth,width:innerWidth})''')
  await page.screenshot(path=str(out/'settings-mobile.png'),full_page=True)
  info['errors']=errors;info['failedResponses']=failed
  (out/'browser-report.json').write_text(json.dumps(info,indent=2))
  print(json.dumps({k:v for k,v in info.items() if k!='controls'},indent=2)); print(json.dumps([x for x in info['controls'] if x['id'] in ['k-prefill_person','k-prefill_src','k-voice_clip','k-voice','k-pose_cfg_on']]))
  await browser.close()
asyncio.run(main())
