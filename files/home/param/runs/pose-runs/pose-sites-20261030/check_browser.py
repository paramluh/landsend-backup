import asyncio,json
from pathlib import Path
from playwright.async_api import async_playwright
URL='https://morris-rack-republic-appear.trycloudflare.com'
OUT=Path('/home/param/runs/pose-runs/pose-sites-20261030/browser'); OUT.mkdir(exist_ok=True)
async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
  for mode in ('monoplex','duplex','duplex&dup_mode=duo'):
   page=await browser.new_page(viewport={'width':1440,'height':1100})
   errors=[]
   page.on('pageerror',lambda e:errors.append(str(e)))
   await page.goto(URL+'/?model='+mode,wait_until='domcontentloaded',timeout=60000)
   await page.wait_for_timeout(5000)
   await page.screenshot(path=str(OUT/(mode+'.png')),full_page=True)
   state=await page.evaluate('''() => ({title:document.title, model:typeof MODEL==='undefined'?null:MODEL, schema:typeof SCHEMA==='undefined'?null:SCHEMA.length,values:typeof VALS==='undefined'?null:VALS,body:document.body.innerText.slice(0,4000)})''')
   data={'mode':mode,'errors':errors,'state':state}
   (OUT/(mode+'.json')).write_text(json.dumps(data,indent=2))
   print(json.dumps({'mode':mode,'errors':errors,'model':state['model'],'schema':state['schema'],'body':state['body'][:600]}),flush=True)
   await page.close()
  await browser.close()
asyncio.run(main())
