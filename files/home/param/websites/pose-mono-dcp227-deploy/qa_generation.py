import asyncio,json,sys,time
from pathlib import Path
from playwright.async_api import async_playwright
BASE='https://awarded-slope-decimal-easier.trycloudflare.com/'
async def main():
 out=Path('/home/param/websites/pose-mono-dcp227-deploy/qa');out.mkdir(exist_ok=True)
 async with async_playwright() as p:
  browser=await p.chromium.launch(executable_path='/home/param/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',args=['--no-sandbox'])
  page=await browser.new_page(viewport={'width':1450,'height':1080})
  errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  await page.goto(BASE+'?model=monoplex',wait_until='domcontentloaded')
  await page.wait_for_function("document.querySelector('#k-clip_s') !== null")
  await page.get_by_role('button',name='Run',exact=True).click()
  await page.locator('#k-clip_s').fill('4')
  await page.locator('#k-clip_s').dispatch_event('change')
  await page.locator('#k-seed').fill('2272026')
  await page.locator('#k-seed').dispatch_event('change')
  await page.get_by_role('button',name='Pose',exact=True).click()
  await page.locator('#k-pose_temp').fill('0.8')
  await page.locator('#k-pose_temp').dispatch_event('change')
  settings=await page.evaluate("()=>({seconds:document.querySelector('#k-clip_s').value,seed:document.querySelector('#k-seed').value,pose_temp:document.querySelector('#k-pose_temp').value,person:document.querySelector('#k-prefill_person').value,voice:document.querySelector('#k-voice').value,voice_clip:document.querySelector('#k-voice_clip').checked,src:document.querySelector('#k-prefill_src').value,model:document.querySelector('#model').innerText})")
  print('CONTROL_SETTINGS',json.dumps(settings),flush=True)
  if '--generate' not in sys.argv:
   await browser.close();return
  async with page.expect_response(lambda r:'/api/gen' in r.url and r.request.method=='POST',timeout=30000) as pending:
   await page.locator('#go').click()
  response=await pending.value
  record={'settings':settings,'status':response.status,'request':response.request.post_data_json,'response':await response.json()}
  rid=record['response']['run_id']
  (out/'generation-request.json').write_text(json.dumps(record,indent=2)+'\n')
  print('GENERATION_REQUEST',json.dumps(record),flush=True)
  for i in range(240):
   api=await page.request.get(BASE+'api/run/'+rid)
   status=await api.json()
   if i%10==0:print('GENERATION_STATUS',status.get('state'),status.get('progress'),status.get('msg'),flush=True)
   if status.get('state') in ['done','error','failed','cancelled']:
    record['result']=status;break
   await asyncio.sleep(2)
  record['browser_errors']=errors
  (out/'generation-result.json').write_text(json.dumps(record,indent=2)+'\n')
  await page.screenshot(path=str(out/'generation-after.png'),full_page=True)
  print('GENERATION_RESULT',json.dumps(record.get('result',{})),flush=True)
  await browser.close()
asyncio.run(main())
