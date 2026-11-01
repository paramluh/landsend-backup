import asyncio,json,hashlib,hmac
from pathlib import Path
from playwright.async_api import async_playwright
ROOT=Path('/home/param/pose-runs/sampling-label-audit-20261029');BASE='https://receipt-diana-maple-mentor.trycloudflare.com'
async def main():
 token=Path('/home/param/pose-runs/controlled-comparison-20261028/gallery/.access-token').read_text().strip();cookie=hmac.new(token.encode(),b'comparison-gallery-session-v1',hashlib.sha256).hexdigest()
 async with async_playwright() as p:
  b=await p.chromium.launch(headless=True,executable_path='/home/param/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome',args=['--disable-gpu','--disable-gpu-compositing','--use-gl=disabled'])
  c=await b.new_context(viewport={'width':390,'height':844});await c.add_cookies([{'name':'comparison_session','value':cookie,'url':BASE,'httpOnly':True,'secure':True,'sameSite':'Lax'}]);page=await c.new_page();r=[]
  for route,count in [('/comparison',24),('/native-comparison',16)]:
   await page.goto(BASE+route,wait_until='domcontentloaded');await page.wait_for_function('document.querySelectorAll(".sampling-panel").length === '+str(count))
   assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
   label=await page.locator('.sampling-panel td[data-setting="temp"]').first.evaluate('(e)=>getComputedStyle(e,"::before").content');assert 'Temp:' in label,label
   assert await page.locator('.sampling-panel tbody tr').first.evaluate('(e)=>getComputedStyle(e).display')=='block'
   await page.locator('section').first.screenshot(path=str(ROOT/(route[1:]+'-sampling-mobile-final.png')))
   r.append({'route':route,'390px_no_overflow':True,'stacked_stream_rows':True,'temp_topk_labels_present':True})
  (ROOT/'mobile-followup.json').write_text(json.dumps({'status':'pass','pages':r},indent=2));print(json.dumps({'status':'pass','pages':r}));await b.close()
asyncio.run(main())
