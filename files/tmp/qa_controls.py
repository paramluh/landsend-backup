import asyncio,json
from playwright.async_api import async_playwright
async def main():
 async with async_playwright() as p:
  b=await p.chromium.launch(executable_path='/home/param/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell',args=['--no-sandbox'])
  page=await b.new_page(viewport={'width':1450,'height':1080})
  await page.goto('https://awarded-slope-decimal-easier.trycloudflare.com/?model=monoplex',wait_until='domcontentloaded')
  await page.wait_for_function("document.querySelector('#k-pose_temp') !== null")
  await page.wait_for_timeout(1000)
  print(json.dumps(await page.locator('button').evaluate_all('(els)=>els.filter(e=>e.offsetWidth&&e.offsetHeight).map(e=>({id:e.id,text:e.innerText,aria:e.getAttribute("aria-label")}))')))
  print(json.dumps(await page.locator('#k-pose_temp').evaluate('(e)=>({visible:!!e.offsetWidth,html:e.outerHTML,parent:e.closest("section")?.id})')))
  await b.close()
asyncio.run(main())
