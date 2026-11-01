import asyncio,hashlib,hmac,json,os
from pathlib import Path
from playwright.async_api import async_playwright
ROOT=Path('/home/param/pose-runs/pose-overlay-ui-audit-20261029')
BASE='https://receipt-diana-maple-mentor.trycloudflare.com'
async def main():
 token=Path('/home/param/pose-runs/controlled-comparison-20261028/gallery/.access-token').read_text().strip()
 cookie=hmac.new(token.encode(),b'comparison-gallery-session-v1',hashlib.sha256).hexdigest()
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,executable_path='/home/param/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome',args=['--disable-gpu','--disable-gpu-compositing','--use-gl=disabled'])
  ctx=await browser.new_context(viewport={'width':1440,'height':1100})
  await ctx.add_cookies([{'name':'comparison_session','value':cookie,'url':BASE,'httpOnly':True,'secure':True,'sameSite':'Lax'}])
  page=await ctx.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  manifests=[];report={'pages':[],'assets_checked':0}
  for route,expected in [('/manifest.json',24),('/native-manifest.json',16),('/api/requests',14)]:
   response=await ctx.request.get(BASE+route);assert response.status==200,(route,response.status)
   data=await response.json();rows=([o for c in data['clips'] for o in c['outputs'].values() if o.get('status')=='complete'] if 'clips' in data else [r for r in data['requests'] if r['status']=='complete'])
   assert len(rows)==expected and all(r.get('pose_video') and r.get('pose_overlay_video') for r in rows),(route,len(rows))
   manifests.append(rows)
   for row in rows:
    for key in ['pose_video','pose_overlay_video']:
     r=await ctx.request.get(BASE+row[key],headers={'Range':'bytes=0-255'})
     assert r.status==206 and len(await r.body())==256,(key,row[key],r.status)
     report['assets_checked']+=1
  for route,expected in [('/comparison',24),('/native-comparison',16),('/',14)]:
   await page.goto(BASE+route,wait_until='networkidle');await page.wait_for_function('document.querySelectorAll(".pose-player").length === '+str(expected))
   player=page.locator('.pose-player').first;video=player.locator('video')
   assert await player.get_attribute('data-view')=='overlay'
   await video.evaluate('(v)=>{v.muted=true;return v.play()}')
   await page.wait_for_function('document.querySelector(".pose-player video").readyState>=3')
   state=await video.evaluate('(v)=>{v.pause();v.volume=.35;v.playbackRate=1.25;v.currentTime=Math.min(7.36,v.duration/3);return {time:v.currentTime,duration:v.duration}}')
   await page.wait_for_function('!document.querySelector(".pose-player video").seeking')
   for mode in ['pose','original','overlay']:
    await player.locator('[data-view="'+mode+'"]').click()
    await page.wait_for_function('!document.querySelector(".pose-player").hasAttribute("aria-busy") && !document.querySelector(".pose-player video").seeking && document.querySelector(".pose-player video").readyState>=2')
    now=await video.evaluate('(v)=>({time:v.currentTime,paused:v.paused,muted:v.muted,volume:v.volume,rate:v.playbackRate,src:v.getAttribute("src")})')
    assert abs(now['time']-state['time'])<.05 and now['paused'] and now['muted'] and abs(now['volume']-.35)<.001 and now['rate']==1.25,(route,mode,now,state)
   await player.screenshot(path=str(ROOT/('live-'+('studio' if route=='/' else route[1:])+'-overlay.png')))
   await player.locator('[data-view="pose"]').click()
   await page.wait_for_function('!document.querySelector(".pose-player").hasAttribute("aria-busy")')
   await video.evaluate('(v)=>v.requestFullscreen()');assert await page.evaluate('Boolean(document.fullscreenElement)');await page.evaluate('document.exitFullscreen()')
   await video.evaluate('(v)=>v.play()');await page.wait_for_timeout(250)
   before=await video.evaluate('(v)=>v.currentTime')
   await player.locator('[data-view="overlay"]').click()
   await page.wait_for_function('!document.querySelector(".pose-player").hasAttribute("aria-busy") && !document.querySelector(".pose-player video").paused')
   after=await video.evaluate('(v)=>v.currentTime');assert after>=before-.05 and after<before+3,(before,after)
   await video.evaluate('(v)=>v.pause()')
   await player.locator('[data-view="pose"]').click();await page.wait_for_timeout(10500)
   assert await player.get_attribute('data-view')=='pose'
   await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('document.documentElement.scrollWidth<=window.innerWidth'),route
   await page.set_viewport_size({'width':1440,'height':1100})
   report['pages'].append({'route':route,'players':expected,'switching_preserves_state':True,'playing_switch':True,'poll_keeps_selection':True,'fullscreen':True,'mobile_no_overflow':True})
  assert not errors,errors
  report.update(status='pass',javascript_errors=errors)
  (ROOT/'live-browser-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
  await browser.close()
asyncio.run(main())
