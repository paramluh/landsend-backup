import asyncio,hashlib,hmac,json,datetime
from pathlib import Path
from playwright.async_api import async_playwright
ROOT=Path('/home/param/pose-runs/sampling-label-audit-20261029')
BASE='https://receipt-diana-maple-mentor.trycloudflare.com'
def num(v,kind):
 if isinstance(v,(int,float)) and not isinstance(v,bool) and v>=0:
  if v==0:return '0 (none)' if kind=='topk' else '0 (greedy)'
  return str(v) if v!=int(v) else str(int(v))
 return 'Not recorded'
async def main():
 token=Path('/home/param/pose-runs/controlled-comparison-20261028/gallery/.access-token').read_text().strip()
 cookie=hmac.new(token.encode(),b'comparison-gallery-session-v1',hashlib.sha256).hexdigest()
 report={'status':'running','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pages':[]}
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,executable_path='/home/param/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome',args=['--disable-gpu','--disable-gpu-compositing','--use-gl=disabled'])
  ctx=await browser.new_context(viewport={'width':1440,'height':1100})
  await ctx.add_cookies([{'name':'comparison_session','value':cookie,'url':BASE,'httpOnly':True,'secure':True,'sameSite':'Lax'}])
  page=await ctx.new_page();errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  for route,manifest,expected in [('/comparison','/manifest.json',24),('/native-comparison','/native-manifest.json',16)]:
   response=await ctx.request.get(BASE+manifest);assert response.status==200
   data=await response.json()
   await page.goto(BASE+route,wait_until='domcontentloaded')
   await page.wait_for_function('document.querySelectorAll(".sampling-panel").length === '+str(expected))
   count=0
   for clip in data['clips']:
    section=page.locator('section[data-clip="'+clip['id']+'"]')
    for mi,model in enumerate(data['models']):
     out=clip['outputs'][model['id']];assert out['status']=='complete'
     meta=out['metadata'];settings=meta.get('sampling',{});panel=section.locator('.cell').nth(mi).locator('.sampling-panel')
     assert await panel.is_visible()
     forced=meta.get('text_mode')=='forced' or meta.get('force_text') is True
     assert await panel.locator('.sampling-mode').inner_text()==('Text: forced' if forced else 'Text: native · sampled')
     for stream in ['audio','pose','text']:
      row=panel.locator('tr[data-stream="'+stream+'"]')
      if stream=='text' and forced:assert await row.locator('td').inner_text()=='Not sampled'
      else:
       for suffix,kind in [('temperature','temp'),('top_k','topk')]:
        actual=await row.locator('td[data-setting="'+kind+'"]').inner_text()
        assert actual==num(settings.get(stream+'_'+suffix),kind),(clip['id'],model['id'],stream,kind,actual,settings)
     if settings.get('residual_codebooks_use_same_settings') is True:assert 'CB0 and residual' in await panel.locator('.sampling-note').inner_text()
     for key,label in [('seed','Generation seed'),('video_seed','Video seed')]:
      if key in meta:assert f'{label}: {meta[key]}' in await panel.inner_text()
     count+=1
   assert count==expected
   await page.locator('.cell').first.screenshot(path=str(ROOT/(route[1:]+'-sampling-desktop.png')))
   await page.set_viewport_size({'width':390,'height':844})
   assert await page.evaluate('document.documentElement.scrollWidth<=window.innerWidth'),route
   await page.locator('section').first.screenshot(path=str(ROOT/(route[1:]+'-sampling-mobile.png')))
   # Exercise the live function with changed metadata but identical video SHA; only in this browser.
   first=data['clips'][0];model=data['models'][0];out=first['outputs'][model['id']]
   await page.evaluate('''({clip,model,out})=>{window.auditPlayer=document.querySelector('.pose-player video');const changed=structuredClone(out);changed.metadata.sampling.audio_temperature=0.37;changed.metadata.sampling.pose_top_k=17;fill(clip,model,changed)}''',{'clip':first,'model':model,'out':out})
   panel=page.locator('.sampling-panel').first
   assert await panel.locator('tr[data-stream="audio"] td[data-setting="temp"]').inner_text()=='0.37'
   assert await panel.locator('tr[data-stream="pose"] td[data-setting="topk"]').inner_text()=='17'
   assert await page.evaluate("window.auditPlayer===document.querySelector('.pose-player video')")
   report['pages'].append({'route':route,'panels_verified':count,'actual_metadata_values_match':True,'forced_native_text_correct':True,'residual_and_seeds_verified':True,'mobile_390px_no_overflow':True,'same_video_sha_settings_refresh_preserves_player':True})
   await page.set_viewport_size({'width':1440,'height':1100})
  r=await ctx.request.get(BASE+'/api/requests');assert r.status==200
  jobs=(await r.json())['requests'];done=[j for j in jobs if j['status']=='complete'];assert len(done)==14,len(done)
  await page.goto(BASE+'/',wait_until='domcontentloaded');await page.wait_for_function('document.querySelectorAll("#jobs .jobparams").length===14')
  cards=await page.locator('#jobs .jobparams').all_inner_texts();assert len(cards)==14
  for text in cards:
   for label in ['Audio temperature','Audio top-k','Pose temperature','Pose top-k','Generation seed','Video seed','Text mode']:assert label in text,(label,text)
  await page.set_viewport_size({'width':390,'height':844});assert await page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
  report['pages'].append({'route':'/','request_cards':14,'existing_sampling_controls_present':True,'mobile_390px_no_overflow':True})
  assert not errors,errors
  report.update(status='pass',javascript_errors=errors)
  (ROOT/'live-browser-audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
  await browser.close()
asyncio.run(main())
