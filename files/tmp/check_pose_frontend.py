import asyncio,json,time
from pathlib import Path
from playwright.async_api import async_playwright
URL='https://morris-rack-republic-appear.trycloudflare.com'
OUT=Path('/home/param/runs/pose-runs/pose-sites-20261030/browser/capabilities'); OUT.mkdir(parents=True,exist_ok=True)
async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
  for name,query in [('mono','model=monoplex'),('mono_old_link','model=monoplex&vw=both&stream=1&stream_set_by_user=1'),('live','model=duplex'),('batch','model=duplex&dup_mode=duo')]:
   page=await browser.new_page(viewport={'width':1440,'height':1100}); errors=[]; consoles=[]
   page.on('pageerror',lambda e:errors.append(str(e))); page.on('console',lambda m:consoles.append(m.text) if m.type=='error' else None)
   await page.goto(URL+'/?'+query+'&ws=capabilitycheck'+str(int(time.time())),wait_until='domcontentloaded',timeout=60000)
   await page.wait_for_function("typeof SCHEMA!=='undefined'&&SCHEMA&&SCHEMA.length>0",timeout=30000)
   await page.wait_for_timeout(3500)
   state=await page.evaluate('''() => ({model:MODEL, step:ckOf(MODEL),preset:VALS.preset,defaultPreset:SCHEMA.find(r=>r.name==='preset')?.default,preset06Label:SCHEMA.find(r=>r.name==='preset')?.choice_labels?.['06_74807fe0'],capabilities:window.GENWEB_CAPS,renderer:VW_X,stream:effStream(),streamDisabled:document.getElementById('streamck').disabled,streamHint:document.getElementById('streamhint').textContent,renderButtons:[...document.querySelectorAll('.vwseg button')].map(b=>({value:b.dataset.v,disabled:b.disabled,title:b.title})),header:document.getElementById('vidchip').textContent,go:document.getElementById('go').innerText,goDisabled:document.getElementById('go').disabled,bodyWidth:document.body.scrollWidth,viewport:innerWidth})''')
   # The comparison widget should be disabled without calling its action or generating a run.
   cmp=await page.evaluate('''() => { const p=document.createElement('div');p.innerHTML='<div class="bact"><span class="grow"></span></div><div class="bgrid"></div>'; vwBatch('Bcapabilitycheck',{cards:[]},p);const b=p.querySelector('#vwcmpb');return {disabled:b.disabled,text:b.textContent,title:b.title}; }''')
   assert not errors,(name,errors)
   assert state['renderer'] is None and state['stream'] is False,(name,state)
   assert state['streamDisabled'] and cmp['disabled'],(name,state,cmp)
   assert all(b['disabled']==(b['value']!='param') for b in state['renderButtons']),(name,state)
   if name=='live':
    assert state['model']=='livedup' and state['defaultPreset']=='12_72cb45d3',(name,state)
    assert 'SAM' not in state['preset06Label'] and not state['goDisabled'],(name,state)
   await page.screenshot(path=str(OUT/(name+'.png')),full_page=True)
   (OUT/(name+'.json')).write_text(json.dumps({'state':state,'comparison':cmp,'errors':errors,'console_errors':consoles},indent=2))
   print(json.dumps({'name':name,'state':state,'comparison':cmp,'errors':errors,'console_errors':consoles}),flush=True)
   if name=='mono':
    await page.set_viewport_size({'width':390,'height':844}); await page.wait_for_timeout(500)
    await page.screenshot(path=str(OUT/'mono-mobile.png'),full_page=True)
    print('mobile',await page.evaluate('JSON.stringify({scroll:document.body.scrollWidth,width:innerWidth})'))
   await page.close()
  await browser.close()
asyncio.run(main())
