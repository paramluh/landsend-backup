import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

URL = 'https://morris-rack-republic-appear.trycloudflare.com'
OUT = Path(__file__).parent
BID = 'B20261031-002002-4952'
RIDS = ['20261031-002003-1ef8', '20261031-002003-f5a6']


async def state(page, selector='#bmvid'):
    return await page.locator(selector).evaluate('''v => {
      const c=[...v.parentNode.querySelectorAll('canvas.pose-overlay-canvas')].find(c=>c.dataset.rid===v.dataset.rid)||v.parentNode.querySelector('canvas.pose-overlay-canvas');
      const r=v.getBoundingClientRect(), cr=c?.getBoundingClientRect();
      let painted=0;
      if(c?.width&&c?.height){const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;for(let i=3;i<d.length;i+=4)if(d[i])painted++;}
      return {rid:v.dataset.rid,src:v.currentSrc,time:v.currentTime,duration:v.duration,paused:v.paused,rate:v.playbackRate,videoWidth:v.videoWidth,videoHeight:v.videoHeight,
        frame:Number(c?.dataset.frame),canvasRid:c?.dataset.rid,canvasHidden:c?.hidden,painted,
        videoRect:{x:r.x,y:r.y,width:r.width,height:r.height},canvasRect:cr?{x:cr.x,y:cr.y,width:cr.width,height:cr.height}:null};
    }''')


async def wait_video(page, rid):
    await page.wait_for_function('''rid=>{const v=document.getElementById('bmvid');return v?.dataset.rid===rid&&v.readyState>=2&&v.videoWidth>0;}''',arg=rid,timeout=30000)


async def wait_pose(page, rid, selector='#bmvid'):
    await page.wait_for_function('''([selector,rid])=>{const v=document.querySelector(selector);const c=v?.parentNode.querySelector('canvas.pose-overlay-canvas');return c&&!c.hidden&&c.dataset.rid===rid&&c.dataset.ready==='true';}''',arg=[selector,rid],timeout=45000)


async def paused_seek(page, target, rid, selector='#bmvid'):
    await page.locator(selector).evaluate('(v,t)=>{v.pause();v.currentTime=t;}',target)
    await page.wait_for_function('''([selector,target,rid])=>{const v=document.querySelector(selector),c=v?.parentNode.querySelector('canvas.pose-overlay-canvas');return v?.paused&&!v.seeking&&Math.abs(v.currentTime-target)<.02&&c?.dataset.rid===rid&&Math.abs(Number(c.dataset.frame)-Math.floor(target*25))<=1;}''',arg=[selector,target,rid],timeout=20000)
    s=await state(page,selector)
    assert s['painted']>50,s
    return s


async def check(browser,mobile=False):
    mode='mobile' if mobile else 'desktop'
    page=await browser.new_page(viewport={'width':390,'height':844} if mobile else {'width':1440,'height':1100},device_scale_factor=1)
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    await page.goto(URL+'/?model=monoplex&theme=dark',wait_until='domcontentloaded',timeout=60000)
    await page.wait_for_selector('#k-prefill_person',state='attached')
    await page.evaluate('localStorage.removeItem(WSK("gw-pose"));PoseOverlay.setEnabled(false);batchOpen('+json.dumps(BID)+')')
    await page.wait_for_selector('#bgrid .bcard')
    before=await page.evaluate('({tab:document.querySelector(".mtabs button.on")?.dataset.m,selected:SEL})')
    assert before['tab']=='batch',before
    await page.locator(f'#bgrid .bcard[data-id="{RIDS[0]}"] .bth').click()
    await page.locator('#bmodal').wait_for(state='visible')
    await wait_video(page,RIDS[0])
    assert await page.locator('#pbatch').is_visible()
    assert await page.locator('#bmprev').is_disabled()
    assert not await page.locator('#bmnext').is_disabled()
    await page.locator('#bmpose').click()
    await wait_pose(page,RIDS[0])
    first=await paused_seek(page,3.2,RIDS[0])
    # The canvas must occupy the video box, not a separate side-by-side panel.
    for k in ('x','y','width','height'):
        assert abs(first['canvasRect'][k]-first['videoRect'][k])<2,(k,first)
    response=await page.request.get(URL+f'/api/run/{RIDS[0]}/pose')
    data=await response.json()
    assert response.status==200 and data['fps']==25 and data['frame_count']==len(data['frames'])
    assert abs(first['duration']-data['frame_count']/data['fps'])<.05,(first,data['frame_count'])
    assert all(len(frame)==55 for frame in data['frames'])
    await page.locator('#bmodal .bmb').evaluate('e=>e.scrollTop=0')
    bounds=await page.locator('#bmodal .bmb').evaluate('e=>({width:e.getBoundingClientRect().width,scroll:e.scrollWidth,client:e.clientWidth,viewport:innerWidth})')
    assert bounds['width']<=bounds['viewport'] and bounds['scroll']<=bounds['client']+1,bounds
    await page.screenshot(path=str(OUT/f'{mode}-overlay-modal.png'),full_page=True)
    await page.locator('#bmvid').evaluate('v=>{v.muted=true;v.playbackRate=1.5;return v.play();}')
    await page.wait_for_timeout(900)
    await page.locator('#bmvid').evaluate('v=>v.pause()')
    await page.wait_for_timeout(100)
    playback=await state(page)
    assert playback['time']>4 and abs(playback['frame']-int(playback['time']*25))<=1,playback
    await page.locator('#bmnext').click()
    await wait_video(page,RIDS[1])
    await wait_pose(page,RIDS[1])
    second=await paused_seek(page,1.2,RIDS[1])
    assert await page.locator('#bmnext').is_disabled()
    assert not await page.locator('#bmprev').is_disabled()
    # Browser-native video controls retain their own arrow-key seeking behavior.
    await page.locator('#bmx').focus()
    await page.keyboard.press('ArrowLeft')
    await wait_video(page,RIDS[0])
    await page.keyboard.press('Escape')
    await page.locator('#bmodal').wait_for(state='hidden')
    after=await page.evaluate('({tab:document.querySelector(".mtabs button.on")?.dataset.m,selected:SEL,overflow:document.body.style.overflow})')
    assert after['tab']==before['tab'] and after['selected']==before['selected'] and after['overflow']!='hidden',(before,after)
    # The text/seed portion of a card has the same modal behavior as its thumbnail.
    await page.locator(f'#bgrid .bcard[data-id="{RIDS[1]}"] .btx').click()
    await wait_video(page,RIDS[1])
    await page.locator('#bmx').click()
    # Main output still supports a real superimposed overlay.
    await page.evaluate('showMain("out");pickRun('+json.dumps(RIDS[0])+')')
    await page.wait_for_function('''()=>document.querySelector('#dav video')?.readyState>=2''')
    await wait_pose(page,RIDS[0],selector='#dav video')
    main=await paused_seek(page,2.4,RIDS[0],selector='#dav video')
    await page.screenshot(path=str(OUT/f'{mode}-overlay-output.png'),full_page=True)
    # Opening the output reel's sibling must retain the Output tab and batch context.
    await page.wait_for_selector(f'#bstiles .btile[data-id="{RIDS[1]}"]')
    await page.locator(f'#bstiles .btile[data-id="{RIDS[1]}"]').click()
    await wait_video(page,RIDS[1])
    assert 'Sample 2 of 2' in await page.locator('#bmtitle').inner_text()
    assert await page.evaluate('document.querySelector(".mtabs button.on")?.dataset.m')=='out'
    await page.locator('#bmx').click()
    assert not errors,errors
    result={'mode':mode,'modal_first':first,'modal_second':second,'playback':playback,'main':main,'batch_context_before':before,'batch_context_after':after,'api':{'fps':data['fps'],'frames':data['frame_count'],'width':data['width'],'height':data['height']},'js_errors':errors}
    (OUT/f'{mode}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'mode':mode,'passed':True,'frames':data['frame_count'],'js_errors':errors}),flush=True)
    await page.close()


async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        await check(b)
        await check(b,True)
        await b.close()


if __name__=='__main__':asyncio.run(main())
