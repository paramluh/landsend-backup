import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

URL = 'https://morris-rack-republic-appear.trycloudflare.com/?model=monoplex&theme=dark'
OUT = Path(__file__).parent


async def open_grid(page):
    await page.locator('.mtabs button[data-m="gen"]').click()
    await page.locator('#form .tabs button[data-t="actor"]').click()
    await page.locator('.agbtn').click()
    await page.locator('#aggridm').wait_for(state='visible')
    await page.wait_for_function('''() => {
      const vs=[...document.querySelectorAll('#aggridm .agt video')];
      return vs.length===8 && vs.every(v=>v.readyState>=2 && v.videoWidth>0);
    }''', timeout=45000)


async def check_page(browser, mobile=False):
    name = 'mobile' if mobile else 'desktop'
    page = await browser.new_page(viewport={'width':390,'height':844} if mobile else {'width':1440,'height':1100})
    errors, failed_media = [], []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.on('response', lambda r: failed_media.append({'url':r.url,'status':r.status}) if r.status>=400 and ('/render/' in r.url or '/api/schema' in r.url) else None)
    await page.goto(URL, wait_until='domcontentloaded', timeout=60000)
    await page.wait_for_selector('#k-prefill_person', state='attached')
    await page.locator('#mt-act').click()
    await page.wait_for_function('''() => {
      const imgs=[...document.querySelectorAll('#pkreel img')];
      return imgs.length===8 && imgs.every(i=>i.complete&&i.naturalWidth>0);
    }''', timeout=45000)
    thumbs = await page.locator('#pkreel img').evaluate_all('(imgs)=>imgs.map(i=>({src:i.src,width:i.naturalWidth,height:i.naturalHeight}))')
    await page.screenshot(path=str(OUT/f'{name}-actor-tab.png'), full_page=True)
    await open_grid(page)
    cards = await page.locator('#aggridm .agt').evaluate_all('''(els)=>els.map(e=>{
      const v=e.querySelector('video'),t=e.querySelector('b');
      return {actor:e.dataset.v,title:t.textContent,titleClipped:t.scrollWidth>t.clientWidth+1||t.scrollHeight>t.clientHeight+1,
        width:v.videoWidth,height:v.videoHeight,readyState:v.readyState,src:v.currentSrc,poster:v.poster};
    })''')
    assert len(cards)==8 and all(not c['titleClipped'] for c in cards), cards
    await page.screenshot(path=str(OUT/f'{name}-grid.png'), full_page=True)
    tile=page.locator('#aggridm .agt').first
    await tile.hover()
    await page.wait_for_timeout(400)
    hover = await tile.locator('video').evaluate('(v)=>({paused:v.paused,time:v.currentTime})')
    assert not hover['paused'], hover
    played = []
    actors = [c['actor'] for c in cards] if not mobile else [cards[0]['actor']]
    for actor in actors:
        await page.locator(f'#aggridm .agt[data-v="{actor}"]').click()
        await page.wait_for_function('''() => {const v=document.querySelector('#agplay video');return v&&v.readyState>=2&&v.videoWidth>0&&!v.paused&&v.currentTime>0;}''', timeout=45000)
        video=await page.locator('#agplay video').evaluate('(v)=>({src:v.currentSrc,width:v.videoWidth,height:v.videoHeight,duration:v.duration,controls:v.controls,muted:v.muted,audioBytes:v.webkitAudioDecodedByteCount,videoBytes:v.webkitVideoDecodedByteCount})')
        assert video['controls'] and not video['muted'] and video['audioBytes']>0, video
        await page.locator('#agplay .aguse').click()
        selected = await page.evaluate('''() => ({person:document.getElementById('k-prefill_person').value,voice:document.getElementById('k-voice').value})''')
        assert selected == {'person':actor,'voice':'same'}, selected
        played.append({'actor':actor,'video':video,'selected':selected})
        if actor != actors[-1]:
            await page.locator('.agbtn').click()
            await page.locator('#aggridm').wait_for(state='visible')
    result={'mode':name,'thumbnails':thumbs,'cards':cards,'hover':hover,'playback':played,'js_errors':errors,'failed_media':failed_media}
    (OUT/f'{name}.json').write_text(json.dumps(result,indent=2))
    assert not errors, errors
    assert not failed_media, failed_media
    await page.close()
    print(json.dumps({'mode':name,'thumbnails':len(thumbs),'cards':len(cards),'played':len(played),'js_errors':errors,'failed_media':failed_media}),flush=True)
    return result


async def main():
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
        await check_page(browser)
        await check_page(browser,mobile=True)
        await browser.close()


asyncio.run(main())
