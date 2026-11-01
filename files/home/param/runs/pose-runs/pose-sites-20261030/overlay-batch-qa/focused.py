import asyncio
import json

from playwright.async_api import async_playwright
from check import URL, OUT, BID, RIDS, state, wait_video, wait_pose, paused_seek


async def check(browser, mobile=False):
    mode = 'mobile' if mobile else 'desktop'
    page = await browser.new_page(viewport={'width': 390, 'height': 844} if mobile else {'width': 1440, 'height': 1100}, device_scale_factor=1)
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    await page.goto(URL+'/?model=monoplex&theme=dark', wait_until='domcontentloaded', timeout=60000)
    await page.wait_for_function('typeof batchOpen === "function" && typeof PoseOverlay === "object"')
    await page.evaluate('(bid)=>{PoseOverlay.setEnabled(true);batchOpen(bid)}', BID)
    await page.locator(f'#bgrid .bcard[data-id="{RIDS[0]}"] .bth').click()
    await wait_video(page, RIDS[0])
    await wait_pose(page, RIDS[0])
    await paused_seek(page, 3.2, RIDS[0])
    await page.locator('#bmodal .bmb').evaluate('e=>e.scrollTop=0')
    nav = await page.locator('#bmnext').evaluate('e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,bottom:r.bottom,width:r.width,height:r.height,viewport:innerHeight}}')
    assert nav['y'] >= 0 and nav['bottom'] < nav['viewport'], nav
    await page.screenshot(path=str(OUT/f'{mode}-focused-modal.png'), full_page=False)
    await page.locator('#bmodal .spdseg button[data-s="1.5"]').click()
    await page.locator('#bmvid').evaluate('v=>{v.muted=true;return v.play()}')
    await page.wait_for_timeout(900)
    await page.locator('#bmvid').evaluate('v=>v.pause()')
    await page.wait_for_timeout(150)
    playback = await state(page)
    assert playback['rate'] == 1.5 and playback['time'] > 4 and abs(playback['frame']-int(playback['time']*25)) <= 1, playback
    await page.locator('#bmfullscreen').click()
    await page.wait_for_function('document.fullscreenElement?.contains(document.getElementById("bmvid"))')
    await page.wait_for_timeout(250)
    fullscreen = await paused_seek(page, 5.12, RIDS[0])
    has_both = await page.evaluate('document.fullscreenElement.contains(document.querySelector("#bmvid"))&&document.fullscreenElement.contains(document.querySelector("#bmvid").parentNode.querySelector("canvas.pose-overlay-canvas"))')
    assert has_both and not fullscreen['canvasHidden'], fullscreen
    for key in ('x', 'y', 'width', 'height'):
        assert abs(fullscreen['canvasRect'][key]-fullscreen['videoRect'][key]) < 2, (key, fullscreen)
    await page.screenshot(path=str(OUT/f'{mode}-fullscreen-overlay.png'), full_page=False)
    await page.locator('.pose-fullscreen-exit:visible').click()
    await page.wait_for_function('!document.fullscreenElement')
    await page.locator('#bmnext').click()
    await wait_video(page, RIDS[1])
    await wait_pose(page, RIDS[1])
    assert 'Sample 2 of 2' in await page.locator('#bmtitle').inner_text()
    await page.locator('#bmx').click()
    assert await page.evaluate('document.querySelector(".mtabs button.on")?.dataset.m') == 'batch'
    await page.locator('#mt-hist').click()
    history = page.locator(f'#phist .hrow[data-id="{BID}"] .hth')
    await history.wait_for(state='visible')
    history_rid = await history.evaluate('e=>e.closest(".hrow")._x.first')
    await history.click()
    await wait_video(page, history_rid)
    assert 'of 2' in await page.locator('#bmtitle').inner_text()
    assert await page.evaluate('document.querySelector(".mtabs button.on")?.dataset.m') == 'hist'
    await page.locator('#bmx').focus()
    await page.keyboard.press('Escape')
    assert await page.locator('#phist').is_visible()
    await page.evaluate('(rid)=>{showMain("out");pickRun(rid)}', RIDS[0])
    await page.wait_for_function('document.querySelector("#dav video")?.readyState>=2')
    await wait_pose(page, RIDS[0], '#dav video')
    await paused_seek(page, 2.4, RIDS[0], '#dav video')
    await page.locator('#left .plmore>button').click()
    await page.locator('#posefullscreen').click()
    await page.wait_for_function('document.fullscreenElement?.contains(document.querySelector("#dav video"))')
    await page.wait_for_timeout(250)
    main_fullscreen = await paused_seek(page, 4.8, RIDS[0], '#dav video')
    assert not main_fullscreen['canvasHidden'] and main_fullscreen['painted'] > 50, main_fullscreen
    for key in ('x', 'y', 'width', 'height'):
        assert abs(main_fullscreen['canvasRect'][key]-main_fullscreen['videoRect'][key]) < 2, (key, main_fullscreen)
    await page.screenshot(path=str(OUT/f'{mode}-main-fullscreen-overlay.png'), full_page=False)
    await page.locator('.pose-fullscreen-exit:visible').click()
    await page.wait_for_function('!document.fullscreenElement')
    extra = {}
    assert not errors, errors
    result = {'mode': mode, 'nav': nav, 'playback_1_5x': playback, 'fullscreen': fullscreen, 'main_fullscreen': main_fullscreen, 'history_rid': history_rid, 'other_models': extra, 'js_errors': errors}
    (OUT/f'{mode}-focused.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'mode': mode, 'passed': True, 'rate': playback['rate'], 'history': history_rid, 'other_models': list(extra), 'js_errors': errors}), flush=True)
    await page.close()


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=['--no-sandbox', '--disable-dev-shm-usage'])
        await check(browser)
        await check(browser, True)
        await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
