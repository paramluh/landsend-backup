from pathlib import Path
import json,time
from playwright.sync_api import sync_playwright

OUT=Path('/home/param/outputs/pose-opd-video-dashboard-stage-20261031/REAL-VIEWER.json')
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--disable-gpu'])
    page=browser.new_page(viewport={'width':1280,'height':1000})
    errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('https://mph-daniel-containers-afternoon.trycloudflare.com/samples/index.html',wait_until='domcontentloaded')
    page.wait_for_function("Array.from(document.querySelector('#rollout').options).some(o=>o.text==='Step 1')")
    page.locator('#rollout').select_option(label='Step 1')
    page.locator('#sample').select_option('0')
    page.wait_for_selector('.pose-wrap > video:not([hidden])',timeout=30000)
    assert page.locator('article.sample').count()==1
    assert page.locator('video').count()==1
    video=page.locator('.pose-wrap > video');wrap=page.locator('.pose-wrap')
    wb=wrap.bounding_box();vb=video.bounding_box()
    assert abs(wb['width']/wb['height']-8/15)<.001
    assert all(abs(wb[k]-vb[k])<1 for k in ('x','y','width','height'))
    assert video.evaluate('v=>v.muted')
    button=page.locator('.transport button');button.click()
    page.wait_for_function("document.querySelector('.pose-wrap video').currentTime>.3&&!document.querySelector('.pose-wrap video').paused",timeout=30000)
    button.click();paused=video.evaluate('v=>v.currentTime');page.wait_for_timeout(500)
    assert abs(video.evaluate('v=>v.currentTime')-paused)<.06
    button.click();page.wait_for_function('(t)=>document.querySelector(".pose-wrap video").currentTime>t+.25',arg=paused,timeout=15000)
    resumed=video.evaluate('v=>v.currentTime');button.click()
    result={'verified_unix':time.time(),'public_url':page.url,'sample':'Step1 / sample1','video_src':video.get_attribute('src'),
        'portrait_bounds':wb,'video_bounds':vb,'single_sample':True,'single_video':True,'video_muted':True,
        'audio_lane_controls':page.locator('.lane button').count(),'paused_at_seconds':paused,'resumed_at_seconds':resumed,'page_errors':errors}
    assert not errors,errors
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    browser.close()
