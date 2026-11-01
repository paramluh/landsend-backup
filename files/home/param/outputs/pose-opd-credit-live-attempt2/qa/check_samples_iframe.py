import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
out=Path('/home/param/outputs/pose-opd-credit-live-attempt2/qa')
url='https://mph-daniel-containers-afternoon.trycloudflare.com/'
report={'url':url,'viewport':{'width':1440,'height':900},'errors':[]}
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True)
 page=browser.new_page(viewport=report['viewport'],device_scale_factor=1)
 page.on('pageerror',lambda error:report['errors'].append(str(error)))
 page.goto(url,wait_until='domcontentloaded',timeout=60000)
 page.get_by_role('button',name='SAMPLES',exact=True).click()
 iframe=page.locator('iframe[src="samples/index.html"]')
 iframe.wait_for(timeout=30000)
 frame=iframe.content_frame
 frame.locator('.sample').wait_for(timeout=30000)
 frame.locator('canvas[data-ready="true"]').wait_for(timeout=60000)
 iframe.scroll_into_view_if_needed()
 report['iframe_box']=iframe.bounding_box()
 report['main_overflow']=page.evaluate('document.documentElement.scrollWidth>innerWidth')
 report['sample_count']=frame.locator('.sample').count()
 report['selectors']={selector:frame.locator(selector).bounding_box() for selector in ['#rollout','#sample']}
 report['stage']=frame.locator('.pose-wrap').bounding_box()
 report['transport']=frame.locator('.transport').bounding_box()
 report['iframe_layout']=frame.locator('body').evaluate('e=>({width:innerWidth,height:innerHeight,bodyScrollHeight:document.body.scrollHeight,docScrollWidth:document.documentElement.scrollWidth,stage:(()=>{let r=document.querySelector(".pose-wrap").getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}})(),transport:(()=>{let r=document.querySelector(".transport").getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}})()})')
 assert report['sample_count']==1,report
 assert not report['main_overflow'],report
 assert report['iframe_layout']['docScrollWidth']<=report['iframe_layout']['width'],report
 stage=report['iframe_layout']['stage']; assert abs(stage['width']/stage['height']-8/15)<.001,report
 transport=report['iframe_layout']['transport']; assert transport['y']+transport['height']<report['iframe_layout']['height'],report
 assert frame.locator('#sample').input_value()=='0',report
 frame.locator('#sample').select_option('2')
 frame.locator('article[aria-label="Sample 3"] canvas[data-ready="true"]').wait_for(timeout=60000)
 report['selection_changes_work']=frame.locator('.sample').count()==1
 frame.locator('#sample').select_option('0')
 frame.locator('article[aria-label="Sample 1"] canvas[data-ready="true"]').wait_for(timeout=60000)
 page.screenshot(path=str(out/'samples-dashboard-iframe.png'),full_page=True)
 iframe.screenshot(path=str(out/'samples-dashboard-iframe-detail.png'))
 (out/'samples-dashboard-iframe.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
 browser.close()
