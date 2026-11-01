import json
from pathlib import Path
from urllib.parse import urlencode
from playwright.sync_api import sync_playwright
root=Path(__file__).parent
query='factorised_text=1&cat_temp=1&pad_bias=0&adaptive_pad_on=1&pad_silent_s=3&pad_bias_silent=-0.1&pad_bias_other=0&text_temp=1&text_topk=0&text_topp=1&minp=0.1&minp_mode=relative&freq_pen=0.3&freq_win=50&text_rep=1&text_rep_win=64&word_greedy=0&cb0_temp=1&cb0_topk=50&light_topk=100&light_temp=1&audio_rep=1&audio_rep_win=64&adaptive_rep_on=0&speak_window_frames=25&audio_rep_speaking=1&audio_rep_silent=1&text_rep_speaking=1&text_rep_silent=1&cfg_on=0&cfg_s=1.5&cfg_cb0=0&cfg_uncond_src=quiet_frames&a_gain=0&a_lufs_on=0&a_target_lufs=-23&a_slew_db_s=3&a_g0_db=0&lim_on=0&lim_ceiling=0.98&lim_attack_ms=2&lim_release_ms=50&seed=random&max_len_s=600&ngram_rep_on=0&ngram_rep_win_s=10&ngram_rep_min=3&ngram_rep_allow=1&ngram_rep_pen=2&ngram_rep_growth=2&ngram_rep_len=1.5&model=rosie&preset=rosie-09bb73ac&clip=preset:rosie-09bb73ac&persona=none&aec=1&ns=1&vclip=1&oboost=0&pbuf=120&cumax=1.08&ck_step=108&ck=rosie-full-cotrain-1030-1710%2Fexport-108'
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,executable_path='/home/param/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome',args=['--no-sandbox'])
 page=browser.new_page()
 page.route('**/history/**',lambda r:r.abort())
 page.route('**/clipwav**',lambda r:r.abort())
 page.goto('https://rack-depends-cadillac-nitrogen.trycloudflare.com/?'+query,wait_until='domcontentloaded',timeout=30000)
 page.wait_for_function("typeof P !== 'undefined' && P && typeof PERSIST !== 'undefined' && PERSIST.restored",timeout=30000)
 result=page.evaluate("({settings:startMsg(),model:P.ck,step:P.step,minpSpecs:KIT.specs().filter(s=>KIT.canon(s.key||s.id).startsWith('minp')),url:location.href})")
 (root/'browser-effective-settings.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
 browser.close()
