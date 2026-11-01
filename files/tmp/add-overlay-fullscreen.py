from pathlib import Path
p=Path('/home/param/websites/pose-playground/index.html');s=p.read_text()
nav='<div class="bnav"><button type="button" class="sm" id="bmprev">‹ Previous</button><button type="button" class="sm" id="bmnext">Next ›</button></div>'
assert s.count(nav)==1
s=s.replace(nav,'').replace('<h3 id="bmtitle"></h3>','<h3 id="bmtitle"></h3>'+nav)
s=s.replace("id=\"bmpose\" aria-pressed=\"false\">Pose overlay</button><button", "id=\"bmpose\" aria-pressed=\"false\">Pose overlay</button><button type=\"button\" class=\"sm\" id=\"bmfullscreen\">Fullscreen</button><button")
s=s.replace("    if(window.PoseOverlay) PoseOverlay.attach(v,c.run_id,g('bmpose'));", "    if(window.PoseOverlay){ PoseOverlay.attach(v,c.run_id,g('bmpose')); const fs=g('bmfullscreen'); if(fs){ fs.hidden=!document.fullscreenEnabled; fs.disabled=!c.src; fs.title='Fullscreen video with its pose overlay'; fs.onclick=()=>PoseOverlay.fullscreen(v); } }")
s=s.replace("if(e.key==='Escape'){ e.preventDefault(); closeModal(); return; }", "if(e.key==='Escape'){ if(document.fullscreenElement) return; e.preventDefault(); closeModal(); return; }")
s=s.replace("function closeModal(){ ++MLOAD; clearTimeout(MPOLL);", "function closeModal(){ ++MLOAD; clearTimeout(MPOLL); if(document.fullscreenElement&&g('bmodal').contains(document.fullscreenElement)) document.exitFullscreen().catch(()=>{});")
s=s.replace(".pose-overlay-fullscreen{position:relative;background:#000;display:flex;align-items:center;justify-content:center}\n.pose-overlay-fullscreen video{max-height:100vh!important;max-width:100vw!important;object-fit:contain!important}", '''.pose-overlay-parent:fullscreen{width:100vw!important;height:100vh!important;max-width:none!important;max-height:none!important;aspect-ratio:auto!important;display:flex!important;align-items:center;justify-content:center;background:#000;overflow:hidden;border-radius:0}
.pose-overlay-parent:fullscreen video{position:static!important;inset:auto!important;width:100%!important;height:100%!important;max-width:none!important;max-height:none!important;aspect-ratio:auto!important;object-fit:contain!important;border-radius:0}
.pose-fullscreen-exit{display:none;position:absolute;top:16px;right:16px;z-index:4;background:rgba(0,0,0,.75);color:#fff;border:1px solid rgba(255,255,255,.45);border-radius:8px;padding:8px 12px;cursor:pointer}
.pose-overlay-parent:fullscreen>.pose-fullscreen-exit{display:block}
.plpop .pose-fullscreen{margin:0;min-height:32px;text-align:left;border:0;background:none;color:var(--ink2);padding:0 6px;font-size:12px;font-weight:600}''')
s=s.replace("    if(button){ s.buttons.add(button); button.onclick=()=>enabled(!ON); }", """    if(document.fullscreenEnabled){
      try{video.controlsList?.add('nofullscreen');}catch(e){}
      const host=video.parentNode; if(!host.querySelector(':scope > .pose-fullscreen-exit')){ const exit=document.createElement('button'); exit.type='button'; exit.className='pose-fullscreen-exit'; exit.textContent='Exit fullscreen'; exit.onclick=()=>document.exitFullscreen().catch(()=>{}); host.appendChild(exit); }
    }
    if(button){ s.buttons.add(button); button.onclick=()=>enabled(!ON); }""")
s=s.replace("  window.PoseOverlay={attach,setEnabled:enabled,isEnabled:()=>ON,draw:video=>{const s=byVideo.get(video);if(s)draw(s);}};", """  async function fullscreen(video){ const s=byVideo.get(video); if(!s||!video.parentNode.requestFullscreen) return;
    try{ await video.parentNode.requestFullscreen(); draw(s); }catch(e){ for(const b of s.buttons) b.title='Fullscreen is unavailable in this browser'; }
  }
  window.PoseOverlay={attach,fullscreen,setEnabled:enabled,isEnabled:()=>ON,draw:video=>{const s=byVideo.get(video);if(s)draw(s);}};
  document.addEventListener('fullscreenchange',()=>{ for(const s of states) draw(s); });""")
s=s.replace("if(rid&&video&&act){ let b=document.getElementById('posebtn'); if(!b){ b=document.createElement('button'); b.type='button'; b.className='sm posebtn'; b.id='posebtn'; act.prepend(b); } attach(video,rid,b); }", """if(rid&&video&&act){ let b=document.getElementById('posebtn'); if(!b){ b=document.createElement('button'); b.type='button'; b.className='sm posebtn'; b.id='posebtn'; act.prepend(b); } attach(video,rid,b);
      const menu=document.querySelector('#left .plpop'); if(menu&&document.fullscreenEnabled){ let fs=document.getElementById('posefullscreen'); if(!fs){ fs=document.createElement('button'); fs.type='button'; fs.id='posefullscreen'; fs.className='pose-fullscreen'; fs.textContent='Fullscreen'; fs.title='Fullscreen video with its pose overlay'; fs.setAttribute('role','menuitem'); menu.appendChild(fs); }
        fs.disabled=!video.getAttribute('src'); fs.onclick=()=>{ menu.hidden=true; const more=menu.closest('.plmore'); more?.classList.remove('open'); more?.querySelector('button')?.setAttribute('aria-expanded','false'); fullscreen(video); };
      }
    }""")
p.write_text(s)
print('Fullscreen containers and mobile navigation updated')
