# 10-26: version badge + changelog panel check (desktop + 375 px): page errors, badge text, panel rows / entries, horizontal scroll
import asyncio, sys
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        for w, h in ((1280, 800), (375, 740)):
            pg = await b.new_page(viewport={"width": w, "height": h}); errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
            await pg.goto(sys.argv[1]); await pg.wait_for_function("!document.getElementById('vbadge').hidden", timeout=30000)
            badge = await pg.evaluate("document.getElementById('vbadge').textContent")
            await pg.click("#vbadge"); await pg.wait_for_timeout(1500)
            r = await pg.evaluate("""({rows:[...document.querySelectorAll('#clrun dt')].map(e=>e.textContent), entries:document.querySelectorAll('#cllog li').length,
              open:!document.getElementById('clpanel').hidden, sw:document.documentElement.scrollWidth, cw:document.documentElement.clientWidth,
              pr:Math.round(document.getElementById('clpanel').getBoundingClientRect().right), first:(document.querySelector('#cllog li')||{}).textContent})""")
            if len(sys.argv) > 2: await pg.screenshot(path=f"{sys.argv[2]}-{w}.png")
            await pg.keyboard.press("Escape"); await pg.wait_for_timeout(300)
            closed = await pg.evaluate("document.getElementById('clpanel').hidden")
            print(w, "badge", badge, "| open", r["open"], "rows", r["rows"], "entries", r["entries"], "| scrollW", r["sw"], "clientW", r["cw"], "panelR", r["pr"], "| esc closes", closed, "| errors", errs[:2])
            print("   first:", (r["first"] or "")[:150])
            await pg.close()
        await b.close()
asyncio.run(main())
