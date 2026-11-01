"""headless check of the live page through the public tunnel: no JS errors, the 14B series present in the main chart, header text.
usage: LD_LIBRARY_PATH=/data/home/amartyavarma/.local/chromelibs/root/usr/lib/x86_64-linux-gnu venvs/pagecheck/bin/python -I pagecheck.py URL OUT.png"""
import asyncio
import json
import sys

from playwright.async_api import async_playwright

URL, OUT = sys.argv[1], sys.argv[2]


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        res = {}
        for name, vp in (("desktop", {"width": 1400, "height": 1100}), ("phone", {"width": 390, "height": 844})):
            pg = await b.new_page(viewport=vp)
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.on("console", lambda m: errs.append("console." + m.type + ": " + m.text) if m.type == "error" else None)
            await pg.goto(URL, wait_until="networkidle")
            await pg.wait_for_timeout(3000)
            ds = await pg.evaluate("(()=>{const c=Chart.getChart('nword');const n=Object.keys(Chart.instances).length;return c?{charts:n,legend:c.legend.legendItems.map(i=>i.text),series:c.data.datasets.map(d=>[d.label,d.data.length,!!d._raw,!!d.hidden])}:null})()")
            hdr = await pg.evaluate("document.getElementById('hdr').innerText + ' || ' + document.getElementById('sub').innerText")
            sw = await pg.evaluate("document.documentElement.scrollWidth")
            res[name] = {"errors": errs, "nword_datasets": ds, "hdr": hdr[:900], "scrollWidth": sw}
            if name == "desktop":
                await pg.screenshot(path=OUT, full_page=False)
        await b.close()
    print(json.dumps(res, indent=1))


asyncio.run(main())
