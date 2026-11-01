"""patch_theme.py <in> <out>: put the shared klweb-theme snippet (theme_snip.html) into one live-curve template or page, right after its first </title>
(else at the very top). Idempotent: an existing klweb-theme block is replaced with the current snippet. Changes nothing else."""
import sys, re, os
src, dst = sys.argv[1:3]
snip = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "theme_snip.html")).read().rstrip("\n") + "\n"
t = open(src).read()
t = re.sub(r"<!-- klweb-theme .*?<!-- /klweb-theme -->\n?", "", t, flags=re.S)
i = t.find("</title>")
t = t[:i + 8] + "\n" + snip + t[i + 8:].lstrip("\n") if i >= 0 else snip + t
open(dst + ".tmp", "w").write(t); os.replace(dst + ".tmp", dst); print("theme snippet in", dst)
