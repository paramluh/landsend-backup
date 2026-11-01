# Redactions

Secret scan run on every file before the first commit. Patterns: HF tokens (`hf_...`), GitHub tokens (`ghp_`/`gho_`/`github_pat_`...),
`sk-...` keys, AWS keys, `token=` literals, `Authorization` / `Bearer` headers, `.netrc` / cookies, password / api-key assignments, and
public tunnel URLs (trycloudflare, ngrok, ...). Values are never written here.

Result: **no tokens, keys or credentials found.** The live demo proxies read their token from a token file at run time (`--token-file`);
no token file was copied. The only sensitive items were **live, unauthenticated demo tunnel URLs** (Cloudflare quick tunnels: anyone with
the link reaches the running demo), redacted as follows:

| file | what was there | replaced by |
|---|---|---|
| `infer/posedup/proxy.py` | hard-coded playground tunnel URL (`PLAYGROUND = "https://<host>.trycloudflare.com/?model=duplex"`) | `os.environ.get("PLAYGROUND_URL", "") + "/?model=duplex"` (env var reference) |
| `infer/playground/index.html` | hard-coded fallback tunnel URL (`const FALLBACK_URL='https://<host>.trycloudflare.com'`) | placeholder `'__PLAYGROUND_URL__'` (substitute at deploy) |
| `infer/playground/LIVEDUP_STATUS.md` (8), `infer/playground/DUPUI_PROMPT.md` (1), `infer/playground/LIVEDUP_PROMPT.md` (1), `infer/posedup/CHANGELOG.json` (1), `infer/posedup/posefix/PROMPT.md` (1), `pages/samples/xabi-ft-17641-x76l/ROLLOUT_K60_PROMPT_1023.md` (1) | tunnel URLs in notes / changelog | `${DEMO_URL}` |

Total: 15 URL occurrences in 8 files. The `MANIFEST.tsv` sha256 of these 8 files is of the redacted copy (it differs from the source).
Regex-only references to the tunnel domain (`grep -o 'https://[a-z0-9-]*\.trycloudflare\.com'` in the tunnel scripts) were kept: they
contain no URL. The scripts that write live URL files (`URL.txt`, `URL`, `ALT_URLS.txt`) still do so at run time; those files were not
copied.
