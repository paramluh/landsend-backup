# Secret scan (before upload, 2026-10-25; re-run after the self-contained update)

**Result: clean. Nothing was found, so nothing was stripped or replaced.** This report contains no secret values.

## What was scanned

- **The staging dir.** 139 regular files: `README.md`, `QUICKSTART.md`, `aloe.env`, `inference_defaults.json`, `SECRET_SCAN.md`,
  `env/*` (including `ASSETS_PINNED.json` and `FA3_INSTALL.txt`), `assets/presets/*`, `assets/quiet_bank_m2.npz`,
  `weights/vgain_theta_fp32.json`, `weights/SHA256SUMS` and all of `code/`.
- **The Voxtral files shipped in `assets/voxtral-realtime/`.** These are linked from the cluster copy and were scanned there.
- **The small files of the weights export.** `train_cfg.json`, `COMPLETE` and `.metadata`.

## Pattern rules

The scan is regex-based and reports file, line, rule and a masked snippet only. It looks for:

- HF tokens (`hf_...`), and `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` or `HUGGINGFACE_TOKEN` assignments with values.
- `token="..."` keyword arguments.
- `BEGIN ... PRIVATE KEY` blocks.
- GCP service-account JSON (`"type": "service_account"`, `private_key`, `gserviceaccount.com`) and GCP API keys.
- AWS keys.
- Cloudflared tunnel credentials or tokens (`TunnelSecret`, `AccountTag`, `--token eyJ...`, `cert.pem`) and JWT-like strings.
- OpenAI, Anthropic, GitHub, Slack and W&B keys.
- `password=`, `secret=`, `api_key=` and `auth_token=` assignments.
- Bearer and `Authorization:` values.
- URLs with embedded `user:pass@`.
- Credential-like file names (`*cred*`, `*secret*`, `*token*`, `*.pem`, `*.key`, service-account or cloudflared `*.json`).

## Literal-value check

Every staged file, and all 67 export files, was also searched byte for byte for the actual values of the cluster's known secrets:
the demo backend token file and the local HF token files. Only counts were printed. There were 0 hits in both scans.

## Findings

| finding | count | handling |
|---|---|---|
| HF tokens, keys, private keys, service accounts, cloudflared credentials, passwords, bearer tokens, credential URLs | 0 | nothing to strip |
| Literal values of the known local secrets | 0 | nothing to strip |
| References to a token *file path or argument*, with no value (`--token-file`, `ALOE_TOKEN_FILE` in `aloe.env` / `demo.sbatch` / `proxy.py` / `server.py` / `serve_release.sh` / README / QUICKSTART) | 30 | kept: the code reads the backend token from a user-supplied file at run time; no token is shipped |
| Credential-like file names | 2 | benign: `code/models/Qwen3-32B/tokenizer.json` and `tokenizer_config.json` (the Qwen tokenizer) |
| `env/pip_freeze.txt` URL or VCS installs that could embed credentials | 0 | all entries are plain `name==version` |

`serve_release.sh` creates a fresh random backend token at run time; none is shipped. Cloudflare quick tunnels need no credentials, so none exist to leak. The demo URL in the README carries no token, because the front
proxy injects the token upstream.
