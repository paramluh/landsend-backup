import sys
from huggingface_hub import get_token
if len(sys.argv) < 2 or sys.argv[1] != "get": raise SystemExit(0)
fields = dict(line.rstrip("\n").split("=", 1) for line in sys.stdin if "=" in line)
if fields.get("host") != "huggingface.co": raise SystemExit(0)
token = get_token()
if not token: raise SystemExit("No cached Hugging Face credential")
print("username=hf_user")
print("password=" + token)
