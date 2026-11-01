"""Keep saved actor choices consistent with the assets hosted on this node."""
import json
import os
from pathlib import Path
from urllib.parse import parse_qsl, urlencode


def _read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def catalog():
    root = Path(os.environ.get("GENWEB_ROOT", Path(__file__).parent))
    people = Path(os.environ.get("GENWEB_PEOPLE", root.parent / "prompts"))
    entries = _read(root / "OPTIONS.json").get("persons", [])
    available = {
        p["id"]: p.get("label", p["id"])
        for p in entries if isinstance(p, dict) and isinstance(p.get("id"), str)
        and "/" not in p["id"] and p["id"] not in (".", "..")
        and all((people / p["id"] / "gt" / name).is_file()
                for name in ("pose_tokens.safetensors", "mimi_local.safetensors"))
    }
    default = _read(root / "MONO_CK.json").get("run_defaults", {}).get("prefill_person")
    if default not in available:
        default = next(iter(available), None)
    return available, default


def normalize_query(query):
    """Repair only an unavailable selection; preserve every sampling setting."""
    pairs = parse_qsl(query, keep_blank_values=True)
    values = dict(pairs)
    if values.get("model", "monoplex") != "monoplex":
        return query
    available, default = catalog()
    old = values.get("prefill_person")
    if not old or not default or old in available:
        return query
    return urlencode([
        (key, default if key == "prefill_person" else "same" if key == "voice" and value == old else value)
        for key, value in pairs
    ])


def request_error(body):
    """Reject stale open tabs before creating a run or claiming GPU work."""
    if body.get("model") == "duplex":
        return None
    person = (body.get("prefill") or {}).get("person")
    available, default = catalog()
    if person and person not in available:
        return (f"Actor {person} is not available on this site. Reload the page "
                f"and choose an actor with a face preview (default: {available.get(default, 'Held-out clip 02')}).")
    return None
