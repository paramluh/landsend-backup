#!/usr/bin/env python3
"""Authorize this VM's external IP on the tiktok_pipeline Cloud SQL instance.

Uses a GCP service account JSON to call the Cloud SQL Admin API directly
(no gcloud required), and idempotently appends this VM's IP/32 to the
instance's authorizedNetworks. Run from setup.sh on a fresh VM, or any time
the VM's public IP changes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import jwt  # PyJWT (apt: python3-jwt)

DEFAULT_TARGET_IP = "34.142.78.212"
DEFAULT_NETWORK_NAME = "pose-tokenizer-vm"
IP_LOOKUP_URLS = (
    "https://ifconfig.me",
    "https://api.ipify.org",
    "https://ipinfo.io/ip",
)


def detect_vm_ip() -> str:
    for url in IP_LOOKUP_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=10) as r:
                ip = r.read().decode().strip()
            if ip and all(c.isdigit() or c == "." for c in ip):
                return ip
        except Exception:
            continue
    raise SystemExit("Could not determine this VM's external IP")


def get_access_token(sa: dict) -> str:
    now = int(time.time())
    assertion = jwt.encode(
        {
            "iss": sa["client_email"],
            "scope": "https://www.googleapis.com/auth/sqlservice.admin",
            "aud": sa["token_uri"],
            "iat": now,
            "exp": now + 3600,
        },
        sa["private_key"],
        algorithm="RS256",
        headers={"kid": sa["private_key_id"]},
    )
    body = urllib.parse.urlencode(
        {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": assertion,
        }
    ).encode()
    req = urllib.request.Request(
        sa["token_uri"],
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["access_token"]


def api(method: str, url: str, token: str, body: dict | None = None) -> dict:
    data = None
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = r.read()
            return json.loads(payload) if payload else {}
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")
        raise SystemExit(f"HTTP {e.code} on {method} {url}\n{msg}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--service-account",
        type=Path,
        default=repo_root / "service.json",
        help="Path to GCP service account JSON (default: <repo>/service.json)",
    )
    ap.add_argument(
        "--target-instance-ip",
        default=DEFAULT_TARGET_IP,
        help=f"Public IP of the Cloud SQL instance (default: {DEFAULT_TARGET_IP})",
    )
    ap.add_argument(
        "--vm-ip",
        default=None,
        help="Override VM external IP (default: auto-detected via ifconfig.me)",
    )
    ap.add_argument(
        "--name",
        default=DEFAULT_NETWORK_NAME,
        help=f"Display name for the authorized network entry (default: {DEFAULT_NETWORK_NAME})",
    )
    args = ap.parse_args()

    if not args.service_account.exists():
        raise SystemExit(f"Service account file not found: {args.service_account}")

    with open(args.service_account) as f:
        sa = json.load(f)
    project = sa["project_id"]

    vm_ip = args.vm_ip or detect_vm_ip()
    cidr = f"{vm_ip}/32"
    print(f"VM IP:      {vm_ip}", file=sys.stderr)
    print(f"Project:    {project}", file=sys.stderr)

    token = get_access_token(sa)

    instances = api(
        "GET",
        f"https://sqladmin.googleapis.com/v1/projects/{project}/instances",
        token,
    ).get("items", [])

    target = next(
        (
            inst
            for inst in instances
            if any(
                a.get("ipAddress") == args.target_instance_ip
                for a in (inst.get("ipAddresses") or [])
            )
        ),
        None,
    )
    if target is None:
        raise SystemExit(
            f"No Cloud SQL instance with public IP {args.target_instance_ip} in {project}"
        )

    name = target["name"]
    print(f"Instance:   {name}", file=sys.stderr)

    ip_cfg = (target["settings"].get("ipConfiguration") or {})
    authorized = list(ip_cfg.get("authorizedNetworks") or [])

    if any(n.get("value") == cidr for n in authorized):
        print(f"Already authorized: {cidr} - nothing to do.", file=sys.stderr)
        return

    authorized.append({"name": args.name, "value": cidr})
    op = api(
        "PATCH",
        f"https://sqladmin.googleapis.com/v1/projects/{project}/instances/{name}",
        token,
        {"settings": {"ipConfiguration": {"authorizedNetworks": authorized}}},
    )
    op_name = op.get("name")
    print(f"Patching {name} (op {op_name})...", file=sys.stderr)

    start = time.time()
    while True:
        status = api(
            "GET",
            f"https://sqladmin.googleapis.com/v1/projects/{project}/operations/{op_name}",
            token,
        )
        st = status.get("status")
        if st == "DONE":
            err = status.get("error")
            if err:
                raise SystemExit(f"Operation failed: {err}")
            print(f"Authorized {cidr} on {name}", file=sys.stderr)
            return
        if time.time() - start > 300:
            raise SystemExit("Timed out waiting for Cloud SQL operation")
        print(f"  status={st}, sleeping 5s...", file=sys.stderr)
        time.sleep(5)


if __name__ == "__main__":
    main()
