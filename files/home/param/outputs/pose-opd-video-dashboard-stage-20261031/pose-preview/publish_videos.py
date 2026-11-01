#!/usr/bin/env python3
"""Publish completed video sidecars immediately, under the dashboard writer lock."""
import argparse
import json
from make_demo import refresh_generated_videos

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--samples-dir',required=True)
args=parser.parse_args()
print(json.dumps(refresh_generated_videos(args.samples_dir)),flush=True)
