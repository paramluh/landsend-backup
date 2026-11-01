#!/usr/bin/env python3
"""Move a local artifact without copying data, retaining its former pathname."""
import argparse
import datetime
import json
import os
from pathlib import Path


def move(source, destination, journal):
    source, destination, journal = map(Path, (source, destination, journal))
    if source.is_symlink() and source.resolve() == destination.resolve() and destination.exists():
        return {"source": str(source), "destination": str(destination), "state": "already_moved"}
    if not source.exists() or source.is_symlink():
        raise ValueError(f"Expected an existing, physical source: {source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    if before.st_dev != destination.parent.stat().st_dev:
        raise ValueError("This helper requires a rename on the same filesystem")
    entry = {"source": str(source), "destination": str(destination),
             "device": before.st_dev, "inode": before.st_ino,
             "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "state": "planned"}
    journal.parent.mkdir(parents=True, exist_ok=True)
    with journal.open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    source.rename(destination)
    try:
        source.symlink_to(destination, target_is_directory=destination.is_dir())
    except BaseException:
        destination.rename(source)
        raise
    after = destination.stat()
    assert (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
    assert source.samefile(destination)
    entry["state"] = "complete"
    with journal.open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--journal", default="/home/param/outputs/organization-20261029/moves.jsonl")
    args = parser.parse_args()
    print(json.dumps(move(args.source, args.destination, args.journal)))
