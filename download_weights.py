"""Download and verify the safetensors files in the AtlaHu-9B GitHub release."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
BLOCK = 4 * 1024 * 1024


def digest_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def destination_lock(directory):
    with (directory / ".atlahu-download.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def check_prefix(stream, parts):
    """Keep complete verified parts from an interrupted download."""
    stream.seek(0)
    offset = 0
    for index, part in enumerate(parts):
        remaining = part["bytes"]
        digest = hashlib.sha256()
        while remaining:
            block = stream.read(min(BLOCK, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
        if remaining or digest.hexdigest() != part["sha256"]:
            stream.seek(offset)
            stream.truncate()
            return index
        offset += part["bytes"]
    stream.truncate(offset)
    return len(parts)


def append_part(output, part, base_url, assets_dir):
    name = part["asset"]
    start = output.tell()
    for attempt in range(4):
        try:
            if assets_dir is None:
                request = Request(f"{base_url}/{name}", headers={"User-Agent": "AtlaHu-9B-downloader"})
                source = urlopen(request, timeout=90)
            else:
                source = (assets_dir / name).open("rb")
            size = 0
            digest = hashlib.sha256()
            with source:
                while block := source.read(BLOCK):
                    size += len(block)
                    if size > part["bytes"]:
                        raise ValueError(f"Oversized release asset: {name}")
                    output.write(block)
                    digest.update(block)
            if size != part["bytes"] or digest.hexdigest() != part["sha256"]:
                raise ValueError(f"Size or SHA-256 mismatch: {name}")
            output.flush()
            return
        except (OSError, EOFError, ValueError) as error:
            output.seek(start)
            output.truncate()
            if assets_dir is not None or attempt == 3:
                raise
            print(f"Retrying {name}: {error}", flush=True)
            time.sleep(2 ** attempt)


def restore(directory, name, spec, base_url, assets_dir):
    target = directory / name
    if target.is_file() and target.stat().st_size == spec["bytes"] and digest_file(target) == spec["sha256"]:
        print(f"Verified {name}", flush=True)
        return
    partial = target.with_name(target.name + ".partial")
    with partial.open("r+b" if partial.exists() else "w+b") as output:
        first = check_prefix(output, spec["parts"])
        for part in spec["parts"][first:]:
            print(f"Reading {part['asset']}", flush=True)
            append_part(output, part, base_url, assets_dir)
    if partial.stat().st_size != spec["bytes"] or digest_file(partial) != spec["sha256"]:
        raise ValueError(f"Reassembled file failed verification: {name}")
    os.replace(partial, target)
    print(f"Verified {name}", flush=True)


def main():
    manifest = json.loads((ROOT / "weights.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "AtlaHu")
    parser.add_argument("--assets-dir", type=Path, help="Read predownloaded release assets from this directory")
    parser.add_argument("--file", action="append", choices=sorted(manifest["files"]), help="Restore only this file (repeatable)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base_url = f"https://github.com/{manifest['repository']}/releases/download/{manifest['tag']}"
    with destination_lock(args.output):
        for name in args.file or manifest["files"]:
            restore(args.output, name, manifest["files"][name], base_url, args.assets_dir)


if __name__ == "__main__":
    main()
