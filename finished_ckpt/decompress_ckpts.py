#!/usr/bin/env python3
"""Restore every split checkpoint under this folder back into its .pt file.

Finds each `foo.pt.manifest.json` (recursively), reassembles + decompresses
`foo.pt.gz.part*` into `foo.pt`, verifies the sha256, then deletes the parts
and the manifest. Inverse of `compress_ckpts.py`.

Usage:  python decompress_ckpts.py [root_dir]   (default: this script's folder)
"""
import hashlib
import json
import sys
import zlib
from pathlib import Path

READ_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(READ_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def cleanup(manifest_path: Path, parts):
    for p in parts:
        if p.exists():
            p.unlink()
    manifest_path.unlink()


def decompress(manifest_path: Path):
    m = json.loads(manifest_path.read_text())
    folder = manifest_path.parent
    out = folder / m["file"]
    parts = [folder / name for name in m["parts"]]

    # Already restored (e.g. an interrupted earlier run): just clean up.
    if out.exists() and out.stat().st_size == m["size"] and sha256_file(out) == m["sha256"]:
        cleanup(manifest_path, parts)
        print("  -> already present and verified; removed parts")
        return

    missing = [p.name for p in parts if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing parts: {missing}")

    tmp = out.with_name(out.name + ".tmp")
    h, size = hashlib.sha256(), 0
    d = zlib.decompressobj()
    with open(tmp, "wb") as fo:
        for p in parts:
            with open(p, "rb") as fi:
                for chunk in iter(lambda: fi.read(READ_SIZE), b""):
                    data = d.decompress(chunk)
                    h.update(data)
                    size += len(data)
                    fo.write(data)
        data = d.flush()
        h.update(data)
        size += len(data)
        fo.write(data)

    if h.hexdigest() != m["sha256"] or size != m["size"]:
        tmp.unlink()
        raise RuntimeError("sha256/size mismatch; parts kept")

    tmp.replace(out)
    cleanup(manifest_path, parts)
    print(f"  -> restored {size/2**20:.1f} MB from {len(parts)} part(s)")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent)
    manifests = sorted(root.rglob("*.pt.manifest.json"))
    if not manifests:
        print(f"No split checkpoints under {root}")
        return
    failed = 0
    for mp in manifests:
        print(f"[decompress] {mp.relative_to(root).with_suffix('').with_suffix('')}")
        try:
            decompress(mp)
        except Exception as e:  # keep going on other files
            failed += 1
            print(f"  !! {e}")
    print(f"Done: {len(manifests) - failed} restored, {failed} failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
