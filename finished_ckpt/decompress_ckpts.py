#!/usr/bin/env python3
"""Restore checkpoints in finished_ckpt/ that compress_ckpts.py split into parts.

Works on any file or folder in finished_ckpt/: give folders (searched recursively) and/or
single checkpoints (foo.pt or foo.pt.manifest.json), relative to the repo root or to
finished_ckpt/; with no arguments, every split checkpoint in finished_ckpt/ is restored.
For each foo.pt.manifest.json, foo.pt.gz.part* are reassembled + decompressed into foo.pt, the
sha256 is checked, then the parts and the manifest are deleted.

If foo.pt already exists: identical to the parts → the parts are removed; different (e.g. a
newer foo.pt written after compressing) → both are left alone and a warning is printed, so a
newer checkpoint is never overwritten.

Usage (Python 3.6+):
    python finished_ckpt/decompress_ckpts.py                            # all of finished_ckpt/
    python finished_ckpt/decompress_ckpts.py 1_teacher                  # one folder
    python finished_ckpt/decompress_ckpts.py 1_teacher/dit_cfm_20M/best.pt   # one checkpoint
    python finished_ckpt/decompress_ckpts.py --dry_run                  # list only, change nothing
"""
import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path

READ_SIZE = 8 * 1024 * 1024
HERE      = Path(__file__).resolve().parent          # finished_ckpt/


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(READ_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def cleanup(manifest_path, parts):
    for p in parts:
        if p.exists():
            p.unlink()
    manifest_path.unlink()


def decompress(manifest_path):
    m = json.loads(manifest_path.read_text())
    folder = manifest_path.parent
    out = folder / m["file"]
    parts = [folder / name for name in m["parts"]]

    if out.exists():
        if out.stat().st_size == m["size"] and sha256_file(out) == m["sha256"]:
            cleanup(manifest_path, parts)
            print("  -> already present and identical; removed the parts")
        else:
            print("  !! {} exists and differs from its parts (probably newer); left both alone. "
                  "Delete the parts + manifest yourself if they are stale.".format(out.name))
            return "skipped"
        return "restored"

    missing = [p.name for p in parts if not p.exists()]
    if missing:
        raise FileNotFoundError("missing parts: {}".format(missing))

    tmp = out.with_name(out.name + ".tmp")
    h, size = hashlib.sha256(), 0
    d = zlib.decompressobj()
    with open(str(tmp), "wb") as fo:
        for p in parts:
            with open(str(p), "rb") as fi:
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
    print("  -> restored {:.1f} MB from {} part(s)".format(size / 2**20, len(parts)))
    return "restored"


def resolve(arg):
    """A path as given (repo-root relative or absolute), else relative to finished_ckpt/."""
    p = Path(arg)
    return p if p.exists() or Path(arg + ".manifest.json").exists() else HERE / arg


def collect(args):
    """Every manifest named directly (as foo.pt or its manifest) or under a named folder, each once."""
    found = []
    for arg in args:
        p = resolve(arg)
        if p.is_dir():
            found += sorted(p.rglob("*.pt.manifest.json"))
        elif p.name.endswith(".pt.manifest.json") and p.is_file():
            found.append(p)
        elif p.with_name(p.name + ".manifest.json").is_file():
            found.append(p.with_name(p.name + ".manifest.json"))
        else:
            print("[warn] no split checkpoint for: {}".format(arg))
    seen, out = set(), []
    for p in found:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser(description="Restore split checkpoints back into .pt files")
    parser.add_argument("paths", nargs="*", default=[str(HERE)],
                        help="folders and/or checkpoints in finished_ckpt/ (default: all of it)")
    parser.add_argument("--dry_run", action="store_true", help="only list what would be restored")
    args = parser.parse_args()

    manifests = collect(args.paths)
    if not manifests:
        print("No split checkpoints in: {}".format(" ".join(args.paths)))
        return
    failed = skipped = 0
    for mp in manifests:
        print("[decompress] {}".format(mp.with_name(mp.name[: -len(".manifest.json")])))
        if args.dry_run:
            continue
        try:
            skipped += decompress(mp) == "skipped"
        except Exception as e:  # keep going on other files
            failed += 1
            print("  !! {}".format(e))
    if not args.dry_run:
        print("Done: {} restored, {} left alone, {} failed.".format(
            len(manifests) - failed - skipped, skipped, failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
