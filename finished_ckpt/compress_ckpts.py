#!/usr/bin/env python3
"""Compress .pt checkpoints over 99 MB in finished_ckpt/ into <=99 MB parts (GitHub's limit).

Works on any file or folder in finished_ckpt/: give folders (searched recursively) and/or
.pt files, relative to the repo root or to finished_ckpt/; with no arguments, every .pt in
finished_ckpt/ is checked. Files at or under 99 MB are left untouched. For each larger foo.pt:
  foo.pt.gz.part000, foo.pt.gz.part001, ...   zlib-compressed chunks (<=99 MB each)
  foo.pt.manifest.json                        sha256/size of the original + part list

Each split is verified by streaming it back and checking the sha256 before the original
foo.pt is deleted. decompress_ckpts.py restores the .pt files.

Usage (Python 3.6+):
    python finished_ckpt/compress_ckpts.py                              # all of finished_ckpt/
    python finished_ckpt/compress_ckpts.py 1_teacher                    # one folder
    python finished_ckpt/compress_ckpts.py 2_architecture/distill_4_256 3_gru
    python finished_ckpt/compress_ckpts.py 1_teacher/dit_cfm_20M/best.pt   # one file
    python finished_ckpt/compress_ckpts.py --dry_run                    # list only, change nothing
"""
import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path

PART_SIZE = 99 * 1024 * 1024
READ_SIZE = 8 * 1024 * 1024
PART_GLOB = ".gz.part*"
HERE      = Path(__file__).resolve().parent          # finished_ckpt/


class SplitWriter:
    """File-like sink that rolls over to a new part file every PART_SIZE bytes."""

    def __init__(self, base):
        self.base = base
        self.parts = []
        self._fh = None
        self._written = 0

    def _roll(self):
        if self._fh:
            self._fh.close()
        path = self.base.with_name("{}.gz.part{:03d}".format(self.base.name, len(self.parts)))
        self.parts.append(path)
        self._fh = open(str(path), "wb")
        self._written = 0

    def write(self, data):
        view = memoryview(data)
        while view:
            if self._fh is None or self._written >= PART_SIZE:
                self._roll()
            n = min(len(view), PART_SIZE - self._written)
            self._fh.write(view[:n])
            self._written += n
            view = view[n:]

    def close(self):
        if self._fh is None:  # empty input still gets one (empty) part
            self._roll()
        self._fh.close()


def read_chunks(path):
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(READ_SIZE), b""):
            yield chunk


def stream_parts_sha256(parts):
    h, size = hashlib.sha256(), 0
    d = zlib.decompressobj()
    for p in parts:
        for chunk in read_chunks(p):
            out = d.decompress(chunk)
            h.update(out)
            size += len(out)
    out = d.flush()
    h.update(out)
    size += len(out)
    return h.hexdigest(), size


def remove_stale(pt):
    for old in pt.parent.glob(pt.name + PART_GLOB):
        old.unlink()
    manifest = pt.with_name(pt.name + ".manifest.json")
    if manifest.exists():
        manifest.unlink()


def compress(pt):
    remove_stale(pt)
    h, size = hashlib.sha256(), 0
    c = zlib.compressobj(6)
    w = SplitWriter(pt)
    for chunk in read_chunks(pt):
        h.update(chunk)
        size += len(chunk)
        w.write(c.compress(chunk))
    w.write(c.flush())
    w.close()

    digest = h.hexdigest()
    check, check_size = stream_parts_sha256(w.parts)
    if check != digest or check_size != size:
        for p in w.parts:
            p.unlink()
        raise RuntimeError("verification failed for {}; original kept".format(pt))

    manifest = {"file": pt.name, "size": size, "sha256": digest, "parts": [p.name for p in w.parts]}
    pt.with_name(pt.name + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    pt.unlink()
    comp = sum(p.stat().st_size for p in w.parts)
    print("  -> {} part(s), {:.1f} MB -> {:.1f} MB".format(len(w.parts), size / 2**20, comp / 2**20))


def resolve(arg):
    """A path as given (repo-root relative or absolute), else relative to finished_ckpt/."""
    p = Path(arg)
    return p if p.exists() else HERE / arg


def collect(args):
    """Every .pt named directly or found under a named folder, each once."""
    found = []
    for arg in args:
        p = resolve(arg)
        if p.is_dir():
            found += sorted(p.rglob("*.pt"))
        elif p.is_file() and p.suffix == ".pt":
            found.append(p)
        else:
            print("[warn] not a folder or .pt file in finished_ckpt/: {}".format(arg))
    seen, out = set(), []
    for p in found:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def main():
    parser = argparse.ArgumentParser(description="Split .pt files over 99 MB into compressed parts")
    parser.add_argument("paths", nargs="*", default=[str(HERE)],
                        help="folders and/or .pt files in finished_ckpt/ (default: all of it)")
    parser.add_argument("--dry_run", action="store_true", help="only list what would be compressed")
    args = parser.parse_args()

    pts = collect(args.paths)
    big = [p for p in pts if p.stat().st_size > PART_SIZE]
    for p in pts:
        if p not in big:
            print("[skip] {} ({:.1f} MB, under limit)".format(p, p.stat().st_size / 2**20))
    if not big:
        print("No .pt files over {} MB in: {}".format(PART_SIZE // 2**20, " ".join(args.paths)))
        return
    failed = 0
    for pt in big:
        print("[compress] {} ({:.1f} MB)".format(pt, pt.stat().st_size / 2**20))
        if args.dry_run:
            continue
        try:
            compress(pt)
        except Exception as e:  # keep going on other files
            failed += 1
            print("  !! {}".format(e))
    if not args.dry_run:
        print("Done: {} compressed, {} failed.".format(len(big) - failed, failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
