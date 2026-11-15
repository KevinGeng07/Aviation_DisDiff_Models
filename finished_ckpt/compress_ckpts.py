#!/usr/bin/env python3
"""Compress every .pt file under this folder into <=99 MB split parts.

For each `foo.pt` found (recursively) next to this script:
  foo.pt.gz.part000, foo.pt.gz.part001, ...   zlib-compressed chunks (<=99 MB each)
  foo.pt.manifest.json                        sha256/size of the original + part list

Each split is verified by streaming it back and checking the sha256 before the
original .pt is deleted. Run `decompress_ckpts.py` to restore.

Usage:  python compress_ckpts.py [root_dir]     (default: this script's folder)
"""
import hashlib
import json
import sys
import zlib
from pathlib import Path

PART_SIZE = 99 * 1024 * 1024
READ_SIZE = 8 * 1024 * 1024
PART_GLOB = ".gz.part*"


class SplitWriter:
    """File-like sink that rolls over to a new part file every PART_SIZE bytes."""

    def __init__(self, base: Path):
        self.base = base
        self.parts: list[Path] = []
        self._fh = None
        self._written = 0

    def _roll(self):
        if self._fh:
            self._fh.close()
        path = self.base.with_name(f"{self.base.name}.gz.part{len(self.parts):03d}")
        self.parts.append(path)
        self._fh = open(path, "wb")
        self._written = 0

    def write(self, data: bytes):
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


def stream_parts_sha256(parts: list[Path]) -> tuple[str, int]:
    h, size = hashlib.sha256(), 0
    d = zlib.decompressobj()
    for p in parts:
        with open(p, "rb") as f:
            while chunk := f.read(READ_SIZE):
                out = d.decompress(chunk)
                h.update(out)
                size += len(out)
    out = d.flush()
    h.update(out)
    size += len(out)
    return h.hexdigest(), size


def remove_stale(pt: Path):
    for old in pt.parent.glob(pt.name + PART_GLOB):
        old.unlink()
    pt.with_name(pt.name + ".manifest.json").unlink(missing_ok=True)


def compress(pt: Path):
    remove_stale(pt)
    h, size = hashlib.sha256(), 0
    c = zlib.compressobj(level=6)
    w = SplitWriter(pt)
    with open(pt, "rb") as f:
        while chunk := f.read(READ_SIZE):
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
        raise RuntimeError(f"verification failed for {pt}; original kept")

    manifest = {
        "file": pt.name,
        "size": size,
        "sha256": digest,
        "parts": [p.name for p in w.parts],
    }
    pt.with_name(pt.name + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    pt.unlink()
    comp = sum(p.stat().st_size for p in w.parts)
    print(f"  -> {len(w.parts)} part(s), {size/2**20:.1f} MB -> {comp/2**20:.1f} MB")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent)
    pts = sorted(root.rglob("*.pt"))
    if not pts:
        print(f"No .pt files under {root}")
        return
    failed = 0
    for pt in pts:
        print(f"[compress] {pt.relative_to(root)}")
        try:
            compress(pt)
        except Exception as e:  # keep going on other files
            failed += 1
            print(f"  !! {e}")
    print(f"Done: {len(pts) - failed} compressed, {failed} failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
