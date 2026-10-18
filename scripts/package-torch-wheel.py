#!/usr/bin/env python3
"""package-torch-wheel.py — repack an extracted sm_61 torch tree into a pip-installable wheel.

Why: the sm_61 torch is built from source inside the build image (hours). The build
recipe already extracts the result into ``artifacts/torch/site-packages/<subdir>/``
(see ``scripts/extract-sm61-torch-artifact.sh``) so later builds can reuse it
(``SM61_USE_TORCH_ARTIFACT=1``). That tree is exactly the content of a wheel that
pip produced during the build, so it can be turned back into a wheel without
touching the binaries:

  * a wheel is a plain zip with a ``<name>-<version>.dist-info/`` directory;
  * ``RECORD`` is regenerated (hashes for every member). pip does not verify
    RECORD on install, but a correct one keeps ``pip uninstall``/``pip check``
    honest and makes the artifact self-consistent;
  * interpreter cruft from the build environment is dropped: ``__pycache__/*.pyc``,
    ``direct_url.json`` (contains the build path), ``REQUESTED``, ``INSTALLER``.

Publishing the wheel (GitHub Releases) lets users skip the multi-hour torch build:

    pip install --no-deps torch-<version>-cp312-cp312-linux_x86_64.whl

...and the same file unpacks into ``artifacts/torch/site-packages/current/`` for
the Docker build (``scripts/fetch-torch-artifact.sh`` does both).

Usage:
    python3 scripts/package-torch-wheel.py                     # -> dist/
    python3 scripts/package-torch-wheel.py --src artifacts/torch/site-packages/current
    python3 scripts/package-torch-wheel.py --out dist --compresslevel 6 --dry-run
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

CHUNK = 4 * 1024 * 1024

# Files that exist in the extracted tree only because pip put them there while
# installing from a local directory; they must never leak into a published wheel.
DIST_INFO_DROP = {"direct_url.json", "REQUESTED", "INSTALLER", "RECORD"}
SKIP_DIRS = {"__pycache__"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} GiB"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def record_hash(h: "hashlib._Hash") -> str:
    """PEP 427 RECORD hash = urlsafe base64 of the digest, padding stripped."""
    return base64.urlsafe_b64encode(h.digest()).decode("ascii").rstrip("=")


def read_metadata(dist_info: Path) -> dict:
    meta = {}
    for line in (dist_info / "METADATA").read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line[0].isspace():
            continue
        if line.startswith("Name: "):
            meta["name"] = line[len("Name: "):].strip()
        elif line.startswith("Version: "):
            meta["version"] = line[len("Version: "):].strip()
        elif line.startswith("License-Expression: "):
            meta["license"] = line[len("License-Expression: "):].strip()
    wheel_tags = [
        ln.split(":", 1)[1].strip()
        for ln in (dist_info / "WHEEL").read_text(encoding="utf-8").splitlines()
        if ln.startswith("Tag:")
    ]
    meta["tags"] = wheel_tags
    return meta


def collect(src: Path, dist_info_name: str) -> tuple[list[Path], list[str]]:
    """Return (files to pack, dropped paths relative to src)."""
    files: list[Path] = []
    dropped: list[str] = []
    for root, dirs, names in os.walk(src):
        root_path = Path(root)
        rel_root = root_path.relative_to(src)
        if rel_root.parts and rel_root.parts[0] in SKIP_DIRS:
            dirs[:] = []
            dropped.append(str(rel_root))
            continue
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(names):
            path = root_path / name
            rel = path.relative_to(src)
            if path.suffix in SKIP_SUFFIXES:
                dropped.append(str(rel))
                continue
            if rel.parts[0] == dist_info_name and name in DIST_INFO_DROP:
                dropped.append(str(rel))
                continue
            if path.is_symlink() or not path.is_file():
                dropped.append(str(rel))
                continue
            files.append(path)
    return files, dropped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="artifacts/torch/site-packages/current",
                    help="extracted torch tree (default: artifacts/torch/site-packages/current)")
    ap.add_argument("--out", default="dist", help="output directory for the wheel (default: dist)")
    ap.add_argument("--compresslevel", type=int, default=6, help="zlib level 1-9 (default 6)")
    ap.add_argument("--dry-run", action="store_true", help="report what would be packed, write nothing")
    ap.add_argument("--json", action="store_true", help="print a JSON summary on success")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    if not src.is_dir():
        print(f"error: source tree not found: {src}", file=sys.stderr)
        return 2

    dist_infos = sorted(p for p in src.iterdir() if p.is_dir() and p.name.startswith("torch-") and p.name.endswith(".dist-info"))
    if len(dist_infos) != 1:
        print(f"error: expected exactly one torch-*.dist-info in {src}, found {len(dist_infos)}", file=sys.stderr)
        return 2
    dist_info = dist_infos[0]
    meta = read_metadata(dist_info)
    for key in ("name", "version"):
        if not meta.get(key):
            print(f"error: {dist_info}/METADATA is missing {key}", file=sys.stderr)
            return 2
    if not meta.get("tags"):
        print(f"error: {dist_info}/WHEEL has no Tag entry", file=sys.stderr)
        return 2

    required = {"torch", "torchgen", "functorch"}
    missing = sorted(d for d in required if not (src / d).is_dir())
    if missing:
        print(f"error: source tree lacks package dirs: {', '.join(missing)}", file=sys.stderr)
        return 2

    files, dropped = collect(src, dist_info.name)
    total_bytes = sum(p.stat().st_size for p in files)
    wheel_name = f"{meta['name']}-{meta['version']}-{meta['tags'][0]}.whl"

    print(f"source      : {src}")
    print(f"package     : {meta['name']} {meta['version']} ({meta['tags'][0]})")
    print(f"license     : {meta.get('license', 'n/a')}")
    print(f"members     : {len(files)} files, {human(total_bytes)} uncompressed")
    print(f"dropped     : {len(dropped)} build-environment files (__pycache__, *.pyc, direct_url.json, ...)")
    print(f"wheel       : {wheel_name}")

    if args.dry_run:
        biggest = sorted(files, key=lambda p: p.stat().st_size, reverse=True)[:5]
        print("largest members:")
        for p in biggest:
            print(f"  {human(p.stat().st_size):>10}  {p.relative_to(src)}")
        print("(dry-run: nothing written)")
        return 0

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = out_dir / wheel_name
    tmp_path = out_dir / (wheel_name + ".part")

    record_lines: list[str] = []
    started = time.time()
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED, compresslevel=args.compresslevel,
                         allowZip64=True) as zf:
        for path in files:
            arcname = path.relative_to(src).as_posix()
            h = hashlib.sha256()
            size = 0
            with path.open("rb") as fsrc, zf.open(arcname, "w", force_zip64=True) as fdst:
                for block in iter(lambda: fsrc.read(CHUNK), b""):
                    h.update(block)
                    size += len(block)
                    fdst.write(block)
            record_lines.append(f"{arcname},sha256={record_hash(h)},{size}")
        record_name = f"{dist_info.name}/RECORD"
        record_lines.append(f"{record_name},,")
        zf.writestr(record_name, "\n".join(record_lines) + "\n")

    tmp_path.replace(wheel_path)

    wheel_size = wheel_path.stat().st_size
    wheel_sha = sha256_file(wheel_path)
    elapsed = time.time() - started
    summary = {
        "wheel": wheel_path.name,
        "path": str(wheel_path),
        "bytes": wheel_size,
        "sha256": wheel_sha,
        "torch": f"{meta['name']} {meta['version']}",
        "tag": meta["tags"][0],
        "license": meta.get("license"),
        "members": len(files) + 1,
        "uncompressed_bytes": total_bytes,
        "dropped": len(dropped),
        "seconds": round(elapsed, 1),
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"written     : {wheel_path}")
        print(f"size        : {human(wheel_size)} (from {human(total_bytes)}, {wheel_size / max(total_bytes, 1) * 100:.0f}%)")
        print(f"sha256      : {wheel_sha}")
        print(f"took        : {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
