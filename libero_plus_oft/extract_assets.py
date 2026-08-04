#!/usr/bin/env python3
"""Safely extract the official assets needed by the selected standard-object task."""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
from pathlib import Path


def within(root: Path, candidate: Path) -> bool:
    return os.path.commonpath((str(root.resolve()), str(candidate.resolve()))) == str(root.resolve())


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: extract_assets.py ASSETS_ZIP LIBERO_PLUS_ROOT")
    zip_path = Path(sys.argv[1]).resolve()
    plus_root = Path(sys.argv[2]).resolve()
    libero_root = plus_root / "libero" / "libero"

    with zipfile.ZipFile(zip_path) as archive:
        files = []
        skipped_new_objects = 0
        for entry in archive.infolist():
            if entry.is_dir() or "/assets/" not in entry.filename:
                continue
            relative = entry.filename.split("/assets/", 1)[1]
            # The selected language-perturbation Spatial task uses standard LIBERO
            # objects. LIBERO-Plus new-object variants contain 432k extra mesh
            # fragments and are intentionally not needed for this one episode.
            if relative.startswith("new_objects/"):
                skipped_new_objects += 1
                continue
            files.append((entry, relative))
        if not files:
            raise RuntimeError("assets.zip contains no files.")
        destination = libero_root / "assets"
        destination.mkdir(parents=True, exist_ok=True)

        for index, (entry, relative) in enumerate(files, start=1):
            target = destination / relative
            if not within(destination, target):
                raise RuntimeError(f"Unsafe path in assets.zip: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != entry.file_size:
                    raise RuntimeError(f"Existing asset has a different size; refusing to overwrite: {target}")
                continue
            with archive.open(entry) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if index % 250 == 0 or index == len(files):
                print(f"Extracted {index}/{len(files)} files", flush=True)

    assets_root = libero_root / "assets"
    count = sum(1 for path in assets_root.rglob("*") if path.is_file())
    if count == 0:
        raise RuntimeError(f"No assets found at {assets_root}")
    print(f"LIBERO-Plus task assets ready: {assets_root} ({count} files)")
    print(f"Skipped {skipped_new_objects} new-object mesh fragments not used by task ID 988.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
