"""Materialize the BEN-14K cross-modal subset into the SatQuery trainer layout.

Input
-----
* ``data/BigEarthNet_14K.zip`` - BEN_14k/BigEarthNet-S1/<split>/*.tif and
  BEN_14k/BigEarthNet-S2/<split>/*.tif (paired Sentinel-1 / Sentinel-2 patches).
* ``data/BEN14K_metadata.parquet`` - HF metadata: patch_id, labels (19-class
  names), split, s1_name, s2v1_name, cloud/snow flags.

Output
------
``data/BEN14K_materialized/<patch_id>/`` with ``B02.tif B03.tif B04.tif B08.tif``
(the trainer's 4-channel recipe) plus ``labels_19.json`` and a small
``meta.json`` recording the S1 counterpart and quality flags.

Usage
-----
    python scripts/materialize_ben14k.py [--limit 4000] [--splits train val]
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "data" / "BigEarthNet_14K.zip"
META_PATH = ROOT / "data" / "BEN14K_metadata.parquet"
OUT_DIR = ROOT / "data" / "BEN14K_materialized"

# GFM-Bench S2 band order inside the 12-band stack (B1..B12).
S2_BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
                 "B08", "B8A", "B09", "B11", "B12"]
KEEP_BANDS = ["B02", "B03", "B04", "B08"]
KEEP_IDX = [S2_BAND_NAMES.index(b) for b in KEEP_BANDS]

# GFM-Bench 19-class strings -> SatQuery BIGEARTHNET_LABELS (canonical).
# A few GFM classes merge two of our classes, so values can be lists.
LABEL_MAP = {
    "Urban fabric": ["Urban fabric"],
    "Industrial or commercial units": ["Industrial or commercial units"],
    "Arable land": ["Arable land"],
    "Permanent crops": ["Permanent crops"],
    "Pastures": ["Pastures"],
    "Complex cultivation patterns": ["Complex cultivation patterns"],
    "Land principally occupied by agriculture, with significant areas of "
    "natural vegetation": ["Agriculture with natural vegetation"],
    "Agro-forestry areas": ["Agriculture with natural vegetation"],
    "Broad-leaved forest": ["Broad-leaved forest"],
    "Coniferous forest": ["Coniferous forest"],
    "Mixed forest": ["Broad-leaved forest", "Coniferous forest"],
    "Natural grassland and sparsely vegetated areas":
        ["Natural grassland", "Sparsely vegetated areas"],
    "Moors, heathland and sclerophyllous vegetation":
        ["Moors and heathland", "Sclerophyllous vegetation"],
    "Transitional woodland, shrub": ["Transitional woodland/shrub"],
    "Beaches, dunes, sands": ["Bare rock"],
    "Inland wetlands": ["Coastal wetlands"],
    "Coastal wetlands": ["Coastal wetlands"],
    "Inland waters": ["Inland waters"],
    "Marine waters": ["Marine waters"],
}


def canonical_labels(raw_labels: list) -> list:
    """Map GFM-Bench label strings onto the canonical BIGEARTHNET_LABELS set."""
    out: list = []
    for label in raw_labels:
        for mapped in LABEL_MAP.get(label.strip(), []):
            if mapped not in out:
                out.append(mapped)
    return out


def load_metadata() -> "object":  # noqa: ANN401 - narrow return below
    import pyarrow.parquet as pq

    return pq.read_table(META_PATH).to_pylist()


def materialize_one(zf: zipfile.ZipFile, row: dict, keep_idx: list) -> str:
    """Extract + band-select one patch; returns a status string."""
    import rasterio

    patch_id = row["patch_id"]
    zpath = row["_zpath"]
    patch_dir = OUT_DIR / patch_id
    patch_dir.mkdir(parents=True, exist_ok=True)

    with zf.open(zpath) as raw:
        with rasterio.open(raw) as src:
            arr = src.read()  # (bands, H, W)
            profile = src.profile

    if arr.shape[0] < max(keep_idx) + 1:
        return f"skip {patch_id}: {arr.shape[0]}-band stack"

    for band_name, idx in zip(KEEP_BANDS, keep_idx):
        band = arr[idx].astype(np.float32)
        prof = profile.copy()
        prof.update(count=1, dtype="float32", nodata=None)
        prof.pop("band_names", None)
        with rasterio.open(patch_dir / f"{band_name}.tif", "w", **prof) as dst:
            dst.write(band, 1)

    (patch_dir / "labels_19.json").write_text(
        json.dumps({"labels": row["_canon"], "labels_raw": row["labels"]},
                   ensure_ascii=False, indent=1),
        encoding="utf-8")
    (patch_dir / "meta.json").write_text(json.dumps({
        "patch_id": patch_id,
        "split": row["split"],
        "country": row["country"],
        "s1_name": row["s1_name"],
        "contains_seasonal_snow": bool(row["contains_seasonal_snow"]),
        "contains_cloud_or_shadow": bool(row["contains_cloud_or_shadow"]),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return patch_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="materialize at most N patches (for quick runs)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep-cloudy", action="store_true",
                    help="keep patches flagged with cloud/shadow or seasonal snow "
                         "(excluded by default, per BigEarthNet guidance)")
    args = ap.parse_args()

    rows = load_metadata()
    wanted = [r for r in rows if r["split"] in args.splits]
    print(f"metadata rows: {len(rows)} | after split filter: {len(wanted)}")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
        s2_by_stem = {
            Path(n).stem: n for n in names
            if n.startswith("BEN_14k/BigEarthNet-S2/") and n.endswith(".tif")
        }
        s1_by_stem = {
            Path(n).stem: n for n in names
            if n.startswith("BEN_14k/BigEarthNet-S1/") and n.endswith(".tif")
        }
        print(f"zip: {len(s2_by_stem)} S2 tifs, {len(s1_by_stem)} S1 tifs")

        jobs = []
        missing_labels = 0
        quality_filtered = 0
        unmappable = 0
        for row in wanted:
            if args.limit is not None and len(jobs) >= args.limit:
                break
            if not row["labels"]:
                missing_labels += 1
                continue
            if not args.keep_cloudy and (
                row["contains_cloud_or_shadow"] or row["contains_seasonal_snow"]
            ):
                quality_filtered += 1
                continue
            canon = canonical_labels(row["labels"])
            if not canon:
                unmappable += 1
                continue
            zpath = s2_by_stem.get(row["s2v1_name"]) or s2_by_stem.get(row["patch_id"])
            if zpath is None:
                continue
            row = dict(row, _zpath=zpath, _canon=canon)
            jobs.append(row)
        print(f"jobs: {len(jobs)} | quality-filtered: {quality_filtered} | "
              f"no-labels: {missing_labels} | unmappable: {unmappable}")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(materialize_one, zf, row, KEEP_IDX) for row in jobs]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001 - log and continue
                    print(f"  ERROR: {type(exc).__name__}: {exc}")
                    continue
                done += 1
                if done % 500 == 0:
                    print(f"  materialized {done}/{len(jobs)}...")

    print(f"materialized: {done} | no-labels skipped: {missing_labels}")
    print(f"output: {OUT_DIR}")


if __name__ == "__main__":
    main()
