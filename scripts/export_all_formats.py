"""
Export All Datasets in All Formats + All Visuals in All Image Formats
======================================================================

For every parquet file under datasets/ this script exports to:
  Tabular formats  : parquet (already exists), csv, npz, h5 (HDF5), feather, pkl, json
                     (json skipped for vehicle datasets > 5 MB to avoid huge files)

For every existing image (.png) this script re-saves to:
  Image formats    : png (already exists), pdf, svg, eps

All outputs land alongside the source file — same directory, different extension.

Usage
-----
  python scripts/export_all_formats.py

Dependencies: pandas, pyarrow, numpy, tables (PyTables for HDF5)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT  = Path(__file__).resolve().parent.parent
DATASETS   = REPO_ROOT / "datasets"

# Parquet files inside perception/cache/ are raw API blobs — skip them
SKIP_DIRS  = {"cache"}

# Skip json export for files whose DataFrame exceeds this size (rows × cols)
JSON_MAX_CELLS = 500_000


# ---------------------------------------------------------------------------
# Tabular export
# ---------------------------------------------------------------------------
def export_parquet(src: Path) -> None:
    """Export a single parquet file to all tabular formats."""
    try:
        df = pd.read_parquet(src)
    except Exception as e:
        logger.warning(f"  Cannot read {src.name}: {e}")
        return

    n_cells = df.shape[0] * df.shape[1]
    stem = src.stem
    out_dir = src.parent
    logger.info(f"  {src.parent.relative_to(DATASETS)}/{src.name}  "
                f"({df.shape[0]:,} rows × {df.shape[1]} cols)")

    # --- CSV (skip if already exists and is same size) ---
    csv_path = out_dir / f"{stem}.csv"
    if not csv_path.exists():
        df.to_csv(csv_path, index=False)
        logger.info(f"    → csv  ({csv_path.stat().st_size // 1024} KB)")
    else:
        logger.info(f"    csv  already exists")

    # --- NPZ ---
    npz_path = out_dir / f"{stem}.npz"
    if not npz_path.exists():
        numeric = df.select_dtypes(include=[np.number])
        np.savez_compressed(npz_path, **{col: numeric[col].to_numpy() for col in numeric.columns})
        logger.info(f"    → npz  ({npz_path.stat().st_size // 1024} KB)")
    else:
        logger.info(f"    npz  already exists")

    # --- HDF5 (PyTables) ---
    h5_path = out_dir / f"{stem}.h5"
    if not h5_path.exists():
        try:
            # Sanitise column names: HDF5 keys can't contain special chars
            df_h5 = df.copy()
            df_h5.columns = [c.replace("(", "").replace(")", "").replace(" ", "_")
                             .replace("/", "_").replace("-", "_") for c in df_h5.columns]
            df_h5.to_hdf(h5_path, key="data", mode="w", complevel=5, complib="blosc")
            logger.info(f"    → h5   ({h5_path.stat().st_size // 1024} KB)")
        except Exception as e:
            logger.warning(f"    h5 failed: {e}")
    else:
        logger.info(f"    h5   already exists")

    # --- Feather ---
    feather_path = out_dir / f"{stem}.feather"
    if not feather_path.exists():
        try:
            df.reset_index(drop=True).to_feather(feather_path)
            logger.info(f"    → feather  ({feather_path.stat().st_size // 1024} KB)")
        except Exception as e:
            logger.warning(f"    feather failed: {e}")
    else:
        logger.info(f"    feather  already exists")

    # --- Pickle ---
    pkl_path = out_dir / f"{stem}.pkl"
    if not pkl_path.exists():
        with open(pkl_path, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"    → pkl  ({pkl_path.stat().st_size // 1024} KB)")
    else:
        logger.info(f"    pkl  already exists")

    # --- JSON (skip for large datasets) ---
    json_path = out_dir / f"{stem}.json"
    if not json_path.exists():
        if n_cells <= JSON_MAX_CELLS:
            df.to_json(json_path, orient="records", indent=2, default_handler=str)
            logger.info(f"    → json  ({json_path.stat().st_size // 1024} KB)")
        else:
            logger.info(f"    json  skipped (too large: {n_cells:,} cells > {JSON_MAX_CELLS:,})")
    else:
        logger.info(f"    json  already exists")


# ---------------------------------------------------------------------------
# Image export
# ---------------------------------------------------------------------------
IMAGE_FORMATS = ["png", "pdf", "svg", "eps"]

def export_image(src: Path) -> None:
    """Re-save a .png as pdf, svg, and eps alongside it."""
    try:
        img = mpimg.imread(str(src))
    except Exception as e:
        logger.warning(f"  Cannot read {src.name}: {e}")
        return

    logger.info(f"  {src.parent.relative_to(DATASETS)}/{src.name}")
    h, w = img.shape[:2]
    dpi = 150
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(img)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)

    for fmt in IMAGE_FORMATS:
        if fmt == "png":
            continue  # source already png
        out_path = src.with_suffix(f".{fmt}")
        if not out_path.exists():
            try:
                fig.savefig(out_path, format=fmt, dpi=dpi, bbox_inches="tight",
                            pad_inches=0)
                logger.info(f"    → {fmt}  ({out_path.stat().st_size // 1024} KB)")
            except Exception as e:
                logger.warning(f"    {fmt} failed: {e}")
        else:
            logger.info(f"    {fmt}  already exists")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Walk datasets/
# ---------------------------------------------------------------------------
def find_parquets() -> list[Path]:
    paths = []
    for p in sorted(DATASETS.rglob("*.parquet")):
        # Skip anything inside a cache/ directory
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        paths.append(p)
    return paths


def find_images() -> list[Path]:
    paths = []
    for p in sorted(DATASETS.rglob("*.png")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parquets = find_parquets()
    images   = find_images()

    logger.info(f"Found {len(parquets)} parquet files and {len(images)} PNG images")
    logger.info("="*60)

    logger.info("\n[1/2] Exporting tabular formats...")
    for src in parquets:
        export_parquet(src)

    logger.info("\n[2/2] Exporting image formats...")
    for src in images:
        export_image(src)

    # Summary
    all_files = list(DATASETS.rglob("*"))
    by_ext: dict[str, int] = {}
    for f in all_files:
        if f.is_file() and not any(part in SKIP_DIRS for part in f.parts):
            ext = f.suffix.lower()
            by_ext[ext] = by_ext.get(ext, 0) + 1

    logger.info("\n" + "="*60)
    logger.info("OUTPUT FILE COUNTS BY EXTENSION")
    logger.info("="*60)
    for ext, count in sorted(by_ext.items()):
        logger.info(f"  {ext:<12} {count:>4} files")
    logger.info("="*60)
    logger.info("Done.")


if __name__ == "__main__":
    main()
