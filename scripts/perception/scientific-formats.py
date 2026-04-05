#!/usr/bin/env python3
"""
Multi-Format Scientific Export Pipeline — Defense eVTOL Perception Dataset.

Exports the perception dataset and statistics in every scientifically
relevant format for archival, citation, and downstream toolchain use.

Output directory: outputs/perception_dataset/exports/

Format matrix:
  Tabular data     : CSV, Parquet (snappy), HDF5 (compressed), NumPy .npz
  Statistics       : JSON, CSV
  Documentation    : Markdown (already at doc/), HTML, LaTeX .tex
  Metadata/prov.   : JSON-LD, YAML, BibTeX (.bib), Dublin Core XML

All formats written to exports/ with a manifest file listing every file
and its SHA-256 checksum for reproducibility.

Usage:
    python scripts/export_scientific_formats.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import h5py
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("export_scientific")

# ---------------------------------------------------------------------------
DATA_DIR    = REPO_ROOT / "outputs" / "perception_dataset"
EXPORT_DIR  = DATA_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

FULL_CSV    = DATA_DIR / "perception_full_dataset.csv"
META_PATH   = DATA_DIR / "perception_metadata.json"

# Theatre & dataset constants (replicated here for self-contained export)
THEATRE = "Delhi NCR outskirts, India"
LAT_BOUNDS = (28.40, 28.90)
LON_BOUNDS = (76.90, 77.50)
ALT_LEVELS = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500]
N_ROWS = 1_057_714
N_COLS = 28
N_SPATIAL = 75_551
GENERATED_AT = "2026-04-04T07:13:16Z"

MANIFEST: list[dict] = []

# ===========================================================================
# Utilities
# ===========================================================================

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def record(path: Path, fmt: str, desc: str) -> None:
    MANIFEST.append({
        "file": path.name,
        "format": fmt,
        "description": desc,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    })
    logger.info("  [%s]  %s  (%.1f MB)", fmt, path.name, path.stat().st_size / 1e6)


def load_csv_chunks(chunk_size: int = 100_000):
    """Yield chunks of CSV rows as lists of dicts."""
    with open(FULL_CSV, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        buf = []
        for row in reader:
            buf.append(row)
            if len(buf) == chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf


def csv_to_numpy_arrays() -> tuple[list[str], dict[str, np.ndarray]]:
    """Load full CSV into per-column numpy arrays (memory efficient)."""
    logger.info("  Loading CSV into memory (column-wise)...")
    text_cols = {"surface_type", "obstacle_type"}
    with open(FULL_CSV, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = list(reader.fieldnames)
        # Pre-allocate lists per column
        buffers: dict[str, list] = {c: [] for c in cols}
        for row in reader:
            for c in cols:
                buffers[c].append(row[c])
    arrays = {}
    for col in cols:
        if col in text_cols:
            arrays[col] = np.array(buffers[col], dtype="U32")
        else:
            arrays[col] = np.array(buffers[col], dtype=np.float32)
        buffers[col] = []   # free memory
    return cols, arrays


# ===========================================================================
# 1. Parquet (Apache Arrow / Snappy compressed)
# ===========================================================================

def export_parquet(cols: list[str], arrays: dict) -> None:
    logger.info("[1] Exporting Parquet (Snappy) ...")
    text_cols = {"surface_type", "obstacle_type"}
    pa_arrays = {}
    for col in cols:
        if col in text_cols:
            pa_arrays[col] = pa.array(arrays[col].tolist(), type=pa.string())
        else:
            pa_arrays[col] = pa.array(arrays[col].tolist(), type=pa.float32())

    table = pa.table(pa_arrays)
    path = EXPORT_DIR / "perception_full_dataset.parquet"
    pq.write_table(table, str(path), compression="snappy",
                   write_statistics=True, row_group_size=50_000)
    record(path, "Parquet/Snappy", "Full dataset — columnar format, compressed")

    # Also write per-layer parquet files
    for layer, layer_cols in [
        ("terrain",  ["lat","lon","elev_m","slope_deg","roughness_m","surface_type","terrain_clearance_m"]),
        ("wind",     ["lat","lon","alt_m","wind_u_mps","wind_v_mps","wind_w_mps","wind_speed_mps","wind_dir_deg","turbulence_intensity"]),
        ("obstacle", ["lat","lon","nearest_obstacle_dist_m","nearest_obstacle_height_m","obstacle_type"]),
        ("threat",   ["lat","lon","alt_m","T1_detect_prob","T2_detect_prob","T3_detect_prob","combined_threat_prob"]),
        ("fusion",   ["lat","lon","alt_m","terrain_cost","wind_cost","obstacle_cost","threat_cost","energy_cost","fused_cost"]),
    ]:
        pa_sub = {c: pa_arrays[c] for c in layer_cols if c in pa_arrays}
        tbl_sub = pa.table(pa_sub)
        p = EXPORT_DIR / f"perception_{layer}.parquet"
        pq.write_table(tbl_sub, str(p), compression="snappy")
        record(p, "Parquet/Snappy", f"{layer.capitalize()} layer subset")


# ===========================================================================
# 2. HDF5 (h5py, GZIP level 6)
# ===========================================================================

def export_hdf5(cols: list[str], arrays: dict) -> None:
    logger.info("[2] Exporting HDF5 ...")
    path = EXPORT_DIR / "perception_full_dataset.h5"
    text_cols = {"surface_type", "obstacle_type"}
    with h5py.File(str(path), "w") as hf:
        hf.attrs["theatre"] = THEATRE
        hf.attrs["generated_at"] = GENERATED_AT
        hf.attrs["n_rows"] = N_ROWS
        hf.attrs["n_spatial"] = N_SPATIAL
        hf.attrs["lat_bounds"] = LAT_BOUNDS
        hf.attrs["lon_bounds"] = LON_BOUNDS
        hf.attrs["alt_levels"] = ALT_LEVELS
        hf.attrs["description"] = (
            "Defense eVTOL perception dataset. 5 layers: terrain, wind, "
            "obstacle, threat, fusion. 1.057M rows x 28 columns."
        )

        # Top-level groups by layer
        groups = {
            "position":  ["lat", "lon", "alt_m"],
            "terrain":   ["elev_m", "slope_deg", "roughness_m", "terrain_clearance_m"],
            "wind":      ["wind_u_mps", "wind_v_mps", "wind_w_mps",
                          "wind_speed_mps", "wind_dir_deg", "turbulence_intensity"],
            "obstacle":  ["nearest_obstacle_dist_m", "nearest_obstacle_height_m"],
            "threat":    ["T1_detect_prob", "T2_detect_prob", "T3_detect_prob",
                          "max_threat_prob", "combined_threat_prob"],
            "cost":      ["terrain_cost", "wind_cost", "obstacle_cost",
                          "threat_cost", "energy_cost", "fused_cost"],
        }
        for grp_name, col_list in groups.items():
            grp = hf.create_group(grp_name)
            for col in col_list:
                if col in arrays and col not in text_cols:
                    ds = grp.create_dataset(col, data=arrays[col],
                                             compression="gzip", compression_opts=6,
                                             shuffle=True, chunks=(min(10000, N_ROWS),))
                    ds.attrs["unit"] = _units.get(col, "")
                    ds.attrs["description"] = _descriptions.get(col, "")

        # Text datasets (UTF-8 variable-length strings)
        for col in text_cols:
            if col in arrays:
                dt = h5py.special_dtype(vlen=str)
                hf.create_dataset(col, data=arrays[col].tolist(), dtype=dt)

    record(path, "HDF5/GZIP-6", "Full dataset — hierarchical, compressed, self-describing")


_units = {
    "lat": "degrees_north", "lon": "degrees_east", "alt_m": "m",
    "elev_m": "m_MSL", "slope_deg": "degrees", "roughness_m": "m",
    "terrain_clearance_m": "m",
    "wind_u_mps": "m/s", "wind_v_mps": "m/s", "wind_w_mps": "m/s",
    "wind_speed_mps": "m/s", "wind_dir_deg": "degrees", "turbulence_intensity": "dimensionless",
    "nearest_obstacle_dist_m": "m", "nearest_obstacle_height_m": "m",
    "T1_detect_prob": "probability", "T2_detect_prob": "probability",
    "T3_detect_prob": "probability", "max_threat_prob": "probability",
    "combined_threat_prob": "probability",
    "terrain_cost": "dimensionless", "wind_cost": "dimensionless",
    "obstacle_cost": "dimensionless", "threat_cost": "dimensionless",
    "energy_cost": "dimensionless", "fused_cost": "dimensionless",
}
_descriptions = {
    "lat": "WGS-84 latitude",
    "lon": "WGS-84 longitude",
    "alt_m": "Query altitude above mean sea level",
    "elev_m": "SRTM surface elevation (EGM96 vertical datum)",
    "slope_deg": "Terrain slope magnitude from gradient of elevation",
    "roughness_m": "RMS elevation variability in 3x3 kernel",
    "terrain_clearance_m": "alt_m minus elev_m",
    "wind_u_mps": "Eastward wind component (GFS/HRES forecast)",
    "wind_v_mps": "Northward wind component",
    "wind_w_mps": "Vertical wind component (estimated from shear)",
    "wind_speed_mps": "Horizontal wind speed magnitude",
    "wind_dir_deg": "Meteorological FROM direction (degrees clockwise from North)",
    "turbulence_intensity": "Normalized wind shear turbulence intensity",
    "nearest_obstacle_dist_m": "KDTree nearest OSM obstacle distance",
    "nearest_obstacle_height_m": "Height of nearest obstacle",
    "T1_detect_prob": "Swerling-I P_d: S-300V site A",
    "T2_detect_prob": "Swerling-I P_d: SA-11 site B",
    "T3_detect_prob": "Swerling-I P_d: SA-22 site C",
    "combined_threat_prob": "Probabilistic-OR of all SAM P_d values",
    "fused_cost": "Weighted sum: 0.15*terrain+0.10*wind+0.20*obstacle+0.40*threat+0.15*energy",
}


# ===========================================================================
# 3. NumPy compressed arrays (.npz)
# ===========================================================================

def export_numpy(cols: list[str], arrays: dict) -> None:
    logger.info("[3] Exporting NumPy .npz ...")
    text_cols = {"surface_type", "obstacle_type"}
    numeric = {c: arrays[c] for c in cols if c not in text_cols}
    path = EXPORT_DIR / "perception_full_dataset.npz"
    np.savez_compressed(str(path), **numeric)
    record(path, "NumPy/NPZ", "Numeric columns only, compressed")

    # 3D reshaped arrays per layer (alt, lat, lon)
    lats_u = np.unique(arrays["lat"])[::-1]   # descending
    lons_u = np.unique(arrays["lon"])
    alts_u = np.unique(arrays["alt_m"])
    nl, nlo, na = len(lats_u), len(lons_u), len(alts_u)

    path3d = EXPORT_DIR / "perception_3d_arrays.npz"
    out = {}
    lat_idx = {v: i for i, v in enumerate(lats_u)}
    lon_idx = {v: i for i, v in enumerate(lons_u)}
    alt_idx = {v: i for i, v in enumerate(alts_u)}

    key_cols = [
        "elev_m", "slope_deg", "wind_speed_mps", "wind_dir_deg",
        "nearest_obstacle_dist_m", "combined_threat_prob", "fused_cost",
    ]
    arrs3d = {c: np.full((na, nl, nlo), np.nan, dtype=np.float32) for c in key_cols}

    for k in range(N_ROWS):
        la = arrays["lat"][k]
        lo = arrays["lon"][k]
        alt = arrays["alt_m"][k]
        ai_ = alt_idx[alt]
        li_ = lat_idx[la]
        loi = lon_idx[lo]
        for c in key_cols:
            arrs3d[c][ai_, li_, loi] = arrays[c][k]

    out["lats"] = lats_u
    out["lons"] = lons_u
    out["alts"] = alts_u
    for c in key_cols:
        out[c] = arrs3d[c]

    np.savez_compressed(str(path3d), **out)
    record(path3d, "NumPy/NPZ", "3D arrays (alt, lat, lon) for key columns")


# ===========================================================================
# 4. Summary Statistics CSV + JSON
# ===========================================================================

def export_statistics(cols: list[str], arrays: dict) -> None:
    logger.info("[4] Exporting statistics ...")
    text_cols = {"surface_type", "obstacle_type"}
    stats_rows = []
    for col in cols:
        if col in text_cols:
            continue
        v = arrays[col]
        stats_rows.append({
            "column": col,
            "unit": _units.get(col, ""),
            "n": int(len(v)),
            "mean": float(np.nanmean(v)),
            "std":  float(np.nanstd(v)),
            "min":  float(np.nanmin(v)),
            "p5":   float(np.nanpercentile(v, 5)),
            "p25":  float(np.nanpercentile(v, 25)),
            "p50":  float(np.nanmedian(v)),
            "p75":  float(np.nanpercentile(v, 75)),
            "p95":  float(np.nanpercentile(v, 95)),
            "max":  float(np.nanmax(v)),
            "nan_pct": float(np.sum(np.isnan(v)) / len(v) * 100),
        })

    # CSV
    path_csv = EXPORT_DIR / "dataset_statistics.csv"
    with open(path_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=stats_rows[0].keys())
        writer.writeheader()
        writer.writerows(stats_rows)
    record(path_csv, "CSV", "Descriptive statistics for all numeric columns")

    # JSON
    path_json = EXPORT_DIR / "dataset_statistics.json"
    meta = json.loads(META_PATH.read_text())
    export_doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_provenance": meta,
        "column_statistics": stats_rows,
        "column_definitions": _descriptions,
        "column_units": _units,
    }
    with open(path_json, "w", encoding="utf-8") as fh:
        json.dump(export_doc, fh, indent=2)
    record(path_json, "JSON", "Statistics + provenance, machine-readable")


# ===========================================================================
# 5. LaTeX table of statistics (.tex)
# ===========================================================================

def export_latex_stats(cols: list[str], arrays: dict) -> None:
    logger.info("[5] Exporting LaTeX statistics table ...")
    text_cols = {"surface_type", "obstacle_type"}
    rows = []
    for col in cols:
        if col in text_cols:
            continue
        v = arrays[col]
        rows.append((
            col.replace("_", r"\_"),
            _units.get(col, "—").replace("/", r"/"),
            f"{np.nanmean(v):.3f}",
            f"{np.nanstd(v):.3f}",
            f"{np.nanmin(v):.3f}",
            f"{np.nanmax(v):.3f}",
        ))

    tex = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{geometry}
\geometry{margin=1.5cm}
\begin{document}

\begin{center}
\textbf{Perception Dataset Statistics — Delhi NCR Theatre}\\
\textit{1,057,714 rows $\times$ 28 columns, five perception layers}
\end{center}

\begin{longtable}{lrrrrr}
\toprule
\textbf{Column} & \textbf{Unit} & \textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Max} \\
\midrule
\endfirsthead
\toprule
\textbf{Column} & \textbf{Unit} & \textbf{Mean} & \textbf{Std} & \textbf{Min} & \textbf{Max} \\
\midrule
\endhead
\bottomrule
\endfoot
"""
    for col_name, unit, mean, std, mn, mx in rows:
        tex += f"{col_name} & {unit} & {mean} & {std} & {mn} & {mx} \\\\\n"
    tex += r"""
\end{longtable}

\section*{Dataset Provenance}
\begin{itemize}
  \item \textbf{Terrain}: NASA SRTM 30\,m via Open-Meteo Elevation API (EGM96 vertical datum)
  \item \textbf{Wind}: GFS/HRES forecast via Open-Meteo Weather API (t=0 snapshot)
  \item \textbf{Obstacles}: OpenStreetMap contributors (ODbL), 18,690 structures
  \item \textbf{Threat}: Analytical Swerling-I radar range equation [Skolnik 2008]
  \item \textbf{Fusion}: Weighted linear sum $C = \sum_i w_i C_i$
\end{itemize}

\begin{thebibliography}{9}
\bibitem{skolnik2008} Skolnik, M.~I. (2008). \textit{Radar Handbook}, 3rd Ed. McGraw-Hill.
\bibitem{rodriguez2006} Rodriguez, E., Morris, C.~S., Belz, J.~E. (2006). A Global Assessment of the SRTM Performance. \textit{PE\&RS}, 72(3):249--260.
\bibitem{swerling1954} Swerling, P. (1954). Probability of Detection for Fluctuating Targets. RAND RM-1217.
\bibitem{osm2024} OpenStreetMap contributors (2024). Planet dump. ODbL.
\end{thebibliography}

\end{document}
"""
    path = EXPORT_DIR / "dataset_statistics.tex"
    path.write_text(tex, encoding="utf-8")
    record(path, "LaTeX", "Statistics table suitable for journal papers")


# ===========================================================================
# 6. BibTeX Citation File
# ===========================================================================

def export_bibtex() -> None:
    logger.info("[6] Exporting BibTeX ...")
    bib = textwrap.dedent(f"""\
    @misc{{evtol_perception_dataset_2026,
      title        = {{Defense {{eVTOL}} Perception Dataset: Delhi {{NCR}} Theatre}},
      author       = {{Defense eVTOL Research Team}},
      year         = {{2026}},
      month        = {{4}},
      howpublished = {{Generated dataset, 1,057,714 rows}},
      note         = {{Terrain: NASA SRTM/Open-Meteo;
                       Wind: GFS/HRES/Open-Meteo;
                       Obstacles: OpenStreetMap (ODbL);
                       Threat: Swerling-I analytical model}},
    }}

    @article{{rodriguez2006srtm,
      title   = {{A Global Assessment of the {{SRTM}} Topographic Products}},
      author  = {{Rodriguez, Ernesto and Morris, Charles S and Belz, J Esteban}},
      journal = {{Photogrammetric Engineering \\& Remote Sensing}},
      volume  = {{72}},
      number  = {{3}},
      pages   = {{249--260}},
      year    = {{2006}},
      publisher={{American Society for Photogrammetry and Remote Sensing}},
    }}

    @book{{skolnik2008radar,
      title     = {{Radar Handbook}},
      author    = {{Skolnik, Merrill I}},
      edition   = {{3rd}},
      year      = {{2008}},
      publisher = {{McGraw-Hill}},
      address   = {{New York}},
    }}

    @article{{swerling1960probability,
      title   = {{Probability of Detection for Fluctuating Targets}},
      author  = {{Swerling, Peter}},
      journal = {{IRE Transactions on Information Theory}},
      volume  = {{6}},
      number  = {{2}},
      pages   = {{269--308}},
      year    = {{1960}},
      publisher={{IEEE}},
    }}

    @misc{{openstreetmap2024,
      title        = {{OpenStreetMap}},
      author       = {{OpenStreetMap contributors}},
      year         = {{2024}},
      howpublished = {{\\url{{https://www.openstreetmap.org}}}},
      note         = {{Data licensed under the Open Database License (ODbL)}},
    }}

    @misc{{openmeteo2024,
      title        = {{Open-Meteo API}},
      author       = {{Zippenfenig, Patrick}},
      year         = {{2024}},
      howpublished = {{\\url{{https://open-meteo.com}}}},
      note         = {{Free weather API providing GFS/HRES/ECMWF forecast data}},
    }}
    """)
    path = EXPORT_DIR / "references.bib"
    path.write_text(bib, encoding="utf-8")
    record(path, "BibTeX", "Citation file for all data sources and key references")


# ===========================================================================
# 7. JSON-LD Metadata (Linked Data / schema.org)
# ===========================================================================

def export_jsonld() -> None:
    logger.info("[7] Exporting JSON-LD metadata ...")
    meta = json.loads(META_PATH.read_text())
    doc = {
        "@context": "https://schema.org/",
        "@type": "Dataset",
        "name": "Defense eVTOL Perception Dataset — Delhi NCR Theatre",
        "description": (
            "Multi-layer perception dataset for defense eVTOL trajectory optimization. "
            "1,057,714 rows covering terrain (SRTM), wind (GFS/HRES), obstacles (OSM), "
            "and SAM threat assessment (Swerling-I) over the Delhi NCR theatre."
        ),
        "url": "https://github.com/evtol-research/trajectory-optimization-in-defense-evtols",
        "version": "2.0.0",
        "datePublished": meta["generated_at"],
        "creator": {
            "@type": "Organization",
            "name": "Defense eVTOL Research Team",
        },
        "spatialCoverage": {
            "@type": "Place",
            "name": meta["theatre"],
            "geo": {
                "@type": "GeoShape",
                "box": f"{meta['lat_bounds'][0]} {meta['lon_bounds'][0]} "
                       f"{meta['lat_bounds'][1]} {meta['lon_bounds'][1]}",
            },
        },
        "temporalCoverage": meta["generated_at"][:10],
        "variableMeasured": [
            {"@type": "PropertyValue", "name": c, "unitText": _units.get(c, "")}
            for c in _units
        ],
        "license": "https://opendatacommons.org/licenses/odbl/1-0/",
        "isBasedOn": [
            {"@type": "CreativeWork", "name": "NASA SRTM",
             "url": "https://www2.jpl.nasa.gov/srtm/"},
            {"@type": "CreativeWork", "name": "Open-Meteo API",
             "url": "https://open-meteo.com"},
            {"@type": "CreativeWork", "name": "OpenStreetMap",
             "url": "https://www.openstreetmap.org"},
        ],
        "keywords": [
            "eVTOL", "UAM", "trajectory optimization", "threat assessment",
            "terrain perception", "wind forecasting", "obstacle detection",
            "SAM avoidance", "defense aviation", "Delhi NCR",
        ],
        "distribution": [
            {
                "@type": "DataDownload",
                "encodingFormat": "text/csv",
                "contentUrl": "perception_full_dataset.csv",
            },
            {
                "@type": "DataDownload",
                "encodingFormat": "application/x-parquet",
                "contentUrl": "exports/perception_full_dataset.parquet",
            },
            {
                "@type": "DataDownload",
                "encodingFormat": "application/x-hdf",
                "contentUrl": "exports/perception_full_dataset.h5",
            },
        ],
        "measurementTechnique": [
            "SRTM interferometric SAR (terrain elevation)",
            "GFS/HRES numerical weather prediction (wind)",
            "OpenStreetMap Overpass API (obstacle inventory)",
            "Swerling Case I radar range equation (threat probability)",
        ],
    }
    path = EXPORT_DIR / "dataset_metadata.jsonld"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    record(path, "JSON-LD", "Schema.org Dataset descriptor for semantic web / Zenodo")


# ===========================================================================
# 8. HTML Statistics Report
# ===========================================================================

def export_html_report(cols: list[str], arrays: dict) -> None:
    logger.info("[8] Exporting HTML statistics report ...")
    text_cols = {"surface_type", "obstacle_type"}
    rows_html = ""
    for col in cols:
        if col in text_cols:
            continue
        v = arrays[col]
        rows_html += (
            f"<tr><td>{col}</td><td>{_units.get(col,'')}</td>"
            f"<td>{np.nanmean(v):.4f}</td>"
            f"<td>{np.nanstd(v):.4f}</td>"
            f"<td>{np.nanmin(v):.4f}</td>"
            f"<td>{np.nanmax(v):.4f}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Perception Dataset — Statistics Report</title>
<style>
body {{ font-family: Georgia, serif; margin: 40px; max-width: 1100px; }}
h1 {{ color: #1a3a5c; }}
h2 {{ color: #2c5f8a; border-bottom: 2px solid #ccc; padding-bottom: 6px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th {{ background: #1a3a5c; color: white; padding: 8px 12px; text-align: left; }}
tr:nth-child(even) {{ background: #f5f7fa; }}
td {{ padding: 6px 12px; border-bottom: 1px solid #ddd; }}
.meta {{ background: #eef4fb; padding: 16px; border-left: 4px solid #2c5f8a; margin-bottom: 20px; }}
code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; font-size: 12px; }}
</style>
</head>
<body>
<h1>Defense eVTOL — Perception Dataset Statistics Report</h1>
<div class="meta">
  <strong>Theatre:</strong> {THEATRE}<br>
  <strong>Bounds:</strong> {LAT_BOUNDS[0]}&ndash;{LAT_BOUNDS[1]}&deg;N, {LON_BOUNDS[0]}&ndash;{LON_BOUNDS[1]}&deg;E<br>
  <strong>Total rows:</strong> {N_ROWS:,} &nbsp;|&nbsp; <strong>Spatial points:</strong> {N_SPATIAL:,} &nbsp;|&nbsp; <strong>Altitude levels:</strong> {len(ALT_LEVELS)}<br>
  <strong>Generated:</strong> {GENERATED_AT}<br>
  <strong>Sources:</strong> NASA SRTM (Open-Meteo), GFS/HRES (Open-Meteo), OpenStreetMap (ODbL), Swerling-I analytical
</div>

<h2>Column Statistics</h2>
<table>
<thead><tr><th>Column</th><th>Unit</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr></thead>
<tbody>
{rows_html}</tbody>
</table>

<h2>Data Sources</h2>
<ul>
  <li><strong>Terrain</strong>: NASA SRTM 30m via Open-Meteo Elevation API. Elevation range: 190&ndash;305 m MSL. Bilinear interpolation to 222 m grid.</li>
  <li><strong>Wind</strong>: GFS/HRES forecast via Open-Meteo Weather API. t=0 snapshot. 14 altitude bands. Mean speed: 6.2 m/s.</li>
  <li><strong>Obstacles</strong>: OpenStreetMap Overpass API. 18,690 structures: buildings (42%), power pylons (47%), towers, chimneys, religious structures.</li>
  <li><strong>Threat</strong>: Analytical Swerling Case I radar range equation. 3 SAM systems (S-300V, SA-11, SA-22). P_fa=10<sup>&minus;6</sup>.</li>
  <li><strong>Fusion</strong>: Weighted sum with weights (terrain=0.15, wind=0.10, obstacle=0.20, threat=0.40, energy=0.15).</li>
</ul>

<h2>Radar Equation (Threat Layer)</h2>
<p>
Detection probability under Swerling Case I (many equal-amplitude scatterers, scan-to-scan fluctuation):<br>
<code>P_d = P_fa ^ (1 / (1 + SNR))</code><br>
<code>SNR(R) = SNR_max * (R_max / R)^4</code><br>
where <code>SNR_max = log(P_fa) / log(0.9) - 1 &asymp; 13.15</code> sets P_d=0.9 at R_max.
</p>

<h2>References</h2>
<ol>
  <li>Rodriguez et al. (2006). A Global Assessment of the SRTM Performance. <em>PE&amp;RS</em> 72(3):249&ndash;260.</li>
  <li>Skolnik, M.I. (2008). <em>Radar Handbook</em>, 3rd Ed. McGraw-Hill.</li>
  <li>Swerling, P. (1960). Probability of Detection for Fluctuating Targets. <em>IRE Trans. IT</em> 6(2):269&ndash;308.</li>
  <li>OpenStreetMap contributors (2024). ODbL.</li>
  <li>Open-Meteo API (2024). https://open-meteo.com</li>
</ol>
</body>
</html>
"""
    path = EXPORT_DIR / "dataset_statistics_report.html"
    path.write_text(html, encoding="utf-8")
    record(path, "HTML", "Human-readable statistics report (browser viewable)")


# ===========================================================================
# 9. YAML Dataset Card
# ===========================================================================

def export_yaml_card() -> None:
    logger.info("[9] Exporting YAML dataset card ...")
    card = textwrap.dedent(f"""\
    # Dataset Card — Defense eVTOL Perception Dataset

    dataset_name: perception_delhi_ncr_2026
    version: "2.0.0"
    generated_at: "{GENERATED_AT}"

    description: >
      Multi-layer environmental perception dataset for defense eVTOL trajectory
      optimization. Covers terrain, wind, obstacle, threat, and fused cost fields
      over the Delhi NCR operational theatre.

    theatre:
      name: "Delhi NCR outskirts, India"
      lat_bounds: [{LAT_BOUNDS[0]}, {LAT_BOUNDS[1]}]
      lon_bounds: [{LON_BOUNDS[0]}, {LON_BOUNDS[1]}]
      area_km2: 3025  # ~55 km × 55 km

    grid:
      lat_spacing_deg: 0.002
      lon_spacing_deg: 0.002
      lat_spacing_m: 222
      lon_spacing_m: 195
      n_lat: 251
      n_lon: 301
      n_spatial_points: 75551
      n_altitude_levels: {len(ALT_LEVELS)}
      altitude_levels_m: {ALT_LEVELS}
      total_rows: {N_ROWS}
      n_columns: {N_COLS}

    data_sources:
      terrain:
        name: "NASA SRTM (Open-Meteo Elevation API)"
        url: "https://open-meteo.com/en/docs/elevation-api"
        resolution_m: 30
        vertical_datum: "EGM96"
        accuracy_m_1sigma: 16
        license: "Public Domain (NASA)"
      wind:
        name: "GFS/HRES (Open-Meteo Forecast API)"
        url: "https://open-meteo.com/en/docs"
        model: "NOAA GFS 0.25deg + ECMWF HRES 0.1deg"
        forecast_hours: 6
        license: "CC BY 4.0"
      obstacles:
        name: "OpenStreetMap"
        url: "https://www.openstreetmap.org"
        license: "ODbL 1.0"
        n_obstacles: 18690
      threat:
        name: "Analytical (Swerling Case I)"
        model: "Radar range equation, Swerling 1960"
        n_sam_systems: 3
        evtol_rcs_sqm: 0.5

    quality_notes:
      - "Surface type = flat for all points: correct for Gangetic alluvial plain (slope<2.5 deg)"
      - "Combined threat Pd ~= 1 everywhere: theatre fully inside SAM envelopes (min range 29.6 km vs R_max 100-150 km)"
      - "Wind: single t=0 forecast snapshot, no temporal ensemble"

    usage:
      planning_layer:
        - "fused_cost: edge cost for RRT* planner"
        - "combined_threat_prob: SAM avoidance constraints"
        - "nearest_obstacle_dist_m: static obstacle list"
        - "wind_u/v_mps: wind injection for vehicle simulation"
      ml_training:
        - "Cost classification: terrain/obstacle/threat contribution prediction"
        - "Safe-zone segmentation (binary: fused_cost < 0.5)"
        - "Wind field interpolation networks"

    citation: >
      Defense eVTOL Research Team (2026). Defense eVTOL Perception Dataset.
      Delhi NCR Theatre. Generated 2026-04-04. 5-layer perception dataset,
      1,057,714 rows, sources: NASA SRTM, GFS/HRES, OpenStreetMap (ODbL),
      Swerling-I analytical threat model.

    license: "ODbL 1.0 (obstacle component) + CC BY 4.0 (wind component) + Public Domain (terrain)"
    """)
    path = EXPORT_DIR / "dataset_card.yaml"
    path.write_text(card, encoding="utf-8")
    record(path, "YAML", "HuggingFace-style dataset card")


# ===========================================================================
# 10. Manifest
# ===========================================================================

def export_manifest() -> None:
    path = EXPORT_DIR / "MANIFEST.json"
    manifest_doc = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_files": len(MANIFEST),
        "total_size_bytes": sum(m["size_bytes"] for m in MANIFEST),
        "files": MANIFEST,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest_doc, fh, indent=2)
    logger.info("[MANIFEST] %s  (%d files)", path.name, len(MANIFEST))


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    t0 = time.time()
    logger.info("=" * 65)
    logger.info("MULTI-FORMAT SCIENTIFIC EXPORT PIPELINE")
    logger.info("Output directory: %s", EXPORT_DIR)
    logger.info("=" * 65)

    # Load data once
    logger.info("Loading full dataset into memory (%d rows) ...", N_ROWS)
    cols, arrays = csv_to_numpy_arrays()
    logger.info("  Loaded. Memory: %.1f MB",
                sum(a.nbytes for a in arrays.values() if hasattr(a, "nbytes")) / 1e6)

    # Run all exports
    export_parquet(cols, arrays)
    export_hdf5(cols, arrays)
    export_numpy(cols, arrays)
    export_statistics(cols, arrays)
    export_latex_stats(cols, arrays)
    export_bibtex()
    export_jsonld()
    export_html_report(cols, arrays)
    export_yaml_card()
    export_manifest()

    elapsed = time.time() - t0
    total_mb = sum(m["size_bytes"] for m in MANIFEST) / 1e6
    logger.info("=" * 65)
    logger.info("DONE in %.1fs  —  %d files  —  %.1f MB total",
                elapsed, len(MANIFEST), total_mb)
    logger.info("Output: %s", EXPORT_DIR)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
