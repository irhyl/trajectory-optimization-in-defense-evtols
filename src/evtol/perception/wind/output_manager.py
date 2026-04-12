"""
Wind Output Manager.

Handles persistence, export, and serialisation of wind field data.
Mirrors the terrain output_manager interface.

Supported export formats:
  - JSON  : Lightweight metadata + summary statistics
  - CSV   : Per-altitude-level wind profile
  - NumPy : Full 3-D field arrays (.npz)
  - Parquet: Tabular per-point wind data for dataset integration
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WindExportConfig:
    """Configuration for wind data export."""
    output_dir:     str  = "outputs/perception/wind"
    prefix:         str  = "wind_field"
    save_json:      bool = True
    save_csv:       bool = True
    save_npz:       bool = True
    save_parquet:   bool = False


@dataclass
class WindFieldSummary:
    """Summary statistics of a wind field."""
    # Spatial extent
    x_min_m:    float = 0.0
    x_max_m:    float = 0.0
    y_min_m:    float = 0.0
    y_max_m:    float = 0.0
    z_min_m:    float = 0.0
    z_max_m:    float = 0.0

    # Wind statistics
    speed_mean_ms:   float = 0.0
    speed_max_ms:    float = 0.0
    speed_p95_ms:    float = 0.0
    u_mean_ms:       float = 0.0
    v_mean_ms:       float = 0.0
    w_mean_ms:       float = 0.0

    # Grid info
    n_points:        int   = 0
    grid_shape:      tuple = (0, 0, 0)
    data_source:     str   = "unknown"
    timestamp:       str   = ""


class WindOutputManager:
    """
    Manages wind field output, persistence, and export.

    Used by the perception layer wind submodule to persist wind data
    alongside terrain and threat data.
    """

    def __init__(self, config: Optional[WindExportConfig] = None):
        self.config = config or WindExportConfig()
        self._outdir = Path(self.config.output_dir)
        self._outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Compute summary
    # ------------------------------------------------------------------

    def compute_summary(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        data_source: str = "open_meteo",
    ) -> WindFieldSummary:
        """Compute summary statistics for a 3-D wind field."""
        speed = np.sqrt(u**2 + v**2 + w**2)
        import datetime
        return WindFieldSummary(
            x_min_m=float(x.min()), x_max_m=float(x.max()),
            y_min_m=float(y.min()), y_max_m=float(y.max()),
            z_min_m=float(z.min()), z_max_m=float(z.max()),
            speed_mean_ms=float(speed.mean()),
            speed_max_ms=float(speed.max()),
            speed_p95_ms=float(np.percentile(speed, 95)),
            u_mean_ms=float(u.mean()),
            v_mean_ms=float(v.mean()),
            w_mean_ms=float(w.mean()),
            n_points=int(speed.size),
            grid_shape=tuple(u.shape),
            data_source=data_source,
            timestamp=datetime.datetime.utcnow().isoformat(),
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        data_source: str = "open_meteo",
    ) -> dict[str, Path]:
        """
        Save wind field data in configured formats.

        Args:
            x, y, z: Coordinate axes (1-D arrays, m)
            u, v, w: Wind components on 3-D grid (m/s)
            data_source: Tag for provenance

        Returns:
            Dict mapping format name to saved file path.
        """
        summary = self.compute_summary(x, y, z, u, v, w, data_source)
        saved: dict[str, Path] = {}
        prefix = self._outdir / self.config.prefix

        if self.config.save_json:
            p = prefix.with_suffix(".json")
            with open(p, "w") as f:
                json.dump(asdict(summary), f, indent=2)
            saved["json"] = p
            logger.info("Wind field summary -> %s", p)

        if self.config.save_npz:
            p = Path(str(prefix) + "_arrays.npz")
            np.savez_compressed(str(p), x=x, y=y, z=z, u=u, v=v, w=w)
            saved["npz"] = p
            logger.info("Wind field arrays  -> %s", p)

        if self.config.save_csv:
            # Export per-altitude profile (averaged over horizontal plane)
            records = []
            for k, alt in enumerate(z):
                speed_layer = np.sqrt(u[:, :, k]**2 + v[:, :, k]**2)
                records.append({
                    "altitude_m":     float(alt),
                    "u_mean_ms":      float(u[:, :, k].mean()),
                    "v_mean_ms":      float(v[:, :, k].mean()),
                    "w_mean_ms":      float(w[:, :, k].mean()),
                    "speed_mean_ms":  float(speed_layer.mean()),
                    "speed_max_ms":   float(speed_layer.max()),
                })
            p = prefix.with_suffix(".csv")
            pd.DataFrame(records).to_csv(p, index=False)
            saved["csv"] = p
            logger.info("Wind field profile -> %s", p)

        if self.config.save_parquet:
            # Full point cloud (may be large)
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            df = pd.DataFrame({
                "x_m": X.ravel(), "y_m": Y.ravel(), "z_m": Z.ravel(),
                "u_ms": u.ravel(), "v_ms": v.ravel(), "w_ms": w.ravel(),
                "speed_ms": np.sqrt(u**2 + v**2 + w**2).ravel(),
            })
            p = prefix.with_suffix(".parquet")
            df.to_parquet(p, index=False)
            saved["parquet"] = p
            logger.info("Wind field parquet -> %s", p)

        return saved

    def load_npz(self, path: str | Path) -> dict:
        """Load wind arrays from a previously saved .npz file."""
        data = np.load(str(path))
        return {k: data[k] for k in data.files}
