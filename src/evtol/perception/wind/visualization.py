"""
Wind Field Visualization - publication-quality figures for wind data analysis.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    _MPL = True
except ImportError:
    _MPL = False

DPI, FMTS = 300, ["png", "pdf"]

def _save(fig, name, out_dir):
    for fmt in FMTS:
        fig.savefig(out_dir / f"{name}.{fmt}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

class WindVisualizer:
    """Generates wind field visualization figures."""

    def __init__(self, output_dir="visuals/wind"):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def plot_altitude_profile(self, z_m, speed_mean, speed_max=None, name="W1_altitude_profile"):
        if not _MPL: return
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.plot(speed_mean, z_m, "b-o", label="Mean")
        if speed_max is not None:
            ax.plot(speed_max, z_m, "r--", alpha=0.6, label="Max")
        ax.set_xlabel("Wind Speed (m/s)"); ax.set_ylabel("Altitude (m)")
        ax.set_title("Wind Speed Altitude Profile"); ax.legend(); ax.grid(alpha=0.3)
        _save(fig, name, self.out)

    def plot_horizontal_map(self, x_m, y_m, u, v, altitude_m=100.0, name="W2_horizontal_map"):
        if not _MPL: return
        speed = np.sqrt(u**2 + v**2)
        fig, ax = plt.subplots(figsize=(10, 8))
        cf = ax.contourf(x_m/1000, y_m/1000, speed.T, cmap="Blues", levels=20)
        plt.colorbar(cf, ax=ax, label="Wind Speed (m/s)")
        stride = max(1, len(x_m)//15)
        ax.quiver(x_m[::stride]/1000, y_m[::stride]/1000,
                  u[::stride,::stride].T, v[::stride,::stride].T,
                  scale=200, width=0.003, color="white", alpha=0.8)
        ax.set_xlabel("North (km)"); ax.set_ylabel("East (km)")
        ax.set_title(f"Wind at {altitude_m:.0f} m")
        _save(fig, name, self.out)

    def plot_wind_rose(self, u, v, name="W3_wind_rose"):
        if not _MPL: return
        dirs = np.arctan2(u.ravel(), v.ravel())
        spds = np.sqrt(u.ravel()**2 + v.ravel()**2)
        bins = np.linspace(-np.pi, np.pi, 17)
        freq = np.array([((dirs>=bins[i])&(dirs<bins[i+1])).sum() for i in range(16)], dtype=float)
        mean_s = np.array([spds[(dirs>=bins[i])&(dirs<bins[i+1])].mean() if ((dirs>=bins[i])&(dirs<bins[i+1])).any() else 0.0 for i in range(16)])
        freq /= freq.sum() + 1e-9
        angles = 0.5*(bins[:-1]+bins[1:])
        fig, ax = plt.subplots(figsize=(8,8), subplot_kw={"projection":"polar"})
        ax.bar(angles, freq, width=2*np.pi/16, color=cm.Blues(mean_s/(mean_s.max()+1e-9)), edgecolor="white")
        ax.set_title("Wind Rose", pad=20)
        _save(fig, name, self.out)

    def plot_turbulence_profile(self, z_m, ti, name="W4_turbulence_profile"):
        if not _MPL: return
        fig, ax = plt.subplots(figsize=(6,7))
        ax.fill_betweenx(z_m, 0, ti, alpha=0.4, color="orange")
        ax.plot(ti, z_m, "darkorange", lw=2)
        ax.axvline(0.05, ls="--", color="green", label="Low TI")
        ax.axvline(0.15, ls="--", color="red",   label="High TI")
        ax.set_xlabel("Turbulence Intensity"); ax.set_ylabel("Altitude (m)")
        ax.set_title("Turbulence Intensity Profile"); ax.legend(); ax.grid(alpha=0.3)
        _save(fig, name, self.out)

    def plot_energy_impact(self, x_m, y_m, headwind_ms, name="W5_energy_impact"):
        if not _MPL: return
        fig, ax = plt.subplots(figsize=(10,8))
        vmax = max(float(np.abs(headwind_ms).max()), 1.0)
        cf = ax.contourf(x_m/1000, y_m/1000, headwind_ms.T, cmap="RdBu_r", levels=21, vmin=-vmax, vmax=vmax)
        plt.colorbar(cf, ax=ax, label="Headwind (m/s)")
        ax.set_xlabel("North (km)"); ax.set_ylabel("East (km)")
        ax.set_title("Wind Energy Impact Map")
        _save(fig, name, self.out)

    def generate_all(self, x_m, y_m, z_m, u, v, w):
        speed = np.sqrt(u**2+v**2+w**2)
        self.plot_altitude_profile(z_m, speed.mean(axis=(0,1)), speed.max(axis=(0,1)))
        k = min(len(z_m)-1, len(z_m)//2)
        self.plot_horizontal_map(x_m, y_m, u[:,:,k], v[:,:,k], z_m[k])
        self.plot_wind_rose(u, v)
        ms = speed.mean(axis=(0,1))+1e-6
        self.plot_turbulence_profile(z_m, speed.std(axis=(0,1))/ms)
        hw = -(v[:,:,k]*1.0 + u[:,:,k]*0.0)
        self.plot_energy_impact(x_m, y_m, hw)
        logger.info("Wind visualizations complete -> %s", self.out)
