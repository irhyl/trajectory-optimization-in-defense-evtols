"""
ML Baseline Experiments — All Non-Delhi Regions
=================================================

Runs the same 3-task ML baseline suite (risk classification, energy regression,
abort classification) on each of the 5 non-Delhi regions:
  Mumbai | Bangalore | Arunachal Pradesh | Odisha | Ladakh

Results are saved per-region to:
  datasets/<region>/ml/baseline_results.csv
  datasets/<region>/ml/baseline_results.json

A combined summary across all regions is saved to:
  datasets/ml_all_regions_summary.csv

Usage
-----
  python scripts/ml/run_all_regions.py
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    accuracy_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=ConvergenceWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS  = REPO_ROOT / "datasets"

# Region config: name → (planning_parquet_glob, control_parquet)
REGIONS = {
    "mumbai":    ("planning_dataset/planning_mumbai.parquet",    "control/control_dataset.parquet"),
    "bangalore": ("planning_dataset/planning_bangalore.parquet", "control/control_dataset.parquet"),
    "arunachal": ("planning/planning_arunachal.parquet",         "control/control_dataset.parquet"),
    "odisha":    ("planning_dataset/planning_odisha.parquet",    "control/control_dataset.parquet"),
    "ladakh":    ("planning_dataset/planning_ladakh.parquet",    "control/control_dataset.parquet"),
}

# Feature sets (same as baseline_experiments.py)
PLAN_FEATURES = [
    "path_length_m", "n_waypoints", "time_cost_s", "energy_cost_wh",
    "threat_cost", "terrain_cost_mean", "wind_cost_mean", "obstacle_cost_mean",
]
ENERGY_FEATURES = [
    "path_length_m", "n_waypoints", "time_cost_s", "cruise_speed_ref_ms",
    "cruise_altitude_ref_m", "threat_cost", "terrain_cost_mean", "risk_label",
]
ABORT_FEATURES = [
    "path_length_m", "mission_time_s", "cruise_speed_ref_ms", "cruise_altitude_ref_m",
    "n_waypoints", "feasible", "hover_frac_ctrl", "cruise_frac_ctrl",
    "pos_error_mean_m", "vel_error_mean_ms", "alt_error_mean_m",
    "att_error_mean_rad", "thrust_cmd_mean_N", "itae_pos", "itae_alt", "n_saturations",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning(f"  Not found: {path}")
        return None
    df = pd.read_parquet(path)
    logger.info(f"  Loaded {len(df):,} rows from {path.name}")
    return df


def _select(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    available = [f for f in feats if f in df.columns]
    missing = set(feats) - set(available)
    if missing:
        logger.warning(f"  Missing features (skipped): {sorted(missing)}")
    return df[available].copy()


def _clf_models():
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                               learning_rate=0.05, random_state=42)),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                  learning_rate_init=0.001, random_state=42)),
        ]),
    }


def _reg_models():
    return {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                              learning_rate=0.05, random_state=42)),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                                 learning_rate_init=0.001, random_state=42)),
        ]),
    }


def run_classification(X, y, models, task_name, region):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for name, pipeline in models.items():
        t0 = time.perf_counter()
        y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y, y_prob)
        f1  = f1_score(y, y_pred)
        acc = accuracy_score(y, y_pred)
        elapsed = time.perf_counter() - t0
        results.append({
            "region": region, "task": task_name, "model": name, "type": "classification",
            "metric1": "AUC-ROC", "value1": round(auc, 4),
            "metric2": "F1",      "value2": round(f1, 4),
            "metric3": "Accuracy","value3": round(acc, 4),
            "n_samples": len(y), "n_features": X.shape[1],
            "cv_folds": 5, "runtime_s": round(elapsed, 2),
        })
        logger.info(f"    {name}: AUC={auc:.4f} F1={f1:.4f} Acc={acc:.4f} ({elapsed:.1f}s)")
    return results


def run_regression(X, y, models, task_name, region):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for name, pipeline in models.items():
        t0 = time.perf_counter()
        y_pred = cross_val_predict(pipeline, X, y, cv=cv)
        r2  = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        elapsed = time.perf_counter() - t0
        results.append({
            "region": region, "task": task_name, "model": name, "type": "regression",
            "metric1": "R²",  "value1": round(r2, 4),
            "metric2": "MAE", "value2": round(mae, 4),
            "metric3": "—",   "value3": None,
            "n_samples": len(y), "n_features": X.shape[1],
            "cv_folds": 5, "runtime_s": round(elapsed, 2),
        })
        logger.info(f"    {name}: R²={r2:.4f} MAE={mae:.4f} ({elapsed:.1f}s)")
    return results


def create_splits(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    strat = df["risk_label"].to_numpy() if "risk_label" in df.columns else None
    idx = np.arange(len(df))
    idx_train, idx_tmp = train_test_split(idx, test_size=0.20, random_state=42, stratify=strat)
    strat_tmp = strat[idx_tmp] if strat is not None else None
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, random_state=42, stratify=strat_tmp)
    np.save(out_dir / f"{name}_train_idx.npy", idx_train)
    np.save(out_dir / f"{name}_val_idx.npy",   idx_val)
    np.save(out_dir / f"{name}_test_idx.npy",  idx_test)
    logger.info(f"  Splits: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")


# ---------------------------------------------------------------------------
# Per-region runner
# ---------------------------------------------------------------------------
def run_region(region: str, plan_rel: str, ctrl_rel: str) -> list[dict]:
    logger.info(f"\n{'='*60}")
    logger.info(f"REGION: {region.upper()}")
    logger.info(f"{'='*60}")

    base = DATASETS / region
    df_plan = _load(base / plan_rel)
    df_ctrl = _load(base / ctrl_rel)

    if df_plan is None or df_ctrl is None:
        logger.error(f"  Skipping {region} — missing datasets")
        return []

    ml_dir = base / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = base / "splits"

    # Splits
    create_splits(df_plan, splits_dir, "planning")
    create_splits(df_ctrl, splits_dir, "control")

    # Merge for energy task (needs planning features + control target)
    shared = list(set(df_plan.columns) & set(df_ctrl.columns))
    df_merged = pd.concat([df_plan, df_ctrl.drop(columns=shared, errors="ignore")], axis=1)

    results: list[dict] = []

    # Task 1: Risk classification
    if "risk_label" in df_plan.columns:
        logger.info(f"\n--- T1:RiskClassification ({region}) ---")
        X = _select(df_plan, PLAN_FEATURES).fillna(df_plan.median(numeric_only=True)).to_numpy(dtype=np.float64)
        y = df_plan["risk_label"].to_numpy(dtype=int)
        logger.info(f"  n={len(y)}  pos_rate={y.mean():.2%}")
        results.extend(run_classification(X, y, _clf_models(), "T1:RiskClassification", region))
    else:
        logger.warning(f"  risk_label not found in {region} planning dataset — skipping T1")

    # Task 2: Energy regression
    if "energy_consumed_wh" in df_merged.columns:
        logger.info(f"\n--- T2:EnergyRegression ({region}) ---")
        X = _select(df_merged, ENERGY_FEATURES).fillna(df_merged.median(numeric_only=True)).to_numpy(dtype=np.float64)
        y = df_merged["energy_consumed_wh"].to_numpy(dtype=np.float64)
        logger.info(f"  n={len(y)}  mean={y.mean():.1f} Wh")
        results.extend(run_regression(X, y, _reg_models(), "T2:EnergyRegression", region))
    else:
        logger.warning(f"  energy_consumed_wh not found in {region} merged dataset — skipping T2")

    # Task 3: Abort classification
    if "mission_abort" in df_ctrl.columns:
        abort_rate = df_ctrl["mission_abort"].mean()
        if 0.02 < abort_rate < 0.98:
            logger.info(f"\n--- T3:AbortClassification ({region}) ---")
            X = _select(df_ctrl, ABORT_FEATURES).fillna(df_ctrl.median(numeric_only=True)).to_numpy(dtype=np.float64)
            y = df_ctrl["mission_abort"].to_numpy(dtype=int)
            results.extend(run_classification(X, y, _clf_models(), "T3:AbortClassification", region))
        else:
            logger.warning(f"  Abort rate {abort_rate:.2%} is near-constant — skipping T3")

    # Save per-region results
    if results:
        df_r = pd.DataFrame(results)
        df_r.to_csv(ml_dir / "baseline_results.csv", index=False)
        (ml_dir / "baseline_results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )
        logger.info(f"  Saved results to {ml_dir}")

    return results


# ---------------------------------------------------------------------------
# Cross-region comparison plots
# ---------------------------------------------------------------------------
def plot_cross_region_summary(all_results: list[dict], out_dir: Path) -> None:
    """Bar charts comparing AUC (T1) and R² (T2) across all regions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)

    IMAGE_FORMATS = ["png", "pdf", "svg", "eps"]

    # --- T1: AUC-ROC per region × model ---
    t1 = df[(df["task"] == "T1:RiskClassification") & (df["metric1"] == "AUC-ROC")]
    if not t1.empty:
        regions  = t1["region"].unique()
        models   = t1["model"].unique()
        x        = np.arange(len(regions))
        width    = 0.25
        fig, ax  = plt.subplots(figsize=(10, 5))
        for i, model in enumerate(models):
            vals = [t1[(t1["region"] == r) & (t1["model"] == model)]["value1"].values[0]
                    if len(t1[(t1["region"] == r) & (t1["model"] == model)]) > 0 else 0
                    for r in regions]
            ax.bar(x + i * width, vals, width, label=model)
        ax.set_xticks(x + width)
        ax.set_xticklabels([r.title() for r in regions])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUC-ROC")
        ax.set_title("T1: Risk Classification AUC-ROC — Cross-Region Comparison")
        ax.legend()
        ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="0.9 threshold")
        fig.tight_layout()
        for fmt in IMAGE_FORMATS:
            fig.savefig(out_dir / f"t1_auc_by_region.{fmt}", dpi=150, format=fmt)
        plt.close(fig)
        logger.info(f"  Saved T1 AUC comparison plot ({', '.join(IMAGE_FORMATS)})")

    # --- T2: R² per region × model ---
    t2 = df[(df["task"] == "T2:EnergyRegression") & (df["metric1"] == "R²")]
    if not t2.empty:
        regions  = t2["region"].unique()
        models   = t2["model"].unique()
        x        = np.arange(len(regions))
        width    = 0.25
        fig, ax  = plt.subplots(figsize=(10, 5))
        for i, model in enumerate(models):
            vals = [t2[(t2["region"] == r) & (t2["model"] == model)]["value1"].values[0]
                    if len(t2[(t2["region"] == r) & (t2["model"] == model)]) > 0 else 0
                    for r in regions]
            ax.bar(x + i * width, vals, width, label=model)
        ax.set_xticks(x + width)
        ax.set_xticklabels([r.title() for r in regions])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("R²")
        ax.set_title("T2: Energy Regression R² — Cross-Region Comparison")
        ax.legend()
        fig.tight_layout()
        for fmt in IMAGE_FORMATS:
            fig.savefig(out_dir / f"t2_r2_by_region.{fmt}", dpi=150, format=fmt)
        plt.close(fig)
        logger.info(f"  Saved T2 R² comparison plot ({', '.join(IMAGE_FORMATS)})")

    # --- Risk label distribution per region ---
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(14, 4), sharey=True)
    for ax, (region, (plan_rel, _)) in zip(axes, REGIONS.items()):
        df_p = _load(DATASETS / region / plan_rel)
        if df_p is not None and "risk_label" in df_p.columns:
            counts = df_p["risk_label"].value_counts().sort_index()
            ax.bar(["Low Risk\n(0)", "High Risk\n(1)"], counts.values,
                   color=["#4CAF50", "#F44336"])
            ax.set_title(region.title())
            ax.set_ylabel("Count")
    fig.suptitle("Risk Label Distribution — Non-Delhi Regions", fontsize=12)
    fig.tight_layout()
    for fmt in IMAGE_FORMATS:
        fig.savefig(out_dir / f"risk_label_distribution.{fmt}", dpi=150, format=fmt)
    plt.close(fig)
    logger.info(f"  Saved risk label distribution plot ({', '.join(IMAGE_FORMATS)})")

    # --- Energy distribution per region ---
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(14, 4), sharey=False)
    for ax, (region, (_, ctrl_rel)) in zip(axes, REGIONS.items()):
        df_c = _load(DATASETS / region / ctrl_rel)
        if df_c is not None and "energy_consumed_wh" in df_c.columns:
            ax.hist(df_c["energy_consumed_wh"].dropna(), bins=30, color="#2196F3", edgecolor="white", linewidth=0.3)
            ax.set_title(region.title())
            ax.set_xlabel("Energy (Wh)")
    fig.suptitle("Energy Consumed Distribution — Non-Delhi Regions", fontsize=12)
    fig.tight_layout()
    for fmt in IMAGE_FORMATS:
        fig.savefig(out_dir / f"energy_distribution.{fmt}", dpi=150, format=fmt)
    plt.close(fig)
    logger.info(f"  Saved energy distribution plot ({', '.join(IMAGE_FORMATS)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    all_results: list[dict] = []

    for region, (plan_rel, ctrl_rel) in REGIONS.items():
        results = run_region(region, plan_rel, ctrl_rel)
        all_results.extend(results)

    if not all_results:
        logger.error("No results produced across any region.")
        return

    # Save combined summary
    df_all = pd.DataFrame(all_results)
    summary_path = DATASETS / "ml_all_regions_summary.csv"
    df_all.to_csv(summary_path, index=False)
    (DATASETS / "ml_all_regions_summary.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )
    logger.info(f"\nCombined summary saved to {summary_path}")

    # Cross-region comparison plots
    plots_dir = DATASETS / "ml_plots"
    plot_cross_region_summary(all_results, plots_dir)

    # Print summary table
    print("\n" + "="*80)
    print("CROSS-REGION ML BASELINE SUMMARY")
    print("="*80)
    for _, row in df_all.iterrows():
        print(f"  {row['region']:<12} {row['task']:<28} {row['model']:<22} "
              f"{row['metric1']}={row['value1']:.4f}  {row['metric2']}={row['value2']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
