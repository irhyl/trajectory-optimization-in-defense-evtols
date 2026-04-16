"""
ML Baseline Experiments for NeurIPS Submission
===============================================

Runs 3 core ML tasks on the defense eVTOL dataset:

  Task 1: Risk label classification (binary)
          Dataset  : planning layer (2,000 rows)
          Target   : risk_label (0=low-risk, 1=high-risk)
          Features : path geometry + cost field means
          Models   : LogisticRegression, GradientBoostingClassifier, MLP

  Task 2: Energy consumption regression
          Dataset  : control layer (2,000 rows)
          Target   : energy_consumed_wh
          Features : mission parameters + control performance metrics
          Models   : Ridge, GradientBoostingRegressor, MLP

  Task 3: Mission abort classification (binary)
          Dataset  : control layer (2,000 rows)
          Target   : mission_abort (1 = abort triggered)
          Features : mission parameters + control metrics
          Models   : LogisticRegression, GradientBoostingClassifier, MLP

Each task uses:
  - 5-fold stratified cross-validation (classification) or
    5-fold KFold (regression)
  - Metrics: AUC-ROC + F1 (classification), R² + MAE (regression)
  - StandardScaler preprocessing

Results are printed to stdout and saved to
  outputs/ml/baseline_results.csv

Usage
-----
  python scripts/ml/baseline_experiments.py
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    cross_val_predict,
    cross_validate,
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
OUTPUTS      = REPO_ROOT / "outputs"
CTRL_PATH    = OUTPUTS / "delhi" / "control" / "control_dataset.parquet"
ML_OUT       = OUTPUTS / "delhi" / "ml"
SPLITS_OUT   = OUTPUTS / "delhi" / "splits"

# Prefer delhi/ subdirectory (canonical), fall back to legacy flat paths
NEW_PLAN_PATH = OUTPUTS / "delhi" / "planning_dataset" / "planning_dataset_10k.parquet"
PLAN_PATH     = OUTPUTS / "planning_dataset" / "planning_dataset_10k.parquet"
_LEGACY_PATH  = OUTPUTS / "planning_dataset" / "test_final.parquet"


# ---------------------------------------------------------------------------
# Feature definitions per task
# ---------------------------------------------------------------------------
# Task 1: Risk classification on planning dataset.
# NOTE: fused_cost_mean is deliberately EXCLUDED because risk_label is defined
# as int(fused_cost_mean >= 0.55).  Including it would make the task trivial
# (AUC → 1.0).  Instead we use the raw component costs and path geometry so
# that models must learn a non-trivial mapping.
PLAN_FEATURES = [
    "path_length_m",
    "n_waypoints",
    "time_cost_s",
    "energy_cost_wh",
    "threat_cost",
    "terrain_cost_mean",
    "wind_cost_mean",
    "obstacle_cost_mean",
    # Deliberately exclude: fused_cost_mean, max_combined_threat
    # (fused_cost_mean directly encodes risk_label; max_combined_threat saturates at 1.0)
]

# Task 2: Energy regression.
# Uses only PLANNING-layer features so prediction flows from trajectory geometry
# to closed-loop energy — no circular control→control prediction.
#
# Features excluded and why:
#   mission_time_s          : energy ≈ power × time  (near-identity, r=0.9995)
#   thrust_cmd_mean_N       : from same control sim as target
#   motor_T_mean_N          : from same control sim as target
#   pwm_mean_us             : from same control sim as target
#   hover_frac_ctrl         : control-layer output predicting control-layer output (circular)
#   transition_frac_ctrl    : same
#   cruise_frac_ctrl        : same
#   soc_initial             : constant (always 1.0) across dataset
#
# The retained features (path geometry + speed + altitude) give R²=0.997 via GBM,
# which is the honest physical signal: energy ∝ path_length × f(speed, altitude).
ENERGY_FEATURES = [
    "path_length_m",          # primary energy driver (longer path → more energy)
    "n_waypoints",            # proxy for route complexity / total flight time
    "time_cost_s",            # planning-layer estimated flight time
    "cruise_speed_ref_ms",    # speed determines aerodynamic power draw
    "cruise_altitude_ref_m",  # altitude affects air density and rotor efficiency
    "threat_cost",            # high-threat routes tend to be longer/lower
    "terrain_cost_mean",      # rough terrain → lower cruise altitude → more power
    "risk_label",             # encodes high-cost trajectory characteristics
    # Deliberately exclude: mission_time_s, thrust_cmd_mean_N, motor_T_mean_N,
    # pwm_mean_us, hover_frac_ctrl, transition_frac_ctrl, cruise_frac_ctrl, soc_initial
]

# Task 3: Abort classification on control dataset
ABORT_FEATURES = [
    "path_length_m",
    "mission_time_s",
    "cruise_speed_ref_ms",
    "cruise_altitude_ref_m",
    "n_waypoints",
    "feasible",
    "hover_frac_ctrl",
    "cruise_frac_ctrl",
    "pos_error_mean_m",
    "vel_error_mean_ms",
    "alt_error_mean_m",
    "att_error_mean_rad",
    "thrust_cmd_mean_N",
    "itae_pos",
    "itae_alt",
    "n_saturations",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_df(path: Path, fallback: Path | None = None) -> pd.DataFrame:
    """Load parquet with optional fallback path."""
    if path.exists():
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df):,} rows from {path.name}")
        return df
    if fallback and fallback.exists():
        df = pd.read_parquet(fallback)
        logger.info(f"Loaded {len(df):,} rows from {fallback.name} (fallback)")
        return df
    raise FileNotFoundError(f"Dataset not found: {path}")


def _select_features(df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Select only existing feature columns (skip any absent)."""
    available = [f for f in feats if f in df.columns]
    missing = set(feats) - set(available)
    if missing:
        logger.warning(f"Missing features (skipping): {sorted(missing)}")
    return df[available].copy()


def _run_classification(
    X: np.ndarray,
    y: np.ndarray,
    models: dict,
    n_splits: int = 5,
    task_name: str = "",
) -> list[dict]:
    """Run cross-validated classification for each model in `models`."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for name, pipeline in models.items():
        logger.info(f"  [{task_name}] Fitting {name}...")
        t0 = time.perf_counter()

        # Predict probabilities/labels for each held-out fold
        y_prob = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc   = roc_auc_score(y, y_prob)
        f1    = f1_score(y, y_pred)
        acc   = accuracy_score(y, y_pred)
        elapsed = time.perf_counter() - t0

        results.append({
            "task":    task_name,
            "model":   name,
            "type":    "classification",
            "metric1": "AUC-ROC",
            "value1":  round(auc, 4),
            "metric2": "F1",
            "value2":  round(f1, 4),
            "metric3": "Accuracy",
            "value3":  round(acc, 4),
            "n_samples": len(y),
            "n_features": X.shape[1],
            "cv_folds": n_splits,
            "runtime_s": round(elapsed, 2),
        })
        logger.info(
            f"    AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}  "
            f"({elapsed:.1f}s)"
        )
    return results


def _run_regression(
    X: np.ndarray,
    y: np.ndarray,
    models: dict,
    n_splits: int = 5,
    task_name: str = "",
) -> list[dict]:
    """Run cross-validated regression for each model in `models`."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    for name, pipeline in models.items():
        logger.info(f"  [{task_name}] Fitting {name}...")
        t0 = time.perf_counter()

        y_pred = cross_val_predict(pipeline, X, y, cv=cv)

        r2  = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        elapsed = time.perf_counter() - t0

        results.append({
            "task":    task_name,
            "model":   name,
            "type":    "regression",
            "metric1": "R²",
            "value1":  round(r2, 4),
            "metric2": "MAE",
            "value2":  round(mae, 4),
            "metric3": "—",
            "value3":  None,
            "n_samples": len(y),
            "n_features": X.shape[1],
            "cv_folds": n_splits,
            "runtime_s": round(elapsed, 2),
        })
        logger.info(
            f"    R²={r2:.4f}  MAE={mae:.4f}  ({elapsed:.1f}s)"
        )
    return results


def _print_table(results: list[dict]) -> None:
    """Print a formatted results table to stdout."""
    header = f"{'Task':<28} {'Model':<28} {'Type':<16} {'Metric1':<12} {'Val1':>8} {'Metric2':<8} {'Val2':>8}"
    print()
    print("=" * len(header))
    print("ML BASELINE RESULTS — 5-Fold Cross-Validation")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['task']:<28} {r['model']:<28} {r['type']:<16} "
            f"{r['metric1']:<12} {r['value1']:>8.4f} {r['metric2']:<8} {r['value2']:>8.4f}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------
def task1_risk_classification(df: pd.DataFrame) -> list[dict]:
    """Task 1: Risk label binary classification on planning dataset."""
    task_name = "T1:RiskClassification"
    logger.info(f"\n--- {task_name} ---")
    logger.info(f"  Target: risk_label  n={len(df)}  "
                f"pos_rate={df['risk_label'].mean():.2%}")

    X_df = _select_features(df, PLAN_FEATURES)
    X = X_df.fillna(X_df.median()).to_numpy(dtype=np.float64)
    y = df["risk_label"].to_numpy(dtype=int)

    models = {
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
    return _run_classification(X, y, models, task_name=task_name)


def task2_energy_regression(df: pd.DataFrame) -> list[dict]:
    """Task 2: Energy consumption regression on control dataset."""
    task_name = "T2:EnergyRegression"
    logger.info(f"\n--- {task_name} ---")

    if "energy_consumed_wh" not in df.columns:
        logger.warning("energy_consumed_wh not found, skipping Task 2")
        return []

    logger.info(f"  Target: energy_consumed_wh  n={len(df)}  "
                f"mean={df['energy_consumed_wh'].mean():.1f} Wh")

    X_df = _select_features(df, ENERGY_FEATURES)
    X = X_df.fillna(X_df.median()).to_numpy(dtype=np.float64)
    y = df["energy_consumed_wh"].to_numpy(dtype=np.float64)

    models = {
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
    return _run_regression(X, y, models, task_name=task_name)


def task3_abort_classification(df: pd.DataFrame) -> list[dict]:
    """Task 3: Mission abort binary classification on control dataset."""
    task_name = "T3:AbortClassification"
    logger.info(f"\n--- {task_name} ---")

    if "mission_abort" not in df.columns:
        logger.warning("mission_abort not found, skipping Task 3")
        return []

    abort_rate = df["mission_abort"].mean()
    logger.info(f"  Target: mission_abort  n={len(df)}  "
                f"abort_rate={abort_rate:.2%}")

    if abort_rate < 0.02 or abort_rate > 0.98:
        logger.warning(
            f"  Abort rate {abort_rate:.2%} is nearly constant — "
            "classification will be trivial. Skipping."
        )
        return []

    X_df = _select_features(df, ABORT_FEATURES)
    X = X_df.fillna(X_df.median()).to_numpy(dtype=np.float64)
    y = df["mission_abort"].to_numpy(dtype=int)

    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42,
                                       class_weight="balanced")),
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
    return _run_classification(X, y, models, task_name=task_name)


# ---------------------------------------------------------------------------
# Train / val / test split utility
# ---------------------------------------------------------------------------
def create_splits(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    """
    Create stratified 80/10/10 train/val/test splits by risk_label.

    Saves index arrays to `out_dir/<name>_{train,val,test}_idx.npy`.
    """
    from sklearn.model_selection import train_test_split

    out_dir.mkdir(parents=True, exist_ok=True)

    if "risk_label" in df.columns:
        stratify_col = df["risk_label"].to_numpy()
    else:
        stratify_col = None

    n = len(df)
    idx = np.arange(n)

    # 80/20 split → then 50/50 of the 20% to get 10/10
    idx_train, idx_tmp = train_test_split(
        idx, test_size=0.20, random_state=42,
        stratify=stratify_col,
    )
    strat_tmp = stratify_col[idx_tmp] if stratify_col is not None else None
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=0.50, random_state=42,
        stratify=strat_tmp,
    )

    np.save(out_dir / f"{name}_train_idx.npy", idx_train)
    np.save(out_dir / f"{name}_val_idx.npy",   idx_val)
    np.save(out_dir / f"{name}_test_idx.npy",  idx_test)

    logger.info(
        f"  Splits saved: train={len(idx_train)} val={len(idx_val)} "
        f"test={len(idx_test)} -> {out_dir}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ML_OUT.mkdir(parents=True, exist_ok=True)
    SPLITS_OUT.mkdir(parents=True, exist_ok=True)

    # ---- Load datasets ----
    try:
        df_plan = _load_df(NEW_PLAN_PATH, fallback=PLAN_PATH)
    except FileNotFoundError:
        logger.error("No planning dataset found. Run scripts/planning/dataset.py first.")
        return

    try:
        df_ctrl = _load_df(CTRL_PATH)
    except FileNotFoundError:
        logger.error("No control dataset found. Run scripts/control/dataset.py first.")
        return

    # ---- Create train/val/test splits ----
    logger.info("\n[Splits] Creating 80/10/10 splits...")
    create_splits(df_plan, SPLITS_OUT, "planning")
    create_splits(df_ctrl, SPLITS_OUT, "control")

    # ---- Merge planning + control (row-aligned by mission index) ----
    # Combine so T2 can access planning-layer features (path geometry, costs)
    # alongside the control-layer target (energy_consumed_wh).
    shared_cols = list(set(df_plan.columns) & set(df_ctrl.columns))
    df_merged = pd.concat(
        [df_plan, df_ctrl.drop(columns=shared_cols, errors="ignore")],
        axis=1,
    )

    # ---- Run ML tasks ----
    all_results: list[dict] = []
    all_results.extend(task1_risk_classification(df_plan))
    all_results.extend(task2_energy_regression(df_merged))
    all_results.extend(task3_abort_classification(df_ctrl))

    if not all_results:
        logger.warning("No results produced — check dataset availability.")
        return

    # ---- Print and save results ----
    _print_table(all_results)

    df_results = pd.DataFrame(all_results)
    out_path = ML_OUT / "baseline_results.csv"
    df_results.to_csv(out_path, index=False)
    logger.info(f"\nResults saved to {out_path}")

    # Also save JSON for programmatic use
    (ML_OUT / "baseline_results.json").write_text(
        json.dumps(all_results, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
