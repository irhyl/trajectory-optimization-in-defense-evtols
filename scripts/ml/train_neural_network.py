"""
train_neural_network.py
=======================
Exhaustive training and evaluation of the multi-task neural network on the
defense eVTOL trajectory optimization dataset.

Tasks
-----
T1 : risk_label          — binary classification  (AUC-ROC, F1, accuracy)
T2 : energy_consumed_wh  — energy regression      (R², MAE, RMSE)
T3 : max_combined_threat — threat regression      (R², MAE, RMSE)
T4 : alt_error_mean_m    — altitude error regr.   (R², MAE, RMSE)

Experiments
-----------
1. Per-region 5-fold cross-validation (all 6 regions)
2. Cross-region generalization: train on Delhi (largest), test on each other
3. Single-task vs multi-task ablation (4 single-task models vs shared MTL)
4. Learning curves (varying training set fraction: 10%, 25%, 50%, 75%, 100%)
5. Permutation feature importance (top-20 features per task)

Outputs
-------
  outputs/ml/nn_results/
    training_history_<region>.json      — per-epoch loss curves
    cv_results_<region>.csv             — fold-level metrics
    summary_all_regions.csv             — aggregated per-region means ± std
    cross_region_generalization.csv     — source→target AUC/R² matrix
    ablation_singletask_vs_mtl.csv      — MTL vs individual task comparison
    learning_curves_<region>.csv        — metrics vs training fraction
    feature_importance_<region>.csv     — permutation importance per task
    model_<region>.npz                  — saved best model weights
    scaler_<region>.npz                 — saved StandardScaler params
    nn_results_report.md                — full markdown results report

Usage
-----
  python scripts/ml/train_neural_network.py
  python scripts/ml/train_neural_network.py --region delhi --folds 3
  python scripts/ml/train_neural_network.py --epochs 50 --batch_size 256
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from evtol.ml.neural_network import (
    MultiTaskNN, StandardScalerNP,
    auc_roc_numpy, f1_numpy, r2_numpy, mae_numpy, rmse_numpy,
    kfold_indices,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGIONS = ['delhi', 'mumbai', 'bangalore', 'arunachal', 'odisha', 'ladakh']

PLANNING_FEATURES = [
    'path_length_m', 'time_cost_s', 'energy_cost_wh', 'threat_cost',
    'terrain_cost_mean', 'wind_cost_mean', 'obstacle_cost_mean',
    'fused_cost_mean', 'max_combined_threat', 'n_waypoints',
    'start_alt_m', 'goal_alt_m',
]

VEHICLE_FEATURES = [
    'cruise_speed_ms', 'cruise_altitude_m', 'hover_time_s',
    'power_hover_elec_W', 'power_cruise_elec_W', 'figure_of_merit',
    'LD_ratio', 'propulsive_efficiency', 'soc_final', 'soh_final',
    'battery_peak_temp_C', 'motor_temp_final_C', 'energy_consumed_wh',
    'pack_capacity_wh', 'spl_hover_a_dB', 'spl_cruise_a_dB',
    'rcs_hover_x_dBsm', 'rcs_cruise_x_dBsm',
]

CONTROL_FEATURES = [
    'pos_error_mean_m', 'pos_error_rms_m', 'vel_error_mean_ms',
    'alt_error_mean_m', 'alt_error_rms_m', 'att_error_mean_rad',
    'itae_pos', 'itae_alt', 'itae_att',
    'thrust_cmd_mean_N', 'thrust_cmd_std_N',
    'pwm_utilisation_pct', 'n_stall_events', 'n_saturations',
    'n_wp_reached', 'mission_abort',
    'cruise_speed_actual_ms', 'speed_variance_ms',
]

TASK_TARGETS = {
    'T1_risk':      ('risk_label',         'classification'),
    'T2_energy':    ('energy_consumed_wh', 'regression'),
    'T3_threat':    ('max_combined_threat','regression'),
    'T4_alt_error': ('alt_error_mean_m',   'regression'),
}

OUT_DIR = REPO_ROOT / 'outputs' / 'ml' / 'nn_results'
DATASET_ROOT = REPO_ROOT / 'datasets'

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_csv(region: str, layer: str) -> Optional[Path]:
    """Locate best available CSV for this region and layer."""
    candidates = {
        'planning': [
            DATASET_ROOT / region / 'planning_dataset' / f'planning_dataset_10k.csv',
            DATASET_ROOT / region / 'planning_dataset' / f'planning_{region}.csv',
            DATASET_ROOT / region / 'planning_dataset' / 'planning_dataset.csv',
            DATASET_ROOT / region / 'planning'         / f'planning_{region}.csv',
        ],
        'vehicle': [
            DATASET_ROOT / region / 'vehicle' / 'vehicle_dataset.csv',
        ],
        'control': [
            DATASET_ROOT / region / 'control' / 'control_dataset.csv',
        ],
    }
    for path in candidates.get(layer, []):
        if path.exists():
            return path
    return None


def load_region(region: str) -> Optional[pd.DataFrame]:
    """
    Load and join planning + vehicle + control CSVs for one region.
    Returns a merged DataFrame with all features and target columns.
    Falls back gracefully if vehicle/control are missing.
    """
    plan_path = _find_csv(region, 'planning')
    if plan_path is None:
        log.warning(f'[{region}] No planning CSV found — skipping')
        return None

    log.info(f'[{region}] Loading planning: {plan_path.name}')
    plan_df = pd.read_csv(plan_path)

    veh_path = _find_csv(region, 'vehicle')
    ctrl_path = _find_csv(region, 'control')

    # Trim all to equal length (inner join by row index)
    n = len(plan_df)
    if veh_path:
        log.info(f'[{region}] Loading vehicle: {veh_path.name}')
        veh_df = pd.read_csv(veh_path).iloc[:n].reset_index(drop=True)
        n = min(n, len(veh_df))
    else:
        veh_df = pd.DataFrame()

    if ctrl_path:
        log.info(f'[{region}] Loading control: {ctrl_path.name}')
        ctrl_df = pd.read_csv(ctrl_path).iloc[:n].reset_index(drop=True)
        n = min(n, len(ctrl_df))
    else:
        ctrl_df = pd.DataFrame()

    plan_df = plan_df.iloc[:n].reset_index(drop=True)

    # Merge, suffixing duplicates
    df = plan_df.copy()
    for col in veh_df.columns:
        if col not in df.columns:
            df[col] = veh_df[col].values
        elif col in ('energy_consumed_wh', 'risk_label', 'max_combined_threat'):
            # Prefer vehicle/control version when it exists (more accurate)
            df[col] = veh_df[col].values
    for col in ctrl_df.columns:
        if col not in df.columns:
            df[col] = ctrl_df[col].values
        elif col in ('energy_consumed_wh', 'risk_label', 'max_combined_threat',
                     'alt_error_mean_m'):
            df[col] = ctrl_df[col].values

    log.info(f'[{region}] Merged dataset: {len(df)} rows, {len(df.columns)} cols')
    return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """
    Extract feature matrix X and target arrays from merged DataFrame.
    Missing feature columns are filled with 0. Missing target columns with NaN.
    Returns (X, targets_dict, feature_names).
    """
    all_features = PLANNING_FEATURES + VEHICLE_FEATURES + CONTROL_FEATURES
    # Only keep features that exist and are numeric
    feature_names = []
    for f in all_features:
        if f in df.columns:
            try:
                _ = pd.to_numeric(df[f], errors='raise')
                feature_names.append(f)
            except Exception:
                pass

    X = df[feature_names].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float64)

    targets: Dict[str, np.ndarray] = {}
    for task, (col, _) in TASK_TARGETS.items():
        if col in df.columns:
            targets[task] = pd.to_numeric(df[col], errors='coerce').values.astype(np.float64)
        else:
            targets[task] = np.full(len(df), np.nan)

    return X, targets, feature_names


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_predictions(task: str, task_type: str,
                          y_true: np.ndarray, y_pred: np.ndarray
                          ) -> Dict[str, float]:
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {}
    if task_type == 'classification':
        return {
            'auc_roc': auc_roc_numpy(y_true, y_pred),
            'f1':      f1_numpy(y_true, y_pred),
            'acc':     float(np.mean((y_pred >= 0.5) == y_true)),
        }
    else:
        return {
            'r2':   r2_numpy(y_true, y_pred),
            'mae':  mae_numpy(y_true, y_pred),
            'rmse': rmse_numpy(y_true, y_pred),
        }


# ---------------------------------------------------------------------------
# Cross-validation for one region
# ---------------------------------------------------------------------------

def cross_validate_region(
    region: str,
    df: pd.DataFrame,
    n_folds: int = 5,
    epochs: int = 80,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 12,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[dict], MultiTaskNN, StandardScalerNP, List[str]]:
    """
    Run k-fold cross-validation on one region.
    Returns (fold_results_df, training_histories, best_model, scaler, feature_names).
    """
    X, targets, feature_names = build_feature_matrix(df)
    n = len(X)
    splits = kfold_indices(n, k=n_folds, seed=seed)

    fold_rows = []
    histories = []
    best_val_loss = np.inf
    best_model = None
    best_scaler = None

    for fold_i, (train_idx, val_idx) in enumerate(splits):
        log.info(f'[{region}] Fold {fold_i+1}/{n_folds} — '
                 f'train={len(train_idx)}, val={len(val_idx)}')

        # Scale features
        scaler = StandardScalerNP()
        X_tr = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])

        tgt_tr = {t: v[train_idx] for t, v in targets.items()}
        tgt_val = {t: v[val_idx] for t, v in targets.items()}

        model = MultiTaskNN(d_in=X_tr.shape[1], seed=seed + fold_i)
        t0 = time.time()
        history = model.fit(
            X_tr, tgt_tr,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            val_fraction=0.0,  # we already split manually
            verbose=False,
        )
        elapsed = time.time() - t0
        histories.append(history)

        # Evaluate on validation fold
        preds = model.predict(X_val)
        row = {'region': region, 'fold': fold_i + 1,
               'train_size': len(train_idx), 'val_size': len(val_idx),
               'train_time_s': round(elapsed, 2),
               'final_val_loss': history['val_loss'][-1] if history['val_loss'] else np.nan,
               'n_epochs': len(history['train_loss'])}

        for task, (_, task_type) in TASK_TARGETS.items():
            metrics = evaluate_predictions(
                task, task_type, tgt_val[task], preds[task]
            )
            for metric_name, val in metrics.items():
                row[f'{task}_{metric_name}'] = round(val, 6)

        fold_rows.append(row)

        val_loss = history['val_loss'][-1] if history['val_loss'] else np.inf
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_scaler = scaler

        log.info(f'[{region}] Fold {fold_i+1} done in {elapsed:.1f}s | '
                 f'val_loss={val_loss:.4f}')

    fold_df = pd.DataFrame(fold_rows)
    return fold_df, histories, best_model, best_scaler, feature_names


# ---------------------------------------------------------------------------
# Cross-region generalization experiment
# ---------------------------------------------------------------------------

def cross_region_generalization(
    region_data: Dict[str, Tuple],
    epochs: int = 80,
    batch_size: int = 512,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each source region: train full model, evaluate on every target region.
    Returns a DataFrame with source, target, and all task metrics.
    """
    rows = []
    trained_models: Dict[str, Tuple[MultiTaskNN, StandardScalerNP]] = {}

    for src in REGIONS:
        if src not in region_data:
            continue
        X_src, tgt_src, _, df_src = region_data[src]
        scaler = StandardScalerNP()
        X_s = scaler.fit_transform(X_src)
        model = MultiTaskNN(d_in=X_s.shape[1], seed=seed)
        log.info(f'Cross-region: training source={src} ({len(X_s)} samples)')
        model.fit(X_s, tgt_src, epochs=epochs, batch_size=batch_size,
                  patience=10, val_fraction=0.1, verbose=False)
        trained_models[src] = (model, scaler)

    for src, (src_model, src_scaler) in trained_models.items():
        for tgt in REGIONS:
            if tgt not in region_data:
                continue
            X_tgt, tgt_labels, _, _ = region_data[tgt]
            X_t = src_scaler.transform(X_tgt)
            preds = src_model.predict(X_t)
            row = {'source_region': src, 'target_region': tgt}
            for task, (_, task_type) in TASK_TARGETS.items():
                metrics = evaluate_predictions(task, task_type, tgt_labels[task], preds[task])
                for m, v in metrics.items():
                    row[f'{task}_{m}'] = round(v, 6)
            rows.append(row)
            log.info(f'Cross-region {src}→{tgt}: done')

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ablation: single-task vs multi-task
# ---------------------------------------------------------------------------

def ablation_single_vs_multi(
    region: str,
    X_tr: np.ndarray,
    X_val: np.ndarray,
    tgt_tr: Dict[str, np.ndarray],
    tgt_val: Dict[str, np.ndarray],
    epochs: int = 80,
    batch_size: int = 512,
    seed: int = 42,
) -> pd.DataFrame:
    """Train one model per task (single-task) vs the full MTL model."""
    rows = []

    # Multi-task model
    mt_model = MultiTaskNN(d_in=X_tr.shape[1], seed=seed)
    mt_model.fit(X_tr, tgt_tr, epochs=epochs, batch_size=batch_size,
                 patience=10, val_fraction=0.0, verbose=False)
    mt_preds = mt_model.predict(X_val)
    for task, (_, task_type) in TASK_TARGETS.items():
        metrics = evaluate_predictions(task, task_type, tgt_val[task], mt_preds[task])
        for m, v in metrics.items():
            rows.append({'region': region, 'model': 'multi_task',
                         'task': task, 'metric': m, 'value': round(v, 6)})

    # Single-task models — zero out other task weights
    for active_task in TASK_TARGETS:
        weights = {t: (1.0 if t == active_task else 0.0) for t in TASK_TARGETS}
        st_model = MultiTaskNN(d_in=X_tr.shape[1], task_weights=weights, seed=seed)
        st_model.fit(X_tr, {active_task: tgt_tr[active_task], **{t: None for t in TASK_TARGETS if t != active_task}},
                     epochs=epochs, batch_size=batch_size,
                     patience=10, val_fraction=0.0, verbose=False)
        st_preds = st_model.predict(X_val)
        _, task_type = TASK_TARGETS[active_task]
        metrics = evaluate_predictions(active_task, task_type,
                                       tgt_val[active_task], st_preds[active_task])
        for m, v in metrics.items():
            rows.append({'region': region, 'model': 'single_task',
                         'task': active_task, 'metric': m, 'value': round(v, 6)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def learning_curves(
    region: str,
    X: np.ndarray,
    targets: Dict[str, np.ndarray],
    fractions: List[float] = None,
    epochs: int = 60,
    batch_size: int = 256,
    seed: int = 42,
) -> pd.DataFrame:
    """Train at increasing data fractions, measure validation metrics."""
    if fractions is None:
        fractions = [0.10, 0.25, 0.50, 0.75, 1.00]

    rng = np.random.default_rng(seed)
    n = len(X)
    n_val = max(50, int(n * 0.15))
    idx = rng.permutation(n)
    val_idx = idx[:n_val]
    train_pool = idx[n_val:]

    scaler_full = StandardScalerNP().fit(X[train_pool])
    X_val = scaler_full.transform(X[val_idx])
    tgt_val = {t: v[val_idx] for t, v in targets.items()}

    rows = []
    for frac in fractions:
        n_train = max(32, int(len(train_pool) * frac))
        tr_idx = train_pool[:n_train]
        scaler = StandardScalerNP()
        X_tr = scaler.fit_transform(X[tr_idx])
        tgt_tr = {t: v[tr_idx] for t, v in targets.items()}

        model = MultiTaskNN(d_in=X_tr.shape[1], seed=seed)
        model.fit(X_tr, tgt_tr, epochs=epochs, batch_size=min(batch_size, n_train),
                  patience=8, val_fraction=0.0, verbose=False)

        preds = model.predict(X_val)
        row = {'region': region, 'fraction': frac, 'n_train': n_train}
        for task, (_, task_type) in TASK_TARGETS.items():
            metrics = evaluate_predictions(task, task_type, tgt_val[task], preds[task])
            for m, v in metrics.items():
                row[f'{task}_{m}'] = round(v, 6)
        rows.append(row)
        log.info(f'[{region}] Learning curve frac={frac:.0%} n_train={n_train}: done')

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Permutation feature importance
# ---------------------------------------------------------------------------

def permutation_importance(
    model: MultiTaskNN,
    scaler: StandardScalerNP,
    X_raw: np.ndarray,
    targets: Dict[str, np.ndarray],
    feature_names: List[str],
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = scaler.transform(X_raw)

    # Baseline metrics
    baseline_preds = model.predict(X)
    baseline: Dict[str, float] = {}
    for task, (_, task_type) in TASK_TARGETS.items():
        m = evaluate_predictions(task, task_type, targets[task], baseline_preds[task])
        if task_type == 'classification':
            baseline[task] = m.get('auc_roc', 0.0)
        else:
            baseline[task] = m.get('r2', 0.0)

    rows = []
    for fi, feat in enumerate(feature_names):
        for task in TASK_TARGETS:
            drops = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                X_perm[:, fi] = rng.permutation(X_perm[:, fi])
                p = model.predict(X_perm)[task]
                _, task_type = TASK_TARGETS[task]
                m = evaluate_predictions(task, task_type, targets[task], p)
                key = 'auc_roc' if task_type == 'classification' else 'r2'
                drops.append(baseline[task] - m.get(key, 0.0))
            rows.append({
                'feature': feat,
                'task': task,
                'importance_mean': round(float(np.mean(drops)), 6),
                'importance_std':  round(float(np.std(drops)), 6),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown report generation
# ---------------------------------------------------------------------------

def generate_report(
    cv_summary: pd.DataFrame,
    cross_region_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
) -> str:
    lines = [
        '# Neural Network Results Report',
        '',
        'Auto-generated by `scripts/ml/train_neural_network.py`',
        '',
        '## 1. Cross-Validation Summary (mean ± std across folds)',
        '',
    ]

    # CV summary table
    metric_cols = [c for c in cv_summary.columns
                   if any(t in c for t in ['T1', 'T2', 'T3', 'T4'])]
    grouped = cv_summary.groupby('region')[metric_cols].agg(['mean', 'std'])
    lines.append('| Region | T1 AUC | T1 F1 | T2 R² | T2 MAE | T3 R² | T4 R² |')
    lines.append('|--------|--------|-------|-------|--------|-------|-------|')
    for region, row in grouped.iterrows():
        def _fmt(col):
            try:
                m = row[(col, 'mean')]
                s = row[(col, 'std')]
                return f'{m:.3f}±{s:.3f}'
            except Exception:
                return 'N/A'
        lines.append(
            f'| {region} '
            f'| {_fmt("T1_risk_auc_roc")} '
            f'| {_fmt("T1_risk_f1")} '
            f'| {_fmt("T2_energy_r2")} '
            f'| {_fmt("T2_energy_mae")} '
            f'| {_fmt("T3_threat_r2")} '
            f'| {_fmt("T4_alt_error_r2")} |'
        )

    lines += ['', '## 2. Cross-Region Generalization', '']
    if not cross_region_df.empty:
        lines.append('T1 AUC-ROC matrix (rows=source, cols=target):')
        lines.append('')
        regions_present = cross_region_df['source_region'].unique().tolist()
        header = '| Source \\ Target | ' + ' | '.join(regions_present) + ' |'
        lines.append(header)
        lines.append('|' + '---|' * (len(regions_present) + 1))
        for src in regions_present:
            vals = []
            for tgt in regions_present:
                sub = cross_region_df[
                    (cross_region_df['source_region'] == src) &
                    (cross_region_df['target_region'] == tgt)
                ]
                if not sub.empty and 'T1_risk_auc_roc' in sub.columns:
                    vals.append(f'{sub["T1_risk_auc_roc"].values[0]:.3f}')
                else:
                    vals.append('N/A')
            lines.append(f'| {src} | ' + ' | '.join(vals) + ' |')

    lines += ['', '## 3. Ablation: Single-Task vs Multi-Task', '']
    if not ablation_df.empty:
        lines.append('| Region | Task | Model | Metric | Value |')
        lines.append('|--------|------|-------|--------|-------|')
        for _, row in ablation_df.iterrows():
            lines.append(f'| {row.get("region","?")} | {row["task"]} | '
                         f'{row["model"]} | {row["metric"]} | {row["value"]:.4f} |')

    lines += ['', '## 4. Key Findings', '']
    if not cv_summary.empty and 'T1_risk_auc_roc' in cv_summary.columns:
        best_auc = cv_summary.groupby('region')['T1_risk_auc_roc'].mean().idxmax()
        best_val = cv_summary.groupby('region')['T1_risk_auc_roc'].mean().max()
        lines.append(f'- Best T1 risk classification AUC: **{best_val:.4f}** (region: {best_auc})')
    if not cv_summary.empty and 'T2_energy_r2' in cv_summary.columns:
        best_r2_region = cv_summary.groupby('region')['T2_energy_r2'].mean().idxmax()
        best_r2 = cv_summary.groupby('region')['T2_energy_r2'].mean().max()
        lines.append(f'- Best T2 energy R²: **{best_r2:.4f}** (region: {best_r2_region})')

    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--region', default='all', help='Region to run (or "all")')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--skip_cross_region', action='store_true')
    p.add_argument('--skip_ablation', action='store_true')
    p.add_argument('--skip_learning_curves', action='store_true')
    p.add_argument('--skip_importance', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    regions_to_run = REGIONS if args.region == 'all' else [args.region]

    # -----------------------------------------------------------------------
    # 1. Per-region cross-validation
    # -----------------------------------------------------------------------
    all_cv_rows: List[pd.DataFrame] = []
    all_histories: Dict[str, List[dict]] = {}
    region_data: Dict[str, Tuple] = {}  # for cross-region experiment

    for region in regions_to_run:
        log.info(f'=== Region: {region} ===')
        df = load_region(region)
        if df is None:
            continue

        X, targets, feat_names = build_feature_matrix(df)
        if len(X) < 50:
            log.warning(f'[{region}] Too few samples ({len(X)}), skipping')
            continue

        region_data[region] = (X, targets, feat_names, df)

        fold_df, histories, best_model, best_scaler, feat_names = cross_validate_region(
            region, df,
            n_folds=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )

        all_cv_rows.append(fold_df)
        all_histories[region] = histories

        # Save per-region artifacts
        fold_df.to_csv(OUT_DIR / f'cv_results_{region}.csv', index=False)

        with open(OUT_DIR / f'training_history_{region}.json', 'w') as f:
            json.dump(histories, f, indent=2)

        if best_model is not None:
            best_model.save(str(OUT_DIR / f'model_{region}.npz'))
        if best_scaler is not None:
            best_scaler.save(str(OUT_DIR / f'scaler_{region}.npz'))

        # --- Learning curves ---
        if not args.skip_learning_curves:
            lc_df = learning_curves(region, X, targets, epochs=min(args.epochs, 50))
            lc_df.to_csv(OUT_DIR / f'learning_curves_{region}.csv', index=False)

        # --- Permutation feature importance ---
        if not args.skip_importance and best_model is not None and best_scaler is not None:
            log.info(f'[{region}] Computing permutation importance...')
            imp_df = permutation_importance(
                best_model, best_scaler, X, targets, feat_names, n_repeats=3
            )
            imp_df.to_csv(OUT_DIR / f'feature_importance_{region}.csv', index=False)

        # --- Per-region ablation (single-task vs MTL) ---
        if not args.skip_ablation:
            log.info(f'[{region}] Ablation: single-task vs multi-task...')
            rng = np.random.default_rng(42)
            n = len(X)
            n_val = max(50, int(n * 0.2))
            idx = rng.permutation(n)
            val_idx, tr_idx = idx[:n_val], idx[n_val:]
            scaler_ab = StandardScalerNP()
            X_tr_ab = scaler_ab.fit_transform(X[tr_idx])
            X_val_ab = scaler_ab.transform(X[val_idx])
            tgt_tr_ab = {t: v[tr_idx] for t, v in targets.items()}
            tgt_val_ab = {t: v[val_idx] for t, v in targets.items()}
            abl_df = ablation_single_vs_multi(
                region, X_tr_ab, X_val_ab, tgt_tr_ab, tgt_val_ab,
                epochs=min(args.epochs, 50)
            )
            abl_df.to_csv(OUT_DIR / f'ablation_{region}.csv', index=False)

    # -----------------------------------------------------------------------
    # 2. Summary across regions
    # -----------------------------------------------------------------------
    if all_cv_rows:
        cv_all = pd.concat(all_cv_rows, ignore_index=True)
        cv_all.to_csv(OUT_DIR / 'summary_all_regions.csv', index=False)

        # Aggregate: mean ± std per region
        metric_cols = [c for c in cv_all.columns
                       if any(t in c for t in ['T1', 'T2', 'T3', 'T4'])]
        agg = cv_all.groupby('region')[metric_cols].agg(['mean', 'std']).round(4)
        agg.to_csv(OUT_DIR / 'summary_aggregated.csv')
        log.info('\n' + agg.to_string())
    else:
        cv_all = pd.DataFrame()

    # -----------------------------------------------------------------------
    # 3. Cross-region generalization
    # -----------------------------------------------------------------------
    cross_region_df = pd.DataFrame()
    if not args.skip_cross_region and len(region_data) >= 2:
        log.info('=== Cross-region generalization experiment ===')
        cross_region_df = cross_region_generalization(
            region_data, epochs=min(args.epochs, 60)
        )
        cross_region_df.to_csv(OUT_DIR / 'cross_region_generalization.csv', index=False)

    # -----------------------------------------------------------------------
    # 4. Combined ablation report
    # -----------------------------------------------------------------------
    abl_paths = list(OUT_DIR.glob('ablation_*.csv'))
    ablation_df = pd.concat([pd.read_csv(p) for p in abl_paths], ignore_index=True) \
        if abl_paths else pd.DataFrame()
    if not ablation_df.empty:
        ablation_df.to_csv(OUT_DIR / 'ablation_singletask_vs_mtl.csv', index=False)

    # -----------------------------------------------------------------------
    # 5. Markdown report
    # -----------------------------------------------------------------------
    report = generate_report(cv_all, cross_region_df, ablation_df)
    report_path = OUT_DIR / 'nn_results_report.md'
    report_path.write_text(report)
    log.info(f'Report saved to {report_path}')

    log.info('=== All experiments complete ===')
    log.info(f'Results in: {OUT_DIR}')


if __name__ == '__main__':
    main()
