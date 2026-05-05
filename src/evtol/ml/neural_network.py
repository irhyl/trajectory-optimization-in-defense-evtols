"""
neural_network.py
=================
Pure-NumPy multi-task neural network for defense eVTOL trajectory optimization.

Architecture
------------
Shared backbone:
    Input(d_in) → LayerNorm → Dense(512, ReLU) → Dropout(p=0.3)
                            → Dense(256, ReLU) → Dropout(p=0.3)
                            → Dense(128, ReLU) → Dropout(p=0.3)

Task-specific heads:
    T1 risk_label       : Dense(64, ReLU) → Dense(1, Sigmoid)   [binary classification]
    T2 energy_wh        : Dense(64, ReLU) → Dense(1, Linear)    [regression]
    T3 threat           : Dense(64, ReLU) → Dense(1, Sigmoid)   [bounded regression 0-1]
    T4 alt_error        : Dense(64, ReLU) → Dense(1, Softplus)  [non-negative regression]

Optimizer  : Adam (Kingma & Ba 2015)
Regularizer: L2 weight decay on backbone weights
Loss       : Binary cross-entropy (T1) + Huber (T2, T3, T4)
             Combined as weighted sum: L = w1*L1 + w2*L2 + w3*L3 + w4*L4
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float64)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    pos = x >= 0
    out = np.zeros_like(x)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_neg = np.exp(x[~pos])
    out[~pos] = exp_neg / (1.0 + exp_neg)
    return out

def _softplus(x: np.ndarray) -> np.ndarray:
    # log(1 + exp(x)), numerically stable
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))

def _softplus_grad(x: np.ndarray) -> np.ndarray:
    return _sigmoid(x)

def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray,
                eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    out = gamma * x_hat + beta
    return out, x_hat, var

def _layer_norm_grad(d_out: np.ndarray, x_hat: np.ndarray, var: np.ndarray,
                     gamma: np.ndarray, eps: float = 1e-6
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = d_out.shape[1]
    d_gamma = (d_out * x_hat).sum(axis=0)
    d_beta = d_out.sum(axis=0)
    d_x_hat = d_out * gamma
    d_var = (-0.5 * d_x_hat * x_hat / (var + eps)).sum(axis=1, keepdims=True)
    d_mu = (-d_x_hat / np.sqrt(var + eps)).sum(axis=1, keepdims=True)
    d_x = (d_x_hat / np.sqrt(var + eps)
           + 2.0 * d_var * x_hat / N
           + d_mu / N)
    return d_x, d_gamma, d_beta


def _huber_loss(pred: np.ndarray, target: np.ndarray,
                delta: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    diff = pred - target
    abs_diff = np.abs(diff)
    mask = abs_diff <= delta
    loss = np.where(mask, 0.5 * diff ** 2, delta * (abs_diff - 0.5 * delta))
    grad = np.where(mask, diff, delta * np.sign(diff))
    return loss.mean(), grad / len(diff)


def _bce_loss(prob: np.ndarray, target: np.ndarray,
              pos_weight: float = 1.0,
              eps: float = 1e-7) -> Tuple[float, np.ndarray]:
    """
    Weighted binary cross-entropy.
    pos_weight > 1 up-weights the minority positive class; set to n_neg/n_pos.
    Gradient w.r.t. pre-sigmoid logit = pos_weight*(p-1)*y + p*(1-y).
    """
    p = np.clip(prob, eps, 1.0 - eps)
    loss = -(pos_weight * target * np.log(p) + (1.0 - target) * np.log(1.0 - p))
    # dL/d_logit via chain rule through sigmoid: s*(1-s) cancels neatly
    grad_logit = pos_weight * p * (1.0 - target) - pos_weight * (1.0 - p) * target
    # simplifies when pos_weight=1 to the standard (p - y)
    grad_logit = p - target + (pos_weight - 1.0) * (1.0 - p) * target
    return loss.mean(), grad_logit / len(target)


# ---------------------------------------------------------------------------
# Target preprocessing utilities
# ---------------------------------------------------------------------------

def compute_pos_weight(y: np.ndarray, min_positives: int = 10,
                       max_weight: float = 100.0) -> float:
    """
    Compute BCE positive class weight = n_neg / n_pos.
    Returns 1.0 when there are fewer than `min_positives` (degenerate case),
    capped at `max_weight` to prevent gradient explosion.
    """
    mask = ~np.isnan(y)
    n_pos = int(y[mask].sum())
    n_neg = int((1 - y[mask]).sum())
    if n_pos < min_positives:
        return 1.0
    return min(float(n_neg) / float(n_pos), max_weight)


def logit_transform(y: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Apply logit = log(p/(1-p)) to a [0,1] bounded target.
    Spreads saturated distributions (e.g. max_combined_threat clustered near 1.0).
    Clips y to [eps, 1-eps] to avoid ±inf.
    """
    y_clip = np.clip(y, eps, 1.0 - eps)
    return np.log(y_clip / (1.0 - y_clip))


def logit_inverse(z: np.ndarray) -> np.ndarray:
    """Inverse of logit_transform (sigmoid), maps R → (0,1)."""
    return _sigmoid(z)


# ---------------------------------------------------------------------------
# Dense layer (forward + backward stored in cache)
# ---------------------------------------------------------------------------

class DenseLayer:
    """Single fully-connected layer with He initialisation."""

    def __init__(self, d_in: int, d_out: int, activation: str = 'relu',
                 rng: np.random.Generator = None):
        if rng is None:
            rng = np.random.default_rng(0)
        scale = np.sqrt(2.0 / d_in)  # He initialisation
        self.W = rng.standard_normal((d_in, d_out)) * scale
        self.b = np.zeros(d_out)
        self.activation = activation
        self._cache: dict = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        z = x @ self.W + self.b
        self._cache['x'] = x
        self._cache['z'] = z
        if self.activation == 'relu':
            out = _relu(z)
        elif self.activation == 'sigmoid':
            out = _sigmoid(z)
        elif self.activation == 'softplus':
            out = _softplus(z)
        else:  # linear
            out = z
        self._cache['out'] = out
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        z = self._cache['z']
        x = self._cache['x']
        if self.activation == 'relu':
            d_z = d_out * _relu_grad(z)
        elif self.activation == 'sigmoid':
            s = _sigmoid(z)
            d_z = d_out * s * (1.0 - s)
        elif self.activation == 'softplus':
            d_z = d_out * _softplus_grad(z)
        else:  # linear
            d_z = d_out
        self.dW = x.T @ d_z
        self.db = d_z.sum(axis=0)
        return d_z @ self.W.T

    def params(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return [(self.W, self.dW if hasattr(self, 'dW') else np.zeros_like(self.W)),
                (self.b, self.db if hasattr(self, 'db') else np.zeros_like(self.b))]


class DropoutLayer:
    """Inverted dropout."""

    def __init__(self, p: float = 0.3, rng: np.random.Generator = None):
        self.p = p
        self.rng = rng or np.random.default_rng(0)
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.p == 0.0:
            self._mask = None
            return x
        self._mask = (self.rng.random(x.shape) > self.p) / (1.0 - self.p)
        return x * self._mask

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return d_out
        return d_out * self._mask


# ---------------------------------------------------------------------------
# Multi-task Neural Network
# ---------------------------------------------------------------------------

class MultiTaskNN:
    """
    Multi-task neural network with shared backbone and four task heads.

    Tasks
    -----
    T1 : risk_label          binary classification  (BCE loss)
    T2 : energy_consumed_wh  regression             (Huber loss)
    T3 : max_combined_threat bounded regression     (Huber loss on sigmoid output)
    T4 : alt_error_mean_m    non-neg regression     (Huber loss on softplus output)

    Parameters
    ----------
    d_in        : number of input features
    task_weights: dict mapping task name → loss weight (default equal weighting)
    dropout_p   : dropout probability during training
    l2_lambda   : L2 weight-decay coefficient
    seed        : RNG seed
    """

    TASKS = ['T1_risk', 'T2_energy', 'T3_threat', 'T4_alt_error']

    def __init__(
        self,
        d_in: int,
        task_weights: Optional[Dict[str, float]] = None,
        dropout_p: float = 0.3,
        l2_lambda: float = 1e-4,
        pos_weight: float = 1.0,
        seed: int = 42,
    ):
        self.d_in = d_in
        self.l2_lambda = l2_lambda
        self.pos_weight = pos_weight  # class weight for minority positive in T1
        self.task_weights = task_weights or {t: 1.0 for t in self.TASKS}
        rng = np.random.default_rng(seed)

        # --- Shared backbone ---
        self.backbone: List = [
            DenseLayer(d_in, 512, 'relu', rng),
            DropoutLayer(dropout_p, rng),
            DenseLayer(512, 256, 'relu', rng),
            DropoutLayer(dropout_p, rng),
            DenseLayer(256, 128, 'relu', rng),
            DropoutLayer(dropout_p, rng),
        ]
        # LayerNorm parameters (on raw input)
        self.ln_gamma = np.ones(d_in)
        self.ln_beta = np.zeros(d_in)
        self._ln_cache: dict = {}

        # --- Task heads ---
        self.heads: Dict[str, List] = {
            'T1_risk':      [DenseLayer(128, 64, 'relu', rng),
                             DenseLayer(64, 1, 'sigmoid', rng)],
            'T2_energy':    [DenseLayer(128, 64, 'relu', rng),
                             DenseLayer(64, 1, 'linear', rng)],
            'T3_threat':    [DenseLayer(128, 64, 'relu', rng),
                             DenseLayer(64, 1, 'sigmoid', rng)],
            'T4_alt_error': [DenseLayer(128, 64, 'relu', rng),
                             DenseLayer(64, 1, 'softplus', rng)],
        }

        # Adam optimizer state for all parameters
        self._adam_m: List[np.ndarray] = []
        self._adam_v: List[np.ndarray] = []
        self._adam_t: int = 0
        self._init_adam()

    # ------------------------------------------------------------------
    # Adam initialisation
    # ------------------------------------------------------------------

    def _all_param_pairs(self):
        """Yield (param_array, grad_array) for every trainable parameter."""
        # LayerNorm
        yield self.ln_gamma, getattr(self, '_d_ln_gamma', np.zeros_like(self.ln_gamma))
        yield self.ln_beta,  getattr(self, '_d_ln_beta',  np.zeros_like(self.ln_beta))
        # Backbone Dense layers only
        for layer in self.backbone:
            if isinstance(layer, DenseLayer):
                for p, g in layer.params():
                    yield p, g
        # Heads
        for head_layers in self.heads.values():
            for layer in head_layers:
                if isinstance(layer, DenseLayer):
                    for p, g in layer.params():
                        yield p, g

    def _init_adam(self):
        self._adam_m = [np.zeros_like(p) for p, _ in self._all_param_pairs()]
        self._adam_v = [np.zeros_like(p) for p, _ in self._all_param_pairs()]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray, training: bool = True) -> Dict[str, np.ndarray]:
        # LayerNorm on input
        out, x_hat, var = _layer_norm(X, self.ln_gamma, self.ln_beta)
        self._ln_cache = {'x_hat': x_hat, 'var': var, 'X': X}

        # Shared backbone
        for layer in self.backbone:
            out = layer.forward(out, training=training)
        backbone_out = out

        # Task heads
        predictions: Dict[str, np.ndarray] = {}
        for task, head_layers in self.heads.items():
            h = backbone_out
            for layer in head_layers:
                h = layer.forward(h, training=training)
            predictions[task] = h.squeeze(axis=1) if h.ndim > 1 else h
        return predictions

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(self, predictions: Dict[str, np.ndarray],
                     targets: Dict[str, Optional[np.ndarray]]
                     ) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Returns total loss and per-task output gradients d_loss/d_output.
        Targets with None are masked (task not available for this batch).
        """
        total_loss = 0.0
        grad_out: Dict[str, np.ndarray] = {}
        n = next(iter(predictions.values())).shape[0]

        for task, w in self.task_weights.items():
            y = targets.get(task)
            pred = predictions[task]
            if y is None:
                grad_out[task] = np.zeros(n)
                continue

            mask = ~np.isnan(y)
            if mask.sum() == 0:
                grad_out[task] = np.zeros(n)
                continue

            if task == 'T1_risk':
                # Skip task entirely when fewer than 10 positives in batch
                # (degenerate imbalance: model can't learn anything meaningful)
                n_pos = y[mask].sum()
                if n_pos < 10:
                    grad_out[task] = np.zeros(n)
                    continue
                loss, g = _bce_loss(pred[mask], y[mask], pos_weight=self.pos_weight)
            else:
                loss, g = _huber_loss(pred[mask], y[mask])

            full_g = np.zeros(n)
            full_g[mask] = g
            total_loss += w * loss
            grad_out[task] = w * full_g

        # L2 regularization on backbone weights
        l2 = 0.0
        for layer in self.backbone:
            if isinstance(layer, DenseLayer):
                l2 += 0.5 * self.l2_lambda * np.sum(layer.W ** 2)
        total_loss += l2
        return total_loss, grad_out

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, grad_out: Dict[str, np.ndarray]) -> None:
        """Backpropagate task gradients through heads and shared backbone."""
        n = next(iter(grad_out.values())).shape[0]
        d_backbone = np.zeros((n, 128))

        for task, head_layers in self.heads.items():
            g = grad_out[task].reshape(-1, 1)
            for layer in reversed(head_layers):
                g = layer.backward(g)
            d_backbone += g

        # Backbone backward
        d = d_backbone
        for layer in reversed(self.backbone):
            d = layer.backward(d)
            if isinstance(layer, DenseLayer):
                # L2 gradient
                layer.dW += self.l2_lambda * layer.W

        # LayerNorm backward
        d_ln, d_gamma, d_beta = _layer_norm_grad(
            d, self._ln_cache['x_hat'], self._ln_cache['var'], self.ln_gamma
        )
        self._d_ln_gamma = d_gamma
        self._d_ln_beta = d_beta

    # ------------------------------------------------------------------
    # Adam update step
    # ------------------------------------------------------------------

    def adam_step(self, lr: float = 1e-3, beta1: float = 0.9,
                  beta2: float = 0.999, eps: float = 1e-8) -> None:
        self._adam_t += 1
        t = self._adam_t
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t

        for i, (p, g) in enumerate(self._all_param_pairs()):
            self._adam_m[i] = beta1 * self._adam_m[i] + (1.0 - beta1) * g
            self._adam_v[i] = beta2 * self._adam_v[i] + (1.0 - beta2) * g ** 2
            m_hat = self._adam_m[i] / bc1
            v_hat = self._adam_v[i] / bc2
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        targets: Dict[str, np.ndarray],
        epochs: int = 100,
        batch_size: int = 512,
        lr: float = 1e-3,
        lr_decay: float = 0.98,
        patience: int = 10,
        val_fraction: float = 0.1,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the network.

        Returns history dict with 'train_loss' and 'val_loss' per epoch.
        """
        n = X.shape[0]
        rng = np.random.default_rng(42)

        # Train/val split
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_fraction))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        X_tr, X_val = X[train_idx], X[val_idx]
        tgt_tr = {t: (v[train_idx] if v is not None else None)
                  for t, v in targets.items()}
        tgt_val = {t: (v[val_idx] if v is not None else None)
                   for t, v in targets.items()}

        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        best_val = np.inf
        patience_count = 0
        best_params = None

        for epoch in range(1, epochs + 1):
            # Shuffle training data each epoch
            perm = rng.permutation(len(train_idx))
            X_tr = X_tr[perm]
            tgt_tr = {t: (v[perm] if v is not None else None)
                      for t, v in tgt_tr.items()}

            # Mini-batch SGD
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, len(train_idx), batch_size):
                end = min(start + batch_size, len(train_idx))
                X_b = X_tr[start:end]
                tgt_b = {t: (v[start:end] if v is not None else None)
                         for t, v in tgt_tr.items()}

                preds = self.forward(X_b, training=True)
                loss, grad_out = self.compute_loss(preds, tgt_b)
                self.backward(grad_out)
                self.adam_step(lr=lr)

                epoch_loss += loss
                n_batches += 1

            train_loss = epoch_loss / n_batches

            # Validation
            preds_val = self.forward(X_val, training=False)
            val_loss, _ = self.compute_loss(preds_val, tgt_val)

            history['train_loss'].append(float(train_loss))
            history['val_loss'].append(float(val_loss))
            history['lr'].append(lr)

            if verbose and epoch % 10 == 0:
                print(f"  epoch {epoch:4d} | train_loss={train_loss:.4f} | "
                      f"val_loss={val_loss:.4f} | lr={lr:.2e}")

            # Early stopping
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                patience_count = 0
                best_params = self._snapshot_params()
            else:
                patience_count += 1
                if patience_count >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch} (best val_loss={best_val:.4f})")
                    break

            # LR decay
            lr *= lr_decay

        # Restore best parameters
        if best_params is not None:
            self._restore_params(best_params)

        return history

    # ------------------------------------------------------------------
    # Parameter snapshot / restore (for early stopping)
    # ------------------------------------------------------------------

    def _snapshot_params(self) -> List[np.ndarray]:
        return [p.copy() for p, _ in self._all_param_pairs()]

    def _restore_params(self, snapshot: List[np.ndarray]) -> None:
        for (p, _), s in zip(self._all_param_pairs(), snapshot):
            p[:] = s

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        return self.forward(X, training=False)

    def predict_task(self, X: np.ndarray, task: str) -> np.ndarray:
        return self.predict(X)[task]

    # ------------------------------------------------------------------
    # Serialisation (numpy .npz)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        arrays = {}
        arrays['ln_gamma'] = self.ln_gamma
        arrays['ln_beta'] = self.ln_beta
        arrays['d_in'] = np.array([self.d_in])
        arrays['l2_lambda'] = np.array([self.l2_lambda])

        def _save_dense(prefix, layer):
            arrays[f'{prefix}_W'] = layer.W
            arrays[f'{prefix}_b'] = layer.b

        for i, layer in enumerate(self.backbone):
            if isinstance(layer, DenseLayer):
                _save_dense(f'bb_{i}', layer)

        for task, head_layers in self.heads.items():
            for j, layer in enumerate(head_layers):
                if isinstance(layer, DenseLayer):
                    _save_dense(f'head_{task}_{j}', layer)

        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str, **kwargs) -> 'MultiTaskNN':
        data = np.load(path, allow_pickle=False)
        d_in = int(data['d_in'][0])
        l2_lambda = float(data['l2_lambda'][0])
        obj = cls(d_in=d_in, l2_lambda=l2_lambda, **kwargs)
        obj.ln_gamma[:] = data['ln_gamma']
        obj.ln_beta[:] = data['ln_beta']

        for i, layer in enumerate(obj.backbone):
            if isinstance(layer, DenseLayer):
                key_W = f'bb_{i}_W'
                if key_W in data:
                    layer.W[:] = data[key_W]
                    layer.b[:] = data[f'bb_{i}_b']

        for task, head_layers in obj.heads.items():
            for j, layer in enumerate(head_layers):
                if isinstance(layer, DenseLayer):
                    key_W = f'head_{task}_{j}_W'
                    if key_W in data:
                        layer.W[:] = data[key_W]
                        layer.b[:] = data[f'head_{task}_{j}_b']
        return obj


# ---------------------------------------------------------------------------
# Feature preprocessing (pure numpy — no sklearn dependency)
# ---------------------------------------------------------------------------

class StandardScalerNP:
    """Minimal StandardScaler using numpy only."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'StandardScalerNP':
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-10] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.std_ + self.mean_

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean_, std=self.std_)

    @classmethod
    def load(cls, path: str) -> 'StandardScalerNP':
        data = np.load(path)
        obj = cls()
        obj.mean_ = data['mean']
        obj.std_ = data['std']
        return obj


# ---------------------------------------------------------------------------
# Metrics (pure numpy)
# ---------------------------------------------------------------------------

def auc_roc_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Trapezoidal AUC-ROC (numpy only)."""
    sorted_idx = np.argsort(-y_score)
    y_true_sorted = y_true[sorted_idx]
    npos = y_true.sum()
    nneg = len(y_true) - npos
    if npos == 0 or nneg == 0:
        return float('nan')
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    tpr = tp / npos
    fpr = fp / nneg
    # Prepend (0, 0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return float(np.trapz(tpr, fpr))


def f1_numpy(y_true: np.ndarray, y_pred: np.ndarray,
             threshold: float = 0.5) -> float:
    pred_bin = (y_pred >= threshold).astype(int)
    tp = ((pred_bin == 1) & (y_true == 1)).sum()
    fp = ((pred_bin == 1) & (y_true == 0)).sum()
    fn = ((pred_bin == 0) & (y_true == 1)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    return float(2 * prec * rec / (prec + rec + 1e-9))


def r2_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def mae_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# K-fold cross-validation utility
# ---------------------------------------------------------------------------

def kfold_indices(n: int, k: int = 5,
                  shuffle: bool = True,
                  seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    folds = np.array_split(idx, k)
    splits = []
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train, val))
    return splits
