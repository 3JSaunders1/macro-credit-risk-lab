# pylint: disable=no-name-in-module
"""
models/var_utils.py
-------------------
Shared VAR math: companion matrix, stability, moving-average coefficients,
forecast error variances, FEVD, and normal forecast intervals.

Lag matrices follow the convention y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + e_t,
passed as an array of shape (p, n, n).
"""
import numpy as np
from scipy.stats import norm


def companion(coefs: np.ndarray) -> np.ndarray:
    p, n, _ = coefs.shape
    C = np.zeros((n * p, n * p))
    C[:n, :] = np.hstack(list(coefs))
    if p > 1:
        C[n:, :-n] = np.eye(n * (p - 1))
    return C


def is_stable(coefs: np.ndarray) -> bool:
    """All companion eigenvalues inside the unit circle."""
    return bool(np.max(np.abs(np.linalg.eigvals(companion(coefs)))) < 1)


def ma_coefficients(coefs: np.ndarray, horizon: int) -> np.ndarray:
    """Psi_0, ..., Psi_{horizon-1}: responses to one-unit reduced-form innovations."""
    p, n, _ = coefs.shape
    C, M, out = companion(coefs), np.eye(n * p), []
    for _ in range(horizon):
        out.append(M[:n, :n].copy())
        M = M @ C
    return np.array(out)


def forecast_mse(coefs: np.ndarray, sigma: np.ndarray, steps: int) -> np.ndarray:
    """Forecast error covariance for horizons 1..steps: sum of Psi_j Sigma Psi_j'."""
    psis = ma_coefficients(coefs, steps)
    acc, out = np.zeros_like(sigma), []
    for h in range(steps):
        acc = acc + psis[h] @ sigma @ psis[h].T
        out.append(acc.copy())
    return np.array(out)


def fevd_from_irfs(irfs: np.ndarray) -> np.ndarray:
    """FEVD from structural IRFs of shape (H, n, n): share of each shock in each variable."""
    sq = np.cumsum(np.asarray(irfs) ** 2, axis=0)
    return sq / sq.sum(axis=2, keepdims=True)


def normal_interval(point: np.ndarray, variances: np.ndarray, alpha: float):
    """Lower and upper bounds: point +/- z * sqrt(variance)."""
    z = norm.ppf(1 - alpha / 2)
    se = np.sqrt(np.maximum(variances, 0))
    return point - z * se, point + z * se
