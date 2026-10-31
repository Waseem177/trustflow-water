"""
Time-windowing utilities for the reconstruction detector.

* make_windows(X, W): causal sliding windows, flattened -> (N-W+1, W*F).
  Window i covers rows [i, i+W) and is used to score time t = i+W-1.
* smooth_errors(err, k): causal rolling mean over the last k scores. Sustained
  attacks stay high; isolated noisy spikes get averaged down.
"""
import numpy as np
import pandas as pd


def make_windows(X: np.ndarray, W: int) -> np.ndarray:
    if W == 1:
        return X
    N, F = X.shape
    idx = np.arange(W)[None, :] + np.arange(N - W + 1)[:, None]   # (N-W+1, W)
    return X[idx].reshape(N - W + 1, W * F)


def smooth_errors(err: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return err
    return pd.Series(err).rolling(k, min_periods=1).mean().values


def episode_stats(y: np.ndarray, pred: np.ndarray):
    """Contiguous attack episodes: how many got >=1 alarm, and mean detection delay."""
    starts = np.where(np.diff(np.r_[0, y]) == 1)[0]
    ends = np.where(np.diff(np.r_[y, 0]) == -1)[0]
    detected, delays = 0, []
    for s, e in zip(starts, ends):
        hits = np.where(pred[s:e + 1] == 1)[0]
        if len(hits):
            detected += 1; delays.append(int(hits[0]))
    return {"episodes": len(starts), "episodes_detected": detected,
            "mean_delay_steps": round(float(np.mean(delays)), 1) if delays else None}
