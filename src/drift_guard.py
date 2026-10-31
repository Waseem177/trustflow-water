"""
Rate-capped running references (STATUS gap 5).

A trailing median absorbs any change that lasts longer than its window, so an
attacker who ramps a reading slowly drags the reference along and the alarm
goes quiet. The fix: a reference may move no faster than the plant itself
legitimately drifts. The cap is learned from attack-free data, as the
tolerances are: 1.2 x the fastest change of the uncapped reference over one
hour. A logged clean (CIP for membrane quantities, CEB for UF) resets the
reference, as before.

Limit, by construction: a ramp slower than the cap looks exactly like fouling
or probe drift and is not caught by this mechanism.

Alarm hold (optional, 23 Oct): given the checked quantity q and its tolerance,
the reference does not move on any row where q is already out of tolerance
against it, so a detected drift is not absorbed while the alarm is on. It
moves again, at the capped rate, once q is back within tolerance; a logged
clean still resets it.

Works on the outputs of invariants_swro.trailing_reference and
energy_check.trailing, without changing them, so earlier results stand.
"""
from __future__ import annotations
import numpy as np

MARGIN = 1.2
HORIZON_MIN = 60.0


def _groups(n, key=None):
    if key is None:
        return [np.arange(n)]
    key = np.asarray(key)
    return [np.flatnonzero(key == g) for g in np.unique(key)]


def fit_cap(ref, t_min, group=None, resets=None):
    """Largest legitimate rate of change (units per minute) of a reference over HORIZON_MIN."""
    ref, t_min = np.asarray(ref, float), np.asarray(t_min, float)
    seg = np.cumsum(np.isin(np.arange(len(ref)), resets)) if resets is not None and len(resets) else np.zeros(len(ref), int)
    worst = 0.0
    for idx in _groups(len(ref), group):
        r, t, s = ref[idx], t_min[idx], seg[idx]
        j = np.searchsorted(t, t - HORIZON_MIN, side="right") - 1      # last row at least an hour back
        ok = (j >= 0) & np.isfinite(r)
        jj = np.clip(j, 0, None)
        ok &= np.isfinite(r[jj]) & (s[jj] == s) & (t - t[jj] > 0)
        if ok.any():
            rate = np.abs(r[ok] - r[jj][ok]) / (t[ok] - t[jj][ok])
            worst = max(worst, float(rate.max()))
    return MARGIN * worst


def cap(ref, t_min, rate, group=None, resets=None, q=None, tol=None):
    """Causal rate limiter: each step moves towards the raw reference by at most rate * dt.
    With q and tol, also hold the reference wherever |q - reference| > tol (alarm hold)."""
    ref, t_min = np.asarray(ref, float), np.asarray(t_min, float)
    q = None if q is None else np.asarray(q, float)
    out = np.full_like(ref, np.nan)
    reset = np.isin(np.arange(len(ref)), resets) if resets is not None and len(resets) else np.zeros(len(ref), bool)
    for idx in _groups(len(ref), group):
        prev, prev_t = np.nan, None
        for i in idx:
            r = ref[i]
            if not np.isfinite(r):
                prev, prev_t = np.nan, None           # warm-up after a reset: no reference yet
                continue
            if reset[i] or not np.isfinite(prev):
                v = r
            elif q is not None and np.isfinite(q[i]) and abs(q[i] - prev) > tol:
                v = prev                               # in alarm: hold
            else:
                step = rate * (t_min[i] - prev_t)
                v = min(max(r, prev - step), prev + step)
            out[i], prev, prev_t = v, v, t_min[i]
    return out
