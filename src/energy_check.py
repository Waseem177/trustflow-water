"""
Energy check: the HP pump's electrical power must match the hydraulic work the
process readings say it is doing.

    W_hp  ~=  Q_feed [m3/h] / 3600 * (P_feed - P_SUC) [bar] * 1e5 / eta / 1000   [kW]

Residual = ln W_hp - ln(Q_feed (P_feed - P_SUC) / 36) = -ln eta, compared with a
trailing median (efficiency drifts with wear and with the operating point) or a
fixed reference. Tolerance = 1.2 x worst deviation on attack-free data, the same
rule as every other check.

Why it is different from the other eight: W_hp comes from the electrical
system's own meter, which in most plants sits on a different network from the
process SCADA. An attacker who controls every process tag, even one who replays
genuine normal data, still has to explain the electricity the pump draws.

Kept separate from invariants_swro.py so the published eight-check results do
not change. Run the experiment with eval_energy.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from swro_sim import P_SUC

MARGIN = 1.2


def residual(rep):
    """ln W - ln(hydraulic power); NaN where the pump is not lifting."""
    head = rep["P_feed"].values - P_SUC
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(rep["W_hp"].values) - np.log(rep["Q_feed"].values * head / 36.0)
    return np.where((head > 1) & (rep["Q_feed"].values > 1), r, np.nan)


def trailing(r, window_min=360, lag_min=60):
    return pd.Series(r).rolling(window_min, min_periods=window_min // 4).median().shift(lag_min).values


def fit(r_cal, ref_cal=None):
    """Tolerance from attack-free data; ref_cal = trailing reference over the same rows, or None for fixed."""
    if ref_cal is None:
        ref = float(np.nanmedian(r_cal))
        return {"ref": ref, "tol": MARGIN * float(np.nanmax(np.abs(r_cal - ref)))}
    return {"ref": None, "tol": MARGIN * float(np.nanmax(np.abs(r_cal - ref_cal)))}


def violations(r, cal, ref=None):
    ref = cal["ref"] if ref is None else ref
    d = np.abs(r - ref) / cal["tol"]
    return np.where(np.isfinite(d), d > 1, False)
