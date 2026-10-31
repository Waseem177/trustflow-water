"""
SWRO physics invariants: the four relations promised in the white paper, as
eight residuals over the reported SCADA tags (names and units in swro_sim.TAGS).

  1 flow_balance    Q_feed = Q_perm + Q_conc
  2 salt_balance    Q_feed C_feed = Q_perm C_perm + Q_conc C_conc
  3 ro_water_perm   ASTM D4516 normalised permeate flow: A25 = Q_perm / (NDP * TCF)
  4 ro_salt_perm    ASTM D4516 normalised salt passage:  B25 = Q_perm C_perm / ((C_fc - C_perm) TCF)
  5 ro_brine_dp     feed-to-concentrate dP tracks flow:  dP / Q_avg^1.5
  6 uf_resistance   UF filter resistance (Darcy):        TMP / (mu(T) Q_uf)
  7 orp_pretreat    ORP sits on the probe line implied by free chlorine and pH
  8 orp_ro_feed     RO-feed ORP agrees with the chlorine left after the SBS dose

Physics fixes the *form* of each relation; only a reference value and a
tolerance are fitted on attack-free data (ORP also fits its probe line). A row
violates an invariant when its normalised residual exceeds 1 in magnitude.

Invariants 3-8 track quantities that drift legitimately (membrane fouling,
UF cake, probe drift). Against a fixed reference their tolerance has to cover
a whole cleaning cycle, which leaves an attacker a lot of room. With
fit(..., trailing=...) they are instead compared with a trailing median of the
plant's own recent reported values, the way operators normalise RO data.

Everything here is torch and differentiable, so the invariant-aware attacker
can optimise against exactly the code the defence runs. The trailing
reference itself is not differentiated through; an attacker sees it as fixed.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from swro_sim import TAGS, SBS_STOCK

NAMES = ["flow_balance", "salt_balance", "ro_water_perm", "ro_salt_perm",
         "ro_brine_dp", "uf_resistance", "orp_pretreat", "orp_ro_feed"]
DRIFTING = ["ro_water_perm", "ro_salt_perm", "ro_brine_dp", "uf_resistance",
            "orp_pretreat", "orp_ro"]
RESET_BY_CIP = ["ro_water_perm", "ro_salt_perm", "ro_brine_dp"]   # an RO clean moves these only
U_TCF = 2640.0     # K, ASTM D4516 temperature correction constant
DP_EXP = 1.5       # brine-side pressure drop exponent
SBS_STOICH = 1.34  # mg SBS per mg Cl2
FAC_GATE = 0.10    # mg/L left after SBS above which RO-feed ORP must read oxidising
MARGIN = 1.2       # tolerance = MARGIN x worst deviation on calibration data


def cols(x):
    """Accepts a DataFrame, or a [N, len(TAGS)] array/tensor in TAGS order."""
    if isinstance(x, pd.DataFrame):
        x = x[list(TAGS)].values
    if not torch.is_tensor(x):
        x = torch.tensor(np.array(x, dtype=np.float32))
    return {k: x[:, i] for i, k in enumerate(TAGS)}


def pi_bar(c, t):
    return 0.0385 * c * (t + 273.15) / (1000.0 - c / 1000.0) / 14.5038


def mu_water(t):
    """Pure-water viscosity, Vogel equation (Pa.s)."""
    return 2.414e-5 * 10 ** (247.8 / (t + 133.15))


def tcf(t):
    return torch.exp(U_TCF * (1 / 298.15 - 1 / (t + 273.15)))


def orp_line(fac, ph, k):
    return k[0] + k[1] * torch.log10(fac.clamp(min=0) + 0.01) + k[2] * ph


def quantities(x, orp_k):
    """Per-row invariant quantities. orp_k is the fitted ORP probe line."""
    c, eps = cols(x), 1e-6
    Qf, Qp, Qc = c["Q_feed"], c["Q_perm"], c["Q_conc"]
    Cf, Cp, Cc, T = c["C_feed"], c["C_perm"], c["C_conc"], c["T_feed"]
    Y = (Qp / Qf).clamp(0.01, 0.95)
    C_fc = Cf * torch.log(1 / (1 - Y)) / Y                    # log-mean feed/brine conc
    dP = c["P_feed"] - c["P_conc"]
    ndp = c["P_feed"] - dP / 2 - c["P_perm"] - pi_bar(C_fc, T) + pi_bar(Cp, T)
    dose = c["Q_sbs"] * SBS_STOCK / (Qf * 1000)
    return {
        "flow_balance": (Qf - Qp - Qc) / Qf,
        "salt_balance": (Qf * Cf - Qp * Cp - Qc * Cc) / (Qf * Cf),
        "ro_water_perm": torch.log(Qp.clamp(min=eps) / (ndp.clamp(min=0.1) * tcf(T))),
        "ro_salt_perm": torch.log((Qp * Cp).clamp(min=eps) / ((C_fc - Cp).clamp(min=1.0) * tcf(T))),
        "ro_brine_dp": torch.log(dP.clamp(min=1e-3) / ((Qf + Qc) / 2) ** DP_EXP),
        "uf_resistance": torch.log(c["TMP_uf"].clamp(min=1e-4) / (mu_water(T) * c["Q_uf"].clamp(min=eps))),
        "orp_pretreat": c["ORP_pre"] - orp_line(c["FAC_pre"], c["pH_feed"], orp_k),
        "orp_ro": c["ORP_ro"],
        "fac_left": torch.relu(c["FAC_pre"] - dose / SBS_STOICH),
        "orp_ro_floor": orp_line(torch.relu(c["FAC_pre"] - dose / SBS_STOICH), c["pH_feed"], orp_k),
    }


def trailing_reference(q, window_min, lag_min, events=None):
    """
    Median of each drifting quantity over [t - lag - window, t - lag].

    events: the plant's own maintenance log, dict with optional keys
      "cip":      rows where an RO clean finished. For the membrane quantities,
                  history before a clean is dropped, and the reference is NaN
                  (static fallback) until a full window has built up.
      "uf_phase": minutes since the last UF backwash, per row. UF resistance
                  rebuilds the same way each filtration cycle (cake grows with
                  filtered volume), so it is compared with the same point in
                  the cycles inside the window, not with a plain median.
      "ceb":      rows where a UF chemically enhanced backwash finished; resets
                  the UF reference as "cip" does the membrane ones.
    """
    events = events or {}
    n = len(q["orp_ro"])

    def segments(key):
        seg = np.zeros(n, dtype=int)
        if len(events.get(key, [])):
            seg[np.asarray(events[key])] = 1
        return np.cumsum(seg)

    seg, seg_uf = segments("cip"), segments("ceb")
    ref = {}
    for k in DRIFTING:
        s = pd.Series(q[k].detach().cpu().numpy().astype(np.float64))
        if k == "uf_resistance" and events.get("uf_phase") is not None:
            phase = np.asarray(events["uf_phase"])
            period = int(phase.max()) + 1
            w, lag = max(1, window_min // period), -(-lag_min // period)
            # a CEB happens daily, so restart after 2 cycles rather than a full window
            med = s.groupby([phase, seg_uf]).transform(
                lambda v: v.rolling(w, min_periods=min(w, 2)).median().shift(lag))
        else:
            g = seg if k in RESET_BY_CIP else np.zeros(n, dtype=int)
            med = s.groupby(g).transform(
                lambda v: v.rolling(window_min, min_periods=window_min).median().shift(lag_min))
        ref[k] = torch.tensor(med.values, dtype=torch.float32)
    return ref


def _deviations(q, cal, ref):
    """Signed deviation per invariant, plus a mask of rows using a trailing reference."""
    d, trail = {}, {}
    for k in NAMES[:7]:
        static = torch.full_like(q[k], cal["ref"][k])
        if ref is not None and k in ref:
            ok = ~torch.isnan(ref[k])
            d[k], trail[k] = q[k] - torch.where(ok, ref[k], static), ok
        else:
            d[k], trail[k] = q[k] - static, torch.zeros_like(q[k], dtype=torch.bool)
    # orp_ro_feed has two sides, blended by how much chlorine the SBS dose leaves
    gate = torch.sigmoid((q["fac_left"] - FAC_GATE) / 0.02)
    too_low = torch.relu(q["orp_ro_floor"] - q["orp_ro"])
    centre = torch.full_like(q["orp_ro"], cal["ref"]["orp_ro"])
    t_ro = torch.zeros_like(gate, dtype=torch.bool)
    if ref is not None and "orp_ro" in ref:
        t_ro = ~torch.isnan(ref["orp_ro"])
        centre = torch.where(t_ro, ref["orp_ro"], centre)
    d["orp_ro_feed"] = (gate, too_low, (q["orp_ro"] - centre).abs())
    trail["orp_ro_feed"] = t_ro
    return d, trail


def _reference(q, cal, ref, events):
    if ref is None and cal.get("trailing"):
        ref = trailing_reference(q, **cal["trailing"], events=events)
    return ref


def fit(x_normal, trailing=None, events=None):
    """
    Calibrate on attack-free reported data.
    trailing: dict(window_min=..., lag_min=...) to use trailing references for
    the drifting invariants; events: the maintenance log for x_normal's rows
    (see trailing_reference).
    Tolerances are fitted separately for rows on a trailing reference and rows
    on the static fallback, so the warm-up after a CIP does not widen both.
    """
    c = cols(x_normal)
    X = torch.stack([torch.ones_like(c["FAC_pre"]), torch.log10(c["FAC_pre"].clamp(min=0) + 0.01),
                     c["pH_feed"]], 1).double()
    k = torch.linalg.lstsq(X, c["ORP_pre"].double()[:, None]).solution[:, 0].float()
    q = quantities(x_normal, k)
    cal = {"orp_k": k, "trailing": trailing,
           "ref": {n: float(q[n].median()) for n in NAMES[:7] + ["orp_ro"]},
           "tol": {}, "tol_trail": {}}
    ref = _reference(q, cal, None, events)
    d, trail = _deviations(q, cal, ref)

    def worst(v, m):
        return MARGIN * float(v[m].abs().max()) if bool(m.any()) else float("nan")

    for n in NAMES[:7]:
        cal["tol"][n] = worst(d[n], ~trail[n])
        cal["tol_trail"][n] = worst(d[n], trail[n])
    gate, too_low, off = d["orp_ro_feed"]
    cal["tol"]["orp_ro_low"] = max(MARGIN * float((gate * too_low).max()), 20.0)
    cal["tol"]["orp_ro_band"] = worst(off * (1 - gate), ~trail["orp_ro_feed"])
    cal["tol_trail"]["orp_ro_band"] = worst(off * (1 - gate), trail["orp_ro_feed"])
    return cal


def residuals(x, cal, ref=None, events=None):
    """
    [N, 8] normalised residuals in NAMES order; |r| > 1 is a violation.
    If the calibration uses trailing references and ref is None, they are
    computed here from x itself, i.e. from whatever the plant reported.
    """
    q = quantities(x, cal["orp_k"])
    ref = _reference(q, cal, ref, events)
    d, trail = _deviations(q, cal, ref)

    def tol(n, key=None):
        key = key or n
        return torch.where(trail[n], torch.tensor(cal["tol_trail"].get(key, np.nan)),
                           torch.tensor(cal["tol"][key]))

    out = [d[n] / tol(n) for n in NAMES[:7]]
    gate, too_low, off = d["orp_ro_feed"]
    out.append(torch.maximum(gate * too_low / cal["tol"]["orp_ro_low"],
                             (1 - gate) * off / tol("orp_ro_feed", "orp_ro_band")))
    return torch.stack(out, 1)


def violations(x, cal, ref=None, events=None):
    return (residuals(x, cal, ref, events).abs() > 1).detach().cpu().numpy()
