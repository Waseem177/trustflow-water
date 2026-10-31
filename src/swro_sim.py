"""
Seawater RO train simulator: UF pretreatment, chlorination and SBS
dechlorination, one RO pass. Returns the true process state and what the
sensors report, so a concealment attack can later edit the reports while the
true state carries on underneath.

It is deliberately *richer* than the invariants in invariants_swro.py, so the
invariants are not merely checking the simulator against its own equations:

  simulator                                    invariant assumes
  -------------------------------------------  ----------------------------------
  7-segment vessel, concentration polarisation  log-mean feed/brine conc, no CP
  seawater viscosity (Sharqawy et al. 2010)     pure-water viscosity (Vogel)
  two-branch water TCF, separate salt TCF       one ASTM D4516 TCF, U = 2640 K
  brine-side dP ~ Q^1.6                         dP ~ Q^1.5
  HOBr speciation, SBS reducing term, 90%       ORP linear in log10(FAC) and pH,
    SBS efficiency, probe lag and drift           stoichiometric SBS (1.34)
  meter gain bias, noise, TDS nonlinearity      exact sensors

This still does not answer circularity on its own: the white paper commits to
checking at least one invariant on real Port Hueneme UF data (STATUS item 7).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import lfilter

TAGS = {
    "Q_uf": "m3/h",     # UF filtrate flow
    "TMP_uf": "bar",    # UF transmembrane pressure
    "T_feed": "degC",
    "FAC_pre": "mg/L",  # free chlorine after UF, before SBS (as Cl2)
    "ORP_pre": "mV",
    "pH_feed": "-",
    "Q_sbs": "L/h",     # sodium bisulphite stock dosing flow
    "ORP_ro": "mV",     # RO feed, after dechlorination
    "Q_feed": "m3/h", "Q_perm": "m3/h", "Q_conc": "m3/h",
    "P_feed": "bar", "P_conc": "bar", "P_perm": "bar",
    "C_feed": "mg/L", "C_perm": "mg/L", "C_conc": "mg/L",   # TDS from conductivity
    "W_hp": "kW",       # HP pump electrical power, from the electrical system's own meter
}

SBS_STOCK = 100_000.0   # mg/L, 10% w/v; a plant constant both sides know
UF_AREA = 16_700.0      # m2, ~60 LMH at 1000 m3/h
STEP_MIN = 1            # one row per minute
P_SUC = 2.0             # bar, HP pump suction (booster outlet); a plant constant both sides know
Q_BEP = 1000.0          # m3/h, HP pump best-efficiency flow


def pi_bar(c, t):
    """Osmotic pressure (bar) of NaCl-dominated water, TDS c in mg/L (Hydranautics form)."""
    return 0.0385 * c * (t + 273.15) / (1000.0 - c / 1000.0) / 14.5038


def seawater_viscosity(t, c):
    """Sharqawy, Lienhard & Zubair (2010), Desal. Water Treat. 16, eq. 22-23. Pa.s"""
    mu_w = 4.2844e-5 + 1.0 / (0.157 * (t + 64.993) ** 2 - 91.296)
    s = c / 1e6
    a = 1.541 + 1.998e-2 * t - 9.52e-5 * t ** 2
    b = 7.974 - 7.561e-2 * t + 4.724e-4 * t ** 2
    return mu_w * (1 + a * s + b * s ** 2)


def hp_power(Q, P, day, days):
    """HP pump electrical power (kW). Pump curve: efficiency falls off either side of the
    best-efficiency point; motor 95%; wear costs 1% of efficiency over the run. No energy
    recovery device in this train, so the pump carries the whole feed flow.
    The energy check assumes none of this: it uses a constant efficiency."""
    eta = 0.82 * (1 - 0.6 * (Q / Q_BEP - 1) ** 2) * 0.95 * (1 - 0.01 * day / days)
    return Q / 3600 * (P - P_SUC) * 1e5 / eta / 1000


def ar1(n, rho, sigma, rng):
    """Stationary AR(1) noise with marginal std sigma."""
    return lfilter([sigma * np.sqrt(1 - rho ** 2)], [1, -rho], rng.standard_normal(n))


def schedule(n, levels, min_days, max_days, rng, ramp_min=10):
    """Piecewise-constant operator setpoint, held 'min_days'..'max_days', ramped."""
    out, i = np.empty(n), 0
    while i < n:
        hold = int(rng.uniform(min_days, max_days) * 1440)
        out[i:i + hold] = rng.choice(levels)
        i += hold
    return np.convolve(np.pad(out, (ramp_min - 1, 0), mode="edge"),
                       np.ones(ramp_min) / ramp_min, mode="valid")


def simulate(days=60, seed=0, cip_days=(21, 49), algal=((10, 3, 2.0), (40, 2.5, 2.5)),
             T_start=24.0, T_end=30.0):
    """
    Returns (truth, reported, events), each a DataFrame indexed by minute.
    algal: (start_day, duration_days, extra solids factor) bloom episodes.
    """
    rng = np.random.default_rng(seed)
    n = int(days * 1440 / STEP_MIN)
    day = np.arange(n) * STEP_MIN / 1440.0

    # ---- environment -------------------------------------------------------
    T = (T_start + (T_end - T_start) * day / days + 0.6 * np.sin(2 * np.pi * (day - 0.6))
         + ar1(n, 0.999, 0.3, rng))
    Cf = 42_000 + 700 * np.sin(2 * np.pi * day * 24 / 12.42) + ar1(n, 0.998, 400, rng)
    pH = 8.1 + 0.04 * np.sin(2 * np.pi * (day - 0.6)) + ar1(n, 0.99, 0.02, rng)
    solids = np.ones(n)
    for d0, dur, extra in algal:
        bump = np.clip(np.minimum(day - d0, d0 + dur - day) / 0.5, 0, 1)   # 12 h ramps
        solids += extra * bump

    # ---- operation ---------------------------------------------------------
    Qp = schedule(n, [380.0, 420.0, 450.0], 2, 5, rng) * (1 + ar1(n, 0.95, 0.003, rng))
    Y = schedule(n, [0.40, 0.42, 0.44], 3, 7, rng)
    Qf = Qp / Y * (1 + ar1(n, 0.95, 0.002, rng))
    Qc = Qf - Qp
    Yt = Qp / Qf

    # ---- RO membrane state: fouling, CIP -----------------------------------
    cip_rows = np.array([int(c * 1440 / STEP_MIN) for c in cip_days if c < days], dtype=int)
    last_cip = np.zeros(n, dtype=int)
    last_cip[cip_rows] = cip_rows
    last_cip = np.maximum.accumulate(last_cip)            # row of the most recent clean
    since = day - day[last_cip]
    n_cip = np.searchsorted(cip_rows, np.arange(n), side="right")
    algal_load = np.cumsum(solids - 1) * STEP_MIN / 1440.0   # bloom-days since start
    algal_since = algal_load - algal_load[last_cip]
    A25 = 24.0 * 0.98 ** n_cip * (1 - 0.004 * since - 0.004 * algal_since)
    B25 = 1.6 * 1.03 ** n_cip * (1 + 0.003 * since)
    kdp = 1.5 / 790 ** 1.6 * (1 + 0.006 * since + 0.01 * algal_since)

    U_w = np.where(T >= 25, 2640.0, 3020.0)
    A_T = A25 * np.exp(U_w * (1 / 298.15 - 1 / (T + 273.15)))
    B_T = B25 * np.exp(3000.0 * (1 / 298.15 - 1 / (T + 273.15)))

    seg = (np.arange(7) + 0.5) / 7
    Cb = Cf[:, None] / (1 - Yt[:, None] * seg[None, :])
    Cw = Cb * np.exp(0.7 * Yt / 7)[:, None]              # concentration polarisation
    pi_w = pi_bar(Cw, T[:, None]).mean(1)
    Cp = B_T * Cw.mean(1) / (Qp + B_T)
    dP = kdp * ((Qf + Qc) / 2) ** 1.6
    Pp = 0.8 + ar1(n, 0.95, 0.02, rng)
    Pf = Qp / A_T + dP / 2 + Pp + pi_w - pi_bar(Cp, T)   # what the HP pump must deliver
    Pc = Pf - dP
    Cc = (Qf * Cf - Qp * Cp) / Qc

    # ---- UF: cake builds per filtered volume, backwash every 40 min, CEB daily
    Quf = Qf * (1 + ar1(n, 0.98, 0.01, rng))             # feed tank buffers the mismatch
    R_m, R_irr, R_rev, R = 2.0e12, 0.0, 0.0, np.empty(n)
    for i in range(n):
        R_rev += 9e8 * solids[i] * Quf[i] / 60
        R[i] = R_m + R_irr + R_rev
        if i % 40 == 39:
            R_irr += 0.02 * R_rev; R_rev = 0.0
        if i % 1440 == 1439:
            R_irr *= 0.15
    TMP = seawater_viscosity(T, Cf) * R * (Quf / 3600 / UF_AREA) / 1e5

    # ---- chlorination and SBS dechlorination -------------------------------
    FAC = np.clip(schedule(n, [0.2, 0.3, 0.4], 1, 3, rng) + ar1(n, 0.99, 0.03, rng), 0.05, None)
    dose = schedule(n, [2.0, 2.5, 3.0], 3, 8, rng)
    Qsbs = dose * Qf * 1000 / SBS_STOCK * (1 + ar1(n, 0.9, 0.02, rng))
    dose_true = Qsbs * SBS_STOCK / (Qf * 1000)
    FAC_ro = np.maximum(0, FAC - 0.9 * dose_true / 1.34)
    excess = np.maximum(0, 0.9 * dose_true - 1.34 * FAC)
    alpha = 1 / (1 + 10 ** (pH - 8.6))                   # HOBr share in seawater
    ORP_pre = 750 + 75 * np.log10(FAC * alpha + 1e-5)
    ORP_ro = 750 + 75 * np.log10(FAC_ro * alpha + 1e-5) - 60 * np.log10(1 + 5 * excess)

    truth = pd.DataFrame(dict(
        Q_uf=Quf, TMP_uf=TMP, T_feed=T, FAC_pre=FAC, ORP_pre=ORP_pre, pH_feed=pH,
        Q_sbs=Qsbs, ORP_ro=ORP_ro, Q_feed=Qf, Q_perm=Qp, Q_conc=Qc,
        P_feed=Pf, P_conc=Pc, P_perm=Pp, C_feed=Cf, C_perm=Cp, C_conc=Cc,
        W_hp=hp_power(Qf, Pf, day, days)))
    reported = sense(truth, rng)
    events = pd.DataFrame(dict(day=day, cip=np.isin(np.arange(n), cip_rows),
                               uf_phase=np.arange(n) % 40, ceb=np.arange(n) % 1440 == 0,
                               solids=solids, A25=A25, B25=B25, kdp=kdp, uf_R=R))
    return truth, reported, events


def sense(truth, rng):
    """What the transmitters report: fixed gain error, noise, probe lag and drift."""
    n, rep = len(truth), {}
    rel = lambda g, s: (1 + rng.normal(0, g)) * (1 + s * rng.standard_normal(n))
    for q in ("Q_uf", "Q_feed", "Q_perm", "Q_conc"):
        rep[q] = truth[q].values * rel(0.003, 0.003)
    rep["Q_sbs"] = truth["Q_sbs"].values * rel(0.01, 0.01)
    for p in ("P_feed", "P_conc"):
        rep[p] = truth[p].values * (1 + rng.normal(0, 0.002)) + rng.normal(0, 0.05, n)
    rep["P_perm"] = truth["P_perm"].values + rng.normal(0, 0.02, n)
    rep["TMP_uf"] = truth["TMP_uf"].values + rng.normal(0, 0.005, n)
    rep["T_feed"] = truth["T_feed"].values + rng.normal(0, 0.1) + rng.normal(0, 0.05, n)
    rep["pH_feed"] = truth["pH_feed"].values + rng.normal(0, 0.03) + rng.normal(0, 0.01, n)
    rep["FAC_pre"] = np.clip(truth["FAC_pre"].values * (1 + rng.normal(0, 0.03))
                             + rng.normal(0, 0.01, n), 0, None)
    for c in ("C_feed", "C_conc"):     # EC->TDS conversion is mildly nonlinear
        v = truth[c].values
        rep[c] = v * (1 + 0.01 * (v / 42_000 - 1)) * rel(0.005, 0.003)
    rep["C_perm"] = truth["C_perm"].values * rel(0.02, 0.01)
    for o in ("ORP_pre", "ORP_ro"):    # 3-min probe lag, slow drift, noise
        lagged = lfilter([1 - np.exp(-1 / 3)], [1, -np.exp(-1 / 3)], truth[o].values,
                         zi=[truth[o].values[0] * np.exp(-1 / 3)])[0]
        rep[o] = lagged + np.cumsum(rng.normal(0, 0.07, n)) + rng.normal(0, 2.0, n)
    # power meter last, so adding it leaves every earlier tag's random draws unchanged
    rep["W_hp"] = truth["W_hp"].values * rel(0.005, 0.005)
    return pd.DataFrame(rep)[list(TAGS)]


if __name__ == "__main__":
    truth, rep, ev = simulate()
    print(rep.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    print("A25 range", ev.A25.min().round(2), ev.A25.max().round(2))
