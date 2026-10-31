"""
Optional task: is the Port Hueneme feed pump a usable invariant?  Pump affinity
laws say discharge pressure follows speed N and flow Q:
    PT-100 ~ a*N^2 + b*N*Q + c*Q^2          (N = VFD-100-FB, Q = FE/FT-100)
Question: does PT-100 follow N and Q tightly enough to flag a spoof of PT-100,
the pump speed signal or the flow meter?

Roles. VFD-100-FB = feed pump speed feedback (about 1.0 = nominal, documented by
the readme's variable order). PT-100 is INFERRED to be the pump discharge
pressure (about 40 psi, ahead of the UF inlet PT-101 at about 33 psi).

What the training days showed (exploration, 24 long training files; no test day
was touched):
  - Pooled over days the pump curve fails: mean PT-100 sits between 34 and 48 psi
    from campaign to campaign at the same speed and flow, because the downstream
    UF/RO resistance (fouling) moves it. A fixed curve cannot be absolute.
  - Within a day, speed and flow explain about 50% of the variance of PT-100
    (R2 0.2-0.85, median 0.48; residual sd 0.8 psi on 40 psi, i.e. 2%).
So the usable form is relative, as for every other check here: take the residual
of the pump curve and compare it with the same point in the filtration cycle
over recent cycles. Whether the pump curve ADDS anything to that is the test.

Design fixed BEFORE running the test days (26 Oct):
  - Same files, steady-filtration mask, 6 h trailing median 1 h old per 1-min
    phase bin, tolerance 1.2 x 99.9th percentile of training |deviation|, 30 s
    persistence, spoof levels +/-2/5/10/20/50% with the reference frozen, as in
    hueneme_uf.py. Same steady rows for every spoof (the mask is not recomputed).
  - Pump-curve coefficients: pooled within-day (per-file intercept) least squares
    of PT-100 on [N^2, N*Q, Q^2] over the training days, rows thinned 1 in 5.
  - Two deviations, same tolerance rule:
      pump   dev = (P - curve) - trailing(P - curve)
      plain  dev = P - trailing(P)                 (no speed or flow: baseline)
    The pump curve earns a place only if it catches smaller PT-100 spoofs than
    plain, or catches N or Q spoofs that the existing flow-balance and
    UF-resistance checks do not.
  - Spoof targets: PT-100, N (VFD-100-FB), Q (FE/FT-100). Detection reported for
    the pump check alone, and for pump or the existing hueneme_uf checks.
  - Tolerance rule reported as p999 only (the 'worst' rule was shown useless on
    real data in hueneme_uf.py).

Run: .venv/bin/python src/hueneme_pump_curve.py  ->  results/hueneme_pump_curve.json
"""
from __future__ import annotations
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
import hueneme_uf as H

ROOT = Path(__file__).resolve().parent.parent
LEVELS = H.LEVELS
MARGIN = H.MARGIN
THIN = 5


def parts(d):
    """Steady mask, phase, and the model inputs."""
    q, phase, steady = H.quantities(d)
    return q, phase, steady


def curve_design(N, Q):
    return np.c_[N ** 2, N * Q, Q ** 2]


def fit_curve(train):
    """Within-day least squares: demean each file before regressing."""
    Xs, ys = [], []
    for _, d in train:
        q, phase, steady = parts(d)
        s = d[steady].iloc[::THIN]
        if len(s) < 500:
            continue
        X = curve_design(s["VFD-100-FB"].values, s["FE/FT-100"].values)
        y = s["PT-100"].values
        Xs.append(X - X.mean(0)); ys.append(y - y.mean())
    X, y = np.vstack(Xs), np.concatenate(ys)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ coef
    return coef, 1 - r.var() / y.var(), float(r.std())


def pump_quantity(d, coef, steady):
    """P - curve on the clean steady rows; NaN elsewhere."""
    P = d["PT-100"].values
    cur = curve_design(d["VFD-100-FB"].values, d["FE/FT-100"].values) @ coef
    return pd.Series(np.where(steady, P - cur, np.nan), index=d.index)


def plain_quantity(d, steady):
    return pd.Series(np.where(steady, d["PT-100"].values, np.nan), index=d.index)


def deviations(d, coef):
    q, phase, steady = parts(d)
    pq = pump_quantity(d, coef, steady)
    lq = plain_quantity(d, steady)
    dev = {"pump": pq - H.trailing(pq, phase), "plain": lq - H.trailing(lq, phase)}
    ref = {"pump": pq - dev["pump"], "plain": lq - dev["plain"]}
    return dev, ref, phase, steady


def main():
    data = {s: [] for s in ("training", "test")}
    for split in data:
        for p in sorted(glob.glob(str(H.BASE / split / "*.csv"))):
            d, why = H.load(p)
            if d is not None:
                data[split].append((Path(p).stem, d))
    coef, r2, rs = fit_curve(data["training"])

    cal_dev = {"pump": [], "plain": [], "flow_balance": [], "uf_resistance": []}
    for _, d in data["training"]:
        dev, *_ = deviations(d, coef)
        dh, *_ = H.deviations(d)
        cal_dev["pump"].append(dev["pump"].dropna().values)
        cal_dev["plain"].append(dev["plain"].dropna().values)
        for n in ("flow_balance", "uf_resistance"):
            cal_dev[n].append(dh[n].dropna().values)
    tol = {}
    for n, v in cal_dev.items():
        a = np.abs(np.concatenate(v))
        tol[n] = {"p999": MARGIN * float(np.quantile(a, 0.999)), "sd": float(np.concatenate(v).std()),
                  "rows": int(len(a))}

    NAMES = ["pump", "plain", "any_existing", "pump_or_existing"]
    fa = {n: [0, 0] for n in NAMES}
    TARGETS = ("PT-100", "N", "Q")
    sens = {t: {n: {l: [0, 0] for l in LEVELS} for n in NAMES} for t in TARGETS}
    sens_sign = {t: {n: {(l, s): [0, 0] for l in LEVELS for s in (1, -1)} for n in NAMES} for t in TARGETS}
    for name, d in data["test"]:
        dev, ref, phase, steady = deviations(d, coef)
        dh, qh, ph, st = H.deviations(d)
        t = d["Time Elapsed"].values
        ref_bal = qh["flow_balance"] - dh["flow_balance"]
        ref_r = qh["uf_resistance"] - dh["uf_resistance"]
        okp = steady & np.isfinite(dev["pump"].values) & np.isfinite(dev["plain"].values)
        exist_dev = (np.abs(dh["flow_balance"].values) > tol["flow_balance"]["p999"]) | \
                    (np.abs(dh["uf_resistance"].values) > tol["uf_resistance"]["p999"])
        flags = {"pump": H.persist(np.abs(dev["pump"].values) > tol["pump"]["p999"], t),
                 "plain": H.persist(np.abs(dev["plain"].values) > tol["plain"]["p999"], t)}
        ex = H.persist(exist_dev & np.isfinite(dh["flow_balance"].values), t)
        flags["any_existing"] = ex
        flags["pump_or_existing"] = flags["pump"] | ex
        for n in NAMES:
            fa[n][0] += int(flags[n][okp].sum()); fa[n][1] += int(okp.sum())
        for lvl in LEVELS:
            for sign in (+1, -1):
                for target in TARGETS:
                    x = d.copy()
                    col = {"PT-100": "PT-100", "N": "VFD-100-FB", "Q": "FE/FT-100"}[target]
                    x[col] = x[col] * (1 + sign * lvl)
                    pq = pump_quantity(x, coef, steady)
                    lq = plain_quantity(x, steady)
                    fl = {"pump": H.persist((np.abs(pq.values - ref["pump"].values) > tol["pump"]["p999"]) & okp, t),
                          "plain": H.persist((np.abs(lq.values - ref["plain"].values) > tol["plain"]["p999"]) & okp, t)}
                    # existing checks on the spoofed copy (they recompute their own mask)
                    qx, _, stx = H.quantities(x)
                    okx = stx & np.isfinite(ref_bal.values) & np.isfinite(ref_r.values)
                    fb = np.abs(qx["flow_balance"].values - ref_bal.values) > tol["flow_balance"]["p999"]
                    fr = np.abs(qx["uf_resistance"].values - ref_r.values) > tol["uf_resistance"]["p999"]
                    fl["any_existing"] = H.persist((fb | fr) & okx, t)
                    fl["pump_or_existing"] = fl["pump"] | fl["any_existing"]
                    for n in NAMES:
                        sens_sign[target][n][(lvl, sign)][0] += int(fl[n][okp].sum())
                        sens_sign[target][n][(lvl, sign)][1] += int(okp.sum())
    for t_ in TARGETS:
        for n in NAMES:
            for lvl in LEVELS:
                for s in (1, -1):
                    a, b = sens_sign[t_][n][(lvl, s)]
                    sens[t_][n][lvl][0] += a; sens[t_][n][lvl][1] += b

    def share(a, b): return round(a / max(b, 1), 3)
    sens_out = {t_: {n: {f"+/-{int(l*100)}%": {"+": share(*sens_sign[t_][n][(l, 1)]), "-": share(*sens_sign[t_][n][(l, -1)])}
                         for l in LEVELS} for n in NAMES} for t_ in TARGETS}

    def smallest(t_, n):
        for l in LEVELS:
            if min(share(*sens_sign[t_][n][(l, 1)]), share(*sens_sign[t_][n][(l, -1)])) >= 0.9:
                return f"{int(l*100)}%"
        return "none up to 50%"
    summary = {t_: {n: smallest(t_, n) for n in NAMES} for t_ in TARGETS}
    out = {"design": __doc__.split("Design fixed BEFORE")[1].split("Run:")[0].strip(),
           "coefficients_a_b_c": [float(c) for c in coef], "within_day_R2_training": round(r2, 4),
           "within_day_residual_sd_psi": round(rs, 4),
           "tolerances_p999x1.2": {n: {k: round(v, 5) if isinstance(v, float) else v for k, v in c.items()} for n, c in tol.items()},
           "false_alarm_share_test": {n: share(*v) for n, v in fa.items()},
           "false_alarm_rows": {n: v[1] for n, v in fa.items()},
           "smallest_spoof_caught_90pct_both_signs": summary,
           "sensitivity": sens_out,
           "files_used": {s: [n for n, _ in v] for s, v in data.items()}}
    json.dump(out, open(ROOT / "results" / "hueneme_pump_curve.json", "w"), indent=1)
    print(f"curve coefficients a,b,c = {[round(float(c), 3) for c in coef]}; within-day R2 {r2:.3f}, residual sd {rs:.3f} psi")
    print("tolerances (p999 x1.2):", {n: round(c["p999"], 3) for n, c in tol.items()},
          " sd:", {n: round(c["sd"], 3) for n, c in tol.items()})
    print("false alarms on test days:", out["false_alarm_share_test"], out["false_alarm_rows"])
    print("smallest spoof caught on >=90% of steady rows, both signs:")
    for t_ in TARGETS:
        print(f"  {t_:7s}", summary[t_])
    for t_ in TARGETS:
        print(f"\nspoof {t_}: share flagged (+ / -)")
        for n in ("pump", "plain", "any_existing"):
            print(f"  {n:13s}", {k: (v['+'], v['-']) for k, v in sens_out[t_][n].items()})


if __name__ == "__main__":
    main()
