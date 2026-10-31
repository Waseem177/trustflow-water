"""
Turbidity-linked UF resistance check on Port Hueneme (STATUS item 7b).

Question: the existing UF check (hueneme_uf.py) has 0% false alarms on test days but
only catches flow spoofs >=20% and TMP spoofs ~50%, because resistance legitimately
swings with feed-water quality. Cake-filtration theory says resistance gained per m3
filtered is proportional to feed solids, so adding the turbidity tag Tu/TuT-100
should tighten the tolerance. Does it?

Same data, same rules as hueneme_uf.py (imported, not copied): training/ days to
calibrate, test/ days to measure false alarms, tolerance = 1.2 x 99.9th percentile
of attack-free |deviation| (the 1.2 x worst rule is also reported, as there), alarm
needs >= 30 s over tolerance, spoofs of +/-2, 5, 10, 20, 50% on FT-100 and on the TMP
(via PT-101) at onset with the clean reference frozen.

Exploration was done on TRAINING days only (test-day results not looked at):
  - Turbidity-valid (>0 NTU) on 52% of steady training rows; 35 of 118 training files
    are valid on >90% of steady rows. Turbidity is low (median 0.03-1.6 NTU by day).
  - Cake-slope version (suggested in the task): per filtration cycle, slope of
    R = TMP / (mu * FT-100 / n_active) against volume filtered, 464 cycles on valid
    training days. R rises only ~2% per cycle (715 -> 730), the slope has 48%
    relative spread, and it is NOT related to mean turbidity (Pearson r = -0.16,
    Spearman -0.12; slope against cumulative solids load is 130% spread). A
    per-cycle slope check is therefore not used: it would be looser than the level
    check and gives one verdict per ~13 min cycle, not per row.
  - Level version: the existing deviation dev = ln R - trailing phase-bin median
    has sd 0.053 on valid training days; its correlation with the change in
    ln(turbidity) against its own trailing median is -0.06 (R^2 0.3%).
    That is the pre-registered design below; this exploration says to expect
    little gain, and the result is reported either way.

Design fixed BEFORE running on test days (20 Oct):
  - Days: files used by hueneme_uf.py (standard 18-column CSVs, interval <= 10 s) that
    are turbidity-valid (> 0 NTU) on >= 90% of their steady rows, in both splits.
    Rows with turbidity <= 0.005 NTU (probe floor) are not evaluated.
  - Steady rows, flow balance, TMP and resistance R exactly as hueneme_uf.py.
  - Turbidity feature: x = ln(Tu), Tu smoothed by a 60 s rolling median;
    dx = x - trailing(x) with the same 6 h window and 1 h lag, no phase bins.
  - Turbidity-linked resistance deviation:
        dev_T = [ ln R - trailing_phase(ln R) ] - beta * dx
    beta = least-squares slope of the resistance deviation on dx over the training
    turbidity-valid rows (one global value, fitted before any test day is touched).
    The check compares ln R against reference_T = trailing_phase(ln R) + beta * dx.
  - Tolerance from training turbidity-valid days: 1.2 x 99.9th pct of |dev_T|
    (and 1.2 x worst, reported).
  - Comparators, both on the SAME turbidity-valid days: (a) the existing check
    (no turbidity term) with tolerance calibrated on the same training subset, and
    (b) the existing check as published (all training days, results/hueneme_uf.json).
    (a) separates the effect of the turbidity term from the effect of restricting to
    clean-sensor days, which by itself changes the tolerance.
  - Sensitivity: share of steady rows flagged by flow_balance OR resistance, for
    each spoof level, sign pooled and per sign; "smallest spoof caught on 90% of
    rows" = smallest level at which both signs reach >= 0.9, from the five levels.

Run: .venv/bin/python src/hueneme_uf_turbidity.py  ->  results/hueneme_uf_turbidity.json
"""
from __future__ import annotations
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path

import hueneme_uf as H

ROOT = H.ROOT
TU_COL = "Tu/TuT-100"
MIN_VALID, TU_FLOOR, SMOOTH = 0.9, 0.005, "60s"
LEVELS = H.LEVELS


def tu_feature(d, steady):
    """dx = ln(smoothed turbidity) minus its trailing reference (NaN off steady rows)."""
    tu = d[TU_COL].rolling(SMOOTH, min_periods=1).median()
    x = np.log(tu.where(steady & (tu > TU_FLOOR)))
    return x - H.trailing(x)


def valid_fraction(d, steady):
    return float(((d[TU_COL] > 0) & steady).sum() / max(int(steady.sum()), 1))


def load_days():
    data, skipped = {"training": [], "test": []}, []
    for split in data:
        for p in sorted(glob.glob(str(H.BASE / split / "*.csv"))):
            d, why = H.load(p)
            if d is None:
                skipped.append(f"{split}/{Path(p).name}: {why}")
                continue
            _, _, steady = H.quantities(d)
            f = valid_fraction(d, steady)
            if f >= MIN_VALID:
                data[split].append((Path(p).stem, d))
            else:
                skipped.append(f"{split}/{Path(p).name}: turbidity valid on {f:.0%} of steady rows")
    return data, skipped


def day_devs(d, beta):
    """Deviations for the existing and turbidity-linked checks; also frozen references."""
    dev, q, phase, steady = H.deviations(d)
    dx = tu_feature(d, steady)
    dev_t = dev["uf_resistance"] - (beta * dx if beta is not None else 0.0)
    return dev, dev_t, dx, q, phase, steady


def first_level_90(share):
    """Smallest level where both signs reach 0.9 (None if never)."""
    for l in LEVELS:
        if share[f"+{int(l*100)}%"] >= 0.9 and share[f"-{int(l*100)}%"] >= 0.9:
            return f"{int(l*100)}%"
    return None


def main():
    data, skipped = load_days()
    print(f"turbidity-valid files: training {len(data['training'])}, test {len(data['test'])}")

    # beta and calibration (training only)
    xs, ys = [], []
    for _, d in data["training"]:
        dev, _, dx, *_ = day_devs(d, None)
        m = np.isfinite(dev["uf_resistance"].values) & np.isfinite(dx.values)
        xs.append(dx.values[m]); ys.append(dev["uf_resistance"].values[m])
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    beta = float(np.polyfit(xs, ys, 1)[0])
    r2 = float(np.corrcoef(xs, ys)[0, 1] ** 2)
    print(f"beta = {beta:.5f}, R^2 of resistance deviation on turbidity change = {r2:.4f}, rows {len(xs)}")

    dev_tr = {"flow_balance": [], "uf_res_existing": [], "uf_res_turbidity": []}
    for _, d in data["training"]:
        dev, dev_t, dx, *_ = day_devs(d, beta)
        m = np.isfinite(dx.values)
        dev_tr["flow_balance"].append(dev["flow_balance"].dropna().values)
        dev_tr["uf_res_existing"].append(dev["uf_resistance"].values[m & np.isfinite(dev["uf_resistance"].values)])
        dev_tr["uf_res_turbidity"].append(dev_t.values[m & np.isfinite(dev_t.values)])
    cal = {}
    for n, v in dev_tr.items():
        a = np.abs(np.concatenate(v))
        cal[n] = {"worst": H.MARGIN * float(a.max()), "p999": H.MARGIN * float(np.quantile(a, 0.999)),
                  "sd": float(a.std()), "rows": int(len(a))}

    variants = {"turbidity": "uf_res_turbidity", "existing_same_days": "uf_res_existing"}
    rules = ("worst", "p999")
    fa = {v: {r: {"resistance": [0, 0], "any": [0, 0]} for r in rules} for v in variants}
    sens = {v: {r: {tg: {f"{s}{int(l*100)}%": [0, 0] for l in LEVELS for s in "+-"}
                    for tg in ("FT-100", "TMP", )} for r in rules} for v in variants}
    sens_res_only = {v: {tg: {f"{s}{int(l*100)}%": [0, 0] for l in LEVELS for s in "+-"}
                         for tg in ("FT-100", "TMP")} for v in variants}   # p999 rule, resistance check alone

    for name, d in data["test"]:
        dev, dev_t, dx, q, phase, steady = day_devs(d, beta)
        t = d["Time Elapsed"].values
        ok_tu = np.isfinite(dx.values)
        devs = {"uf_res_turbidity": dev_t.values, "uf_res_existing": dev["uf_resistance"].values}
        for v, key in variants.items():
            for r in rules:
                fres = persist_flag(np.abs(devs[key]) > cal[key][r], t) & ok_tu
                fbal = persist_flag(np.abs(dev["flow_balance"].values) > cal["flow_balance"][r], t)
                ok_r = ok_tu & np.isfinite(devs[key])
                fa[v][r]["resistance"][0] += int(fres[ok_r].sum()); fa[v][r]["resistance"][1] += int(ok_r.sum())
                anyf = (fres | (fbal & np.isfinite(dev["flow_balance"].values)))
                ok_a = ok_tu & steady
                fa[v][r]["any"][0] += int(anyf[ok_a].sum()); fa[v][r]["any"][1] += int(ok_a.sum())
        ref_bal = q["flow_balance"] - dev["flow_balance"]
        ref_r = q["uf_resistance"] - dev["uf_resistance"]          # existing reference
        ref_t = ref_r + beta * dx                                    # turbidity-linked reference
        refs = {"uf_res_turbidity": ref_t.values, "uf_res_existing": ref_r.values}
        for lvl in LEVELS:
            for sign in (+1, -1):
                for target in ("FT-100", "TMP"):
                    x = d.copy()
                    if target == "FT-100":
                        x["FE/FT-100"] = x["FE/FT-100"] * (1 + sign * lvl)
                    else:
                        x["PT-101"] = x["PT-104"] + (x["PT-101"] - x["PT-104"]) * (1 + sign * lvl)
                    qx, _, st = H.quantities(x)
                    lab = f"{'+' if sign > 0 else '-'}{int(lvl*100)}%"
                    for v, key in variants.items():
                        ok = st & np.isfinite(ref_bal.values) & np.isfinite(refs[key]) & ok_tu
                        dres = np.abs(qx["uf_resistance"].values - refs[key])
                        dbal = np.abs(qx["flow_balance"].values - ref_bal.values)
                        for r in rules:
                            fb = dbal > cal["flow_balance"][r]
                            fr = dres > cal[key][r]
                            hit = H.persist((fb | fr) & ok, t)
                            sens[v][r][target][lab][0] += int(hit[ok].sum()); sens[v][r][target][lab][1] += int(ok.sum())
                        hit_r = H.persist((dres > cal[key]["p999"]) & ok, t)
                        sens_res_only[v][target][lab][0] += int(hit_r[ok].sum()); sens_res_only[v][target][lab][1] += int(ok.sum())

    def share(dct):
        return {k: round(a / max(b, 1), 3) for k, (a, b) in dct.items()}

    out = {"design": __doc__.split("Design fixed BEFORE")[1].split("Run:")[0].strip(),
           "beta": round(beta, 5), "r2_resistance_deviation_on_turbidity_change": round(r2, 4),
           "files_used": {s: [n for n, _ in v] for s, v in data.items()},
           "files_not_used": skipped,
           "tolerances": {n: {k: round(v, 5) if isinstance(v, float) else v for k, v in c.items()} for n, c in cal.items()},
           "false_alarms": {v: {r: {k: round(a / max(b, 1), 5) for k, (a, b) in fa[v][r].items()}
                                 | {"rows": {k: b for k, (a, b) in fa[v][r].items()}} for r in rules} for v in variants},
           "sensitivity_flow_or_resistance": {v: {r: {tg: share(s) for tg, s in sens[v][r].items()} for r in rules} for v in variants},
           "sensitivity_resistance_only_p999": {v: {tg: share(s) for tg, s in sens_res_only[v].items()} for v in variants},
           }
    summ = {}
    for v in variants:
        summ[v] = {tg: {"smallest_spoof_caught_90pct_both_signs": first_level_90(share(sens[v]["p999"][tg]))}
                   for tg in ("FT-100", "TMP")}
        summ[v]["false_alarm_any_p999"] = out["false_alarms"][v]["p999"]["any"]
    out["summary_p999"] = summ
    out["published_check_all_days"] = {"source": "results/hueneme_uf.json",
                                       "note": "1.2 x p99.9 rule, 118 train / 21 test files, pooled signs"}
    json.dump(out, open(ROOT / "results" / "hueneme_uf_turbidity.json", "w"), indent=1)
    print(json.dumps({"summary_p999": summ, "tolerances": out["tolerances"]}, indent=1))
    for v in variants:
        print(f"\n== {v}")
        print(" false alarms p999:", out["false_alarms"][v]["p999"])
        for tg in ("FT-100", "TMP"):
            print(f" spoof {tg} (flow or resistance), p999:", out["sensitivity_flow_or_resistance"][v]["p999"][tg])
            print(f" spoof {tg} (resistance only), p999:", out["sensitivity_resistance_only_p999"][v][tg])


def persist_flag(flag, times):
    flag = np.asarray(flag, bool) & True
    return H.persist(flag, times)


if __name__ == "__main__":
    main()
