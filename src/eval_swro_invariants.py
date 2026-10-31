"""
How good are the SWRO invariants before any attacker optimises against them?

  1. False alarms on attack-free data: calibrate on days 0-30 of a simulated
     60-day run, count violations on days 30-60. Repeated over seeds.
  2. Single-sensor spoof sensitivity: bias one reported tag at a time over the
     test period and record which invariants fire, and the smallest bias that
     at least 90% of rows flag. References are frozen from the clean data,
     i.e. this measures a spoof at onset, before a trailing reference could
     absorb it.

Both use a static reference and a trailing reference (6 h median, 1 h lag;
UF resistance against the same point in the backwash cycles of that window).
This is the *naive* view; item 8 (adaptive attacker vs the full set) is the
real test.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

from swro_sim import simulate, TAGS
from invariants_swro import NAMES, fit, residuals, quantities, trailing_reference

ROOT = Path(__file__).resolve().parent.parent
DAYS, CAL_DAYS, SEEDS = 60, 30, range(5)
TRAILING = dict(window_min=360, lag_min=60)
ADDITIVE = {"T_feed": [0.25, 0.5, 1, 2, 5], "pH_feed": [0.05, 0.1, 0.2, 0.5, 1.0],
            "ORP_pre": [5, 10, 25, 50, 100, 200], "ORP_ro": [5, 10, 25, 50, 100, 200]}
RELATIVE = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
DETECT = 0.9


def run(seed):
    _, rep, ev = simulate(days=DAYS, seed=seed)
    m = (ev.day < CAL_DAYS).values
    log = {"cip": np.flatnonzero(ev.cip.values), "ceb": np.flatnonzero(ev.ceb.values),
           "uf_phase": ev.uf_phase.values}
    log_cal = {"cip": log["cip"][log["cip"] < m.sum()], "ceb": log["ceb"][log["ceb"] < m.sum()],
               "uf_phase": log["uf_phase"][m]}
    cals = {"static": fit(rep[m]),
            "trailing": fit(rep[m], trailing=TRAILING, events=log_cal)}
    return rep, log, m, cals


def false_alarms():
    out = {mode: [] for mode in ("static", "trailing")}
    for seed in SEEDS:
        rep, log, m, cals = run(seed)
        for mode, cal in cals.items():
            V = (residuals(rep, cal, events=log).abs() > 1).numpy()[~m]
            out[mode].append({"any": float(V.any(1).mean()),
                              **{n: float(v) for n, v in zip(NAMES, V.mean(0))}})
    summary = {}
    for mode, runs in out.items():
        summary[mode] = {k: {"mean": round(float(np.mean([r[k] for r in runs])), 5),
                             "max": round(float(np.max([r[k] for r in runs])), 5)}
                         for k in ["any"] + NAMES}
    return summary


def sensitivity():
    rep, log, m, cals = run(0)
    test = rep[~m].reset_index(drop=True)
    out = {}
    for mode, cal in cals.items():
        ref = None
        if cal["trailing"]:
            full = trailing_reference(quantities(rep, cal["orp_k"]), **TRAILING, events=log)
            ref = {k: v[~m] for k, v in full.items()}
        res = {}
        for tag in TAGS:
            levels = ADDITIVE.get(tag, RELATIVE)
            rows = []
            for lvl in levels:
                for sign in (+1, -1):
                    x = test.copy()
                    x[tag] = x[tag] + sign * lvl if tag in ADDITIVE else x[tag] * (1 + sign * lvl)
                    V = (residuals(x, cal, ref=ref).abs() > 1).numpy()
                    rows.append({"bias": sign * lvl, "any": round(float(V.any(1).mean()), 4),
                                 "by": {n: round(float(v), 4) for n, v in zip(NAMES, V.mean(0)) if v > 0.01}})
            det = [abs(r["bias"]) for r in rows if r["any"] >= DETECT]
            # smallest level caught in BOTH directions
            both = [l for l in levels if all(r["any"] >= DETECT for r in rows if abs(r["bias"]) == l)]
            res[tag] = {"unit": "relative" if tag not in ADDITIVE else TAGS[tag],
                        "min_detectable_both_signs": min(both) if both else None,
                        "min_detectable_either_sign": min(det) if det else None,
                        "levels": rows}
        out[mode] = res
    return out, cals


def main():
    fa = false_alarms()
    sens, cals = sensitivity()
    tol = {mode: {"static": {k: round(v, 4) for k, v in c["tol"].items() if v == v},
                  "trailing": {k: round(v, 4) for k, v in c["tol_trail"].items() if v == v}}
           for mode, c in cals.items()}
    out = {"setup": {"days": DAYS, "calibration_days": CAL_DAYS, "seeds": len(SEEDS),
                     "trailing": TRAILING, "detect_threshold": DETECT,
                     "tolerance_note": "flow/salt balance: fraction of feed; ro_*/uf_*: natural-log "
                                       "units (0.05 ~ 5%); orp_*: mV"},
           "tolerances": tol, "false_alarms": fa, "sensitivity": sens}
    json.dump(out, open(ROOT / "results" / "swro_invariants.json", "w"), indent=1)

    print("false alarms on held-out normal data (mean / worst over seeds):")
    for mode, s in fa.items():
        print(f"  {mode:8s} any {s['any']['mean']:.4%} / {s['any']['max']:.4%}   " +
              "  ".join(f"{k}={v['mean']:.3%}" for k, v in s.items() if k != "any" and v["max"] > 0))
    print("\nsmallest bias caught on >=90% of rows, both directions (static | trailing):")
    for tag in TAGS:
        a, b = (sens[mo][tag]["min_detectable_both_signs"] for mo in ("static", "trailing"))
        fmt = lambda v: "-" if v is None else (f"{v:.0%}" if tag not in ADDITIVE else f"{v:g} {TAGS[tag]}")
        catch = {n for r in sens["trailing"][tag]["levels"] for n in r["by"]}
        print(f"  {tag:8s} {fmt(a):>10s} | {fmt(b):>10s}   via {', '.join(sorted(catch)) or 'nothing'}")


if __name__ == "__main__":
    main()
