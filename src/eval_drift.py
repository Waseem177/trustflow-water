"""
Does capping how fast the running reference may move stop slow-drift spoofs?

Design fixed before running (20 Oct). Simulator, 60 days, seeds 0-4; calibrate
on days 0-30 (tolerances exactly as eval_swro_invariants; caps from
drift_guard.fit_cap on the same days); test on days 30-60.

  False alarms   days 30-60, eight process checks and the energy check,
                 running reference uncapped vs capped. Same tolerances.
  Attack A       reported permeate TDS lowered by x (10%, 20%), ramped
                 linearly over D (0 = step, 2 h, 12 h, 48 h), then held 12 h;
                 onset day 45. Caught by the salt-passage check if at all.
  Attack B       HP pump head changed by +x (5%, 10%) with the same ramps
                 while every process tag replays the clean record; only the
                 electrical meter shows the truth (as eval_energy.py).
  References are computed from what the plant reports, as deployed: the
  attacker's ramp does reach the reference.
  Metrics        share of attack rows flagged; first alarm (min after onset);
                 share flagged in the final 2 h of the attack.
  Also reported  each cap as the slowest drift it lets through: a ramp slower
                 than the cap looks like fouling and is not caught by design.

Run: .venv/bin/python src/eval_drift.py  ->  results/drift_guard.json
"""
from __future__ import annotations
import json
import numpy as np
import torch
from pathlib import Path

from swro_sim import simulate
from invariants_swro import (NAMES, DRIFTING, fit, residuals, quantities, trailing_reference)
import energy_check as E
import drift_guard as G

ROOT = Path(__file__).resolve().parent.parent
DAYS, CAL_DAYS, SEEDS = 60, 30, range(5)
TRAILING = dict(window_min=360, lag_min=60)
ONSET_MIN, HOLD_MIN = 45 * 1440, 12 * 60
RAMPS = [0, 120, 720, 2880]
LEVELS_A, LEVELS_B = [0.10, 0.20], [0.05, 0.10]


def ramp(n, D):
    t = np.arange(n) - ONSET_MIN
    frac = np.ones(n) if D == 0 else np.clip(t / D, 0, 1)
    frac[t < 0] = 0.0
    return frac


def window(D, n):
    w = np.zeros(n, bool); w[ONSET_MIN:min(n, ONSET_MIN + D + HOLD_MIN)] = True
    return w


def summary(flag, w):
    f = flag[w]
    hit = np.flatnonzero(f)
    return {"share": float(f.mean()), "first_alarm_min": int(hit[0]) if len(hit) else None,
            "share_last_2h": float(f[-120:].mean())}


def run(seed):
    _, rep, ev = simulate(days=DAYS, seed=seed)
    n = len(rep); t = np.arange(n, dtype=float)
    m = (ev.day < CAL_DAYS).values
    log = {"cip": np.flatnonzero(ev.cip.values), "ceb": np.flatnonzero(ev.ceb.values), "uf_phase": ev.uf_phase.values}
    log_cal = {"cip": log["cip"][log["cip"] < m.sum()], "ceb": log["ceb"][log["ceb"] < m.sum()],
               "uf_phase": log["uf_phase"][m]}
    cal = fit(rep[m], trailing=TRAILING, events=log_cal)
    group = {k: (log["uf_phase"] if k == "uf_resistance" else None) for k in DRIFTING}

    raw_clean = trailing_reference(quantities(rep, cal["orp_k"]), **TRAILING, events=log)
    caps = {k: G.fit_cap(raw_clean[k].numpy()[m], t[m], group=None if group[k] is None else group[k][m])
            for k in DRIFTING}

    def refs(x, capped):
        raw = trailing_reference(quantities(x, cal["orp_k"]), **TRAILING, events=log)
        if not capped:
            return raw
        return {k: torch.tensor(G.cap(raw[k].numpy(), t, caps[k], group=group[k]), dtype=torch.float32)
                for k in DRIFTING}

    def flags8(x, capped):
        return (residuals(x, cal, ref=refs(x, capped), events=log).abs() > 1).numpy().any(1)

    # energy check
    r_clean = E.residual(rep)
    eref_clean = E.trailing(r_clean)
    ecal = E.fit(r_clean[m], eref_clean[m])
    ecap = G.fit_cap(eref_clean[m], t[m])

    def eflag(r, capped):
        ref = E.trailing(r)
        if capped:
            ref = G.cap(ref, t, ecap)
        return E.violations(r, ecal, ref)

    out = {"false_alarms": {}, "A": {}, "B": {},
           "caps_percent_per_hour": {k: round(100 * caps[k] * 60, 4) for k in DRIFTING},
           "energy_cap_percent_per_hour": round(100 * ecap * 60, 4)}
    for capped in (False, True):
        key = "capped" if capped else "uncapped"
        out["false_alarms"][key] = {"process": float(flags8(rep, capped)[~m].mean()),
                                    "energy": float(eflag(r_clean, capped)[~m].mean())}
    for x in LEVELS_A:
        for D in RAMPS:
            xa = rep.copy()
            xa["C_perm"] = xa["C_perm"].values * (1 - x * ramp(n, D))
            w = window(D, n)
            out["A"][f"{x:.0%} over {D} min"] = {("capped" if c else "uncapped"): summary(flags8(xa, c), w)
                                                 for c in (False, True)}
    for x in LEVELS_B:
        for D in RAMPS:
            xb = rep.copy()
            xb["W_hp"] = xb["W_hp"].values * (1 + x * ramp(n, D))
            r = E.residual(xb); w = window(D, n)
            out["B"][f"+{x:.0%} over {D} min"] = {("capped" if c else "uncapped"): summary(eflag(r, c), w)
                                                  for c in (False, True)}
    return out


def mean_of(runs, path):
    vals = []
    for r in runs:
        v = r
        for p in path:
            v = v[p]
        vals.append(v)
    if all(isinstance(v, (int, float)) or v is None for v in vals):
        nums = [v for v in vals if v is not None]
        return None if not nums else round(float(np.mean(nums)), 4)
    return vals


def main():
    runs = [run(s) for s in SEEDS]
    res = {"design": __doc__.split("Design fixed before running (20 Oct).")[1].split("Run:")[0].strip(),
           "false_alarms": {c: {k: {"mean": mean_of(runs, ["false_alarms", c, k]),
                                    "max": round(max(r["false_alarms"][c][k] for r in runs), 5)}
                                for k in ("process", "energy")} for c in ("uncapped", "capped")},
           "caps_percent_per_hour": {k: round(float(np.mean([r["caps_percent_per_hour"][k] for r in runs])), 4)
                                     for k in DRIFTING},
           "energy_cap_percent_per_hour": round(float(np.mean([r["energy_cap_percent_per_hour"] for r in runs])), 4),
           "A_permeate_TDS_ramp": {}, "B_pump_head_ramp": {}}
    for part, key in (("A", "A_permeate_TDS_ramp"), ("B", "B_pump_head_ramp")):
        for scen in runs[0][part]:
            res[key][scen] = {c: {mtr: mean_of(runs, [part, scen, c, mtr]) for mtr in ("share", "first_alarm_min", "share_last_2h")}
                              for c in ("uncapped", "capped")}
    json.dump(res, open(ROOT / "results" / "drift_guard.json", "w"), indent=1)

    print("false alarms, days 30-60 (mean / worst seed):")
    for c in ("uncapped", "capped"):
        fa = res["false_alarms"][c]
        print(f"  {c:9s} process {fa['process']['mean']:.4%} / {fa['process']['max']:.4%}   energy {fa['energy']['mean']:.4%} / {fa['energy']['max']:.4%}")
    print("caps (% per hour):", res["caps_percent_per_hour"], "| energy", res["energy_cap_percent_per_hour"])
    for key, title in (("A_permeate_TDS_ramp", "A: permeate TDS lowered"), ("B_pump_head_ramp", "B: pump head raised, process tags replay")):
        print(f"\n{title}   share flagged | first alarm (min) | share in last 2 h   [uncapped -> capped]")
        for scen, v in res[key].items():
            u, c = v["uncapped"], v["capped"]
            print(f"  {scen:18s} {u['share']:.2f} -> {c['share']:.2f} | {u['first_alarm_min']} -> {c['first_alarm_min']} | {u['share_last_2h']:.2f} -> {c['share_last_2h']:.2f}")


if __name__ == "__main__":
    main()
