"""
Alarm hold on the energy check only, on fresh simulator runs.

Why this exists: eval_alarm_hold.py (seeds 0-4) failed its pre-set criterion
because the process checks' false alarms rose from 0.17 to 1.3 a day, while the
energy check's stayed at 0. Choosing "energy only" after seeing that is a
post-hoc choice, so it is tested here on seeds 5-9, which no hold result has
touched.

Design fixed before running (23 Oct), same scenarios and metrics as
eval_alarm_hold.py part B (pump head +5% / +10%, ramps 0, 2 h, 12 h, 48 h, full
process replay), energy check capped vs capped + hold.
  Pass   adopt the hold for the energy check if its false alarms on days 30-60
         stay at or below 0.1 episodes a day (mean over seeds) and no seed's
         row share exceeds 1%.

Run: .venv/bin/python src/eval_alarm_hold_energy.py  ->  results/alarm_hold_energy.json
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

from swro_sim import simulate
import energy_check as E
import drift_guard as G
from eval_drift import DAYS, CAL_DAYS, RAMPS, LEVELS_B, ramp, window, summary
from alarms_per_day import count

ROOT = Path(__file__).resolve().parent.parent
SEEDS = range(5, 10)
VARIANTS = ("capped", "hold")


def run(seed):
    _, rep, ev = simulate(days=DAYS, seed=seed)
    n = len(rep); t = np.arange(n, dtype=float)
    m = (ev.day < CAL_DAYS).values
    r_clean = E.residual(rep)
    eref_clean = E.trailing(r_clean)
    ecal = E.fit(r_clean[m], eref_clean[m])
    ecap = G.fit_cap(eref_clean[m], t[m])

    def eflag(r, variant):
        kw = {"q": r, "tol": ecal["tol"]} if variant == "hold" else {}
        return E.violations(r, ecal, G.cap(E.trailing(r), t, ecap, **kw))

    out = {"false_alarms": {}, "B": {}}
    for v in VARIANTS:
        f = eflag(r_clean, v)
        out["false_alarms"][v] = {"rows": float(f[~m].mean()), "per_day": count(f, 60, ~m)["episodes_per_day"]}
    for x in LEVELS_B:
        for D in RAMPS:
            xb = rep.copy()
            xb["W_hp"] = xb["W_hp"].values * (1 + x * ramp(n, D))
            r = E.residual(xb); w = window(D, n)
            out["B"][f"+{x:.0%} over {D} min"] = {v: summary(eflag(r, v), w) for v in VARIANTS}
    return out


def mean(vals):
    nums = [v for v in vals if v is not None]
    return None if not nums else round(float(np.mean(nums)), 4)


def main():
    runs = [run(s) for s in SEEDS]
    fa = {v: {k: {"mean": mean([r["false_alarms"][v][k] for r in runs]),
                  "max": round(max(r["false_alarms"][v][k] for r in runs), 5)} for k in ("rows", "per_day")}
          for v in VARIANTS}
    res = {"design": __doc__.split("Design fixed before running (23 Oct),")[1].split("Run:")[0].strip(),
           "seeds": list(SEEDS), "false_alarms": fa, "B_pump_head_ramp": {}}
    for scen in runs[0]["B"]:
        res["B_pump_head_ramp"][scen] = {v: {mt: mean([r["B"][scen][v][mt] for r in runs])
                                             for mt in ("share", "first_alarm_min", "share_last_2h")} for v in VARIANTS}
    res["pass"] = {"adopt": bool(fa["hold"]["per_day"]["mean"] <= 0.1 and fa["hold"]["rows"]["max"] <= 0.01)}
    json.dump(res, open(ROOT / "results" / "alarm_hold_energy.json", "w"), indent=1)
    print("false alarms (mean / worst):", json.dumps(fa))
    print("pass:", res["pass"])
    print("\nshare flagged | first alarm (min) | share in last 2 h   [capped -> hold]")
    for scen, v in res["B_pump_head_ramp"].items():
        c, h = v["capped"], v["hold"]
        print(f"  {scen:18s} {c['share']:.2f} -> {h['share']:.2f} | {c['first_alarm_min']} -> {h['first_alarm_min']} | {c['share_last_2h']:.2f} -> {h['share_last_2h']:.2f}")


if __name__ == "__main__":
    main()
