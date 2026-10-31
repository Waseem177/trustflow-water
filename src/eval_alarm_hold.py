"""
Does holding the reference during an alarm keep a detected drift detected?

Design fixed before running (23 Oct). Same simulator, seeds, calibration, caps,
tolerances and attack scenarios as eval_drift.py (permeate TDS lowered, A;
pump head raised under full replay, B). Two variants, both rate-capped:

  capped   the deployed design (drift_guard.cap)
  hold     the same, plus drift_guard.cap(q=..., tol=...): a check's reference
           does not move on rows where that check is already out of tolerance

  Hold applies to every drifting check: the five with a plain deviation q - ref
  (water and salt permeability, brine dP, UF resistance, pretreatment ORP) with
  their trailing tolerance, RO-feed ORP against its band tolerance, and the
  energy check against its tolerance.

  Metrics  share of attack rows flagged, first alarm, share flagged in the
           final 2 h of the attack (the one hold is meant to raise); and false
           alarms on days 30-60 as rows and as alarm episodes per day
           (alarms_per_day.count), since a hold can latch a harmless shift.
  Pass     the hold is adopted only if false-alarm episodes per day do not
           rise by more than 0.1 (mean over seeds) and no seed's false-alarm
           row share exceeds 1%.

Run: .venv/bin/python src/eval_alarm_hold.py  ->  results/alarm_hold.json
"""
from __future__ import annotations
import json
import numpy as np
import torch
from pathlib import Path

from swro_sim import simulate
from invariants_swro import DRIFTING, fit, residuals, quantities, trailing_reference
import energy_check as E
import drift_guard as G
from eval_drift import DAYS, CAL_DAYS, SEEDS, TRAILING, RAMPS, LEVELS_A, LEVELS_B, ramp, window, summary
from alarms_per_day import count

ROOT = Path(__file__).resolve().parent.parent
HOLD_TOL_KEY = {k: k for k in DRIFTING if k != "orp_ro"} | {"orp_ro": "orp_ro_band"}
VARIANTS = ("capped", "hold")


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

    def flags8(x, variant):
        q = quantities(x, cal["orp_k"])
        raw = trailing_reference(q, **TRAILING, events=log)
        ref = {}
        for k in DRIFTING:
            kw = {}
            if variant == "hold":
                kw = {"q": q[k].numpy(), "tol": cal["tol_trail"][HOLD_TOL_KEY[k]]}
            ref[k] = torch.tensor(G.cap(raw[k].numpy(), t, caps[k], group=group[k], **kw), dtype=torch.float32)
        return (residuals(x, cal, ref=ref, events=log).abs() > 1).numpy().any(1)

    r_clean = E.residual(rep)
    eref_clean = E.trailing(r_clean)
    ecal = E.fit(r_clean[m], eref_clean[m])
    ecap = G.fit_cap(eref_clean[m], t[m])

    def eflag(r, variant):
        kw = {"q": r, "tol": ecal["tol"]} if variant == "hold" else {}
        return E.violations(r, ecal, G.cap(E.trailing(r), t, ecap, **kw))

    out = {"false_alarms": {}, "A": {}, "B": {}}
    for v in VARIANTS:
        fp, fe = flags8(rep, v), eflag(r_clean, v)
        out["false_alarms"][v] = {"process_rows": float(fp[~m].mean()), "energy_rows": float(fe[~m].mean()),
                                  "process_per_day": count(fp, 60, ~m)["episodes_per_day"],
                                  "energy_per_day": count(fe, 60, ~m)["episodes_per_day"]}
    for x in LEVELS_A:
        for D in RAMPS:
            xa = rep.copy()
            xa["C_perm"] = xa["C_perm"].values * (1 - x * ramp(n, D))
            w = window(D, n)
            out["A"][f"{x:.0%} over {D} min"] = {v: summary(flags8(xa, v), w) for v in VARIANTS}
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
                  "max": round(max(r["false_alarms"][v][k] for r in runs), 5)}
              for k in runs[0]["false_alarms"][v]} for v in VARIANTS}
    res = {"design": __doc__.split("Design fixed before running (23 Oct).")[1].split("Run:")[0].strip(),
           "false_alarms": fa, "A_permeate_TDS_ramp": {}, "B_pump_head_ramp": {}}
    for part, key in (("A", "A_permeate_TDS_ramp"), ("B", "B_pump_head_ramp")):
        for scen in runs[0][part]:
            res[key][scen] = {v: {mt: mean([r[part][scen][v][mt] for r in runs])
                                  for mt in ("share", "first_alarm_min", "share_last_2h")} for v in VARIANTS}
    rise = max(fa["hold"][k]["mean"] - fa["capped"][k]["mean"] for k in ("process_per_day", "energy_per_day"))
    worst_rows = max(fa["hold"][k]["max"] for k in ("process_rows", "energy_rows"))
    res["pass"] = {"episodes_per_day_rise": round(rise, 3), "worst_seed_row_share": worst_rows,
                   "adopt": bool(rise <= 0.1 and worst_rows <= 0.01)}
    json.dump(res, open(ROOT / "results" / "alarm_hold.json", "w"), indent=1)

    print("false alarms, days 30-60 (mean / worst seed):")
    for v in VARIANTS:
        f = fa[v]
        print(f"  {v:7s} process {f['process_rows']['mean']:.4%} rows, {f['process_per_day']['mean']} /day | "
              f"energy {f['energy_rows']['mean']:.4%} rows, {f['energy_per_day']['mean']} /day | worst rows "
              f"{max(f['process_rows']['max'], f['energy_rows']['max']):.4%}")
    print("pass:", res["pass"])
    for key, title in (("A_permeate_TDS_ramp", "A: permeate TDS lowered"), ("B_pump_head_ramp", "B: pump head raised, process tags replay")):
        print(f"\n{title}   share flagged | first alarm (min) | share in last 2 h   [capped -> hold]")
        for scen, v in res[key].items():
            c, h = v["capped"], v["hold"]
            print(f"  {scen:18s} {c['share']:.2f} -> {h['share']:.2f} | {c['first_alarm_min']} -> {h['first_alarm_min']} | {c['share_last_2h']:.2f} -> {h['share_last_2h']:.2f}")


if __name__ == "__main__":
    main()
