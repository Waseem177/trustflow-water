"""
When are the checks switched off, and can an attacker switch them off?

The process checks run only in steady operation, so start-up transients do not
set the tolerances. That leaves windows with no check, and on SWaT the "steady"
gate reads the HP pump's status tag (P501), which an attacker can spoof.

Design fixed before running (24 Oct).
  Part 1, exposure. "Water flowing" is judged from measured flow only:
    SWaT        FIT501 > 0.5 m3/h (RO feed)
    Port Hueneme  FE/FT-100 > 10 (UF feed)
  For each check: share of flowing time it is off; number of off-while-flowing
  windows per day; median and longest window. Data: SWaT 2015 normal week
  (all rows), SWaT 2015 attack file (attack-free rows), Port Hueneme test files.
  Part 2, the status-tag hole (SWaT). Proposed identity check "pump status vs
  flow": alarm when P501 reports not running while FIT501 > 0.5 m3/h, or reports
  running while FIT501 < 0.5 m3/h, for at least STATUS_PERSIST_S = 30 s, with no
  tolerance to fit. Measured: alarm rows and episodes per day on the held-out
  30% of the normal week and on the attack file's attack-free rows; which
  labelled attacks trip it.
  Adopt the status check if it raises no more than 1 alarm episode a day on the
  held-out normal data.

Run: .venv/bin/python src/gating_windows.py  ->  results/gating_windows.json
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

import swat_invariants as SI
import hueneme_uf as HU
from alarms_per_day import runs, count

ROOT = Path(__file__).resolve().parent.parent
FLOW_ON = 0.5
STATUS_PERSIST_S = 30


def exposure(gate, flowing, dt_s):
    off = flowing & ~gate
    lens = [(e - s) * dt_s for a, b in runs(flowing) for s, e in runs(off[a:b])]
    days = flowing.sum() * dt_s / 86400
    return {"flowing_days": round(float(days), 3),
            "share_of_flowing_time_off": round(float(off.sum() / max(flowing.sum(), 1)), 4),
            "windows": len(lens), "windows_per_day": round(len(lens) / days, 2) if days else None,
            "median_window_s": float(np.median(lens)) if lens else None,
            "longest_window_s": float(max(lens)) if lens else None}


def swat_exposure(d):
    g = SI.gates(d)
    flowing = d.FIT501.values > FLOW_ON
    return {k: exposure(g[k], flowing, 1) for k in SI.NAMES}


def status_check(d):
    running = d.P501.values == 2
    flowing = d.FIT501.values > FLOW_ON
    return SI.persist(running != flowing, STATUS_PERSIST_S)


def main():
    normal = pd.read_pickle(SI.BASE / "normal_v1.pkl")
    att = pd.read_pickle(SI.BASE / "attack_v0.pkl")
    is_attack = att["Normal/Attack"].str.replace(" ", "").eq("Attack").values
    n_cal = int(len(normal) * SI.CAL_FRAC)

    res = {"design": __doc__.split("Design fixed before running (24 Oct).")[1].split("Run:")[0].strip()}
    res["swat_2015_normal_week"] = swat_exposure(normal)
    att_free = att[~is_attack].reset_index(drop=True)
    res["swat_2015_attack_file_attack_free"] = swat_exposure(att_free)

    hu = {"flowing_s": 0.0, "off_s": 0.0, "lens": []}
    for p in sorted((HU.BASE / "test").glob("*.csv")):
        d, why = HU.load(str(p))
        if d is None:
            continue
        _, _, steady = HU.quantities(d)
        t = d["Time Elapsed"].values
        dt = np.r_[np.diff(t), 0].clip(0, HU.MAX_DT)
        flowing = d["FE/FT-100"].values > 10
        off = flowing & ~steady
        hu["flowing_s"] += float(dt[flowing].sum()); hu["off_s"] += float(dt[off].sum())
        for a, b in runs(flowing):
            hu["lens"] += [float(dt[a + s:a + e].sum()) for s, e in runs(off[a:b])]
    days = hu["flowing_s"] / 86400
    res["port_hueneme_test_days"] = {"flowing_days": round(days, 3),
                                     "share_of_flowing_time_off": round(hu["off_s"] / max(hu["flowing_s"], 1), 4),
                                     "windows": len(hu["lens"]), "windows_per_day": round(len(hu["lens"]) / days, 2),
                                     "median_window_s": float(np.median(hu["lens"])) if hu["lens"] else None,
                                     "longest_window_s": float(max(hu["lens"])) if hu["lens"] else None,
                                     "note": "off = backwash and the 180 s after it, plus other non-filtration states while feed flows"}

    s_held = status_check(normal.iloc[n_cal:].reset_index(drop=True))
    s_att = status_check(att)
    hits = []
    for e in SI.episodes(att):
        m = e["rows"]
        if m.any() and s_att[m].any():
            hits.append({"id": e["id"], "point": e["point"], "share_flagged": round(float(s_att[m].mean()), 3)})
    held = count(s_held, 1)
    res["status_vs_flow_check"] = {"heldout_normal": held,
                                   "attack_file_attack_free": count(s_att, 1, ~is_attack),
                                   "attacks_tripping_it": hits,
                                   "adopt": bool(held["episodes_per_day"] <= 1)}
    json.dump(res, open(ROOT / "results" / "gating_windows.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "design"}, indent=1))


if __name__ == "__main__":
    main()
