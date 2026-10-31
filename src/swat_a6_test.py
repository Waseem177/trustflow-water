"""
Clean test of the real-data RO rules (swat_invariants.py) on SWaT A6, 6 Dec 2019.

A6 is a different attack from the Dec 2015 file the rules were tuned on:
malware on the SCADA workstation exfiltrates historian data (10:30-11:20, no
process effect intended), then disrupts sensors and actuators in five 3-min
bursts (12:30, 12:43, 12:56, 13:09, 13:22), per data/swat_a6/Log.docx. The log
gives times, not target sensors.

The plant was reconfigured between 2015 and 2019 (recirculation off, feed
conductivity 45 vs 268 uS/cm, concentrate flow 0.11 vs 0.74 m3/h).

Design fixed BEFORE looking at the attack windows:
  A  transferred: 2015 calibration unchanged. Trailing references need 1.5 h
     of data plus a 1 h lag, so the drifting checks only start at ~12:35.
  B  recalibrated on this plant: calibrate on 10:05-11:15, measure false
     alarms on 11:15-12:30 and 13:30-13:45. Within 3.7 h there is no membrane
     drift to track, so all six checks use fixed medians. Same 1.2 x worst
     rule, same steady-state and flow gates, same 10 s persistence.
  Scoring: hit = any alarm inside a burst window [start, start+3 min).
  Alarms in the 10 min "sleep" after each burst are reported separately,
  since a disruption can outlast its burst.

Run: .venv/bin/python src/swat_a6_test.py  ->  results/swat_a6_test.json
Needs data/swat_a6/a6.pkl (Dec2019.xlsx with '.Pv'/'.Status' stripped).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

from swat_invariants import (BASE, CAL_FRAC, MARGIN, NAMES, quantities, gates,
                             persist, fit, violations)

ROOT = Path(__file__).resolve().parent.parent
PERSIST_S = 10
DAY = "2019-12-06 "
BURSTS = ["12:30", "12:43", "12:56", "13:09", "13:22"]
CAL_B = ("10:05", "11:15")
FA_WINDOWS = [("11:15", "12:30"), ("13:30", "13:46")]


def window(t, a, b):
    return ((t >= pd.Timestamp(DAY + a)) & (t < pd.Timestamp(DAY + b))).values


def fit_fixed(d):
    """Variant B: fixed median reference for every check, tolerance 1.2 x worst."""
    q, g = quantities(d), gates(d)
    cal = {}
    for k in NAMES:
        v = q[k][g[k] & np.isfinite(q[k])]
        ref = float(np.median(v))
        cal[k] = {"ref": ref, "tol": MARGIN * float(np.abs(v - ref).max())}
    return cal


def violations_fixed(d, cal, k):
    q, g = quantities(d), gates(d)
    V = {}
    for n in NAMES:
        r = np.abs(q[n] - cal[n]["ref"]) / cal[n]["tol"]
        V[n] = persist(np.where(g[n] & np.isfinite(r), r > 1, False), k)
    return V


def score(V, t):
    any_ = np.any([V[n] for n in NAMES], 0)
    out = {"bursts": [], "sleeps": []}
    for b in BURSTS:
        s = pd.Timestamp(DAY + b)
        for key, (a, z) in (("bursts", (s, s + pd.Timedelta(minutes=3))),
                            ("sleeps", (s + pd.Timedelta(minutes=3), s + pd.Timedelta(minutes=13)))):
            m = ((t >= a) & (t < z)).values
            out[key].append({"start": str(a.time()), "detected": bool(any_[m].any()),
                             "share_flagged": round(float(any_[m].mean()), 3),
                             "by": {n: round(float(V[n][m].mean()), 3) for n in NAMES if V[n][m].any()}})
    fa = np.zeros(len(t), bool)
    for a, z in FA_WINDOWS:
        fa |= window(t, a, z)
    out["false_alarm_rate"] = round(float(any_[fa].mean()), 5)
    out["false_alarm_by"] = {n: round(float(V[n][fa].mean()), 5) for n in NAMES if V[n][fa].any()}
    out["exfiltration_phase_alarm_rate"] = round(float(any_[window(t, "10:30", "11:20")].mean()), 5)
    out["bursts_detected"] = f"{sum(x['detected'] for x in out['bursts'])} of {len(BURSTS)}"
    return out


# Written after the pre-registered scoring, from the plant's state tags; the scores are left as recorded.
DIAGNOSIS = {
    "verdict": "Not a valid test of RO-stage detection. The disruption bursts switched UF/backwash and upstream "
               "actuators (MV301, MV303, P301, P602, P101, P201, P203; P3_STATE sequence changes) and left stages "
               "P4-P5 unchanged (P4_STATE, P5_STATE constant). The RO-stage alarms in both variants come from slow "
               "drift, not the attack: the plant was still settling after start-up (feed conductivity AIT503 rose "
               "29 -> 46 uS/cm over ~90 min; RO-feed ORP AIT502 drifted 190 -> 207 mV all session).",
    "what_it_does_show": [
        "The two pure identities (flow balance, feed meters) transferred from 2015 to a reconfigured 2019 plant "
        "(recirculation off, feed conductivity 45 vs 268 uS/cm) with no alarm at any point in A6.",
        "Ratio and ORP checks need settled-plant calibration over hours; a 3.7 h recording that starts at plant "
        "start-up cannot support them. Variant B (calibrated on 70 min of start-up) false-alarmed on 84% of its "
        "test window.",
        "AIT402 reads 195.05 mV +/- 0.05 all session: likely a stuck sensor, which the current checks do not flag as such."],
    "note": "Diagnosis done after the pre-registered scoring; the scores above are left as recorded.",
}


def main():
    a6 = pd.read_pickle(ROOT / "data" / "swat_a6" / "a6.pkl")
    t = pd.to_datetime(a6["Timestamp"])

    normal = pd.read_pickle(BASE / "normal_v1.pkl")
    cal_a = fit(normal.iloc[:int(len(normal) * CAL_FRAC)])
    res_a = score(violations(a6, cal_a, PERSIST_S), t)

    m_cal = window(t, *CAL_B)
    cal_b = fit_fixed(a6[m_cal])
    res_b = score(violations_fixed(a6, cal_b, PERSIST_S), t)

    res = {"design": __doc__.split("Design fixed")[1].split("Run:")[0].strip(),
           "A_transferred_2015_calibration": res_a,
           "B_recalibrated_on_A6": {"calibration": {k: {kk: round(vv, 5) for kk, vv in v.items()}
                                                    for k, v in cal_b.items()}, **res_b},
           "diagnosis_after_scoring": DIAGNOSIS}
    json.dump(res, open(ROOT / "results" / "swat_a6_test.json", "w"), indent=1)

    for name, r in (("A transferred 2015 calibration", res_a), ("B recalibrated on A6", res_b)):
        print(f"\n== {name}")
        print(f"bursts detected {r['bursts_detected']} | false alarms {r['false_alarm_rate']:.3%} {r['false_alarm_by']}"
              f" | exfiltration phase {r['exfiltration_phase_alarm_rate']:.3%}")
        for x, y in zip(r["bursts"], r["sleeps"]):
            print(f"  burst {x['start']}  {'HIT ' if x['detected'] else 'miss'} {x['share_flagged']:.2f} {x['by']}"
                  f"   | sleep after: {'alarm' if y['detected'] else 'quiet'} {y['share_flagged']:.2f} {y['by']}")


if __name__ == "__main__":
    main()
