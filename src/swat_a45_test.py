"""
Clean test of the real-data RO rules (swat_invariants.py) on SWaT A4/A5, 20 Jul 2019.

Documented in data/SWaT.A4 & A5_Jul 2019/SWaT data collection_20-07-2019 v2.pdf:
plant start 12:35, normal run 12:35-14:50, six attacks 15:08-16:16, stop 16:35
(local time, GMT+8; the spreadsheet is in GMT+0). Two attacks touch the RO
stages our rules cover:
  #1 FIT401 spoofed 0.8 -> 0.5 (intent: switch off UV401)      15:08:46-15:10:31
  #5 MV501 OPEN -> CLOSE (intent: drain water from RO)          15:54:00-15:56:00
The other four (LIT301, P601, MV201+P101, P301) target stages P1-P3 and P6.

Design fixed BEFORE looking at any rule output on this file:
  A  transferred: 2015 calibration unchanged (as in swat_a6_test.py).
  B  recalibrated on this plant: fixed medians, 1.2 x worst deviation, fitted on
     the last hour of the normal run (13:50-14:50), skipping start-up (the A6
     lesson). Same gates and 10 s persistence.
  Score: hit = any alarm inside an attack window. RO attacks (#1, #5) are the
  test; alarms during the other four are reported separately, since they can
  have real knock-on effects on the RO stage. False alarms are measured on the
  attack-free gaps 14:50-15:08:46 and 16:20-16:35.

Clock alignment, added AFTER the first run: the historian clock runs about
105 s ahead of the attack sheet. Measured from the attacked signals alone
(FIT401 reading the spoofed 0.5, MV501 closed, P601 on, P301 off, LIT301 >1000,
MV201+P101 on), every attack appears 91-110 s earlier in the data than listed.
Results are reported both as documented and with every window (attacks and
false-alarm gaps) shifted by ALIGN_S. Unaligned, the real FIT401 spoof fell in
the first "attack-free" gap and was counted as a false alarm.

Run: .venv/bin/python src/swat_a45_test.py  ->  results/swat_a45_test.json
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

from swat_invariants import BASE, CAL_FRAC, NAMES, fit, violations
from swat_a6_test import fit_fixed, violations_fixed

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "SWaT.A4 & A5_Jul 2019" / "SWaT_dataset_Jul 19 v2.xlsx"
CACHE = ROOT / "data" / "SWaT.A4 & A5_Jul 2019" / "a45.pkl"
PERSIST_S = 10
DAY = "2019-07-20 "
ATTACKS = [  # id, target, start, end, touches RO stages
    (1, "FIT401 spoof 0.8 -> 0.5", "15:08:46", "15:10:31", True),
    (2, "LIT301 spoof 835 -> 1024", "15:15:00", "15:19:32", False),
    (3, "P601 OFF -> ON", "15:26:57", "15:30:48", False),
    (4, "MV201 open + P101 on", "15:38:50", "15:46:20", False),
    (5, "MV501 OPEN -> CLOSE", "15:54:00", "15:56:00", True),
    (6, "P301 ON -> OFF", "16:02:56", "16:16:18", False),
]
CAL_B = ("13:50:00", "14:50:00")
FA_WINDOWS = [("14:50:00", "15:08:46"), ("16:20:00", "16:35:00")]
ALIGN_S = -105               # data timestamp = sheet time + ALIGN_S (see docstring)


def load():
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    d = pd.read_excel(SRC, header=1, skiprows=[2], engine="openpyxl")
    d.columns = [str(c).replace(" Status", "").replace(" ", "") for c in d.columns]
    d = d.rename(columns={"GMT+0": "Timestamp"})
    d["Timestamp"] = (pd.to_datetime(d["Timestamp"], utc=True, format="ISO8601")
                      .dt.tz_convert("Asia/Singapore").dt.tz_localize(None))
    for c in d.columns:
        if c != "Timestamp":
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d.to_pickle(CACHE)
    return d


def win(t, a, b, shift=0):
    off = pd.Timedelta(seconds=shift)
    return ((t >= pd.Timestamp(DAY + a) + off) & (t < pd.Timestamp(DAY + b) + off)).values


IDENTITY = ["flow_balance", "feed_meters"]


def score(V, t, shift=0, names=NAMES):
    any_ = np.any([V[n] for n in names], 0)
    out = {"attacks": []}
    for i, target, a, b, ro in ATTACKS:
        m = win(t, a, b, shift)
        out["attacks"].append({"id": i, "target": target, "ro_stage": ro, "seconds": int(m.sum()),
                               "detected": bool(any_[m].any()),
                               "share_flagged": round(float(any_[m].mean()), 3) if m.any() else None,
                               "by": {n: round(float(V[n][m].mean()), 3) for n in names if V[n][m].any()}})
    fa = np.zeros(len(t), bool)
    for a, b in FA_WINDOWS:
        fa |= win(t, a, b, shift)
    out["false_alarm_rate"] = round(float(any_[fa].mean()), 5)
    out["false_alarm_by"] = {n: round(float(V[n][fa].mean()), 5) for n in names if V[n][fa].any()}
    ro = [x for x in out["attacks"] if x["ro_stage"]]
    out["ro_attacks_detected"] = f"{sum(x['detected'] for x in ro)} of {len(ro)}"
    return out


def main():
    d = load()
    t = pd.to_datetime(d["Timestamp"])
    missing = [c for c in ["FIT401", "FIT501", "FIT502", "FIT503", "FIT504", "PIT501", "PIT502",
                           "PIT503", "AIT503", "AIT504", "AIT402", "AIT502", "P501"] if c not in d]
    assert not missing, f"missing tags: {missing}"

    normal = pd.read_pickle(BASE / "normal_v1.pkl")
    Va = violations(d, fit(normal.iloc[:int(len(normal) * CAL_FRAC)]), PERSIST_S)
    cal_b = fit_fixed(d[win(t, *CAL_B)])          # calibration hour is attack-free either way
    Vb = violations_fixed(d, cal_b, PERSIST_S)
    res_a, res_b = score(Va, t), score(Vb, t)
    al_a, al_b = score(Va, t, ALIGN_S), score(Vb, t, ALIGN_S)

    res = {"design": __doc__.split("Design fixed")[1].split("Run:")[0].strip(),
           "rows": len(d), "time_range": [str(t.min()), str(t.max())],
           "A_transferred_2015_calibration": res_a,
           "B_recalibrated_on_A45": {"calibration": {k: {kk: round(vv, 5) for kk, vv in v.items()}
                                                     for k, v in cal_b.items()}, **res_b},
           "aligned_clock_shift_s": ALIGN_S,
           "A_aligned": al_a, "B_aligned": al_b,
           "secondary_identity_checks_only_aligned": {
               "note": "Secondary analysis. The identity-vs-ratio split comes from the A6 diagnosis (18 Oct), "
                       "made before this file was run, but this subset was not the pre-registered score.",
               "A_transferred": score(Va, t, ALIGN_S, IDENTITY), "B_recalibrated": score(Vb, t, ALIGN_S, IDENTITY)}}
    json.dump(res, open(ROOT / "results" / "swat_a45_test.json", "w"), indent=1)
    print("rows", len(d), "from", t.min(), "to", t.max())
    for name, r in (("A transferred 2015 calibration, as documented", res_a),
                    ("B recalibrated on A4/A5, as documented", res_b),
                    (f"A transferred, windows shifted {ALIGN_S} s", al_a),
                    (f"B recalibrated, windows shifted {ALIGN_S} s", al_b)):
        print(f"\n== {name}: RO attacks detected {r['ro_attacks_detected']} | false alarms "
              f"{r['false_alarm_rate']:.3%} {r['false_alarm_by']}")
        for x in r["attacks"]:
            tag = "RO " if x["ro_stage"] else "   "
            print(f"  {tag}#{x['id']} {x['target']:26s} {'HIT ' if x['detected'] else 'miss'} "
                  f"{x['share_flagged']} {x['by']}")


if __name__ == "__main__":
    main()
