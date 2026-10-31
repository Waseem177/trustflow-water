"""
RO-stage physics invariants on REAL data: SWaT (iTrust, Dec 2015), stages P4-P5.

Answers the "simulated data only" limit. SWaT is a fresh-water testbed
(feed ~270 uS/cm, not seawater), so osmotic pressure is negligible and the
salinity side of the SWRO set does not carry over. The flow, pressure and
redundancy relations do.

Tags (SWaT equipment list): FIT401 RO feed (P4), FIT501 RO feed, FIT502
permeate, FIT503 concentrate, FIT504 concentrate recirculation; PIT501 feed
(after HP pump), PIT502 permeate, PIT503 concentrate; AIT503/AIT504 feed and
permeate conductivity; AIT402 ORP after NaHSO3 dosing, AIT502 ORP at RO feed;
P501 HP pump (2 = on).

Invariants, residual in the tag's own units unless stated:
  flow_balance   FIT501 - (FIT502 + FIT503 - FIT504)            m3/h
  feed_meters    FIT401 - FIT501                                 m3/h
  brine_dp       ln[(PIT501-PIT503) / ((FIT501+FIT503)/2)^1.5]
  permeability   ln[FIT502 / (PIT501-PIT502)]
  salt_passage   ln[AIT504 / AIT503]
  orp_pair       AIT402 - AIT502                                 mV

Three things the simulator did not teach us, each found on this data:
  1. Gating. Start-up and shutdown transients set the worst-case deviation,
     so a tolerance fitted on all rows is useless (flow balance came out at
     45% of feed). Process relations are checked only at steady state: HP pump
     on for >= STEADY_S seconds. The two sensor-pair checks hold regardless of
     the pump: feed_meters is always checked, and orp_pair whenever water is
     MEASURED flowing past both probes (a pump status tag can be spoofed, and
     in attack 22 it was).
  2. Drift. The attack file was recorded days after the normal file; with
     fixed references, tight tolerances false-alarm on ~half its normal rows.
     The drifting relations use a trailing median (6 h window, 1 h lag), as in
     the simulated set.
  3. Persistence. An alarm needs PERSIST consecutive seconds over tolerance.
Reference = median (or trailing median), tolerance = 1.2 x worst deviation on
calibration data (first 70% of Normal_v1), the same rule as the simulated set.

CAVEAT: the gating and trailing choices were made after seeing which SWaT
attacks the first version missed. They are physically motivated, but the
attack file is no longer a clean test set for them. A clean test needs a
later SWaT release with different attacks (A3 Jun 2017 onwards).

Run: .venv/bin/python src/swat_invariants.py  ->  results/swat_invariants.json
Needs the pickles cached from the xlsx files (see STATUS, 16 Oct).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data" / "swat_physical" / "Physical"
MARGIN, CAL_FRAC = 1.2, 0.7
PERSIST = (1, 10)                     # seconds; report both
STEADY_S = 300                        # HP pump on this long before process checks apply
FLOW_GATE = 0.5                       # m3/h through both feed meters for the ORP pair
WIN_S, LAG_S = 6 * 3600, 3600         # trailing reference
NAMES = ["flow_balance", "feed_meters", "brine_dp", "permeability", "salt_passage", "orp_pair"]
TRAILING = ["brine_dp", "permeability", "salt_passage", "orp_pair"]
RO_SENSOR_ATTACKS = [10, 11, 19, 20, 22, 37, 38, 39, 40]   # attacks on P4/P5 sensors in the list


def quantities(d):
    with np.errstate(divide="ignore", invalid="ignore"):
        q = {
            "flow_balance": d.FIT501 - (d.FIT502 + d.FIT503 - d.FIT504),
            "feed_meters": d.FIT401 - d.FIT501,
            "brine_dp": np.log((d.PIT501 - d.PIT503) / ((d.FIT501 + d.FIT503) / 2) ** 1.5),
            "permeability": np.log(d.FIT502 / (d.PIT501 - d.PIT502)),
            "salt_passage": np.log(d.AIT504 / d.AIT503),
            "orp_pair": d.AIT402 - d.AIT502,
        }
    return {k: np.asarray(v, dtype=float) for k, v in q.items()}


def gates(d):
    s = d.P501.values
    changed = np.r_[True, s[1:] != s[:-1]]
    since = np.arange(len(s)) - np.maximum.accumulate(np.where(changed, np.arange(len(s)), 0))
    steady = (s == 2) & (since >= STEADY_S)
    flowing = (d.FIT401.values > FLOW_GATE) & (d.FIT501.values > FLOW_GATE)
    return {"feed_meters": np.ones(len(d), bool), "orp_pair": flowing,
            **{k: steady for k in ("flow_balance", "brine_dp", "permeability", "salt_passage")}}


def deviations(d, cal=None):
    """Deviation from reference for every row a check applies to; NaN elsewhere."""
    q, g = quantities(d), gates(d)
    out = {}
    for k in NAMES:
        v = np.where(g[k] & np.isfinite(q[k]), q[k], np.nan)
        if k in TRAILING:
            ref = pd.Series(v).rolling(WIN_S, min_periods=WIN_S // 4).median().shift(LAG_S).values
        else:
            ref = np.nanmedian(v) if cal is None else cal[k]["ref"]
        out[k] = (v - ref, ref if np.isscalar(ref) else None)
    return out


def fit(d):
    cal = {}
    for k, (dev, ref) in deviations(d).items():
        cal[k] = {"ref": ref, "tol": MARGIN * float(np.nanmax(np.abs(dev)))}
    return cal


def persist(flag, k):
    if k <= 1:
        return flag
    run = np.convolve(flag.astype(int), np.ones(k, dtype=int), mode="full")[:len(flag)]
    hit = run >= k
    out = hit.copy()                   # mark the whole run, not just its k-th second
    for s in range(1, k):
        out[:-s] |= hit[s:]
    return out


def violations(d, cal, k):
    V = {}
    for n, (dev, _) in deviations(d, cal).items():
        r = np.abs(dev) / cal[n]["tol"]
        V[n] = persist(np.where(np.isfinite(r), r > 1, False), k)
    return V


def episodes(att):
    lst = pd.read_excel(ROOT / "data" / "List_of_attacks_Final.xlsx")
    lst = lst[pd.to_numeric(lst["Attack #"], errors="coerce").notna()]
    t = pd.to_datetime(att["Timestamp"].str.strip(), format="%d/%m/%Y %I:%M:%S %p")
    out = []
    for _, r in lst.iterrows():
        if pd.isna(r["End Time"]):
            continue                                   # "no physical impact" placeholders
        start = pd.Timestamp(r["Start Time"])
        if start.year == 2015 and start.month == 1:     # list typo: attacks 37-41 are Jan 2016
            start = start.replace(year=2016)
        end = pd.Timestamp.combine(start.date(), pd.Timestamp(str(r["End Time"])).time())
        m = ((t >= start) & (t <= end)).values
        out.append({"id": int(r["Attack #"]), "point": str(r["Attack Point"]).strip(),
                    "attack": str(r["Attack"]).strip(), "rows": m})
    return out


def main():
    normal = pd.read_pickle(BASE / "normal_v1.pkl")
    att = pd.read_pickle(BASE / "attack_v0.pkl")
    n_cal = int(len(normal) * CAL_FRAC)
    cal = fit(normal.iloc[:n_cal])
    is_attack = att["Normal/Attack"].str.replace(" ", "").eq("Attack").values
    eps = episodes(att)

    res = {"setup": {"calibration_rows": n_cal, "margin": MARGIN, "persist_s": list(PERSIST),
                     "steady_s": STEADY_S, "flow_gate_m3h": FLOW_GATE,
                     "trailing": {"window_s": WIN_S, "lag_s": LAG_S, "applies_to": TRAILING},
                     "note": "SWaT is fresh water (~270 uS/cm); salinity physics of SWRO not tested here. "
                             "Gating and trailing choices were made after seeing the first version's misses "
                             "on this attack file, so it is not a clean test set for them."},
           "calibration": {k: {"ref": None if v["ref"] is None else round(v["ref"], 5),
                               "tol": round(v["tol"], 5)} for k, v in cal.items()}}
    for k in PERSIST:
        Vn = violations(normal.iloc[n_cal:], cal, k)
        Va = violations(att, cal, k)
        any_n = np.any([Vn[n] for n in NAMES], 0)
        any_a = np.any([Va[n] for n in NAMES], 0)
        block = {
            "false_alarms_normal_heldout": {"any": round(float(any_n.mean()), 5),
                                            **{n: round(float(Vn[n].mean()), 5) for n in NAMES}},
            "false_alarms_attackfile_normal_rows": {"any": round(float(any_a[~is_attack].mean()), 5),
                                                    **{n: round(float(Va[n][~is_attack].mean()), 5) for n in NAMES}},
            "episodes": [],
        }
        for e in eps:
            m = e["rows"]
            if not m.any():
                continue
            by = {n: round(float(Va[n][m].mean()), 3) for n in NAMES if Va[n][m].any()}
            block["episodes"].append({"id": e["id"], "point": e["point"], "attack": e["attack"][:80],
                                      "ro_sensor_attack": e["id"] in RO_SENSOR_ATTACKS,
                                      "seconds": int(m.sum()), "detected": bool(any_a[m].any()),
                                      "share_of_rows_flagged": round(float(any_a[m].mean()), 3), "by": by})
        ro = [e for e in block["episodes"] if e["ro_sensor_attack"]]
        block["ro_sensor_attacks_detected"] = f"{sum(e['detected'] for e in ro)} of {len(ro)}"
        block["other_attacks_detected"] = [e["id"] for e in block["episodes"]
                                           if e["detected"] and not e["ro_sensor_attack"]]
        res[f"persist_{k}s"] = block
    json.dump(res, open(ROOT / "results" / "swat_invariants.json", "w"), indent=1)

    print("calibration:", {k: round(v["tol"], 4) for k, v in cal.items()})
    for k in PERSIST:
        b = res[f"persist_{k}s"]
        print(f"\n== persistence {k}s")
        print("false alarms, normal held-out:", b["false_alarms_normal_heldout"]["any"],
              "| attack-file normal rows:", b["false_alarms_attackfile_normal_rows"]["any"])
        print("RO-sensor attacks detected:", b["ro_sensor_attacks_detected"],
              "| other attacks detected:", b["other_attacks_detected"])
        for e in b["episodes"]:
            if e["ro_sensor_attack"]:
                print(f"  #{e['id']:>2} {'HIT ' if e['detected'] else 'MISS'} {e['share_of_rows_flagged']:>5.3f}  {e['attack'][:55]:55s} {e['by']}")


if __name__ == "__main__":
    main()
