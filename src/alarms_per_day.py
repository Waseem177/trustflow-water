"""
False alarms as an operator sees them: alarm episodes per day, not flagged rows.

"0.2% of rows" at one row per second reads like 170 alarm-seconds a day, but
flagged seconds cluster. An operator sees one alarm per cluster.

Design fixed BEFORE counting:
  Episode   a maximal run of consecutive flagged rows (any check), on attack-free
            rows only. Primary count, conservative.
  Merged    runs closer than MERGE_S = 300 s counted as one alarm, the way an
            alarm that clears and re-trips within 5 minutes reaches an operator
            once. Secondary.
  Per day   episodes / (attack-free rows x sample interval / 86,400 s).
  Settings  exactly the deployed ones of each source script: SWaT 2015 10 s
            persistence and its calibration; SWaT Jul 2019 variant B (recalibrated)
            with the 105 s clock alignment, on its two attack-free gaps; the
            simulator's trailing references on days 30-60 of seeds 0-4.
  Added after the first run: the July 2019 count for the two identity checks
            only (flow balance, redundant meters), the subset the report claims
            for that plant; the six-check count stays alongside it.
  Also added after the first run (post hoc, so secondary): SWaT 2015 attack-file
            episodes split by whether they start within AFTERMATH_S = 300 s of an
            attack's labelled end, when the plant is still settling.
  Zeros     Port Hueneme and the energy check flagged 0 rows in their own runs
            (results/hueneme_uf.json, results/energy_check.json), so 0 episodes.
  ISA-18.2 guidance for comparison: about 1 alarm per 10 minutes per operator
  (144 a day) is "manageable"; these counts are for one RO train.

Run: .venv/bin/python src/alarms_per_day.py  ->  results/alarms_per_day.json
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

import swat_invariants as SI
import swat_a45_test as A45
from swat_a6_test import fit_fixed, violations_fixed
from eval_swro_invariants import run, SEEDS, DAYS, CAL_DAYS
from invariants_swro import residuals

ROOT = Path(__file__).resolve().parent.parent
MERGE_S = 300
AFTERMATH_S = 300


def runs(flag):
    """(start, end) index pairs of consecutive True runs."""
    f = np.concatenate([[False], np.asarray(flag, bool), [False]])
    d = np.diff(f.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def count(flag, dt_s, mask=None):
    """Episodes on rows where mask is True; runs never bridge excluded rows."""
    flag = np.asarray(flag, bool)
    mask = np.ones(len(flag), bool) if mask is None else np.asarray(mask, bool)
    eps = []
    for a, b in runs(mask):                        # each attack-free stretch on its own
        eps += [(a + s, a + e) for s, e in runs(flag[a:b])]
    merged = 0
    last_end = None
    for s, e in eps:
        if last_end is None or (s - last_end) * dt_s >= MERGE_S:
            merged += 1
        last_end = e
    days = mask.sum() * dt_s / 86400
    lens = [(e - s) * dt_s for s, e in eps]
    return {"days": round(float(days), 3), "flagged_rows": int((flag & mask).sum()),
            "episodes": len(eps), "episodes_per_day": round(len(eps) / days, 2) if days else None,
            "merged_alarms": merged, "merged_per_day": round(merged / days, 2) if days else None,
            "median_episode_s": float(np.median(lens)) if lens else None,
            "longest_episode_s": float(max(lens)) if lens else None}


def swat_2015():
    normal = pd.read_pickle(SI.BASE / "normal_v1.pkl")
    att = pd.read_pickle(SI.BASE / "attack_v0.pkl")
    n_cal = int(len(normal) * SI.CAL_FRAC)
    cal = SI.fit(normal.iloc[:n_cal])
    is_attack = att["Normal/Attack"].str.replace(" ", "").eq("Attack").values
    Vn = SI.violations(normal.iloc[n_cal:], cal, 10)
    Va = SI.violations(att, cal, 10)
    any_n = np.any([Vn[n] for n in SI.NAMES], 0)
    any_a = np.any([Va[n] for n in SI.NAMES], 0)
    ends = np.flatnonzero(np.diff(is_attack.astype(int)) == -1) + 1
    starts = [a + s for a, b in runs(~is_attack) for s, _ in runs(any_a[a:b])]
    after = sum(bool(len(ends[ends <= s])) and s - ends[ends <= s][-1] <= AFTERMATH_S for s in starts)
    quiet_days = (~is_attack).sum() / 86400
    return {"heldout_normal": count(any_n, 1),
            "attack_file_normal_rows": count(any_a, 1, ~is_attack),
            "attack_file_split_post_hoc": {"within_300s_after_an_attack": int(after),
                                           "in_quiet_operation": len(starts) - int(after),
                                           "quiet_operation_per_day": round((len(starts) - after) / quiet_days, 2)}}


def swat_2019():
    d = A45.load()
    t = pd.to_datetime(d["Timestamp"])
    cal_b = fit_fixed(d[A45.win(t, *A45.CAL_B)])
    Vb = violations_fixed(d, cal_b, A45.PERSIST_S)
    any_b = np.any([Vb[n] for n in SI.NAMES], 0)
    fa = np.zeros(len(t), bool)
    for a, b in A45.FA_WINDOWS:
        fa |= A45.win(t, a, b, A45.ALIGN_S)
    ident = np.any([Vb[n] for n in ("flow_balance", "feed_meters")], 0)
    note = "only %.0f attack-free minutes in this release; per-day figures are extrapolations" % (fa.sum() / 60)
    return {"identity_checks": count(ident, 1, fa), "all_six_checks": count(any_b, 1, fa), "note": note}


def simulator():
    per_seed = []
    for seed in SEEDS:
        rep, log, m, cals = run(seed)
        V = (residuals(rep, cals["trailing"], events=log).abs() > 1).numpy()
        per_seed.append(count(V.any(1), 60, ~m))
    return {"per_seed": per_seed,
            "episodes_per_day_mean": round(float(np.mean([s["episodes_per_day"] for s in per_seed])), 2),
            "episodes_per_day_worst": round(float(np.max([s["episodes_per_day"] for s in per_seed])), 2),
            "merged_per_day_mean": round(float(np.mean([s["merged_per_day"] for s in per_seed])), 2),
            "merged_per_day_worst": round(float(np.max([s["merged_per_day"] for s in per_seed])), 2)}


def main():
    hu = json.load(open(ROOT / "results" / "hueneme_uf.json"))
    en = json.load(open(ROOT / "results" / "energy_check.json"))
    out = {"design": __doc__.split("Design fixed BEFORE counting:")[1].split("Run:")[0].strip(),
           "swat_2015": swat_2015(), "swat_jul_2019_recalibrated": swat_2019(),
           "simulator_process_checks": simulator(),
           "port_hueneme_p999": {"false_alarm_rate": hu["results"]["p999"]["false_alarms"], "steady_test_rows": hu["results"]["p999"]["false_alarm_rows"], "episodes": 0},
           "energy_check_sim": {"false_alarm_rate": en["energy_check_false_alarms_days_30_60"], "episodes": 0}}
    json.dump(out, open(ROOT / "results" / "alarms_per_day.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "design"}, indent=1))


if __name__ == "__main__":
    main()
