"""
Does the energy check catch an attack that every process check misses?

Scenario: from day 45 for 6 h the attacker raises the HP pump's head by x
(a membrane over-pressure attack: compaction, seal and vessel damage) or lowers
it (cuts production). To conceal it, the attacker controls EVERY process tag
and reports the plant's own genuine readings as if nothing had happened. That
cover story satisfies all eight process checks by construction: it is real
normal data. It is the replay attacker that defeats physics checks in the
literature (and in our SWaT experiment). The attacker does not control the
electrical meter, which reports the power the pump really draws.

    true power at fixed flow and efficiency scales with head: W' = W (1 + x)

Measured, over 5 seeds, calibrated on days 0-30:
  - false alarms of the energy check on days 30-60
  - share of attack rows flagged, for x from -20% to +20%:
      onset  : reference frozen from the clean record (as in eval_swro_invariants)
      running: trailing reference computed from what is reported, so it can
               slowly absorb a sustained change (STATUS gap 5)
  - the eight process checks on the concealed record (should look normal)
And the limit, by construction: if the attacker also controls the power meter,
it reports W and nothing is caught.

Run: .venv/bin/python src/eval_energy.py  ->  results/energy_check.json
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

from swro_sim import simulate
from invariants_swro import fit as fit8, residuals as residuals8
import energy_check as E

ROOT = Path(__file__).resolve().parent.parent
DAYS, CAL_DAYS, SEEDS = 60, 30, range(5)
TRAILING = dict(window_min=360, lag_min=60)
ATTACK_DAY, ATTACK_H = 45.0, 6
LEVELS = [-0.20, -0.10, -0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20]


def main():
    fa, sweep, proc_alarm, delay = [], {x: {"onset": [], "running": []} for x in LEVELS}, [], {}
    for seed in SEEDS:
        _, rep, ev = simulate(days=DAYS, seed=seed)
        m = (ev.day < CAL_DAYS).values
        r = E.residual(rep)
        ref_clean = E.trailing(r, **TRAILING)
        cal = E.fit(r[m], ref_clean[m])
        fa.append(float(E.violations(r[~m], cal, ref_clean[~m]).mean()))

        a = int(ATTACK_DAY * 1440)
        win = slice(a, a + ATTACK_H * 60)
        for x in LEVELS:
            rep_x = rep.copy()
            rep_x.loc[rep_x.index[win], "W_hp"] *= (1 + x)      # only the meter shows the truth
            r_x = E.residual(rep_x)
            sweep[x]["onset"].append(float(E.violations(r_x[win], cal, ref_clean[win]).mean()))
            run = E.violations(r_x, cal, E.trailing(r_x, **TRAILING))[win]
            sweep[x]["running"].append(float(run.mean()))
            if seed == 0:
                hit = np.flatnonzero(run)
                delay[x] = None if len(hit) == 0 else int(hit[0])
        # the eight process checks see only the genuine normal record during the attack
        log = {"cip": np.flatnonzero(ev.cip.values), "ceb": np.flatnonzero(ev.ceb.values),
               "uf_phase": ev.uf_phase.values}
        log_cal = {"cip": log["cip"][log["cip"] < m.sum()], "ceb": log["ceb"][log["ceb"] < m.sum()],
                   "uf_phase": log["uf_phase"][m]}
        c8 = fit8(rep[m], trailing=TRAILING, events=log_cal)
        V8 = (residuals8(rep, c8, events=log).abs() > 1).numpy()
        proc_alarm.append(float(V8[win].any(1).mean()))

    out = {
        "scenario": "HP pump head changed by x for 6 h from day 45; every process tag replays the genuine "
                    "normal record; only the electrical meter reports the true power",
        "seeds": len(SEEDS),
        "energy_check_false_alarms_days_30_60": {"mean": round(float(np.mean(fa)), 5), "max": round(float(np.max(fa)), 5)},
        "eight_process_checks_alarm_rate_during_attack": round(float(np.mean(proc_alarm)), 5),
        "share_of_attack_rows_flagged": {f"{x:+.0%}": {k: round(float(np.mean(v)), 3) for k, v in d.items()}
                                         for x, d in sweep.items()},
        "first_alarm_minutes_after_onset_running_seed0": {f"{x:+.0%}": delay[x] for x in LEVELS},
        "if_attacker_also_controls_power_meter": "nothing is caught, by construction",
    }
    json.dump(out, open(ROOT / "results" / "energy_check.json", "w"), indent=1)
    print(f"energy check false alarms, days 30-60: mean {np.mean(fa):.4%}, worst seed {np.max(fa):.4%}")
    print(f"eight process checks during the concealed attack: alarm on {np.mean(proc_alarm):.3%} of rows")
    print("share of attack rows flagged by the energy check (onset ref | running trailing ref), first alarm (min):")
    for x in LEVELS:
        o, rr = np.mean(sweep[x]["onset"]), np.mean(sweep[x]["running"])
        print(f"  head {x:+5.0%}:  {o:6.1%} | {rr:6.1%}   first alarm {delay[x]}")


if __name__ == "__main__":
    main()
