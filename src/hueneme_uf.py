"""
White-paper promise (Section 5, "avoiding circularity"): check at least one
pretreatment invariant on real Port Hueneme UF data and report its false-alarm
rate from that data.

Data: Zhou, Khan, Gu, Christofides, Cohen, doi 10.5068/D1310B (CC0). Seawater UF
pretreatment for RO, Port Hueneme CA, 2012-2015, split by the authors into
training/ and test/ days. No attacks are labelled, so this measures false alarms
and spoof sensitivity, not detection of real attacks.

Tag roles. Documented by the readme's variable order: FE/FT-100 = UF inflow,
FE/FT-101..103 = inflow to UF elements E1..E3, FE/FT-104 = backwash flow,
TE/TT-100 = filtrate temperature. INFERRED from behaviour (not documented):
PT-101 = UF inlet pressure, PT-104 = UF filtrate-side pressure (so TMP =
PT-101 - PT-104: about 5 psi in 2012 and 10 psi in 2013 when each element carries
more flow, rising through each filtration cycle as cake builds); PT-102/PT-103
rise only during backwash.

Checks:
  flow_balance  (FT-100 - sum FT-101..103) / FT-100                (documented)
  uf_resistance ln[ TMP / (mu(T) * FT-100 / n_active) ]             (inferred TMP)
                compared with the same point in the filtration cycle over recent
                cycles (1-min phase bins), as the simulated UF check does.

Design fixed BEFORE looking at test-day results (20 Oct):
  - Files: the standard 18-column CSVs with sample interval <= 10 s. Others
    (5/17/19/20-column layouts, 60-120 s logging, xlsx days) are listed, not used.
  - Steady filtration: FT-100 > 10, at least two elements > 2 GPM, pump speed
    > 0.5, no backwash flow, at least 180 s into the filtration segment.
  - Reference: trailing 6 h median, 1 h old, per file (campaigns differ in meter
    offsets); for UF resistance, per 1-min phase bin.
  - Tolerance from training days: (i) the project rule, 1.2 x worst deviation;
    (ii) 1.2 x the 99.9th percentile, reported alongside because real data has
    glitches (the SWaT lesson). Alarm needs >= 30 s over tolerance.
  - False alarms measured on the authors' test days.
  - Sensitivity: on test days, bias FT-100, or the TMP via PT-101, by +/- x%, at
    onset with the clean reference frozen (as in eval_swro_invariants.py).

Run: .venv/bin/python src/hueneme_uf.py  ->  results/hueneme_uf.json
"""
from __future__ import annotations
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data" / "port_hueneme" / "data_in_UF-RO_system" / "data in UF-RO system"
WANT = ["Time", "Time Elapsed", "FE/FT-100", "FE/FT-101", "FE/FT-102", "FE/FT-103", "FE/FT-104",
        "pH/pHT-100", "ORP/ORPT-100", "PT-100", "PT-101", "PT-102", "PT-103", "PT-104", "PT-100x",
        "Tu/TuT-100", "VFD-100-FB", "TE/TT-100"]
ELEMENTS = ["FE/FT-101", "FE/FT-102", "FE/FT-103"]
MAX_DT, STEADY_S, PERSIST_S = 10.0, 180, 30
WIN, LAG = "6h", pd.Timedelta(hours=1)
MARGIN = 1.2
LEVELS = [0.02, 0.05, 0.10, 0.20, 0.50]
NAMES = ["flow_balance", "uf_resistance"]


def mu(T):
    """Water viscosity, Pa.s (Vogel form); filtrate temperature in degC."""
    return 2.414e-5 * 10 ** (247.8 / (T + 273.15 - 140))


def load(path):
    hdr = [h.strip() for h in open(path, encoding="latin-1").readline().strip().split(",")]
    if hdr != WANT:
        return None, f"{len(hdr)}-column layout"
    d = pd.read_csv(path, skiprows=[1], low_memory=False, on_bad_lines="skip", encoding="latin-1")
    d.columns = [c.strip() for c in d.columns]
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Time Elapsed"]).drop_duplicates("Time Elapsed").sort_values("Time Elapsed").reset_index(drop=True)
    dt = float(d["Time Elapsed"].diff().median())
    if not dt or dt > MAX_DT:
        return None, f"sample interval {dt:g} s"
    d.index = pd.to_timedelta(d["Time Elapsed"], unit="s")
    return d, None


def quantities(d):
    el = d[ELEMENTS]
    nact = (el > 2).sum(axis=1)
    filt = (d["FE/FT-100"] > 10) & (nact >= 2) & (d["VFD-100-FB"] > 0.5) & (d["FE/FT-104"] < 0.5)
    seg = (filt != filt.shift()).cumsum()
    t0 = d.groupby(seg.values)["Time Elapsed"].transform("first").values
    since = d["Time Elapsed"].values - t0
    steady = filt.values & (since >= STEADY_S)
    with np.errstate(divide="ignore", invalid="ignore"):
        bal = (d["FE/FT-100"] - el.sum(axis=1)) / d["FE/FT-100"]
        tmp = (d["PT-101"] - d["PT-104"]).clip(lower=0.05)
        lnr = np.log(tmp / (mu(d["TE/TT-100"]) * d["FE/FT-100"] / nact.replace(0, np.nan)))
    q = {"flow_balance": bal.where(steady), "uf_resistance": lnr.where(steady)}
    phase = pd.Series(np.where(steady, (since // 60).astype(int), -1), index=d.index)
    return q, phase, steady


def trailing(series, phase=None):
    """6 h trailing median taken 1 h back; optionally within each cycle-phase bin."""
    def one(s):
        s = s.dropna()
        if s.empty:
            return s
        r = s.rolling(WIN, min_periods=30).median()
        lagged = r.reindex(s.index - LAG, method="ffill")
        lagged.index = s.index
        return lagged
    if phase is None:
        return one(series).reindex(series.index)
    out = pd.Series(np.nan, index=series.index)
    for b, idx in series.groupby(phase.values).groups.items():
        if b < 0:
            continue
        out.loc[idx] = one(series.loc[idx]).reindex(idx).values
    return out


def deviations(d):
    q, phase, steady = quantities(d)
    dev = {"flow_balance": q["flow_balance"] - trailing(q["flow_balance"]),
           "uf_resistance": q["uf_resistance"] - trailing(q["uf_resistance"], phase)}
    return dev, q, phase, steady


def persist(flag, times, k_s=PERSIST_S):
    """Mark rows belonging to a run of consecutive flagged rows lasting >= k_s seconds."""
    flag = np.asarray(flag, bool)
    out = np.zeros_like(flag)
    i, n = 0, len(flag)
    while i < n:
        if not flag[i]:
            i += 1; continue
        j = i
        while j + 1 < n and flag[j + 1]:
            j += 1
        if times[j] - times[i] >= k_s:
            out[i:j + 1] = True
        i = j + 1
    return out


def main():
    files = {s: sorted(glob.glob(str(BASE / s / "*.csv"))) for s in ("training", "test")}
    xlsx = sorted(p.name for s in ("training", "test") for p in (BASE / s).glob("*.xlsx"))
    skipped, data = [], {"training": [], "test": []}
    for split, paths in files.items():
        for p in paths:
            d, why = load(p)
            if d is None:
                skipped.append(f"{split}/{Path(p).name}: {why}")
            else:
                data[split].append((Path(p).stem, d))

    # calibration on training days
    dev_tr = {n: [] for n in NAMES}
    for _, d in data["training"]:
        dev, *_ = deviations(d)
        for n in NAMES:
            dev_tr[n].append(dev[n].dropna().values)
    cal = {}
    for n in NAMES:
        v = np.abs(np.concatenate(dev_tr[n]))
        cal[n] = {"worst": MARGIN * float(v.max()), "p999": MARGIN * float(np.quantile(v, 0.999)),
                  "rows": int(len(v))}

    # false alarms and sensitivity on test days
    res = {rule: {"false_alarms": {}, "sensitivity": {}} for rule in ("worst", "p999")}
    fa_counts = {rule: {n: [0, 0] for n in NAMES + ["any"]} for rule in res}
    sens = {rule: {n: {lvl: [0, 0] for lvl in LEVELS} for n in ("FT-100", "TMP")} for rule in res}
    for name, d in data["test"]:
        dev, q, phase, steady = deviations(d)
        t = d["Time Elapsed"].values
        for rule in res:
            flags = {}
            for n in NAMES:
                f = (np.abs(dev[n].values) > cal[n][rule]) & np.isfinite(dev[n].values)
                flags[n] = persist(f, t)
                ok = np.isfinite(dev[n].values)
                fa_counts[rule][n][0] += int(flags[n][ok].sum()); fa_counts[rule][n][1] += int(ok.sum())
            anyf = flags["flow_balance"] | flags["uf_resistance"]
            ok = steady
            fa_counts[rule]["any"][0] += int(anyf[ok].sum()); fa_counts[rule]["any"][1] += int(ok.sum())
        # sensitivity: spoof with the clean references frozen
        ref_bal = q["flow_balance"] - dev["flow_balance"]
        ref_r = q["uf_resistance"] - dev["uf_resistance"]
        for lvl in LEVELS:
            for sign in (+1, -1):
                for target in ("FT-100", "TMP"):
                    x = d.copy()
                    if target == "FT-100":
                        x["FE/FT-100"] = x["FE/FT-100"] * (1 + sign * lvl)
                    else:   # TMP biased by spoofing the inlet pressure
                        x["PT-101"] = x["PT-104"] + (x["PT-101"] - x["PT-104"]) * (1 + sign * lvl)
                    qx, _, st = quantities(x)
                    ok = st & np.isfinite(ref_bal.values) & np.isfinite(ref_r.values)
                    for rule in res:
                        fb = np.abs(qx["flow_balance"].values - ref_bal.values) > cal["flow_balance"][rule]
                        fr = np.abs(qx["uf_resistance"].values - ref_r.values) > cal["uf_resistance"][rule]
                        hit = persist((fb | fr) & ok, t)
                        sens[rule][target][lvl][0] += int(hit[ok].sum()); sens[rule][target][lvl][1] += int(ok.sum())
    for rule in res:
        res[rule]["false_alarms"] = {n: round(a / max(b, 1), 5) for n, (a, b) in fa_counts[rule].items()}
        res[rule]["false_alarm_rows"] = {n: b for n, (a, b) in fa_counts[rule].items()}
        res[rule]["sensitivity"] = {tg: {f"±{int(l*100)}%": round(a / max(b, 1), 3) for l, (a, b) in v.items()}
                                    for tg, v in sens[rule].items()}
    out = {"design": __doc__.split("Design fixed BEFORE")[1].split("Run:")[0].strip(),
           "files_used": {s: [n for n, _ in v] for s, v in data.items()},
           "files_skipped": skipped, "xlsx_not_used": xlsx,
           "tolerances": {n: {k: round(v, 5) if isinstance(v, float) else v for k, v in c.items()} for n, c in cal.items()},
           "results": res}
    json.dump(out, open(ROOT / "results" / "hueneme_uf.json", "w"), indent=1)
    print(f"files used: training {len(data['training'])}, test {len(data['test'])}; skipped {len(skipped)}, xlsx {len(xlsx)}")
    print("tolerances:", out["tolerances"])
    for rule in res:
        print(f"\n== tolerance rule: {rule}")
        print("  false alarms on test days:", res[rule]["false_alarms"], " rows:", res[rule]["false_alarm_rows"])
        for tg, v in res[rule]["sensitivity"].items():
            print(f"  spoof {tg}: share of steady rows flagged", v)


if __name__ == "__main__":
    main()
