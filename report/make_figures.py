"""
Report figures from the result files (and, for Figure 3, the cached SWaT arrays).

  fig_adaptive.png  Share of each RO attack's detector-caught rows that the physics
                    checks flag: attack as recorded vs after physics-aware concealment.
  fig_energy.png    Energy check: share of attack rows flagged vs HP pump head change,
                    onset reference vs running trailing reference.

Run: .venv/bin/python report/make_figures.py
Colours: reference categorical slots 1-2 (#2a78d6, #eb6834), validated light-mode
with the dataviz validator (all checks pass).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "report"
C1, C2 = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.dpi": 300,
})


def fig_adaptive():
    from swat_adaptive import (FEATS, NAMES, DEV, normalised_residuals, clean_references,
                               violation_flags)
    from swat_invariants import BASE, CAL_FRAC, fit, episodes
    normal = pd.read_pickle(BASE / "normal_v1.pkl")
    att = pd.read_pickle(BASE / "attack_v0.pkl")
    cal = fit(normal.iloc[:int(len(normal) * CAL_FRAC)])
    refs_np, steady_np = clean_references(att, cal)
    refs = {n: torch.tensor(refs_np[n], dtype=torch.float32, device=DEV) for n in NAMES}
    tols = {n: torch.tensor(cal[n]["tol"], dtype=torch.float32, device=DEV) for n in NAMES}
    steady = torch.tensor(steady_np, device=DEV)
    arr = np.load(ROOT / "results" / "swat_adaptive_arrays.npz")
    target = arr["target"]
    x_all = torch.tensor(att[FEATS].values.astype(np.float32), device=DEV)

    def flags(x_mod):
        xf = x_all.clone()
        if x_mod is not None:
            xf[target] = torch.tensor(x_mod, device=DEV)
        with torch.no_grad():
            R = normalised_residuals(xf, refs, tols, steady).cpu().numpy()
        return violation_flags(R, 10)[target]

    before, after = flags(None), flags(arr["x_adaptive"])
    eps = {e["id"]: e for e in episodes(att)}
    per = json.load(open(ROOT / "results" / "swat_adaptive.json"))["per_attack_final"]
    labels = {10: "#10 feed meter spoofed (<0.7)", 39: "#39 feed meter spoofed (0.5)",
              37: "#37 pump off, permeate faked", 40: "#40 feed meter set to 0",
              38: "#38 both ORP probes spoofed", 22: "#22 UV off, ORP spoofed",
              11: "#11 feed meter set to 0"}
    rows = []
    for i, lab in labels.items():
        m = np.isin(target, np.flatnonzero(eps[i]["rows"]))
        rows.append((lab, before[m].mean(), after[m].mean(), int(m.sum())))
    rows.sort(key=lambda r: r[2])

    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    y = np.arange(len(rows)); h = 0.36
    ax.barh(y + h / 2 + 0.02, [r[1] for r in rows], h, color=C1, label="Attack as recorded")
    ax.barh(y - h / 2 - 0.02, [r[2] for r in rows], h, color=C2, label="After physics-aware concealment")
    for k, r in enumerate(rows):
        ax.text(r[2] + 0.01, k - h / 2 - 0.02, f"{r[2]:.0%}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [f"{r[0]}  (n={r[3]})" for r in rows], fontsize=7.8, color=INK)
    ax.set_xlim(0, 1.08); ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Rows flagged by the physics checks (share of detector-caught rows)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7.8)
    fig.tight_layout(); fig.savefig(OUT / "fig_adaptive.png"); plt.close(fig)
    return {r[0]: (round(r[1], 3), round(r[2], 3), r[3]) for r in rows}


def fig_energy():
    e = json.load(open(ROOT / "results" / "energy_check.json"))
    sh = e["share_of_attack_rows_flagged"]
    xs = [int(k.replace("%", "")) for k in sh]
    onset = [sh[k]["onset"] for k in sh]; running = [sh[k]["running"] for k in sh]
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    pos_x = np.arange(len(xs), dtype=float)            # evenly spaced levels (not to scale)
    pos_x[pos_x >= len(xs) // 2] += 1                  # gap at 0
    for s, c, lab in ((onset, C1, "Reference frozen at onset"), (running, C2, "Running 6 h reference")):
        half = len(xs) // 2
        for part in (slice(0, half), slice(half, None)):
            ax.plot(pos_x[part], s[part], color=c, lw=2, marker="o", ms=4,
                    label=lab if part.start == half else None)
    lo, hi = pos_x[xs.index(-2)] - 0.5, pos_x[xs.index(2)] + 0.5
    ax.axvspan(lo, hi, color=GRID, alpha=0.6, lw=0)
    ax.text((lo + hi) / 2, 0.55, "within\nnormal\nscatter", ha="center", va="center", fontsize=7.5, color=INK2)
    ax.set_xticks(pos_x, [f"{v:+d}%" for v in xs], fontsize=7.5)
    ax.set_ylim(-0.03, 1.05); ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("Change in HP pump head during the 6 h attack (levels evenly spaced, not to scale)")
    ax.set_ylabel("Attack rows flagged")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=7.8)
    fig.tight_layout(); fig.savefig(OUT / "fig_energy.png"); plt.close(fig)


if __name__ == "__main__":
    print(fig_adaptive())
    fig_energy()
    print("saved fig_adaptive.png, fig_energy.png")
