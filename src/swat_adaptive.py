"""
Item 8 on REAL RO data: naive vs physics-aware concealment on SWaT stages P4-P5.

Design fixed before running (18 Oct):
  Scope      TrustFlow's claim is the RO stage, so the detector watches the 22
             P4-P5 tags (15 continuous, 7 actuator states) and the test rows are
             the 9 SWaT 2015 attacks on P4-P5 sensors (RO_SENSOR_ATTACKS).
  Detector   Same autoencoder design as BATADAL, trained on Normal_v1 after the
             first 6 h of start-up, up to the 70% calibration split, every 5th
             second. Threshold = 99.5th percentile of validation error.
  Naive      Edits the 15 continuous readings to minimise reconstruction error
             (actuator states are not editable, as in the BATADAL budget).
  Adaptive   Same, plus all six physics checks (swat_invariants, 2015
             calibration) kept within tolerance: penalty on max(0, |r| - 0.9)^2
             with a rising weight. Gates are recomputed from the attacker's own
             reported values every step (an attacker may try to switch a gate
             off); trailing references are fixed from the clean attack file,
             as in the simulated set.
  Scoring    Physics violations with the deployed 10 s persistence (1 s also
             reported). hidden = caught by the detector before, not after.
  Criterion  Fixed in the white paper: if the adaptive attacker still hides more
             than half of what the naive attacker hides, the physics layer is
             reported as insufficient.
  REVISION before any attack result was used (18 Oct): the first run's
             detector raised false alarms on 40% of held-out normal data and
             100% of the attack file's normal rows, so its results were
             discarded. Causes: all 7 actuator states are constant in training
             (spread ~0, so any switch looks extreme), and the analyser tags
             (AIT401/402/501-504) drift between days (77% false alarms with
             them in). The detector now reads the 9 flow, pressure and level
             tags (DET), and its threshold comes from the temporally held-out
             last 30% of the normal file, not a random sample of the training
             period. The attacker edits the detector's inputs. The physics
             checks still read every tag they need. Checkpoints extended to
             30,000 steps because the naive attacker had not converged.
  Out of scope: wholesale replay of recorded normal data. It is physically
             consistent by construction; Palleti et al. (2020) show replay of
             sensors and actuators also defeats model-based checks.

Run: .venv/bin/python src/swat_adaptive.py  ->  results/swat_adaptive.json
"""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from model import Autoencoder
from swat_invariants import (BASE, CAL_FRAC, MARGIN, NAMES, TRAILING, STEADY_S, FLOW_GATE,
                             WIN_S, LAG_S, RO_SENSOR_ATTACKS, fit, episodes, persist)

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
CONT = ["AIT401", "AIT402", "FIT401", "LIT401", "AIT501", "AIT502", "AIT503", "AIT504",
        "FIT501", "FIT502", "FIT503", "FIT504", "PIT501", "PIT502", "PIT503"]
FEATS = CONT + ["P501"]                       # everything the physics checks read
DET = ["FIT401", "LIT401", "FIT501", "FIT502", "FIT503", "FIT504", "PIT501", "PIT502", "PIT503"]
DET_IDX = [FEATS.index(f) for f in DET]
SKIP_START_S, SUB = 6 * 3600, 5
CHECKPOINTS = [300, 1000, 3000, 10000, 20000, 30000]
LR = 0.05


# ---------------------------------------------------------------- detector
def train_detector(normal):
    torch.manual_seed(SEED); np.random.seed(SEED)
    n_cal = int(len(normal) * CAL_FRAC)
    X = normal[DET].values[SKIP_START_S:n_cal:SUB].astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    idx = np.random.default_rng(SEED).permutation(len(Z))
    nv = len(Z) // 10
    zv, zf = torch.tensor(Z[idx[:nv]], device=DEV), torch.tensor(Z[idx[nv:]], device=DEV)
    model = Autoencoder(n_features=len(DET), hidden=(16, 8), bottleneck=4).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, state, wait = 1e9, None, 0
    for ep in range(200):
        model.train()
        for b in torch.randperm(len(zf), device=DEV).split(256):
            opt.zero_grad(); nn.functional.mse_loss(model(zf[b]), zf[b]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(model(zv), zv).item()
        if v < best - 1e-6:
            best, state, wait = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 10:
                break
    model.load_state_dict(state); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    zh = torch.tensor(((normal[DET].values[n_cal:] - mu) / sd).astype(np.float32), device=DEV)
    with torch.no_grad():
        thr = float(np.percentile(model.reconstruction_error(zh).cpu().numpy(), 99.5))
    return model, torch.tensor(mu, device=DEV), torch.tensor(sd, device=DEV), thr, ep + 1


def recon_err(model, mu, sd, x_full):
    """x_full: [N, len(FEATS)] raw tags; the detector reads only DET."""
    z = (x_full[:, DET_IDX] - mu) / sd
    return ((model(z) - z) ** 2).mean(1)


# ---------------------------------------------------------------- physics in torch
def col(x, name):
    return x[:, FEATS.index(name)]


def quantities_t(x):
    f401, f501, f502, f503, f504 = (col(x, n) for n in ("FIT401", "FIT501", "FIT502", "FIT503", "FIT504"))
    p501, p502, p503 = (col(x, n) for n in ("PIT501", "PIT502", "PIT503"))
    eps = 1e-6
    return {
        "flow_balance": f501 - (f502 + f503 - f504),
        "feed_meters": f401 - f501,
        "brine_dp": torch.log((p501 - p503).clamp(min=eps)) - 1.5 * torch.log(((f501 + f503) / 2).clamp(min=eps)),
        "permeability": torch.log(f502.clamp(min=eps)) - torch.log((p501 - p502).clamp(min=eps)),
        "salt_passage": torch.log(col(x, "AIT504").clamp(min=eps)) - torch.log(col(x, "AIT503").clamp(min=eps)),
        "orp_pair": col(x, "AIT402") - col(x, "AIT502"),
    }


def gates_t(x, steady):
    flowing = (col(x, "FIT401") > FLOW_GATE) & (col(x, "FIT501") > FLOW_GATE)
    return {"feed_meters": torch.ones_like(steady), "orp_pair": flowing,
            **{k: steady for k in ("flow_balance", "brine_dp", "permeability", "salt_passage")}}


def normalised_residuals(x, refs, tols, steady):
    """[N, 6] |deviation| / tolerance, zero where a check does not apply."""
    q, g = quantities_t(x), gates_t(x, steady)
    out = []
    for n in NAMES:
        r = (q[n] - refs[n]).abs() / tols[n]
        ok = g[n] & torch.isfinite(refs[n])
        out.append(torch.where(ok, r, torch.zeros_like(r)))
    return torch.stack(out, 1)


def clean_references(att, cal):
    """Reference per check per row, computed from the unmodified attack file (attacker sees it fixed)."""
    s = att.P501.values
    changed = np.r_[True, s[1:] != s[:-1]]
    since = np.arange(len(s)) - np.maximum.accumulate(np.where(changed, np.arange(len(s)), 0))
    steady = (s == 2) & (since >= STEADY_S)
    flowing = (att.FIT401.values > FLOW_GATE) & (att.FIT501.values > FLOW_GATE)
    x = torch.tensor(att[FEATS].values.astype(np.float32))
    q = {k: v.numpy().astype(float) for k, v in quantities_t(x).items()}
    gate = {"feed_meters": np.ones(len(att), bool), "orp_pair": flowing,
            **{k: steady for k in ("flow_balance", "brine_dp", "permeability", "salt_passage")}}
    refs = {}
    for n in NAMES:
        if n in TRAILING:
            v = np.where(gate[n] & np.isfinite(q[n]), q[n], np.nan)
            refs[n] = pd.Series(v).rolling(WIN_S, min_periods=WIN_S // 4).median().shift(LAG_S).values
        else:
            refs[n] = np.full(len(att), cal[n]["ref"])
    return refs, steady


def violation_flags(R_full, k):
    """R_full: [N_file, 6] normalised residuals over the whole attack file -> any-check flag after persistence."""
    return np.any([persist(R_full[:, i] > 1, k) for i in range(len(NAMES))], 0)


# ---------------------------------------------------------------- attacks
def attack(model, mu, sd, x0, edit_mask, refs, tols, steady, adaptive, log):
    step_scale = torch.ones(x0.shape[1], device=x0.device)
    step_scale[DET_IDX] = sd                     # step in detector-standardised units
    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=LR)
    snaps = {}
    for step in range(1, max(CHECKPOINTS) + 1):
        opt.zero_grad()
        x = x0 + delta * step_scale * edit_mask
        loss = recon_err(model, mu, sd, x).sum()
        if adaptive:
            lam = 10.0 ** min(4.0, step / 3000)           # rising penalty weight, capped at 1e4
            r = normalised_residuals(x, refs, tols, steady)
            loss = loss + lam * (torch.relu(r - 0.9) ** 2).sum()
        loss.backward(); opt.step()
        if step in CHECKPOINTS:
            snaps[step] = (x0 + delta * step_scale * edit_mask).detach().clone()
            log(f"    step {step}")
    return snaps


def main():
    t0 = time.time()
    normal = pd.read_pickle(BASE / "normal_v1.pkl")
    att = pd.read_pickle(BASE / "attack_v0.pkl")
    n_cal = int(len(normal) * CAL_FRAC)
    cal = fit(normal.iloc[:n_cal])

    model, mu, sd, thr, epochs = train_detector(normal)
    print(f"detector trained ({epochs} epochs), threshold {thr:.4f}")

    def err_np(df):
        with torch.no_grad():
            return recon_err(model, mu, sd, torch.tensor(df[FEATS].values.astype(np.float32), device=DEV)).cpu().numpy()

    e_norm = err_np(normal.iloc[n_cal:])
    e_att = err_np(att)
    is_att = att["Normal/Attack"].str.replace(" ", "").eq("Attack").values
    eps = {e["id"]: e for e in episodes(att)}
    ro_rows = np.zeros(len(att), bool)
    for i in RO_SENSOR_ATTACKS:
        ro_rows |= eps[i]["rows"]
    det0 = e_att > thr
    detector = {"false_alarms_normal_heldout": round(float((e_norm > thr).mean()), 5),
                "false_alarms_attackfile_normal_rows": round(float(det0[~is_att].mean()), 5),
                "recall_all_attack_rows": round(float(det0[is_att].mean()), 4),
                "recall_ro_attack_rows": round(float(det0[ro_rows].mean()), 4),
                "ro_attack_rows": int(ro_rows.sum()),
                "ro_rows_caught": int((det0 & ro_rows).sum()),
                "ro_episodes_caught": f"{sum(det0[eps[i]['rows']].any() for i in RO_SENSOR_ATTACKS)} of {len(RO_SENSOR_ATTACKS)}"}
    print("detector:", detector)

    refs_np, steady_np = clean_references(att, cal)
    target = np.flatnonzero(det0 & ro_rows)            # what the attacker must hide
    x_all = torch.tensor(att[FEATS].values.astype(np.float32), device=DEV)
    x0 = x_all[target]
    refs = {n: torch.tensor(refs_np[n][target], dtype=torch.float32, device=DEV) for n in NAMES}
    tols = {n: torch.tensor(cal[n]["tol"], dtype=torch.float32, device=DEV) for n in NAMES}
    steady = torch.tensor(steady_np[target], device=DEV)
    edit = torch.tensor([1.0 if f in DET else 0.0 for f in FEATS], device=DEV)

    refs_full = {n: torch.tensor(refs_np[n], dtype=torch.float32, device=DEV) for n in NAMES}
    steady_full = torch.tensor(steady_np, device=DEV)

    def evaluate(x_mod):
        xf = x_all.clone(); xf[target] = x_mod
        with torch.no_grad():
            R = normalised_residuals(xf, refs_full, tols, steady_full).cpu().numpy()
            e = recon_err(model, mu, sd, x_mod).cpu().numpy()
        hidden = e <= thr
        out = {"hidden": int(hidden.sum())}
        for k in (10, 1):
            v = violation_flags(R, k)[target]
            out[f"persist_{k}s"] = {"caught_by_physics": int((hidden & v).sum()),
                                    "evading_both": int((hidden & ~v).sum())}
        R_t = R[target]
        out["checks_binding_on_hidden_rows"] = {n: round(float((R_t[hidden, i] > 0.8).mean()), 3)
                                                for i, n in enumerate(NAMES)} if hidden.any() else {}
        return out

    baseline = evaluate(x0)
    res = {"design": __doc__.split("Design fixed before running (18 Oct):")[1].split("Run:")[0].strip(),
           "detector": detector, "rows_to_hide": int(len(target)),
           "unmodified_attack_rows": baseline, "naive": {}, "adaptive": {}}
    final = {}
    for name, adaptive in (("naive", False), ("adaptive", True)):
        print(f"{name} attack on {len(target)} rows")
        snaps = attack(model, mu, sd, x0, edit, refs, tols, steady, adaptive, print)
        for step, xs in snaps.items():
            res[name][str(step)] = evaluate(xs)
            r = res[name][str(step)]
            print(f"  {name:8s} {step:6d} steps: hidden {r['hidden']}, evading both (10 s) {r['persist_10s']['evading_both']}")

        final[name] = snaps[max(CHECKPOINTS)]

    # per-attack breakdown at the final checkpoint (added after the first rebuilt run showed
    # naive == adaptive: the surviving physics alarms sat on tags the attacker could not edit)
    tpos = {int(t): k for k, t in enumerate(target)}
    per = {}
    for name, xs in final.items():
        xf = x_all.clone(); xf[target] = xs
        with torch.no_grad():
            R = normalised_residuals(xf, refs_full, tols, steady_full).cpu().numpy()
            e = recon_err(model, mu, sd, xs).cpu().numpy()
        V = violation_flags(R, 10)
        for i in RO_SENSOR_ATTACKS:
            rows = [tpos[t] for t in np.flatnonzero(eps[i]["rows"]) if t in tpos]
            if not rows:
                per.setdefault(str(i), {"attack": eps[i]["attack"][:60], "caught_by_detector": 0})
                continue
            rows = np.array(rows); hid = e[rows] <= thr; v = V[target[rows]]
            per.setdefault(str(i), {"attack": eps[i]["attack"][:60], "caught_by_detector": int(len(rows))})
            per[str(i)][name] = {"hidden": int(hid.sum()), "evading_both": int((hid & ~v).sum()),
                                 "physics_checks_firing": {n: round(float((R[target[rows], j] > 1).mean()), 2)
                                                           for j, n in enumerate(NAMES) if (R[target[rows], j] > 1).any()}}
    res["per_attack_final"] = per
    np.savez(ROOT / "results" / "swat_adaptive_arrays.npz", target=target, thr=thr,
             **{f"x_{k}": v.cpu().numpy() for k, v in final.items()})

    last = str(max(CHECKPOINTS))
    nv, ad = res["naive"][last], res["adaptive"][last]
    ratio = ad["persist_10s"]["evading_both"] / max(nv["hidden"], 1)
    res["summary"] = {
        "naive_hidden": nv["hidden"],
        "naive_evading_both_10s": nv["persist_10s"]["evading_both"],
        "adaptive_evading_both_10s": ad["persist_10s"]["evading_both"],
        "adaptive_over_naive_hidden": round(ratio, 3),
        "criterion": "insufficient if > 0.5",
        "verdict": "physics layer INSUFFICIENT by the pre-set criterion" if ratio > 0.5
                   else "physics layer passes the pre-set criterion",
        "runtime_s": round(time.time() - t0, 1),
    }
    print(json.dumps(res["summary"], indent=1))
    for i, p in res["per_attack_final"].items():
        print(f"#{i:>2} {p['attack'][:45]:45s} caught {p['caught_by_detector']:4d}",
              *(f"| {k}: hidden {p[k]['hidden']}, evade both {p[k]['evading_both']} {p[k]['physics_checks_firing']}"
                for k in ("naive", "adaptive") if k in p))
    json.dump(res, open(ROOT / "results" / "swat_adaptive.json", "w"), indent=1)


if __name__ == "__main__":
    main()
