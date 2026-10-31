"""
Invariant-AWARE concealment attack: projected gradient descent that minimises the
detector's reconstruction error while staying inside the physically feasible set.

This is the honest test of the physics layer. The naive attacker in
attack_concealment.py optimises only against the detector and therefore emits
physically impossible reports. Here the attacker knows the invariant and is
projected back onto it after every step, so:

    STATUS_PUi in {0,1}   and   STATUS_PUi = 0  =>  FLOW_PUi = 0
                                STATUS_PUi = 1  =>  FLOW_PUi >= min observed ON flow

If concealment survives this projection, the invariant is not a defense.
"""
from __future__ import annotations
import json
import numpy as np
import torch
import joblib
from pathlib import Path
from data import load_batadal, prepare_arrays
from model import Autoencoder
from attack_concealment import pump_pairs

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, LR = 10000, 0.03   # plateaus here; see results/attack_convergence.json


def main():
    ck = torch.load(ROOT / "models" / "autoencoder.pt", map_location=DEV, weights_only=False)
    model = Autoencoder(**ck["arch"]).to(DEV)
    model.load_state_dict(ck["state_dict"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    thr = ck["threshold"]
    scaler = joblib.load(ROOT / "models" / "scaler.joblib")
    mu = torch.tensor(scaler.mean_, dtype=torch.float32, device=DEV)
    sd = torch.tensor(scaler.scale_, dtype=torch.float32, device=DEV)

    tr, t1, t2 = load_batadal(); d = prepare_arrays(tr, t1, t2)
    feats = d["feats"]; pairs = pump_pairs(feats)

    # minimum observed ON-flow per pump, from attack-free training data
    min_on = {}
    for si, fi in pairs:
        on = tr[feats[fi]].values[tr[feats[si]].values > 0.5]
        min_on[fi] = float(on.min()) if len(on) else 0.0

    atk = d["X_test1"][d["y_test1"] == 1]
    x0_raw = torch.tensor(scaler.inverse_transform(atk), dtype=torch.float32, device=DEV)

    def project(x):
        """Snap a raw sensor matrix into the physically feasible set."""
        x = x.clone()
        for si, fi in pairs:
            s = (x[:, si] > 0.5).float()          # binarise the reported actuator state
            x[:, si] = s
            f = x[:, fi].clamp(min=0.0)
            x[:, fi] = torch.where(s > 0.5, f.clamp(min=min_on[fi]), torch.zeros_like(f))
        return x

    def err_of(x_raw):
        z = (x_raw - mu) / sd
        return ((model(z) - z) ** 2).mean(dim=1)

    base_err = err_of(x0_raw).detach().cpu().numpy()

    # projected gradient descent
    x = project(x0_raw.clone())
    for _ in range(STEPS):
        x = x.detach().requires_grad_(True)
        loss = err_of(x).sum()
        g, = torch.autograd.grad(loss, x)
        x = project(x.detach() - LR * g * sd)      # step in raw units, then project

    adv_err = err_of(x).detach().cpu().numpy()

    det0, det1 = base_err > thr, adv_err > thr
    newly = det0 & ~det1
    res = {
        "n_attack_samples": int(len(atk)),
        "recall_before": round(float(det0.mean()), 4),
        "recall_after_invariant_aware": round(float(det1.mean()), 4),
        "detected_before": int(det0.sum()),
        "newly_hidden_by_invariant_aware_attack": int(newly.sum()),
        "invariant_violations": 0,
        "note": "attacker is projected onto the invariant every step, so violations are 0 by construction",
    }
    print(json.dumps(res, indent=2))
    json.dump(res, open(ROOT / "results" / "adaptive_attack.json", "w"), indent=2)
    np.savez(ROOT / "results" / "adaptive_arrays.npz",
             base_err=base_err, adv_err=adv_err, thr=thr)


if __name__ == "__main__":
    main()
