"""
Concealment attack against the WADI detector.

Same white-box attack as on BATADAL: the attacker edits the reported sensor
vector to minimise the detector's reconstruction error. We report the attack's
actual contribution, i.e. timesteps the detector was catching and now misses.
"""
from __future__ import annotations
import json
import numpy as np, torch
from pathlib import Path
from data_wadi import load_wadi
from model import Autoencoder

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, LR = 300, 0.02


def main():
    ck = torch.load(ROOT / "models" / "autoencoder_wadi.pt", map_location=DEV, weights_only=False)
    model = Autoencoder(**ck["arch"]).to(DEV)
    model.load_state_dict(ck["state_dict"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    thr = ck["threshold"]

    d = load_wadi(verbose=False)
    X, y = d["X_attack"], d["y_attack"]
    atk = X[y == 1]
    x0 = torch.tensor(atk, device=DEV)

    err0 = model.reconstruction_error(x0).cpu().numpy()
    det0 = err0 > thr

    delta = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=LR)
    for _ in range(STEPS):
        opt.zero_grad()
        xa = x0 + delta
        (((model(xa) - xa) ** 2).mean(dim=1).sum()).backward()
        opt.step()
    err1 = model.reconstruction_error((x0 + delta).detach()).cpu().numpy()
    det1 = err1 > thr
    newly = det0 & ~det1

    res = {
        "dataset": "WADI A2, 82 features",
        "attacked_timesteps": int(len(atk)),
        "recall_before": round(float(det0.mean()), 4),
        "recall_after": round(float(det1.mean()), 4),
        "detected_before": int(det0.sum()),
        "detected_after": int(det1.sum()),
        "newly_hidden_by_attack": int(newly.sum()),
        "share_of_caught_timesteps_hidden": round(float(newly.sum() / max(det0.sum(), 1)), 4),
        "median_error_before": round(float(np.median(err0)), 5),
        "median_error_after": round(float(np.median(err1)), 5),
    }
    print(json.dumps(res, indent=2))
    json.dump(res, open(ROOT / "results" / "wadi_attack.json", "w"), indent=2)


if __name__ == "__main__":
    main()
