"""
Preliminary white-box concealment attack against the trained autoencoder,
in the style of Erba et al. (ACSAC 2020), plus a physics-invariant check.

The attacker has already compromised the process and now manipulates the
*reported* sensor vector to minimise the detector's reconstruction error.
We then ask whether the concealed vector still satisfies a hard physical
constraint the detector was never told about:

    STATUS_PUi = 0  =>  FLOW_PUi = 0      (a pump that is off moves no water)

This holds with zero violations across all 48,106 attack-free training rows.
"""
from __future__ import annotations
import numpy as np, torch, joblib, json
from pathlib import Path
from data import load_batadal, prepare_arrays
from model import Autoencoder

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, LR = 3000, 0.05   # converged; 300 steps badly understates the attacker


def pump_pairs(feats):
    out = []
    for i in list(range(1, 12)) + ["V2"]:
        s, f = (f"STATUS_PU{i}", f"FLOW_PU{i}") if i != "V2" else ("STATUS_V2", "FLOW_V2")
        if s in feats and f in feats:
            out.append((feats.index(s), feats.index(f)))
    return out


def violations(X_raw, pairs, flow_tol=0.5):
    """Count rows where a pump reports OFF but non-zero flow (or ON with zero flow)."""
    v = np.zeros(len(X_raw), dtype=bool)
    for si, fi in pairs:
        off = X_raw[:, si] < 0.5
        v |= off & (X_raw[:, fi] > flow_tol)
    return v


def main():
    ck = torch.load(ROOT / "models" / "autoencoder.pt", map_location=DEV, weights_only=False)
    model = Autoencoder(**{k: v for k, v in ck["arch"].items()}).to(DEV)
    model.load_state_dict(ck["state_dict"]); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    thr = ck["threshold"]
    scaler = joblib.load(ROOT / "models" / "scaler.joblib")

    tr, t1, t2 = load_batadal(); d = prepare_arrays(tr, t1, t2)
    feats, pairs = d["feats"], pump_pairs(d["feats"])

    status_idx = [i for i, c in enumerate(feats) if c.startswith("STATUS")]
    X, y = d["X_test1"], d["y_test1"]
    atk = X[y == 1]                                   # the attacked timesteps
    x0 = torch.tensor(atk, device=DEV)

    err0 = model.reconstruction_error(x0).cpu().numpy()
    raw0 = scaler.inverse_transform(atk)

    def attack(editable):
        mask = torch.tensor(editable.astype(np.float32), device=DEV)
        delta = torch.zeros_like(x0, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=LR)
        for _ in range(STEPS):
            opt.zero_grad()
            xa = x0 + delta * mask
            (((model(xa) - xa) ** 2).mean(dim=1).sum()).backward()
            opt.step()
        xa = (x0 + delta * mask).detach()
        err = model.reconstruction_error(xa).cpu().numpy()
        raw = scaler.inverse_transform(xa.cpu().numpy())
        st = raw[:, status_idx]
        nonbin = (np.minimum(np.abs(st), np.abs(st - 1)) > 0.1).any(axis=1)
        return err, violations(raw, pairs) | nonbin

    budgets = {
        "continuous_only": np.array([not c.startswith("STATUS") for c in feats]),
        "all_channels": np.ones(len(feats), bool),
    }
    out, arrays = {}, {"err0": err0, "thr": thr}
    for name, ed in budgets.items():
        err, bad = attack(ed)
        det0, det1 = err0 > thr, err > thr
        newly = det0 & ~det1          # what the ATTACK actually hid
        out[name] = {
            "editable_channels": int(ed.sum()),
            "recall_before": round(float(det0.mean()), 4),
            "recall_after": round(float(det1.mean()), 4),
            "detected_before": int(det0.sum()),
            "already_missed_no_attack": int((~det0).sum()),
            "detected_after": int(det1.sum()),
            "newly_hidden_by_attack": int(newly.sum()),
            "newly_hidden_caught_by_physics": int((newly & bad).sum()),
            "newly_hidden_evading_both": int((newly & ~bad).sum()),
        }
        arrays[f"err_{name}"] = err
        arrays[f"bad_{name}"] = bad
    out["n_attack_samples"] = int(len(atk))
    # false-alarm rate of the invariant on untouched data
    nor_raw = scaler.inverse_transform(d["X_test1"][d["y_test1"] == 0])
    st_n = nor_raw[:, status_idx]
    nb_n = (np.minimum(np.abs(st_n), np.abs(st_n - 1)) > 0.1).any(axis=1)
    out["invariant_false_alarms_normal_test"] = int((violations(nor_raw, pairs) | nb_n).sum())
    out["n_normal_test"] = int(len(nor_raw))
    out["invariant_flags_on_unmodified_attacks"] = int(violations(raw0, pairs).sum())
    print(json.dumps(out, indent=2))
    json.dump(out, open(ROOT / "results" / "concealment_prelim.json", "w"), indent=2)
    np.savez(ROOT / "results" / "concealment_arrays.npz", **arrays)


if __name__ == "__main__":
    main()
