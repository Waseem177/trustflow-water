"""
Train the reconstruction detector on WADI A2 normal data and evaluate it on the
attack set, at the same operating point Erba et al. report (FPR = 1%).

Their published baseline on this dataset: Recall 0.68, FPR 0.01.
"""
from __future__ import annotations
import json, time
import numpy as np, torch, torch.nn as nn, joblib
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from data_wadi import load_wadi
from model import Autoencoder
from windowed import episode_stats

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED, BATCH, LR, EPOCHS, PATIENCE, TARGET_FPR = 42, 512, 1e-3, 60, 8, 0.01

torch.manual_seed(SEED); np.random.seed(SEED)
d = load_wadi()
X = d["X_normal"]; n_val = int(len(X) * 0.1)
Xfit, Xval = X[:-n_val], X[-n_val:]                      # contiguous split
Xfit_t = torch.tensor(Xfit, device=DEV); Xval_t = torch.tensor(Xval, device=DEV)

model = Autoencoder(n_features=X.shape[1], hidden=(64, 32), bottleneck=16).to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=LR); mse = nn.MSELoss()
loader = DataLoader(TensorDataset(Xfit_t), batch_size=BATCH, shuffle=True)

best, state, wait, t0 = 1e9, None, 0, time.time()
for ep in range(1, EPOCHS + 1):
    model.train()
    for (xb,) in loader:
        opt.zero_grad(); loss = mse(model(xb), xb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        v = mse(model(Xval_t), Xval_t).item()
    if v < best - 1e-7:
        best, state, wait = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
    else:
        wait += 1
    if ep % 5 == 0 or ep == 1:
        print(f"  epoch {ep:3d}  val MSE {v:.5f}  (best {best:.5f})")
    if wait >= PATIENCE:
        print(f"  early stop at epoch {ep}"); break
model.load_state_dict(state)
print(f"trained in {time.time()-t0:.0f}s, best val MSE {best:.5f}")


def err(a):
    out = []
    with torch.no_grad():
        for i in range(0, len(a), 20000):
            out.append(model.reconstruction_error(torch.tensor(a[i:i+20000], device=DEV)).cpu().numpy())
    return np.concatenate(out)


val_err = err(Xval)
thr = float(np.percentile(val_err, 100 * (1 - TARGET_FPR)))   # 1% false alarms on normal
e = err(d["X_attack"]); y = d["y_attack"]; pred = (e > thr).astype(int)

recall = float(pred[y == 1].mean()); fpr = float(pred[y == 0].mean())
prec = float((y[pred == 1] == 1).mean()) if pred.sum() else 0.0
es = episode_stats(y, pred)
res = {"dataset": "WADI A2 (19 Nov 2019), Erba 82-feature subset",
       "n_normal": int(len(X)), "n_attack_rows": int(len(y)), "attacked_timesteps": int(y.sum()),
       "threshold_at_FPR": TARGET_FPR, "threshold": round(thr, 6),
       "recall": round(recall, 4), "precision": round(prec, 4), "fpr": round(fpr, 4),
       "f1": round(2 * prec * recall / (prec + recall), 4) if prec + recall else 0.0,
       "roc_auc": round(float(roc_auc_score(y, e)), 4),
       "pr_auc": round(float(average_precision_score(y, e)), 4),
       "episodes_detected": f"{es['episodes_detected']}/{es['episodes']}",
       "erba_published": {"recall": 0.68, "fpr": 0.01}}
print("\n=== WADI result ===")
for k, v in res.items(): print(f"  {k}: {v}")
json.dump(res, open(ROOT / "results" / "wadi_baseline.json", "w"), indent=2)
torch.save({"state_dict": state, "arch": {"n_features": X.shape[1], "hidden": [64, 32], "bottleneck": 16},
            "threshold": thr, "target_fpr": TARGET_FPR}, ROOT / "models" / "autoencoder_wadi.pt")
joblib.dump(d["scaler"], ROOT / "models" / "scaler_wadi.joblib")
