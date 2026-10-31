"""
Sweep input-window size W and error-smoothing k for the autoencoder detector.
Contiguous fit/val split (no window overlap leakage). Saves the best config.
"""
import json, time, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from data import load_batadal, prepare_arrays
from model import Autoencoder
from windowed import make_windows, smooth_errors, episode_stats

ROOT = Path(__file__).resolve().parent.parent
SEED, BATCH, LR, EPOCHS, PATIENCE, PCTILE = 42, 512, 1e-3, 100, 10, 99.5
WINDOWS = [1, 6, 12, 24]
SMOOTHS = [1, 6, 12]
ARCH = {1: ((32, 16), 8), 6: ((128, 32), 16), 12: ((256, 64), 32), 24: ((256, 64), 32)}

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)
tr, t1, t2 = load_batadal(); d = prepare_arrays(tr, t1, t2)
X = d["X_train"]; n_val = int(len(X) * 0.1)
Xfit_raw, Xval_raw = X[:-n_val], X[-n_val:]          # contiguous split


def train(W):
    hidden, bn = ARCH[W]
    Xf = torch.tensor(make_windows(Xfit_raw, W), device=dev)
    Xv = torch.tensor(make_windows(Xval_raw, W), device=dev)
    m = Autoencoder(n_features=Xf.shape[1], hidden=hidden, bottleneck=bn).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=LR); mse = nn.MSELoss()
    loader = DataLoader(TensorDataset(Xf), batch_size=BATCH, shuffle=True)
    best, state, wait = 1e9, None, 0
    for ep in range(EPOCHS):
        m.train()
        for (xb,) in loader:
            opt.zero_grad(); l = mse(m(xb), xb); l.backward(); opt.step()
        m.eval()
        with torch.no_grad(): v = mse(m(Xv), Xv).item()
        if v < best - 1e-6: best, state, wait = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
        else: wait += 1
        if wait >= PATIENCE: break
    m.load_state_dict(state)
    with torch.no_grad(): val_err = m.reconstruction_error(Xv).cpu().numpy()
    return m, state, best, val_err, ep + 1


def score(m, Xnp, W):
    with torch.no_grad():
        return m.reconstruction_error(torch.tensor(make_windows(Xnp, W), device=dev)).cpu().numpy()


rows, best_cfg = [], None
for W in WINDOWS:
    t0 = time.time(); m, state, vloss, val_err, eps = train(W)
    e1, e2 = score(m, d["X_test1"], W), score(m, d["X_test2"], W)
    y1, y2 = d["y_test1"][W - 1:], d["y_test2"][W - 1:]
    print(f"\nW={W:2d}  in={W*43}  val_mse={vloss:.4f}  epochs={eps}  ({time.time()-t0:.0f}s)")
    for k in SMOOTHS:
        thr = float(np.percentile(smooth_errors(val_err, k), PCTILE))
        r = {"W": W, "k": k, "thr": round(thr, 4)}
        for name, e, y in [("t1", e1, y1), ("t2", e2, y2)]:
            es = smooth_errors(e, k); pred = (es > thr).astype(int)
            p, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
            ep_ = episode_stats(y, pred)
            r[name] = {"auc": round(roc_auc_score(y, es), 3), "prauc": round(average_precision_score(y, es), 3),
                       "P": round(p, 3), "R": round(rc, 3), "F1": round(f1, 3),
                       "FPR": round(float(((pred == 1) & (y == 0)).sum() / (y == 0).sum()), 4),
                       "ep": f"{ep_['episodes_detected']}/{ep_['episodes']}", "delay": ep_["mean_delay_steps"]}
        rows.append(r)
        a, b = r["t1"], r["t2"]
        print(f"  k={k:2d} thr={thr:.3f} | t1 AUC {a['auc']} P {a['P']} R {a['R']} F1 {a['F1']} FPR {a['FPR']} ep {a['ep']} "
              f"| t2 AUC {b['auc']} P {b['P']} R {b['R']} F1 {b['F1']} FPR {b['FPR']} ep {b['ep']}")
        meanF1 = (a["F1"] + b["F1"]) / 2
        if best_cfg is None or meanF1 > best_cfg["meanF1"]:
            best_cfg = {"meanF1": meanF1, "W": W, "k": k, "thr": thr, "state": state,
                        "arch": {"n_features": W * 43, "hidden": list(ARCH[W][0]), "bottleneck": ARCH[W][1]}}

json.dump(rows, open(ROOT / "results" / "windowing_sweep.json", "w"), indent=2)
torch.save({"state_dict": best_cfg["state"], "arch": best_cfg["arch"], "window": best_cfg["W"],
            "smooth_k": best_cfg["k"], "threshold": best_cfg["thr"], "threshold_pctile": PCTILE},
           ROOT / "models" / "autoencoder_windowed.pt")
print(f"\nBEST: W={best_cfg['W']} k={best_cfg['k']} meanF1={best_cfg['meanF1']:.3f} -> models/autoencoder_windowed.pt")
