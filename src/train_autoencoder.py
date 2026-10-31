"""
Train the reconstruction autoencoder on attack-free BATADAL data, pick a
detection threshold from the normal reconstruction-error distribution, and
evaluate on the two labeled test sets.

Pipeline:
  1. Load + scale (scaler fit on train only).
  2. Split train into fit/val (val stays attack-free, used for early stopping
     AND for threshold calibration -> no test leakage).
  3. Train with Adam + MSE, early stopping on val loss.
  4. Threshold = high percentile of val reconstruction errors (controls FP rate).
  5. Evaluate: threshold-free (ROC-AUC, PR-AUC) + at-threshold (P/R/F1, confusion).
  6. Save model, scaler, threshold, metrics, and diagnostic plots.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_fscore_support, confusion_matrix,
                             roc_curve, precision_recall_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_batadal, prepare_arrays
from model import Autoencoder

ROOT = Path(__file__).resolve().parent.parent
MODELS, RESULTS = ROOT / "models", ROOT / "results"
MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)

SEED = 42
EPOCHS = 100
BATCH = 256
LR = 1e-3
VAL_FRAC = 0.1
PATIENCE = 10
THRESHOLD_PCTILE = 99.5   # % of normal val errors below threshold


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def evaluate(name, err, y, thr):
    """Threshold-free + at-threshold metrics for one test set."""
    pred = (err > thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary",
                                                  zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "set": name, "n": int(len(y)), "n_attacks": int(y.sum()),
        "roc_auc": round(float(roc_auc_score(y, err)), 4),
        "pr_auc": round(float(average_precision_score(y, err)), 4),
        "precision": round(float(p), 4), "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "fp_rate": round(float(fp / max(tn + fp, 1)), 4),
    }


def main():
    set_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    tr, t1, t2 = load_batadal()
    d = prepare_arrays(tr, t1, t2)
    X = d["X_train"]

    # fit/val split (val is attack-free -> safe for threshold calibration)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * VAL_FRAC)
    val_idx, fit_idx = idx[:n_val], idx[n_val:]
    Xfit, Xval = X[fit_idx], X[val_idx]
    print(f"fit={Xfit.shape} val={Xval.shape}")

    Xfit_t = torch.tensor(Xfit, device=dev)
    Xval_t = torch.tensor(Xval, device=dev)
    loader = DataLoader(TensorDataset(Xfit_t), batch_size=BATCH, shuffle=True)

    model = Autoencoder(n_features=X.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val, best_state, wait = float("inf"), None, 0
    hist = {"train": [], "val": []}
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); tot = 0.0
        for (xb,) in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        tr_loss = tot / len(Xfit)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval_t), Xval_t).item()
        hist["train"].append(tr_loss); hist["val"].append(val_loss)
        if val_loss < best_val - 1e-6:
            best_val, best_state, wait = val_loss, {k: v.cpu().clone()
                                                    for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if ep % 10 == 0 or ep == 1:
            print(f"epoch {ep:3d}  train {tr_loss:.5f}  val {val_loss:.5f}  (best {best_val:.5f})")
        if wait >= PATIENCE:
            print(f"early stop at epoch {ep} (no val improvement for {PATIENCE})")
            break
    model.load_state_dict(best_state)
    print(f"trained in {time.time()-t0:.1f}s, best val MSE {best_val:.5f}")

    # threshold from normal val errors
    model.eval()
    with torch.no_grad():
        val_err = model.reconstruction_error(Xval_t).cpu().numpy()
    thr = float(np.percentile(val_err, THRESHOLD_PCTILE))
    print(f"threshold ({THRESHOLD_PCTILE} pctile of normal val err): {thr:.5f}")

    def err_of(Xnp):
        with torch.no_grad():
            return model.reconstruction_error(torch.tensor(Xnp, device=dev)).cpu().numpy()
    e1, e2 = err_of(d["X_test1"]), err_of(d["X_test2"])

    results = [evaluate("test_1", e1, d["y_test1"], thr),
               evaluate("test_2", e2, d["y_test2"], thr)]
    print("\n=== EVALUATION ===")
    for r in results:
        print(r)

    # save artifacts
    torch.save({"state_dict": best_state,
                "arch": {"n_features": X.shape[1], "hidden": [32, 16], "bottleneck": 8},
                "threshold": thr, "threshold_pctile": THRESHOLD_PCTILE},
               MODELS / "autoencoder.pt")
    import joblib
    joblib.dump(d["scaler"], MODELS / "scaler.joblib")
    json.dump({"seed": SEED, "epochs_ran": len(hist["train"]),
               "best_val_mse": best_val, "threshold": thr,
               "threshold_pctile": THRESHOLD_PCTILE, "results": results},
              open(RESULTS / "metrics.json", "w"), indent=2)

    # plots
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(hist["train"], label="train"); ax[0].plot(hist["val"], label="val")
    ax[0].set(title="Training curve (MSE)", xlabel="epoch", ylabel="MSE"); ax[0].legend()

    y1 = d["y_test1"]
    ax[1].hist(np.log10(e1[y1 == 0] + 1e-9), bins=80, alpha=.6, label="normal", density=True)
    ax[1].hist(np.log10(e1[y1 == 1] + 1e-9), bins=80, alpha=.6, label="attack", density=True)
    ax[1].axvline(np.log10(thr + 1e-9), color="k", ls="--", label="threshold")
    ax[1].set(title="Test-1 recon error (log10)", xlabel="log10(MSE)"); ax[1].legend()

    fpr, tpr, _ = roc_curve(y1, e1)
    prec, rec, _ = precision_recall_curve(y1, e1)
    ax[2].plot(fpr, tpr, label=f"ROC (AUC {results[0]['roc_auc']})")
    ax[2].plot(rec, prec, label=f"PR (AUC {results[0]['pr_auc']})")
    ax[2].set(title="Test-1 ROC / PR", xlabel="FPR or Recall", ylabel="TPR or Precision")
    ax[2].legend()
    plt.tight_layout(); plt.savefig(RESULTS / "training_report.png", dpi=110)
    print(f"\nsaved: {MODELS/'autoencoder.pt'}, {MODELS/'scaler.joblib'}, "
          f"{RESULTS/'metrics.json'}, {RESULTS/'training_report.png'}")


if __name__ == "__main__":
    main()
