"""
BATADAL data loading & preprocessing for TrustFlow.

Design notes
------------
* We train the anomaly detector ONLY on attack-free data (semi-supervised).
* The StandardScaler is fit on TRAIN ONLY, then applied to test sets, to avoid
  information leakage from attack samples into the normalization statistics.
* We keep all 43 features (incl. 4 dead/constant sensors) because the downstream
  Erba concealment attacks (scy-phy/ICS-Evasion-Attacks) operate on the full
  43-dim feature vector. StandardScaler maps constant columns to 0 (scale_=1),
  so they are harmless.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "batadal"

# Columns that are metadata / label, not sensor features.
NON_FEATURE_COLS = ("Unnamed: 0", "DATETIME", "ATT_FLAG")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the 43 sensor feature column names, in file order."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def _read(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name)
    # ATT_FLAG in Erba's files is sometimes -999 for "unlabeled"; coerce to {0,1}.
    if "ATT_FLAG" in df.columns:
        df["ATT_FLAG"] = (df["ATT_FLAG"] == 1).astype(int)
    return df


def load_batadal():
    """Load train / test1 / test2 as DataFrames."""
    train = _read("train_dataset.csv")
    test1 = _read("test_dataset_1.csv")
    test2 = _read("test_dataset_2.csv")
    return train, test1, test2


def prepare_arrays(train, test1, test2):
    """
    Fit scaler on train features, return scaled X arrays + label arrays + scaler.

    Returns dict with X_train, X_test1, y_test1, X_test2, y_test2, feats, scaler.
    """
    feats = feature_columns(train)
    scaler = StandardScaler().fit(train[feats].values)

    def X(df):
        return scaler.transform(df[feats].values).astype(np.float32)

    return {
        "feats": feats,
        "scaler": scaler,
        "X_train": X(train),
        "X_test1": X(test1),
        "y_test1": test1["ATT_FLAG"].values.astype(int),
        "X_test2": X(test2),
        "y_test2": test2["ATT_FLAG"].values.astype(int),
    }


if __name__ == "__main__":
    tr, t1, t2 = load_batadal()
    d = prepare_arrays(tr, t1, t2)
    print("features:", len(d["feats"]))
    print("X_train:", d["X_train"].shape, "attack-free:", (tr["ATT_FLAG"].sum() == 0))
    print("X_test1:", d["X_test1"].shape, "attacks:", d["y_test1"].sum())
    print("X_test2:", d["X_test2"].shape, "attacks:", d["y_test2"].sum())
    print("train scaled mean~0:", round(float(d["X_train"].mean()), 4),
          "std~1:", round(float(d["X_train"].std()), 4))
