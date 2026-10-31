"""
WADI A2 (19 Nov 2019) loader, on Erba et al.'s 82-feature subset.

Two quirks this handles:
  * the attack CSV's real header is on line 2 (line 1 is just 0..130)
  * Erba's feature list names the last column with its A1 Windows path prefix;
    in A2 it is plainly TOTAL_CONS_REQUIRED_FLOW
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "data" / "WaDi" / "WaDi" / "WADI.A2_19 Nov 2019"
NORMAL = BASE / "WADI_14days_new.csv"
ATTACK = BASE / "WADI_attackdataLABLE.csv"
LABEL = "Attack LABLE (1:No Attack, -1:Attack)"


def feature_list() -> list[str]:
    raw = (ROOT / "data" / "wadi_82_features.csv").read_text().strip().split(",")
    return [c.split("\\")[-1] if "\\" in c else c for c in raw]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sensor dropouts appear as NaN; carry the last reading forward."""
    return df.ffill().bfill()


def load_wadi(verbose: bool = True):
    feats = feature_list()
    normal = _clean(pd.read_csv(NORMAL, usecols=feats)[feats].astype(np.float32))

    atk = pd.read_csv(ATTACK, skiprows=1)
    atk.columns = [c.strip() for c in atk.columns]
    lab = next(c for c in atk.columns if c.startswith("Attack LABLE"))
    y = (atk[lab].values == -1).astype(int)          # -1 means under attack
    attack = _clean(atk[feats].astype(np.float32))

    scaler = StandardScaler().fit(normal.values)
    Xn, Xa = scaler.transform(normal.values), scaler.transform(attack.values)
    if verbose:
        print(f"normal: {Xn.shape}  attack: {Xa.shape}  attacked timesteps: {y.sum()} ({y.mean():.1%})")
    return dict(X_normal=Xn.astype(np.float32), X_attack=Xa.astype(np.float32),
                y_attack=y, feats=feats, scaler=scaler)


if __name__ == "__main__":
    d = load_wadi()
    print("features:", len(d["feats"]))
    print("any NaN:", bool(np.isnan(d["X_normal"]).any() or np.isnan(d["X_attack"]).any()))
    print("constant columns in normal:", int((d["X_normal"].std(0) == 0).sum()))
