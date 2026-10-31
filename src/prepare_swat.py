"""
Cache the SWaT spreadsheets as pickles; reading the 2015 xlsx files takes ~50 min,
the pickles load in seconds. Run once after placing the iTrust files:

  data/swat_physical/Physical/SWaT_Dataset_Normal_v1.xlsx   (from SWaT.A1 & A2_Dec_2015/Physical)
  data/swat_physical/Physical/SWaT_Dataset_Attack_v0.xlsx
  data/swat_a6/Dec2019.xlsx                                  (SWaT.A6_Dec 2019/csv/Dec2019.xlsx)

    .venv/bin/python src/prepare_swat.py

Writes normal_v1.pkl and attack_v0.pkl next to the 2015 files, and
data/swat_a6/a6.pkl with tag names normalised ('FIT501.Pv' -> 'FIT501').
The July 2019 (A4/A5) file is cached by swat_a45_test.py itself.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P2015 = ROOT / "data" / "swat_physical" / "Physical"
A6 = ROOT / "data" / "swat_a6"


def main():
    for src, out in (("SWaT_Dataset_Normal_v1.xlsx", "normal_v1.pkl"),
                     ("SWaT_Dataset_Attack_v0.xlsx", "attack_v0.pkl")):
        if (P2015 / out).exists():
            print("exists:", out); continue
        d = pd.read_excel(P2015 / src, header=1, engine="openpyxl")
        d.columns = [str(c).strip() for c in d.columns]
        d.to_pickle(P2015 / out)
        print(out, d.shape)
    if (A6 / "Dec2019.xlsx").exists() and not (A6 / "a6.pkl").exists():
        d = pd.read_excel(A6 / "Dec2019.xlsx", header=9, engine="openpyxl")
        d.columns = [str(c).strip().replace(".Pv", "").replace(".Status", "") for c in d.columns]
        d = d.rename(columns={"t_stamp": "Timestamp"})
        d.to_pickle(A6 / "a6.pkl")
        print("a6.pkl", d.shape)


if __name__ == "__main__":
    main()
