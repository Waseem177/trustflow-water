# TrustFlow

Team Hydra's project for the Khalifa University–UNESCO Global Water Hackathon 2026
(Theme 3: Smart, Digital and AI-Enabled Water Systems, team CMAT-00135).

## The idea in plain words

Most of the drinking water in the Gulf comes from desalination plants that push seawater through
reverse-osmosis (RO) membranes. These plants are full of sensors, and more utilities are now putting
AI on top of those sensors to catch problems early.

The catch is that the AI only knows what the sensors tell it. Someone who gets into the plant's network
can edit the readings so everything looks normal while the real plant drifts into trouble. We built
that attack ourselves: it took a detector from catching 65% of attacked readings to catching none.

TrustFlow is our answer. Besides asking "does this look normal?", it asks "does this obey physics?"
Water going in has to equal water coming out. Two meters on the same pipe have to agree. The membranes
have to behave the way pressure and temperature say they should. And the pump's electricity use has
to match the work the readings claim it is doing. Faking one number is easy. Faking a whole set that
still adds up is much harder, and faking a power meter that sits on a separate network is harder again.

## What we found

- **The AI alone can be fully blinded.** A concealment attack, run until it stopped improving, hid
  every attack the detector had caught.
- **Physics checks hold up on real plant data.** On the RO stage of the SWaT water-treatment testbed
  they caught all 9 attacks on RO sensors with no false alarms on held-out data, and both RO attacks on a
  version of the plant they had never seen (0.05% false alarms).
- **They are not magic.** An attacker who controls every sensor in a check can make those sensors agree.
  That is why we added the energy check, which uses the pump's power meter. In simulation it caught a
  disguised pump change of 5% or more from the first minute, with no false alarms.
- **Real seawater is messy.** On 422 days of real seawater pretreatment data from Port Hueneme, the
  checks raised no false alarms, but they only catch large spoofs there.

Our white paper quoted weaker attack numbers (589 hidden, 248, 58%). Those attacks had not finished
optimising, and the numbers below replace them.

| Experiment | Data | Result | Script |
|---|---|---|---|
| Detector under concealment attack | BATADAL | recall 0.65 → **0.00** | `attack_concealment.py` |
| One physics rule vs an attacker who knows it | BATADAL | hidden 1,283 → **477** (63% removed) | `attack_adaptive.py` |
| Does the attack work elsewhere? | WADI A2 | recall 0.26 → 0.15 (42% of caught attacks hidden) | `attack_wadi.py` |
| Seawater RO checks | our simulator | **0.03%** false alarms; can't check bisulphite dose, permeate pressure, free chlorine with pH | `eval_swro_invariants.py` |
| Checks on real RO data | SWaT 2015 | **9 of 9** RO-sensor attacks, 0% held-out false alarms (we tuned the design on this file) | `swat_invariants.py` |
| Clean test on a plant never seen | SWaT Jul 2019 | identity checks **2 of 2**, 0.05% false alarms (the Dec 2019 attacks missed the RO stage, so that test doesn't apply) | `swat_a45_test.py`, `swat_a6_test.py` |
| Attacker who knows the checks | SWaT 2015 RO stage | beats a check when it controls every sensor in it; still caught where the real water changes | `swat_adaptive.py` |
| Energy check | our simulator | all process tags replayed: process checks silent; pump head ≥5% caught from the first minute, 0% false alarms | `eval_energy.py` |
| UF checks on real seawater | Port Hueneme | **0%** false alarms; flow spoofs caught from 20%, pressure spoofs from about 50% | `hueneme_uf.py` |
| Alarm hold (freeze the reference during an alarm) | our simulator | energy check: a 5% pump attack stays flagged to the end (99% vs 3%), still 0 false alarms on 5 fresh runs; process checks: rejected, false alarms rose to 1.3 a day | `eval_alarm_hold.py`, `eval_alarm_hold_energy.py` |
| When the checks pause, and a pump status vs flow check | SWaT 2015, Port Hueneme | SWaT: process checks pause 0.1–0.3% of flowing time (5 min after each pump start); new status-vs-flow check: 0 false alarms held out, fired on 4 attacks. Port Hueneme: UF checks pause about half the flowing time (backwash) | `gating_windows.py` |
| False alarms an operator would see | SWaT 2015, simulator | none on held-out data; about 3 a day on later SWaT days (12 of 14 right after an attack); about one every 6 days in simulation | `alarms_per_day.py` |
| Slow-drift attacks | our simulator | capping how fast the reference moves keeps a 10% step or 2-hour ramp flagged (99% / 89% of rows, up from 33% / 28%) | `eval_drift.py` |

What it still can't do: stop an attacker who controls every channel, including the power meter, or
tell apart a drift slower than normal membrane fouling (about 0.3% per hour). The energy check has only
been tested in simulation, because no public dataset includes pump power.

## Try the demo

Open `results/trustflow_demo.html` in any browser. It plays out in three acts: the attacker fools the
AI but the physics checks catch it; the attacker replays normal readings and every process check goes
quiet; then the pump's power meter gives the game away. Add `#act=2&t=end` to the
address to jump straight to a moment.

## Running it yourself

### Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Use Python 3.12 (tested with 3.12.3, PyTorch 2.12 and pandas 3.0 on Linux/WSL; `.python-version` pins it for pyenv and uv). The Port Hueneme results also reproduce byte-for-byte on macOS with Python 3.14, CPU only. A GPU helps for the
attack experiments but isn't required.

### Data

None of the data is in this repository. BATADAL and Port Hueneme are public. SWaT and WADI belong to
iTrust (Singapore University of Technology and Design): request access from them, and please don't
redistribute the files.

| Put it here | Where to get it |
|---|---|
| `data/batadal/` | run `bash scripts/download_batadal.sh` (Erba et al.'s 15-minute re-simulation) |
| `data/WaDi/WaDi/WADI.A2_19 Nov 2019/WADI_14days_new.csv`, `WADI_attackdataLABLE.csv` | iTrust WaDi folder: extract `WaDi.A2_19 Nov 2019.zip` |
| `data/wadi_82_features.csv` | already here: just the 82 column names Erba et al. used, no data |
| `data/swat_physical/Physical/SWaT_Dataset_{Normal_v1,Attack_v0}.xlsx` | iTrust folder `SWaT.A1 & A2_Dec_2015`, subfolder `Physical` only (~380 MB). Don't download the whole 101 GB zip: it is mostly network captures |
| `data/List_of_attacks_Final.xlsx` | top level of the same `SWaT.A1 & A2_Dec_2015` folder |
| `data/SWaT.A4 & A5_Jul 2019/SWaT_dataset_Jul 19 v2.xlsx` and `SWaT data collection_20-07-2019 v2.pdf` | iTrust folder `SWaT.A4 & A5_Jul 2019` |
| `data/swat_a6/Dec2019.xlsx` | iTrust folder `SWaT.A6_Dec 2019`, file `csv/Dec2019.xlsx` (skip the pcaps) |
| `data/port_hueneme/data_in_UF-RO_system/data in UF-RO system/{training,test}/` | Zenodo 4630769, doi 10.5068/D1310B (CC0): `data_in_UF-RO_system.rar`, 356 MB, MD5 `317ba90d029d86fd7d69208c53b3c4d6`. Extract inside `data/port_hueneme/` (macOS: `brew install unar`, then `unar data_in_UF-RO_system.rar`; Linux: `unrar x`; Windows: 7-Zip) |

The SWaT spreadsheets are slow to read (about 50 minutes for the 2015 files), so cache them once:

```bash
python src/prepare_swat.py
```

### Check the install

```bash
python -m unittest discover tests
```

Sixteen quick tests of the core pieces (persistence, alarm counting, the rate-capped reference and its alarm
hold, the energy check, and the checks on a short simulated run). No dataset needed; about 15 seconds.

### Reproduce the results

Run these from the repo root. Each one writes its numbers to `results/`.

| Step | Command | Writes | Time |
|---|---|---|---|
| 1 | `python src/train_autoencoder.py` | `models/autoencoder.pt`, `metrics.json` | ~1 min |
| 2 | `python src/attack_concealment.py` | `concealment_prelim.json` | a few GPU minutes |
| 3 | `python src/attack_adaptive.py` | `adaptive_attack.json` | a few GPU minutes |
| 4 | `python src/train_wadi.py`, then `python src/attack_wadi.py` | `wadi_baseline.json`, `wadi_attack.json` | minutes |
| 5 | `python src/eval_swro_invariants.py` | `swro_invariants.json` | ~15 s |
| 6 | `python src/eval_energy.py` | `energy_check.json` | ~30 s |
| 7 | `python src/swat_invariants.py` | `swat_invariants.json` | seconds |
| 8 | `python src/swat_a6_test.py`, `python src/swat_a45_test.py` | `swat_a6_test.json`, `swat_a45_test.json` | seconds (+1 min the first time) |
| 9 | `python src/swat_adaptive.py` | `swat_adaptive.json` | ~13 min on an RTX 4060 |
| 10 | `python src/hueneme_uf.py` | `hueneme_uf.json` | ~40 s |
| 11 | `python src/eval_drift.py` | `drift_guard.json` | ~7 min |
| 11d | `python src/gating_windows.py` | `gating_windows.json` | ~20 s |
| 11c | `python src/eval_alarm_hold.py`, then `python src/eval_alarm_hold_energy.py` | `alarm_hold.json`, `alarm_hold_energy.json` | ~9 min + 1 min |
| 10c | `python src/hueneme_pump_curve.py` (Fawaz; negative result) | `hueneme_pump_curve.json` | ~1 min |
| 11b | `python src/alarms_per_day.py` | `alarms_per_day.json` | ~40 s |
| 12 | `python src/make_demo_acts.py` | `trustflow_demo.html` (`make_demo.py` is an older single-panel version) | ~40 s |
| 13 | `python report/make_figures.py`, `python report/figure_architecture.py` | report figures | ~1 min |

The neural-network numbers (detector, attacks, WADI) shift a little between machines: on a MacBook (arm64, CPU)
the BATADAL detector's threshold moved about 7% and the rule-aware attack hid 452 instead of 477; every headline
claim held. The physics, energy and Port Hueneme results reproduced exactly. The training and attack scripts
overwrite `results/` and `models/`, so run them in a fresh clone if you want to compare against ours.
`requirements.txt` pins the versions we used.

## What's where

```
src/        detectors, attacks, the seawater RO simulator, physics checks, experiments
results/    result files (JSON) and the demo page
report/     figure scripts
models/     trained BATADAL detectors (train_wadi.py rebuilds the WADI one, which is not shared because it is derived from iTrust data)
tests/      unit tests (python -m unittest discover tests)
scripts/    BATADAL download
```

## Team

Team Hydra, Sathyabama Institute of Science and Technology:
Mohamed Waseem Mohamed Kaisar (team leader), Mohammed Fawaz, Mohammed Sharukh and
Muhammad Abdur Rahman Ahamed Kaiser. Academic advisor: Dr. Saranya Velusamy.

## License

The code is released under the MIT License (see `LICENSE`). The datasets keep their own terms: SWaT and WADI
belong to iTrust and are not included.

## Thanks and prior work

SWaT and WADI data come from iTrust, Singapore University of Technology and Design. The Port Hueneme
UF data were released on Zenodo under CC0. Our attack follows Erba et al. (ACSAC 2020), and the idea of
checking readings against plant physics goes back to Cárdenas et al. (2011), Urbina et al. (2016) and
Adepu & Mathur (2016). The full comparison is in Section 4 of our report.
