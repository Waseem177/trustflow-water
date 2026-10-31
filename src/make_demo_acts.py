"""
Three-act demo page for the one-minute video (replaces the single-panel make_demo.py page).
Page design (UAE theme) made with Codex on 26-27 Oct and reviewed by the team; the data it shows is
computed below.

    .venv/bin/python src/make_demo_acts.py  ->  results/trustflow_demo.html

Everything shown is computed here from the simulator (seed 0), not drawn by hand.
Attack onset day 44.0, 6 h. The AI detector is an autoencoder on the 17 process
tags, trained on days 30-43.9 (90%; threshold = 99.5th percentile of the other
10%). The onset was chosen where the detector is quiet on normal data (0.6% of the
unattacked window): from day 45 the plant moves to a new, legitimate operating
point the detector never saw, and it then flags 100% of normal rows, while the
physics checks, which normalise for operating conditions, do not.

  Act 1  Reported permeate TDS lowered 20%; the attacker also edits the other
         process tags (gradient concealment, 2,000 steps) so the AI sees normal
         data. Physics checks evaluated on the concealed record.
  Act 2  True HP pump head +10% (membrane over-pressure); every process tag
         replays the genuine record. AI and physics see normal data.
  Act 3  Same attack; the electrical meter, on its own network, reports the
         true power. Energy check against its rate-capped running reference.
Physics and energy checks use the deployed design: trailing references capped
by drift_guard, tolerances calibrated on days 0-30.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from swro_sim import simulate, TAGS, P_SUC
from invariants_swro import NAMES, DRIFTING, fit, residuals, quantities, trailing_reference
from model import Autoencoder
import energy_check as E
import drift_guard as G

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
ONSET, DUR, PRE = 44 * 1440, 360, 120
SPOOF, HEAD = 0.80, 0.10
TRAILING = dict(window_min=360, lag_min=60)
PROC = [t for t in TAGS if t != "W_hp"]
LABEL = {"flow_balance": "flow balance", "salt_balance": "salt balance", "ro_water_perm": "membrane water flow",
         "ro_salt_perm": "salt passage", "ro_brine_dp": "brine pressure drop", "uf_resistance": "UF resistance",
         "orp_pretreat": "ORP vs chlorine", "orp_ro_feed": "ORP at RO feed"}


def train_detector(X):
    torch.manual_seed(SEED)
    rows = np.random.RandomState(SEED).permutation(np.arange(30 * 1440, int(43.9 * 1440)))
    val, trn = rows[: len(rows) // 10], rows[len(rows) // 10:]
    mu, sd = X[trn].mean(0), X[trn].std(0)
    Z = torch.tensor((X - mu) / sd, device=DEV)
    m = Autoencoder(n_features=X.shape[1], hidden=(16, 8), bottleneck=8).to(DEV)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    tr = Z[trn]
    for _ in range(40):
        for b in torch.randperm(len(tr), device=DEV).split(512):
            opt.zero_grad(); nn.functional.mse_loss(m(tr[b]), tr[b]).backward(); opt.step()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    err = lambda z: ((m(z) - z) ** 2).mean(1)
    with torch.no_grad():
        thr = float(np.percentile(err(Z[val]).cpu().numpy(), 99.5))
    return err, mu, sd, thr


def main():
    np.random.seed(SEED)
    truth, rep, ev = simulate(days=60, seed=SEED)
    n = len(rep); t = np.arange(n, dtype=float)
    m = (ev.day < 30).values
    log = {"cip": np.flatnonzero(ev.cip.values), "ceb": np.flatnonzero(ev.ceb.values), "uf_phase": ev.uf_phase.values}
    log_cal = {"cip": log["cip"][log["cip"] < m.sum()], "ceb": log["ceb"][log["ceb"] < m.sum()],
               "uf_phase": log["uf_phase"][m]}
    cal = fit(rep[m], trailing=TRAILING, events=log_cal)
    group = {k: (log["uf_phase"] if k == "uf_resistance" else None) for k in DRIFTING}
    raw_clean = trailing_reference(quantities(rep, cal["orp_k"]), **TRAILING, events=log)
    caps = {k: G.fit_cap(raw_clean[k].numpy()[m], t[m], group=None if group[k] is None else group[k][m]) for k in DRIFTING}

    def physics(x):
        raw = trailing_reference(quantities(x, cal["orp_k"]), **TRAILING, events=log)
        ref = {k: torch.tensor(G.cap(raw[k].numpy(), t, caps[k], group=group[k]), dtype=torch.float32) for k in DRIFTING}
        return residuals(x, cal, ref=ref, events=log).abs().numpy()

    X = rep[PROC].values.astype(np.float32)
    err, mu, sd, thr = train_detector(X)
    w = np.arange(ONSET, ONSET + DUR)
    show = np.arange(ONSET - PRE, ONSET + DUR)
    ci = PROC.index("C_perm")

    def ai(Xa):
        with torch.no_grad():
            return err(torch.tensor((Xa - mu) / sd, device=DEV)).cpu().numpy()

    # ---- Act 1: spoof + concealment against the AI
    Xs = X.copy(); Xs[w, ci] *= SPOOF
    zs = torch.tensor((Xs - mu) / sd, device=DEV)
    mask = torch.ones(len(PROC), device=DEV); mask[ci] = 0
    z0 = zs[w].clone(); delta = torch.zeros_like(z0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.05)
    for _ in range(2000):
        opt.zero_grad(); err(z0 + delta * mask).sum().backward(); opt.step()
    Xc = Xs.copy(); Xc[w] = (z0 + delta * mask).detach().cpu().numpy() * sd + mu
    x1 = pd.DataFrame(Xc, columns=PROC); x1["W_hp"] = rep["W_hp"].values
    ai_raw, ai_hid, ai_clean = ai(Xs), ai(Xc), ai(X)
    R1 = physics(x1); R0 = physics(rep)

    # ---- Acts 2-3: pump over-pressure, process tags replay the genuine record
    frac = np.zeros(n); frac[w] = 1.0
    true_P = P_SUC + (truth["P_feed"].values - P_SUC) * (1 + HEAD * frac)
    meter = rep["W_hp"].values * (1 + HEAD * frac)
    x3 = rep.copy(); x3["W_hp"] = meter
    r_clean = E.residual(rep); eref_clean = E.trailing(r_clean)
    ecal = E.fit(r_clean[m], eref_clean[m]); ecap = G.fit_cap(eref_clean[m], t[m])
    r3 = E.residual(x3); eref3 = G.cap(E.trailing(r3), t, ecap)
    e_norm = np.abs(r3 - eref3) / ecal["tol"]
    implied = rep["Q_feed"].values * (rep["P_feed"].values - P_SUC) / 36.0 / np.exp(-eref3)

    def rate(flag):
        return round(float(flag[w].mean()), 3)

    stats = {"ai_threshold": round(thr, 4),
             "ai_false_alarm_unattacked_window": rate(ai_clean > thr),
             "act1": {"ai_flags_raw_spoof": rate(ai_raw > thr), "ai_flags_concealed": rate(ai_hid > thr),
                      "physics_flags_concealed": rate((R1 > 1).any(1)),
                      "physics_by_check": {LABEL[nm]: round(float((R1[w, i] > 1).mean()), 3) for i, nm in enumerate(NAMES)},
                      "physics_false_alarm_unattacked": rate((R0 > 1).any(1))},
             "act2": {"ai_flags": rate(ai_clean > thr), "physics_flags": rate((R0 > 1).any(1))},
             "act3": {"energy_flags": rate(e_norm > 1)}}
    top = sorted(((float((R1[w, i] > 1).mean()), LABEL[nm]) for i, nm in enumerate(NAMES)), reverse=True)
    stats["act1"]["main_checks"] = [lab for v, lab in top if v > 0.1][:2]

    rd = lambda a, k=3: [None if not np.isfinite(v) else round(float(v), k) for v in a]
    data = {
        "t": [int(v - ONSET) for v in show], "dur": DUR,
        "act1": {"true": rd(truth["C_perm"].values[show], 1), "shown": rd(x1["C_perm"].values[show], 1),
                 "ai": rd(ai_hid[show] / thr), "ai_uncovered": rd(ai_raw[show] / thr),
                 "phys": rd(R1[show].max(1))},
        "act2": {"true": rd(true_P[show], 2), "shown": rd(rep["P_feed"].values[show], 2),
                 "ai": rd(ai_clean[show] / thr), "phys": rd(R0[show].max(1))},
        "act3": {"meter": rd(meter[show], 0), "implied": rd(implied[show], 0), "energy": rd(e_norm[show])},
        "stats": stats,
    }
    json.dump(stats, open(ROOT / "results" / "trustflow_demo_stats.json", "w"), indent=1)
    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    (ROOT / "results" / "trustflow_demo.html").write_text(html, encoding="utf-8")
    print(json.dumps(stats, indent=1))


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>TrustFlow | Water Security Demonstrator</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
:root{--paper:#f4f1e9;--panel:#fffcf6;--ink:#153e3a;--ink-2:#526863;--ink-3:#727b71;--rule:#dddcd0;--real:#b26c37;--screen:#138779;--alarm:#b74332;--alarm-ink:#fffaf1;--attack:#f3e9dc;--focus:#138779;--sans:'Archivo','Segoe UI',Arial,sans-serif;--mono:'IBM Plex Mono',Consolas,monospace;color-scheme:light}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 var(--sans);-webkit-font-smoothing:antialiased}button,input,select{font:inherit}button{cursor:pointer}button:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid var(--focus);outline-offset:4px}button{transition:background .16s,color .16s}svg{display:block;width:100%;height:auto}.masthead{background:#103d38;color:#f4f0e3;border-bottom:3px solid #b99c62}.nav-inner{max-width:1400px;margin:auto;padding:17px 36px;display:flex;align-items:center;justify-content:space-between;gap:20px}.brand{display:flex;align-items:center;gap:12px}.brand-mark{width:38px;height:42px}.brand b{font-size:25px;letter-spacing:-1px;font-weight:600}.brand small{display:block;font-size:9px;letter-spacing:2.5px;text-transform:uppercase;color:#b8c8b9}.nav-context{text-align:right;font-size:11px;letter-spacing:1.2px;text-transform:uppercase}.nav-context span{display:block;color:#b7c8bd;font-size:10px;margin-top:3px;letter-spacing:1px}.wrap{max-width:1400px;margin:auto;padding:0 36px 24px}.hero{display:grid;grid-template-columns:1.6fr 1fr;align-items:center;min-height:204px;gap:38px;padding:25px 0 22px}.eyebrow{font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;display:flex;gap:12px;align-items:center;margin-bottom:12px}.eyebrow:before{content:'';width:26px;height:1px;background:#ab8a51}h1{font-family:Georgia,'Times New Roman',serif;font-size:clamp(32px,3.2vw,48px);font-weight:400;letter-spacing:-1.5px;line-height:1.08;margin:0 0 12px;text-wrap:balance}.lede{color:var(--ink-2);font-size:14px;max-width:690px;line-height:1.65;margin:0}.coast{position:relative;align-self:stretch;display:flex;align-items:center}.coast svg{max-height:185px;width:100%;filter:drop-shadow(0 5px 8px rgba(58,70,48,.035))}.coast-label{position:absolute;right:0;bottom:0;font-size:9px;letter-spacing:1.6px;color:var(--ink-2);text-transform:uppercase}.steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border:1px solid #ced3c7;margin-bottom:22px;background:#eaece2}.step{border:0;border-right:1px solid #ced3c7;background:transparent;color:var(--ink-2);display:flex;text-align:left;align-items:center;gap:16px;padding:14px 20px;min-width:0}.step:last-child{border-right:0}.step i{font:400 22px/1 Georgia,serif;color:#89917d;font-style:normal}.step strong{font-size:13px;font-weight:600;display:block}.step small{font-size:10px;color:var(--ink-2);display:block;margin-top:3px}.step[aria-selected=true]{background:#17483f;color:#fffdf3}.step[aria-selected=true] i{color:#cfb581}.step[aria-selected=true] small{color:#c1d2c4}.step:hover:not([aria-selected=true]){background:#dfe5d8}.stage{display:grid;grid-template-columns:minmax(0,1.8fr) minmax(0,1fr);gap:20px;align-items:stretch}.panel{border:1px solid var(--rule);background:var(--panel);padding:20px;min-width:0;position:relative}.ptitle{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:15px}.ptitle h2{margin:0;font-size:13px;font-weight:600}.section-id{font:10px var(--mono);letter-spacing:1px;color:var(--ink-3);margin-bottom:7px}.signal-status{font:10px var(--mono);padding:5px 8px;border:1px solid #d7dfce;background:#edf2e7;white-space:nowrap}.signal-status.attack{color:#96502d;background:#f4e7d8;border-color:#e7cfb4}.readings{display:flex;gap:0;border-block:1px solid var(--rule);margin-bottom:15px;padding:13px 0}.reading{width:50%;padding-left:20px;border-left:1px solid var(--rule)}.reading:first-child{padding-left:0;border:0}.reading-label{font-size:10px;text-transform:uppercase;letter-spacing:1px;display:flex;align-items:center;gap:7px;color:var(--ink-2)}.reading-label:before{content:'';height:7px;width:7px;border-radius:50%;background:var(--real)}.reading:nth-child(2) .reading-label:before{background:var(--screen)}.reading-value{font:400 29px/1.4 var(--mono);letter-spacing:-1px}.reading-value small{font:11px var(--sans);letter-spacing:0;color:var(--ink-3);margin-left:7px}.legend{display:none}#chart{touch-action:pan-y;cursor:crosshair}.chart-foot{font-size:10px;color:var(--ink-2);display:flex;justify-content:space-between;padding-top:10px;border-top:1px solid var(--rule);gap:10px}.det{display:grid}.row{padding:14px 0;border-top:1px solid var(--rule);display:grid;grid-template-columns:1fr auto;gap:5px 10px}.row .q{font-size:10px;color:var(--ink-2);margin-bottom:2px}.row .n{font-size:15px;font-weight:600}.row .why{grid-column:1/-1;color:var(--ink-2);font-size:11px;min-height:17px}.row svg{grid-column:1/-1;height:35px}.chip{align-self:center;display:inline-flex;gap:5px;align-items:center;font-size:9px;letter-spacing:1px;text-transform:uppercase;border:1px solid #d8dfd2;background:#eef2e8;color:#4f6e59;padding:5px 8px}.chip:before{content:'';width:5px;height:5px;border-radius:50%;background:currentColor}.row.alarm .chip{background:var(--alarm);color:var(--alarm-ink);border-color:var(--alarm)}.row.off .chip{color:#777c72;background:#f1f0e9;border-color:#dedfd7}.row.off svg{opacity:.4}.row.off .chip:before{background:transparent;border:1px solid currentColor;width:5px;height:5px}.detector-note{font-size:10px;color:var(--ink-3);margin:4px 0 0;padding-top:10px;border-top:1px solid var(--rule)}.bar{margin-top:18px;display:flex;align-items:center;gap:16px;padding:14px 18px;background:#e6e9df;border:1px solid #d4dbcd}.play{display:flex;align-items:center;gap:9px;background:#16483f;color:#fffdf3;border:0;padding:11px 16px;font-size:12px;min-width:89px;justify-content:center}.play:hover{background:#256255}.play svg{width:11px;height:11px}#scrub{flex:1;min-width:70px;accent-color:#236857;cursor:pointer}.clock{font:11px var(--mono);min-width:173px}.clock em{font:10px var(--sans);font-style:normal;color:var(--ink-2);margin-right:7px}.utility{border:1px solid #bec9b7;background:transparent;padding:8px 10px;font-size:10px;color:var(--ink)}.utility:hover{background:#d4ddcd}.footer{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;margin-top:19px;font-size:10px;color:var(--ink-2)}.footer-brand{white-space:nowrap;letter-spacing:1px}.footer details{max-width:760px;text-align:right}.footer summary{cursor:pointer;list-style-position:inside}.fine{font-size:11px;line-height:1.7;margin:10px 0;text-align:left}.mono{font-family:var(--mono)}.key-hint{font-size:9px;color:var(--ink-2);white-space:nowrap}button[aria-selected=true]{box-shadow:inset 0 -2px #bea36c}
@media(min-width:1500px){.hero{min-height:220px}.wrap,.nav-inner{max-width:1480px}.panel{padding:23px}.row{padding:18px 0}}
@media(max-width:1000px){.wrap{padding-inline:22px}.nav-inner{padding-inline:22px}.stage{grid-template-columns:minmax(0,1.55fr) minmax(0,1fr)}.panel{padding:14px}.hero{gap:20px}.step{padding:13px;gap:10px}.step small{font-size:9px}.clock{min-width:148px}.key-hint{display:none}.reading-value{font-size:24px}}
@media(max-width:760px){.wrap{padding-inline:16px}.nav-inner{padding:14px 16px}.nav-context{font-size:9px;letter-spacing:.5px}.nav-context span{font-size:8px}.brand b{font-size:23px}.brand small{font-size:8px}.brand-mark{width:30px}.hero{grid-template-columns:1fr;padding:25px 0}.coast{display:none}h1{font-size:35px}.lede{font-size:13px}.steps{margin-bottom:16px}.step{padding:12px 8px;gap:7px;align-items:flex-start}.step i{font-size:17px}.step strong{font-size:11px}.step small{display:none}.stage{grid-template-columns:1fr}.bar{flex-wrap:wrap;gap:10px;padding:12px}.clock{min-width:auto}.utility{padding:8px}#scrub{min-width:120px}.footer{flex-direction:column;gap:10px}.footer details{text-align:left}.reading-value{font-size:27px}.row svg{height:43px}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}

/* Reference-inspired geometric lattice and woven UAE border. */
.masthead{position:relative;overflow:hidden}.masthead::before{content:"";position:absolute;inset:0;width:100%;background-image:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22192%22%20height%3D%22192%22%20viewBox%3D%220%200%20192%20192%22%3E%3Cdefs%3E%3Cpattern%20id%3D%22ornament%22%20width%3D%2296%22%20height%3D%2296%22%20patternUnits%3D%22userSpaceOnUse%22%3E%3Cg%20fill%3D%22none%22%20stroke%3D%22%23b1945c%22%20stroke-width%3D%22.85%22%3E%3Cpolygon%20points%3D%2286.00%2C48.00%2072.15%2C54.47%2080.91%2C67.00%2065.68%2C65.68%2067.00%2C80.91%2054.47%2C72.15%2048.00%2C86.00%2041.53%2C72.15%2029.00%2C80.91%2030.32%2C65.68%2015.09%2C67.00%2023.85%2C54.47%2010.00%2C48.00%2023.85%2C41.53%2015.09%2C29.00%2030.32%2C30.32%2029.00%2C15.09%2041.53%2C23.85%2048.00%2C10.00%2054.47%2C23.85%2067.00%2C15.09%2065.68%2C30.32%2080.91%2C29.00%2072.15%2C41.53%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(0%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(30%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(60%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(90%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(120%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(150%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(180%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(210%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(240%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(270%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(300%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(330%2048%2048)%22%2F%3E%3Ccircle%20cx%3D%2248%22%20cy%3D%2248%22%20r%3D%2217%22%2F%3E%3Cpolygon%20points%3D%2265.00%2C48.00%2058.63%2C50.85%2062.72%2C56.50%2055.78%2C55.78%2056.50%2C62.72%2050.85%2C58.63%2048.00%2C65.00%2045.15%2C58.63%2039.50%2C62.72%2040.22%2C55.78%2033.28%2C56.50%2037.37%2C50.85%2031.00%2C48.00%2037.37%2C45.15%2033.28%2C39.50%2040.22%2C40.22%2039.50%2C33.28%2045.15%2C37.37%2048.00%2C31.00%2050.85%2C37.37%2056.50%2C33.28%2055.78%2C40.22%2062.72%2C39.50%2058.63%2C45.15%22%2F%3E%3Cpath%20d%3D%22M0%200%2020%200%200%2020Zm96%200H76l20%2020ZM0%2096V76l20%2020Zm96%200V76L76%2096M0%2048l10-10%2010%2010-10%2010Zm96%200L86%2038%2076%2048l10%2010ZM48%200l-10%2010%2010%2010%2010-10Zm0%2096L38%2086l10-10%2010%2010Z%22%2F%3E%3C%2Fg%3E%3C%2Fpattern%3E%3C%2Fdefs%3E%3Crect%20width%3D%22192%22%20height%3D%22192%22%20fill%3D%22url(%23ornament)%22%2F%3E%3C%2Fsvg%3E");background-size:144px 144px;opacity:.13;pointer-events:none}.nav-inner{position:relative}.coast{isolation:isolate}.coast::before{content:"";position:absolute;inset:-10px 0 14px 22%;background-image:url("data:image/svg+xml,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22192%22%20height%3D%22192%22%20viewBox%3D%220%200%20192%20192%22%3E%3Cdefs%3E%3Cpattern%20id%3D%22ornament%22%20width%3D%2296%22%20height%3D%2296%22%20patternUnits%3D%22userSpaceOnUse%22%3E%3Cg%20fill%3D%22none%22%20stroke%3D%22%23b1945c%22%20stroke-width%3D%22.85%22%3E%3Cpolygon%20points%3D%2286.00%2C48.00%2072.15%2C54.47%2080.91%2C67.00%2065.68%2C65.68%2067.00%2C80.91%2054.47%2C72.15%2048.00%2C86.00%2041.53%2C72.15%2029.00%2C80.91%2030.32%2C65.68%2015.09%2C67.00%2023.85%2C54.47%2010.00%2C48.00%2023.85%2C41.53%2015.09%2C29.00%2030.32%2C30.32%2029.00%2C15.09%2041.53%2C23.85%2048.00%2C10.00%2054.47%2C23.85%2067.00%2C15.09%2065.68%2C30.32%2080.91%2C29.00%2072.15%2C41.53%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(0%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(30%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(60%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(90%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(120%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(150%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(180%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(210%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(240%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(270%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(300%2048%2048)%22%2F%3E%3Cpath%20d%3D%22M48%2024%20Q34%209%2048%202%20Q62%209%2048%2024Z%22%20transform%3D%22rotate(330%2048%2048)%22%2F%3E%3Ccircle%20cx%3D%2248%22%20cy%3D%2248%22%20r%3D%2217%22%2F%3E%3Cpolygon%20points%3D%2265.00%2C48.00%2058.63%2C50.85%2062.72%2C56.50%2055.78%2C55.78%2056.50%2C62.72%2050.85%2C58.63%2048.00%2C65.00%2045.15%2C58.63%2039.50%2C62.72%2040.22%2C55.78%2033.28%2C56.50%2037.37%2C50.85%2031.00%2C48.00%2037.37%2C45.15%2033.28%2C39.50%2040.22%2C40.22%2039.50%2C33.28%2045.15%2C37.37%2048.00%2C31.00%2050.85%2C37.37%2056.50%2C33.28%2055.78%2C40.22%2062.72%2C39.50%2058.63%2C45.15%22%2F%3E%3Cpath%20d%3D%22M0%200%2020%200%200%2020Zm96%200H76l20%2020ZM0%2096V76l20%2020Zm96%200V76L76%2096M0%2048l10-10%2010%2010-10%2010Zm96%200L86%2038%2076%2048l10%2010ZM48%200l-10%2010%2010%2010%2010-10Zm0%2096L38%2086l10-10%2010%2010Z%22%2F%3E%3C%2Fg%3E%3C%2Fpattern%3E%3C%2Fdefs%3E%3Crect%20width%3D%22192%22%20height%3D%22192%22%20fill%3D%22url(%23ornament)%22%2F%3E%3C%2Fsvg%3E");background-size:140px 140px;opacity:.13;mask-image:linear-gradient(90deg,transparent,#000 50%,transparent);z-index:-1;pointer-events:none}.coast-label{background:transparent;padding:8px 0 0 14px;isolation:isolate}.coast-label::before{content:"";position:absolute;inset:-8px -6px -5px -14px;z-index:-1;pointer-events:none;background:linear-gradient(to bottom,rgba(244,241,233,0),rgba(244,241,233,.7) 55%,rgba(244,241,233,.95));mask-image:linear-gradient(to right,transparent,#000 12%,#000 90%,transparent);filter:blur(3px)}.footer{border-top:1px solid #d6c8a8;padding-top:14px}@media(max-width:760px){.masthead::before{width:100%;opacity:.10}}
/* Soft corners for the console surfaces. */
.panel{border-radius:18px}
.steps{border-radius:14px;overflow:hidden}
.step:first-child{border-radius:13px 0 0 13px}
.step:last-child{border-radius:0 13px 13px 0}
.bar{border-radius:14px}
.play{border-radius:9px}
.utility{border-radius:8px}
.chip,.signal-status{border-radius:6px}
@media(max-width:760px){.panel{border-radius:14px}.steps,.bar{border-radius:12px}.step:first-child{border-radius:11px 0 0 11px}.step:last-child{border-radius:0 11px 11px 0}}

.takeaway{display:flex;align-items:center;gap:18px;padding:13px 18px;margin-top:18px;border:1px solid #d2d9c9;border-radius:12px;background:#eaf0e5;color:var(--ink)}
.takeaway-label{font:10px var(--mono);letter-spacing:1px;text-transform:uppercase;color:var(--ink-2);white-space:nowrap}
.takeaway p{margin:0;font-size:13px;line-height:1.5}
.speed-control{display:flex;align-items:center;gap:7px;color:var(--ink-2);font-size:10px;white-space:nowrap}
#speed{color:var(--ink);background:var(--panel);border:1px solid #bec9b7;border-radius:8px;padding:7px 8px;font-size:11px;cursor:pointer;max-width:155px}
#speed:focus-visible{outline:3px solid var(--focus);outline-offset:3px}
.bar{flex-wrap:wrap}.bar .key-hint{display:none}
@media(max-width:760px){.takeaway{align-items:flex-start;flex-direction:column;gap:5px;padding:12px 14px}.takeaway p{font-size:12px}}

/* Keep scenario changes from moving the console; retain natural text wrapping. */
.text-slot{display:grid;min-width:0}.text-slot>*{grid-area:1/1;align-self:start}.sizing-copy{visibility:hidden;pointer-events:none;user-select:none}
.panel{box-shadow:0 3px 12px rgba(23,62,50,.035)}
.ptitle{min-height:52px}.ptitle h2{font-size:14px;line-height:1.45;min-height:2.9em}.ptitle>div{min-width:0}
.signal-status{min-width:132px;text-align:center;flex-shrink:0}
.reading-label{min-height:3em;align-items:flex-start;line-height:1.5}.reading-label:before{flex-shrink:0;margin-top:4px}
.reading-value{font-variant-numeric:tabular-nums}
.row .why{min-height:3em;line-height:1.5}.row .n{min-height:3em;line-height:1.5}.chip{min-width:68px;justify-content:center}
.takeaway p{min-height:3em}.clock{font-variant-numeric:tabular-nums;white-space:nowrap}
#chart{overflow:visible}.chart-foot{line-height:1.5}
.step:focus-visible{outline-offset:-4px}
@media(min-width:1200px){.reading-label{min-height:1.5em}.row .n{min-height:1.5em}.takeaway p{min-height:1.5em}}
@media(max-width:760px){.ptitle{align-items:flex-start;min-height:62px}.signal-status{min-width:118px;font-size:9px}.row .n{min-height:1.5em}.takeaway p{min-height:3em}}

#chart text{font-variant-numeric:tabular-nums}.reading-value{white-space:nowrap}
/* Small Arabic accents, with natural Arabic shaping and no letter spacing. */
.arabic{font-family:"Segoe UI",Tahoma,Arial,sans-serif;letter-spacing:normal;text-transform:none;font-weight:400;unicode-bidi:isolate;line-height:1.6}
.header-arabic{display:inline-block;font-size:16px;color:#d6c194;margin-inline-end:15px;padding-inline-end:15px;border-inline-end:1px solid rgba(214,193,148,.35);vertical-align:middle}
.footer-brand{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.footer-arabic{display:inline-block;font-size:14px;color:var(--ink-2);margin-left:9px;padding-left:14px;border-left:1px solid #d6c8a8}
@media(max-width:760px){.header-arabic{display:block;border:0;padding:0;margin:0 0 3px;font-size:14px}.footer-arabic{font-size:13px}.nav-context{max-width:60%}}
/* Give the Arabic header its own line and explicit physical spacing. */
.nav-context{min-width:0;max-width:70%;line-height:1.5;overflow-wrap:break-word}
.header-arabic{display:block;width:fit-content;margin:0 0 5px auto;padding:0;border:0;font-size:14px;line-height:1.7;color:#d6c194;letter-spacing:0}
@media(max-width:760px){.nav-context{max-width:60%}.header-arabic{font-size:13px;margin-bottom:3px}}
/* Refined wordmark and water-verification emblem. */
.brand{gap:14px;flex-shrink:0;align-items:center}
.brand .brand-mark{width:54px;height:54px;flex-shrink:0;filter:drop-shadow(0 3px 7px rgba(0,0,0,.12))}
.brand-type{display:flex;flex-direction:column;gap:5px}
.brand .wordmark{display:block;font-family:var(--sans);font-size:30px;line-height:1;font-weight:500;letter-spacing:-1.1px;color:#fff9eb}
.wordmark-flow{color:#dcc38e}
.brand small{font-size:10px;font-weight:500;line-height:1.4;letter-spacing:1.8px;color:#d0dcd0}
@media(max-width:760px){.brand{gap:10px}.brand .brand-mark{width:42px;height:42px}.brand .wordmark{font-size:25px;letter-spacing:-.8px}.brand small{font-size:8px;letter-spacing:1.3px}.brand-type{gap:4px}}
.brand{min-height:54px}.brand .wordmark{font-size:32px;letter-spacing:-1px}@media(max-width:760px){.brand{min-height:42px}.brand .wordmark{font-size:27px}}
.flow-emblem{width:50px;height:50px;flex:0 0 50px}.brand{gap:13px}@media(max-width:760px){.flow-emblem{width:40px;height:40px;flex-basis:40px}.brand{gap:10px}}

/* Optional compact, fitted recording layout. */
.nav-actions{display:flex;align-items:center;justify-content:flex-end;gap:24px;min-width:0}
.nav-actions .nav-context{max-width:none}
.recording-toggle{flex-shrink:0;border:1px solid #8f9d79;border-radius:8px;background:#17483f;color:#f2e4c3;padding:8px 11px;font-size:11px;line-height:1.4;white-space:nowrap}
.recording-toggle:hover{background:#285d4e}
.recording-toggle[aria-pressed=true]{background:#dcc38e;color:#153e3a;border-color:#dcc38e}
body.recording #demoShell{width:var(--record-width,1280px);max-width:none;zoom:var(--record-scale,1);margin:0 auto}
body.recording .nav-inner{padding:10px 26px;max-width:none}
body.recording .brand{min-height:42px}
body.recording .flow-emblem{width:40px;height:40px;flex-basis:40px}
body.recording .brand .wordmark{font-size:28px}
body.recording .header-arabic{margin-bottom:1px;font-size:12px}
body.recording .wrap{padding:0 26px 10px;max-width:none}
body.recording .hero{grid-template-columns:1.8fr 1fr;min-height:0;padding:14px 0;gap:28px}
body.recording .eyebrow{font-size:9px;margin-bottom:7px}
body.recording h1{font-size:33px;line-height:1.1;margin-bottom:7px}
body.recording .lede{font-size:12px;line-height:1.5;max-width:none}
body.recording .coast{display:flex;min-height:100px}
body.recording .coast svg{max-height:115px}
body.recording .coast-label{font-size:8px;letter-spacing:1px}
body.recording .steps{margin-bottom:12px}
body.recording .step{padding:10px 16px;gap:12px}
body.recording .step strong{font-size:12px}
body.recording .step small{display:block;font-size:9px}
body.recording .stage{grid-template-columns:minmax(0,1.8fr) minmax(0,1fr);gap:14px}
body.recording .panel{padding:14px 16px 10px}
body.recording .ptitle{min-height:38px;margin-bottom:8px;align-items:center}
body.recording .ptitle h2{min-height:1.45em;font-size:13px}
body.recording .section-id{margin-bottom:4px;font-size:9px}
body.recording .readings{padding:8px 0;margin-bottom:8px}
body.recording .reading-label{min-height:1.5em;font-size:9px}
body.recording .reading-value{font-size:26px;line-height:1.3}
body.recording #chart{height:286px;width:100%}
body.recording .chart-foot{padding-top:6px;font-size:9px}
body.recording .row{padding:9px 0;gap:3px 10px}
body.recording .row .n{font-size:14px;min-height:1.5em}
body.recording .row .q{font-size:10px}
body.recording .row .why{min-height:3em;font-size:11px}
body.recording .row svg{height:29px}
body.recording .detector-note{padding-top:6px;font-size:9px}
body.recording .takeaway{margin-top:10px;padding:9px 14px;gap:16px;flex-direction:row;align-items:center}
body.recording .takeaway p{font-size:12px;min-height:1.5em}
body.recording .bar{margin-top:10px;padding:9px 14px;gap:12px;flex-wrap:nowrap}
body.recording .clock{min-width:165px;font-size:11px}
body.recording #scrub{min-width:80px}
body.recording .play{padding:9px 14px}
body.recording .footer{margin-top:10px;padding-top:8px;flex-direction:row;gap:18px}
body.recording .footer-arabic{font-size:12px}
@media(max-width:900px){.nav-actions{gap:10px}.recording-toggle{font-size:10px;padding:7px}.nav-actions .nav-context{font-size:9px}}
@media(max-width:600px){.nav-actions{flex-direction:column;align-items:flex-end;gap:6px}.nav-actions .nav-context{display:none}}

/* Persistent, high-contrast detection state without flashing or layout shifts. */
.detection-banner{display:flex;align-items:center;gap:10px;min-height:70px;padding:10px 12px;margin-bottom:10px;border:1px solid #d7dfce;border-radius:10px;background:#edf2e7;color:#3e6252}
.detection-banner strong{display:block;font-size:15px;line-height:1.35}
.detection-banner span:not(.detection-icon){display:block;font-size:11px;line-height:1.45;margin-top:3px}
.detection-icon{display:grid;place-items:center;flex:0 0 26px;width:26px;height:26px;font:600 18px var(--sans);border:1px solid currentColor;border-radius:50%}
.detection-banner.detected{background:#a63528;border-color:#a63528;color:#fff9ef}
.detection-banner.undetected{background:#f5ead4;border-color:#d8bc80;color:#775316}
.row{padding-inline:10px;margin-inline:-10px;border-radius:8px}
.row.alarm{background:#f9eae2;box-shadow:inset 3px 0 #b74332}
.row.alarm .n{color:#9d3024}.row.alarm .why{color:#713c32}
.chip{font-size:10px;min-width:78px;padding:7px 9px;font-weight:600}
body.recording .row{padding-inline:10px}
body.recording .detection-banner{min-height:60px;margin-bottom:6px;padding:8px 10px}
body.recording .detection-banner strong{font-size:14px}
.detection-icon{display:flex;align-items:center;justify-content:center;flex:0 0 32px;width:32px;height:32px;border:0;border-radius:8px;background:rgba(62,98,82,.09);line-height:1}
.detection-icon svg{display:block;width:22px;height:22px;flex:none;overflow:visible}
.detection-banner.detected .detection-icon{background:rgba(255,255,255,.14)}
.detection-banner.undetected .detection-icon{background:rgba(119,83,22,.09)}

/* Let the chart use the space provided by the full-width recording panels. */
body.recording .stage>section:first-child{display:grid;grid-template-rows:auto auto auto minmax(286px,1fr) auto}
body.recording .stage>section:first-child #chart{height:100%;min-height:286px;width:100%;min-width:0;align-self:stretch}
body.recording .row svg{width:100%}

/* Projection readability: wrapping is allowed; chart geometry remains responsive. */
.lede{font-size:16px;line-height:1.55}
.ptitle h2{font-size:16px}.section-id{font-size:11px}
.reading-label{font-size:13px;letter-spacing:.6px;line-height:1.5;min-height:3em}
.reading-value small{font-size:15px}.reading-value{font-size:31px}
.legend{display:flex;flex-wrap:wrap;gap:8px 20px;color:var(--ink-2);font-size:14px;line-height:1.5;margin-bottom:8px}
.legend span{display:inline-flex;align-items:center;gap:8px}.legend s{display:inline-block;width:24px;height:3px;border-radius:2px;flex-shrink:0}
.row .q{font-size:13px;line-height:1.5}.row .n{font-size:17px;line-height:1.4;min-height:1.4em}.row .why{font-size:14px;line-height:1.5;min-height:3em}
.row{grid-template-columns:minmax(0,1fr) auto}
.chip{font-size:12px;min-width:83px;line-height:1.2}
.row.quiet .chip{background:#e9f0e5;color:#355d45;border-color:#bdcdb6}
.row.off .chip{background:#ecece7;color:#5c625b;border:1px dashed #8b9288}
.row.off .chip::before{display:none}.row.off svg{visibility:hidden}
.scenario-link{grid-column:1/-1;justify-self:start;text-align:left;border:0;background:transparent;color:#236857;font-size:13px;line-height:1.5;padding:0;text-decoration:underline;text-underline-offset:3px}
.row:not(.off) .scenario-link{display:none}
.detection-banner{min-height:84px}.detection-banner strong{font-size:17px}.detection-banner span:not(.detection-icon){font-size:14px;line-height:1.45}
.detector-note,.chart-foot,.fine{font-size:13px;line-height:1.5}
.takeaway p{font-size:15px;line-height:1.5;min-height:3em}.takeaway-label{font-size:11px}
.signal-status{font-size:11px}.clock{font-size:13px}.clock em,.speed-control,.utility{font-size:12px}#speed{font-size:12px;max-width:none}.footer{font-size:12px}
body.recording .lede{font-size:14px;line-height:1.5}
body.recording .ptitle h2{font-size:15px;min-height:2.9em}body.recording .section-id{font-size:10px}
body.recording .reading-label{font-size:12px;min-height:3em}body.recording .reading-value{font-size:29px}body.recording .reading-value small{font-size:14px}
body.recording .legend{font-size:13px}
body.recording .row .q{font-size:12px}body.recording .row .n{font-size:16px}body.recording .row .why{font-size:13px;line-height:1.45;min-height:2.9em}
body.recording .detection-banner{min-height:78px}body.recording .detection-banner strong{font-size:16px}body.recording .detection-banner span:not(.detection-icon){font-size:13px}
body.recording .detector-note,body.recording .chart-foot{font-size:12px}
body.recording .takeaway p{font-size:14px;min-height:3em}body.recording .bar{flex-wrap:wrap}body.recording .clock{font-size:12px}
@media(max-width:1000px){.stage{grid-template-columns:minmax(0,1.45fr) minmax(0,1fr)}.ptitle{flex-wrap:wrap}.takeaway{align-items:flex-start}.chart-foot{flex-wrap:wrap}}
@media(max-width:760px){.stage{grid-template-columns:1fr}.lede{font-size:15px}.reading-label{font-size:12px}.legend{font-size:13px}.takeaway{flex-direction:column;gap:6px}.takeaway p{min-height:0}.row .why{min-height:0}.footer-brand{white-space:normal}.nav-inner{gap:12px}}

/* Recording layout v2: readable at native scale, with a dedicated chart row. */
body.recording #demoShell{width:100%;height:100dvh;min-height:700px;max-width:none;zoom:1;margin:0;display:grid;grid-template-rows:auto minmax(0,1fr)}
body.recording .nav-inner{padding:8px 22px;min-height:64px;max-width:none}
body.recording .brand{min-height:38px}body.recording .flow-emblem{width:36px;height:36px;flex-basis:36px}body.recording .brand .wordmark{font-size:27px}
body.recording .nav-context{font-size:10px}body.recording .nav-context span{font-size:10px}body.recording .header-arabic{font-size:12px;line-height:1.3}
body.recording .wrap{width:100%;max-width:none;min-height:0;padding:10px 22px 14px;display:grid;grid-template-rows:auto auto minmax(0,1fr) auto auto;gap:10px}
body.recording .hero{padding:0;min-height:0;grid-template-columns:minmax(0,1.9fr) minmax(240px,1fr);gap:24px}
body.recording .hero .eyebrow{display:none}body.recording .sizing-copy{display:none}
body.recording .title-slot{min-height:38px}body.recording h1{font-size:30px;line-height:1.15;letter-spacing:-.7px;margin:0 0 4px}
body.recording .lede-slot{min-height:42px}body.recording .lede{font-size:14px;line-height:1.5}
body.recording .coast{min-height:0;max-height:82px}body.recording .coast svg{max-height:82px}body.recording .coast-label{display:none}
body.recording .steps{margin:0;min-height:48px}body.recording .step{padding:8px 14px;gap:10px}body.recording .step strong{font-size:13px}body.recording .step small{display:none}body.recording .step i{font-size:21px}
body.recording .stage{min-height:0;height:100%;grid-template-columns:minmax(0,1.65fr) minmax(350px,1fr);gap:14px;align-items:stretch}
body.recording .panel{min-height:0;padding:12px 15px;border-radius:14px}
body.recording .stage>section:first-child{grid-template-rows:auto auto auto minmax(0,1fr) auto}
body.recording .stage>section:first-child #chart{width:100%;height:100%;min-height:0;min-width:0}
body.recording .stage>section:last-child{display:grid;grid-template-rows:auto auto minmax(0,1fr) auto;gap:7px}
body.recording .ptitle{min-height:0;margin-bottom:5px;flex-wrap:nowrap;gap:8px}body.recording .ptitle h2{font-size:15px;line-height:1.35;min-height:0}
body.recording .section-id{display:none}body.recording .signal-status{font-size:10px;min-width:122px;padding:5px 7px}
body.recording .readings{padding:6px 0;margin-bottom:6px}body.recording .reading-label{font-size:12px;min-height:1.5em;letter-spacing:.4px}body.recording .reading-value{font-size:28px;line-height:1.3}body.recording .reading-value small{font-size:14px}
body.recording .legend{font-size:13px;margin:0 0 2px;gap:6px 16px;line-height:1.4}
body.recording .chart-foot{font-size:11px;padding-top:5px;line-height:1.3}
body.recording .detection-banner{min-height:66px;margin:0;padding:8px 10px;gap:9px}body.recording .detection-banner strong{font-size:16px;line-height:1.25}body.recording .detection-banner span:not(.detection-icon){font-size:12px;line-height:1.35}
body.recording .det{min-height:0;display:grid;grid-template-rows:repeat(3,minmax(0,1fr))}
body.recording .row{min-height:0;padding:7px 9px;margin:0 -7px;gap:3px 8px;align-content:center}
body.recording .row .q{display:none}body.recording .row .n{font-size:15px;line-height:1.35;min-height:0}
body.recording .row .why{font-size:13px;line-height:1.35;min-height:0}body.recording .row svg{height:24px;min-height:0}
body.recording .row.off svg{display:none}body.recording .chip{font-size:11px;padding:5px 7px;min-width:72px}
body.recording .scenario-link{font-size:12px;line-height:1.35}
body.recording .detector-note{font-size:0;line-height:1;margin:0;padding-top:5px}body.recording .detector-note strong,body.recording .detector-note br{display:none}body.recording .detector-note:after{content:'Quiet = no alarm · Off = not evaluated · Dashed line = threshold';font-size:10px;line-height:1.4}
body.recording .takeaway{margin:0;padding:8px 12px;gap:12px;min-height:48px;align-items:center}body.recording .takeaway-label{font-size:10px;flex-shrink:0}body.recording .takeaway p{font-size:14px;line-height:1.4;min-height:0}
body.recording .bar{margin:0;padding:8px 12px;gap:12px;flex-wrap:nowrap;min-height:48px}body.recording .clock{min-width:165px;font-size:12px}body.recording .clock em{font-size:11px}body.recording #scrub{min-width:80px}body.recording .play{padding:9px 14px}body.recording .utility{font-size:11px;padding:7px 9px}body.recording .speed-control{font-size:11px}body.recording #speed{font-size:11px;padding:6px}
body.recording .footer{display:none}
@media(min-width:1600px) and (min-height:900px){body.recording .wrap{padding:14px 32px 18px;gap:14px}body.recording .nav-inner{padding:12px 32px}body.recording h1{font-size:36px}body.recording .title-slot{min-height:46px}body.recording .lede{font-size:17px}body.recording .lede-slot{min-height:51px}body.recording .coast,body.recording .coast svg{max-height:100px}body.recording .step strong{font-size:16px}body.recording .steps{min-height:56px}body.recording .panel{padding:18px 20px}body.recording .ptitle h2{font-size:18px}body.recording .reading-label{font-size:15px}body.recording .reading-value{font-size:34px}body.recording .legend{font-size:15px}body.recording .row .n{font-size:19px}body.recording .row .why{font-size:16px}body.recording .row svg{height:36px}body.recording .chip{font-size:13px}body.recording .detection-banner strong{font-size:20px}body.recording .detection-banner span:not(.detection-icon){font-size:15px}body.recording .detection-banner{min-height:82px}body.recording .takeaway p{font-size:17px}body.recording .scenario-link{font-size:15px}body.recording .bar{min-height:58px}}
@media(max-height:780px){body.recording .nav-inner{min-height:55px;padding-block:6px}body.recording .wrap{gap:7px;padding-block:7px}body.recording .title-slot{min-height:33px}body.recording h1{font-size:27px}body.recording .coast,body.recording .coast svg{max-height:74px}body.recording .steps{min-height:42px}body.recording .row{padding-block:4px}body.recording .row svg{height:18px}body.recording .detector-note{display:none}body.recording .stage>section:last-child{grid-template-rows:auto auto minmax(0,1fr)}body.recording .chart-foot{display:none}body.recording .stage>section:first-child{grid-template-rows:auto auto auto minmax(0,1fr)}body.recording .reading-value{font-size:25px}}
@media(max-width:1050px){body.recording #demoShell{height:auto;min-height:100dvh}body.recording .wrap{grid-template-rows:auto auto auto auto auto}body.recording .stage{height:auto;grid-template-columns:minmax(0,1.4fr) minmax(300px,1fr)}body.recording .stage>section:first-child #chart{min-height:290px}body.recording .det{grid-template-rows:repeat(3,auto)}body.recording .row{padding-block:9px}body.recording .bar{flex-wrap:wrap}body.recording .hero{grid-template-columns:1fr}body.recording .coast{display:none}}
@media(max-width:760px){body.recording .stage{grid-template-columns:1fr}body.recording .takeaway{flex-direction:column;align-items:flex-start;gap:5px}body.recording .nav-actions{gap:8px}body.recording .nav-context{display:none}}

/* Explicit panel sizing: percentage heights must not expand the main content. */
body.recording #demoShell{height:auto;min-height:100dvh;display:block}
body.recording{overflow:hidden}
@media(max-width:1050px){body.recording{overflow:auto}}
body.recording .wrap{display:flex;flex-direction:column;height:auto;min-height:0;gap:10px;padding:10px 24px 14px}
body.recording .hero,body.recording .steps,body.recording .takeaway,body.recording .bar{flex:0 0 auto}
body.recording .stage{flex:0 0 var(--record-panel-height,420px);height:var(--record-panel-height,420px);min-height:0;max-height:480px}
body.recording .stage>.panel{height:100%;max-height:100%;min-height:0;padding:12px 16px}
body.recording .stage>section:first-child{grid-template-rows:auto auto auto minmax(0,1fr) auto}
body.recording .stage>section:first-child #chart{height:100%;min-height:0;max-height:100%;width:100%}
body.recording .ptitle h2{font-size:16px}
body.recording .reading-label{font-size:13px}body.recording .reading-value{font-size:29px}
body.recording .legend{font-size:13px}
body.recording .detection-banner{min-height:62px;padding:7px 10px}body.recording .detection-banner strong{font-size:16px}body.recording .detection-banner span:not(.detection-icon){font-size:12px}
body.recording .row{padding:5px 9px;align-content:center}body.recording .row .n{font-size:16px}body.recording .row .why{font-size:13px}body.recording .row svg{height:20px}
body.recording .chip{font-size:11px}body.recording .scenario-link{font-size:12px}
body.recording .takeaway p{font-size:14px}body.recording .bar{min-height:48px}
@media(max-height:780px){body.recording .wrap{gap:7px;padding-block:7px}body.recording .stage>section:first-child{grid-template-rows:auto auto auto minmax(0,1fr)}body.recording .row svg{height:16px}body.recording .row{padding-block:3px}}
@media(max-width:1050px){body.recording .stage{flex:auto;height:auto;max-height:none}body.recording .stage>.panel{height:auto;max-height:none}body.recording .stage>section:first-child #chart{min-height:290px;height:290px;max-height:none}}
/* Keep wide recording displays balanced without shrinking the typography. */
body.recording .wrap{width:100%;max-width:none;margin-inline:auto;padding-inline:24px}
body.recording .nav-inner{width:100%;max-width:none!important;margin:0!important;padding-inline:24px!important}
.coast-label{left:0;right:auto;padding-left:0}
@media(min-width:1051px){.coast{justify-self:end;width:494px}.coast svg{width:494px}body.recording .coast{width:475px}body.recording .coast svg{width:475px}body.recording .recording-toggle{display:none}}
/* The page scales to the window, so the top section keeps its full design in recording view. */
@media(min-width:1051px){
body.recording .hero{padding:12px 0 8px;min-height:0;grid-template-columns:minmax(0,1.6fr) minmax(300px,1fr);gap:28px;align-items:center}
body.recording .hero .eyebrow{display:flex;font-size:12px;margin-bottom:9px}
body.recording .title-slot,body.recording .lede-slot{min-height:0}
body.recording h1{font-size:50px;line-height:1.08;letter-spacing:-1.1px;margin:0 0 9px}
body.recording .lede{font-size:18px;line-height:1.5;max-width:880px}
body.recording .coast{display:flex;flex-direction:column;min-height:0;max-height:210px}
body.recording .coast svg{max-height:178px}
body.recording .coast-label{display:block;font-size:10px;letter-spacing:1.5px}
body.recording .steps{min-height:0}
body.recording .step{padding:14px 18px;gap:13px}
body.recording .step strong{font-size:17px}
body.recording .step small{display:block;font-size:13px}
}
body.recording .nav-inner{width:100%;max-width:1560px;margin-inline:auto;padding-inline:24px}
body.recording .stage{grid-template-columns:minmax(0,1.5fr) minmax(350px,1fr)}
@media(max-width:1050px){body.recording .stage{grid-template-columns:minmax(0,1.4fr) minmax(300px,1fr)}}
@media(max-width:760px){body.recording .stage{grid-template-columns:1fr}body.recording .wrap,body.recording .nav-inner{padding-inline:16px}}


/* Plain-language presentation: stronger hierarchy and less reserved whitespace. */
.step strong{font-size:17px;line-height:1.3}.step small{font-size:13px;line-height:1.4;margin-top:5px}.step{padding:15px 18px;align-items:center}.step i{font-size:24px}
body.recording .step strong{font-size:16px}.step>span{min-width:0}
.reading-value{font-weight:600;letter-spacing:-.6px}.reading-value small{font-weight:400;margin-left:8px}
.signal-status{font-family:var(--sans);font-weight:600;letter-spacing:.35px;font-size:12px;line-height:1.3}
body.recording .signal-status{font-family:var(--sans);font-weight:600;font-size:11px}
body:not(.recording) .stage{align-items:start}
body:not(.recording) .ptitle{min-height:0;margin-bottom:12px}body:not(.recording) .ptitle h2{min-height:0}
body:not(.recording) .reading-label{min-height:1.5em}
body:not(.recording) .row .why{min-height:1.5em}body:not(.recording) .row.off svg{display:none}
body:not(.recording) .row{padding-block:13px;align-content:start}
.takeaway,body.recording .takeaway{display:grid;grid-template-columns:145px minmax(0,1fr);align-items:center;gap:14px;min-height:0;padding:13px 16px;border-left:3px solid #9b8556}
.takeaway-label,body.recording .takeaway-label{font-family:var(--sans);font-size:12px;font-weight:600;letter-spacing:.5px;line-height:1.4}
.takeaway p,body.recording .takeaway p{min-height:0;margin:0;font-size:17px;font-weight:500;line-height:1.45}
body.recording .takeaway{padding:10px 14px}body.recording .takeaway p{font-size:15px}
@media(max-width:1000px){.step strong{font-size:15px}.step small{font-size:12px}.step{padding:13px 12px;gap:9px}}
@media(max-width:760px){.step strong{font-size:13px}.step small{display:none}.step i{font-size:19px}.takeaway,body.recording .takeaway{grid-template-columns:1fr;gap:5px}.takeaway p{font-size:16px}}
</style></head>
<body>
<div id="demoShell"><header class="masthead"><div class="nav-inner"><div class="brand"><svg class="flow-emblem" viewBox="0 0 56 56" aria-hidden="true"><defs><clipPath id="waterMarkClip"><circle cx="28" cy="28" r="23"/></clipPath></defs><circle cx="28" cy="28" r="23" fill="#17483f" stroke="#d8bf87" stroke-width="1.4"/><g clip-path="url(#waterMarkClip)"><path d="M2 31c9-12 17-14 26-4s17 11 26-1v29H2Z" fill="#d8bf87"/><path d="M2 37c9-12 17-14 26-4s17 11 26-1" fill="none" stroke="#17483f" stroke-width="3"/><path d="M3 44c9-10 17-12 25-3s17 9 25-1" fill="none" stroke="#17483f" stroke-width="2.5"/></g></svg><b class="wordmark">Trust<span class="wordmark-flow">Flow</span></b></div><div class="nav-actions"><button type="button" id="recordingView" class="recording-toggle" aria-pressed="false" title="Fit the full demo into this window">⛶ Recording view</button><div class="nav-context"><bdi class="arabic header-arabic" lang="ar" dir="rtl">أمن المياه</bdi>Water security demonstrator<span>KU · UNESCO Water Hackathon / Simulation</span></div></div></div></header>
<main class="wrap">
<section class="hero"><div><div class="eyebrow">Resilient water infrastructure <span id="actCounter">/ 01</span></div><div class="text-slot title-slot"><h1 id="title"></h1><h1 class="sizing-copy" aria-hidden="true">The screen hides salty water.</h1><h1 class="sizing-copy" aria-hidden="true">The attacker replays a normal day.</h1><h1 class="sizing-copy" aria-hidden="true">The power meter tells the truth.</h1></div><div class="text-slot lede-slot"><p class="lede" id="line"></p><p class="lede sizing-copy" aria-hidden="true">The attacker makes the salt reading look 20% lower. The screen shows less salt than the water really contains. Watch which checks notice.</p><p class="lede sizing-copy" aria-hidden="true">The pump works harder, but the screen replays readings from a normal day. Can the checks tell something is wrong?</p><p class="lede sizing-copy" aria-hidden="true">The attacker cannot change this power meter in the simulation. It measures what the pump really uses. Do the screen readings explain it?</p></div></div>
<div class="coast" aria-hidden="true"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 240" preserveAspectRatio="xMinYMid meet" fill="none"><defs><linearGradient id="sunGold" x1="421" y1="23" x2="421" y2="164" gradientUnits="userSpaceOnUse"><stop stop-color="#c7a563"/><stop offset="1" stop-color="#e6d8b8" stop-opacity=".25"/></linearGradient><linearGradient id="seaTeal" x1="330" y1="173" x2="330" y2="240" gradientUnits="userSpaceOnUse"><stop stop-color="#78a596"/><stop offset="1" stop-color="#d1ded0" stop-opacity=".2"/></linearGradient><linearGradient id="domeIvory" x1="400" y1="92" x2="480" y2="153" gradientUnits="userSpaceOnUse"><stop stop-color="#fffcf2"/><stop offset="1" stop-color="#d6cdb3"/></linearGradient><pattern id="mashrabiya" width="96" height="96" patternUnits="userSpaceOnUse"><g fill="none" stroke="#b1945c" stroke-width=".85"><polygon points="86.00,48.00 72.15,54.47 80.91,67.00 65.68,65.68 67.00,80.91 54.47,72.15 48.00,86.00 41.53,72.15 29.00,80.91 30.32,65.68 15.09,67.00 23.85,54.47 10.00,48.00 23.85,41.53 15.09,29.00 30.32,30.32 29.00,15.09 41.53,23.85 48.00,10.00 54.47,23.85 67.00,15.09 65.68,30.32 80.91,29.00 72.15,41.53"/><path d="M48 24 Q34 9 48 2 Q62 9 48 24Z" transform="rotate(0 48 48)"/><path d="M48 24 Q34 9 48 2 Q62 9 48 24Z" transform="rotate(30 48 48)"/><path d="M48 24 Q34 9 48 2 Q62 9 48 24Z" transform="rotate(60 48 48)"/><path d="M48 24 Q34 9 48 2 Q62 9 48 24Z" transform="rotate(90 48 48)"/><path d="M48 24 Q34 9 48 2 Q62 9 48 24Z" transform="rotate(120 48 48)"/><path d="M48 24 Q34 9 48 2 Q62 9 48 24Z" transform="rotate(150 48 48)"/><path d="M48 24 Q34 9 48 2 Q62 9 48 24Z" transform="rotate(180 48 48)"/><path d="M48 24 Q34 9 48 2 Q62 9 48 24Z" transform="rotate(210 48 48)"/><path d="M48 24 Q34 9 48 2 Q62 9 48 24Z" transform="rotate(240 48 48)"/><path d="M48 24 Q34 9 48 2 Q62 9 48 24Z" transform="rotate(270 48 48)"/><path d="M48 24 Q34 9 48 2 Q62 9 48 24Z" transform="rotate(300 48 48)"/><path d="M48 24 Q34 9 48 2 Q62 9 48 24Z" transform="rotate(330 48 48)"/><circle cx="48" cy="48" r="17"/><polygon points="65.00,48.00 58.63,50.85 62.72,56.50 55.78,55.78 56.50,62.72 50.85,58.63 48.00,65.00 45.15,58.63 39.50,62.72 40.22,55.78 33.28,56.50 37.37,50.85 31.00,48.00 37.37,45.15 33.28,39.50 40.22,40.22 39.50,33.28 45.15,37.37 48.00,31.00 50.85,37.37 56.50,33.28 55.78,40.22 62.72,39.50 58.63,45.15"/><path d="M0 0 20 0 0 20Zm96 0H76l20 20ZM0 96V76l20 20Zm96 0V76L76 96M0 48l10-10 10 10-10 10Zm96 0L86 38 76 48l10 10ZM48 0l-10 10 10 10 10-10Zm0 96L38 86l10-10 10 10Z"/></g></pattern><clipPath id="coastClip"><rect width="640" height="240" rx="3"/></clipPath><g id="arch"><path d="M0 32V12Q0 4 7 0q7 4 7 12v20" fill="#346b5c"/><path d="M3 32V13q0-7 4-10" stroke="#91a593" stroke-width=".8"/></g><g id="palm"><path d="m0 0-3 53h6L2 0" fill="#8e8668"/><path d="M1 2Q-13-17-32-10-14-12 1 5M1 2Q-5-24-17-28-7-25 3 4M1 2Q13-24 28-20 13-14 2 5M1 2Q24-10 38 2 17-3 2 6M1 2Q-24-5-35 10-17 2 2 6M1 2Q5-18 2-33-4-15 1 5" fill="#356b59"/><path d="m-2 15 4 2m-5 8 4 2m-4 9 4 2" stroke="#d1bd91" stroke-width="1.3"/></g></defs><g clip-path="url(#coastClip)"><circle cx="427" cy="95" r="79" stroke="#ceb988" stroke-width=".7" opacity=".45"/><circle cx="427" cy="95" r="66" fill="url(#sunGold)"/><path d="M325 153V82a102 102 0 0 1 204 0v71" fill="url(#mashrabiya)" opacity=".43"/>
<g stroke="#9cae99" stroke-width=".8" fill="#dde1d1"><path d="M286 160V88l11-9 14 9v72M315 160V66l12-12 13 12v94M343 160V43l6-6 7 6v117M368 160V76l12-8 11 8v84"/><path d="M346 37V24m-27 50h17m-17 8h17m-17 8h17m-17 8h17m-17 8h17m-17 8h17m-17 8h17m-46-22h19m-19 8h19m-19 8h19M349 49v107"/></g>
<path d="M8 176Q119 112 212 157T407 153q109-37 226 17v35H8Z" fill="#e5d9ba"/><path d="M3 190Q133 131 262 176t246-8q61-10 132 18v31H3Z" fill="#c9b48b"/><path d="M12 187q132-50 251-10" stroke="#f8f2e4" stroke-width="1.5"/>
<g stroke="#587765" stroke-width="1"><path d="M117 170v-62h125v62" fill="#ede6d2"/><path d="M242 108h16v62h-16" fill="#c5c5ae"/><path d="M113 104h132v8H113z" fill="#faf6e9"/><path d="M145 104V52h47v52" fill="#f4eedc"/><path d="m192 52 16 8v44h-16Z" fill="#c1c4ae"/><path d="M141 47h55v7h-55z" fill="#f9f3e3"/><path d="M151 63h12v27h-12zM173 63h12v27h-12z" fill="#496f5b"/><path d="m155 64 7 25m-10-1 10-23m13 0 9 23m-9 0 9-23" stroke="#9ba88e"/><path d="M150 41v6m12-6v6m12-6v6m13-6v6M146 96h45"/><path d="m198 67 5 3v22l-5-2Z" fill="#6d8770"/><path d="M120 116h119" stroke="#b3aa8c"/></g>
<g><use href="#arch" x="126" y="128"/><use href="#arch" x="150" y="128"/><use href="#arch" x="174" y="128"/><use href="#arch" x="198" y="128"/><use href="#arch" x="222" y="128"/><path d="M112 165h147v6H112z" fill="#e2d8bc" stroke="#74836a" stroke-width=".8"/></g>
<g stroke="#5c7b68" stroke-width=".9"><path d="M376 170v-35h135v35" fill="#faf6e9"/><path d="M382 135c0-25 24-30 34-43 10 13 34 18 34 43Z" fill="url(#domeIvory)"/><path d="M416 92c-7 14-18 27-18 43m18-43c7 14 18 27 18 43" stroke="#bcba9d"/><path d="M416 92V80" stroke="#ad8a4e"/><circle cx="416" cy="78" r="2" fill="#ad8a4e" stroke="none"/><path d="M460 135c0-15 14-19 20-26 6 7 21 11 21 26Z" fill="url(#domeIvory)"/><path d="M480 109v-8" stroke="#ad8a4e"/><path d="M369 169h148v5H369z" fill="#ded7bc"/><path d="M360 169V86h10v83" fill="#f1ead7"/><path d="M357 83h16v5h-16zM360 62h10v21h-10z" fill="#eee8d3"/><path d="m359 62 6-12 6 12Z" fill="#f8f4e5"/><path d="M365 50V39" stroke="#ad8a4e"/></g>
<g><path d="M390 169v-16q6-12 12 0v16m8 0v-16q6-12 12 0v16m8 0v-16q6-12 12 0v16m18 0v-16q6-12 12 0v16m8 0v-16q6-12 12 0v16" fill="#78927d"/><path d="M376 141h135" stroke="#c1b696"/></g>
<use href="#palm" transform="translate(83 121) scale(.95)"/><use href="#palm" transform="translate(281 137) scale(.68)"/><use href="#palm" transform="translate(540 137) scale(.65)"/>
<path d="M0 208q101-39 204-18t203-1q134-26 233 0v51H0Z" fill="url(#seaTeal)"/><path d="M0 208q101-39 204-18t203-1q134-26 233 0" stroke="#427e6a" stroke-width="1.4"/><path d="M12 219q97-28 190-11t205-1q129-22 221-4M65 230q87-14 150-4t173-4m70-3q92-12 161-3" stroke="#f8f5e9" stroke-width="1.3" opacity=".85"/>
<g transform="translate(486 154)"><path d="M-2 43q21 4 45-1l-10 10H9Z" fill="#164b40"/><path d="M22 42V0" stroke="#636950" stroke-width="1.1"/><path d="M20 4Q2 20 2 34h18Z" fill="#fffae8" stroke="#ad9b71" stroke-width=".7"/><path d="m25 15 13 20H25Z" fill="#dfcc9f"/><path d="M2 44h36" stroke="#bf9c5e"/></g><g stroke="#6b8570" stroke-width="1.2" stroke-linecap="round"><path d="M235 46q5-4 10 0 5-4 10 0m-19 13q3-3 7 0 4-3 7 0"/></g></g></svg><span class="coast-label">Emirati heritage. A resilient water future.</span></div></section>
<nav class="steps" role="tablist" aria-label="Attack scenarios"><button class="step" role="tab" id="tab1" data-act="1" aria-controls="scenario"><i>01</i><span><strong>Fake a reading</strong><small>Hide the salt in the water</small></span></button><button class="step" role="tab" id="tab2" data-act="2" aria-controls="scenario"><i>02</i><span><strong>Replay a normal day</strong><small>Make a harder-working pump look normal</small></span></button><button class="step" role="tab" id="tab3" data-act="3" aria-controls="scenario"><i>03</i><span><strong>Check the power meter</strong><small>Check the power against the screen</small></span></button></nav>
<div class="stage" id="scenario" role="tabpanel" aria-labelledby="tab1">
<section class="panel" aria-labelledby="chartTitle"><div class="ptitle"><div><div class="section-id">01 / PROCESS TELEMETRY</div><h2 id="chartTitle"></h2></div><span class="signal-status" id="phase">BASELINE</span></div><div class="readings"><div class="reading"><div class="reading-label" id="labelReal"></div><div class="reading-value" id="valueReal"></div></div><div class="reading"><div class="reading-label" id="labelScreen"></div><div class="reading-value" id="valueScreen"></div></div></div><div class="legend" id="legend"></div><svg id="chart" viewBox="0 0 760 410" role="img" aria-label="Real and reported reading over time"></svg><div class="chart-foot"><span>Simulated seawater reverse osmosis plant</span><span>1 sample / minute</span></div></section>
<section class="panel" aria-label="Detectors"><div class="ptitle"><div><div class="section-id">02 / TRUST VERIFICATION</div><h2>Which checks detect the attack?</h2></div></div><div class="detection-banner" id="detectionBanner" role="status" aria-live="polite" aria-atomic="true"><span class="detection-icon" id="detectionIcon" aria-hidden="true"></span><div><strong id="detectionTitle">Normal baseline</strong><span id="detectionDetail">Waiting for attack onset.</span></div></div><div class="det">
<div class="row" id="R-ai"><div><div class="q">Does it look normal?</div><div class="n">AI detector</div></div><span class="chip" id="C-ai"></span><div class="why" id="W-ai"></div><svg id="G-ai" viewBox="0 0 420 46" preserveAspectRatio="none" role="img" aria-label="AI score relative to alarm threshold"></svg></div>
<div class="row" id="R-phys"><div><div class="q">Does it obey physics?</div><div class="n">Physics checks</div></div><span class="chip" id="C-phys"></span><div class="why" id="W-phys"></div><svg id="G-phys" viewBox="0 0 420 46" preserveAspectRatio="none" role="img" aria-label="Physics score relative to alarm threshold"></svg></div>
<div class="row" id="R-energy"><div><div class="q">Does the power match the work?</div><div class="n">Independent energy check</div></div><span class="chip" id="C-energy"></span><div class="why" id="W-energy"></div><button class="scenario-link" id="energyScenarioLink" type="button">See scenario 3: Check the power meter →</button><svg id="G-energy" viewBox="0 0 420 46" preserveAspectRatio="none" role="img" aria-label="Energy score relative to alarm threshold"></svg></div></div><p class="detector-note"><strong>Quiet:</strong> no current alarm · <strong>Alarm:</strong> threshold exceeded · <strong>Off:</strong> not evaluated.<br>Dashed line = alarm threshold. Simulation scores only.</p></section></div>
<aside class="takeaway" aria-label="Scenario takeaway"><span class="takeaway-label">This simulation</span><p id="takeawayText" aria-live="polite"></p></aside>
<div class="bar"><button class="play" id="play" aria-label="Play"><svg viewBox="0 0 12 12" id="playIcon"><path d="M2 1l9 5-9 5z" fill="currentColor"/></svg><span id="playText">Play</span></button><input type="range" id="scrub" min="0" value="0" aria-label="Simulation time"><span class="clock" id="clock"></span><label class="speed-control" for="speed">Speed<select id="speed" aria-label="Playback speed"><option value="0.25" selected>0.25× · Narrate</option><option value="0.5">0.5× · Walkthrough</option><option value="1">1× · Quick demo</option></select></label><button class="utility" id="jump">Attack onset ↗</button><button class="utility" id="reset">Restart ↺</button><span class="key-hint">SPACE to play · 1–3 scenarios</span></div>
<footer class="footer"><span class="footer-brand">TRUSTFLOW <span style="color:#9b865e">/</span> WATER SECURITY <bdi class="arabic footer-arabic" lang="ar" dir="rtl">مستقبل مائي آمن</bdi></span><details><summary>Simulation only · Method &amp; limitations</summary><p class="fine" id="note"></p></details></footer>
</main></div>
<script>
const D=__DATA__, N=D.t.length, S=D.stats, K0=D.t.indexOf(0);
const ACTS={
 1:{title:"The screen hides salty water.",
    line:"The attacker makes the salt reading look 20% lower. The screen shows less salt than the water really contains. Watch which checks notice.",
    chart:"Salt in the water", unit:"mg/L", dp:1, a:"true", b:"shown", la:"Real water (truth)", lb:"Screen shows (faked)",
    on:{ai:1,phys:1,energy:0},
    why:{ai:"Fooled. The edited readings look normal to it.", phys:"Salt passage no longer matches pressure and flow.", energy:"Not used in this act."}},
 2:{title:"The attacker replays a normal day.",
    line:"The pump works harder, but the screen replays readings from a normal day. Can the checks tell something is wrong?",
    chart:"High-pressure pump", unit:"bar", dp:1, a:"true", b:"shown", la:"Real pressure (truth)", lb:"Screen shows (faked)",
    on:{ai:1,phys:1,energy:0},
    why:{ai:"Sees genuine normal data.", phys:"Replayed data obeys physics.", energy:"Not connected yet."}},
 3:{title:"The power meter tells the truth.",
    line:"The attacker cannot change this power meter in the simulation. It measures what the pump really uses. Do the screen readings explain it?",
    chart:"Pump power", unit:"kW", dp:0, a:"meter", b:"implied", la:"Power meter (truth)", lb:"Readings claim",
    on:{ai:1,phys:1,energy:1},
    why:{ai:"Still sees normal data.", phys:"Still consistent.", energy:"Power doesn't match the work the readings report."}}
};
let act=1,i=0,timer=null,readoutIndex=0,lastReadoutAt=-Infinity;
const $=id=>document.getElementById(id);
let W=760;const H=410,P={l:70,r:212,t:48,b:44};
const X=k=>P.l+(W-P.l-P.r)*k/(N-1);
function series(k){return D["act"+act][k]||D.act2[k]}
function line(a,lo,hi,x0,w,y0,h,upto){let d="",on=false;for(let k=0;k<=upto;k++){const v=a[k];if(v===null){on=false;continue}
 const x=x0+w*k/(N-1),y=y0+h-h*(Math.min(Math.max(v,lo),hi)-lo)/(hi-lo);d+=(on?"L":"M")+x.toFixed(1)+","+y.toFixed(1);on=true}return d}
function nice(lo,hi){const span=hi-lo,step=Math.pow(10,Math.floor(Math.log10(span/4)));const m=[1,2,2.5,5,10].find(f=>span/(step*f)<=5)*step;
 return {lo:Math.floor(lo/m)*m,hi:Math.ceil(hi/m)*m,step:m}}
function fmt(v,dp){return v.toLocaleString("en-US",{minimumFractionDigits:dp,maximumFractionDigits:dp})}
function detectorDescription(k,live,alarm,value){
 if(!live)return 'Normal operation. No attack yet.';
 if(value===null)return 'No score available at this sample.';
 if(k==='ai')return alarm?'The AI sees something unusual in the readings.':'These readings look normal to the AI.';
 if(k==='phys')return alarm?(act===1?'Salt passage is inconsistent with pressure and flow.':'A physics-check score exceeds its threshold.'):(act===1?'Checking salt passage against pressure and flow.':'The replayed readings still obey these physics checks.');
 return alarm?'The pump uses more power than the screen can explain.':'The power gap is not large enough to raise an alarm.';
}
function alarmReached(k){
 if(D.t[i]<0||!ACTS[act].on[k])return false;
 const values=series(k);return !!values&&values.slice(K0,i+1).some(v=>v!==null&&v>1);
}
function updateTakeaway(){
 let message;
 if(D.t[i]<0)message='The plant is running normally. The attack has not started.';
 else if(act===1){
  message=alarmReached('phys')?'The AI missed it. Physics caught it.':'The screen is hiding salt. Physics has not raised an alarm yet.';
 }else if(act===3){
  message=alarmReached('energy')?'The power meter exposed the mismatch.':'The screen looks normal. The power check has not raised an alarm yet.';
 }else{
  message='The replay has fooled both checks.';   // a stray single-row blip is noise, not a detection
 }
 if($('takeawayText').textContent!==message)$('takeawayText').textContent=message;
}
function chartLabel(lab,x){const split=lab.indexOf(' (');return split<0?lab:'<tspan x="'+x+'">'+lab.slice(0,split)+'</tspan><tspan x="'+x+'" dy="19">'+lab.slice(split+1)+'</tspan>';}
function displayAct(){const A=ACTS[act];return {...A,lb:D.t[i]<0?(act===1?'Screen shows':act===2?'Screen shows':A.lb):A.lb};}
function draw(forceReadout=true){
 const chartElement=$('chart');
 const fitted=document.body?.classList.contains('recording');
 W=fitted&&chartElement.clientWidth&&chartElement.clientHeight?Math.max(500,Math.round(chartElement.clientWidth*H/chartElement.clientHeight)):760;
 chartElement.setAttribute('viewBox','0 0 '+W+' '+H);
 const now=performance.now();
 if(forceReadout||now-lastReadoutAt>=250||i===K0||i===N-1){readoutIndex=i;lastReadoutAt=now;}
 const ri=readoutIndex;
 const A=displayAct(),pa=series(A.a),pb=series(A.b);
 const vals=pa.concat(pb).filter(v=>v!==null);let {lo,hi,step}=nice(Math.min(...vals),Math.max(...vals));
 if(hi===lo){hi+=step;lo-=step}
 const ih=H-P.t-P.b,iw=W-P.l-P.r,Y=v=>P.t+ih-ih*(v-lo)/(hi-lo);
 const xa=X(K0);let s=`<rect x="${xa}" y="${P.t}" width="${X(N-1)-xa}" height="${ih}" fill="var(--attack)"/>`+
  `<text x="${xa+8}" y="${P.t-9}" font-size="15" font-weight="600" letter-spacing=".04em" fill="var(--real)">ATTACK START</text>`;
 s+=`<text x="${P.l}" y="${P.t-9}" font-size="14" fill="var(--ink-2)">BASELINE</text>`;
 for(let v=lo;v<=hi+step/2;v+=step){const y=Y(v);s+=`<line x1="${P.l}" x2="${W-P.r}" y1="${y}" y2="${y}" stroke="var(--rule)"/>`+
  `<text x="${P.l-10}" y="${y+4}" text-anchor="end" font-size="16" font-family="var(--mono)" fill="var(--ink-2)">${fmt(v,step<1?1:0)}</text>`}
 for(const m of [-120,0,120,240,360]){const k=D.t.indexOf(m>=D.t[N-1]?D.t[N-1]:m);if(k<0)continue;
  s+=`<text x="${X(k)}" y="${H-10}" text-anchor="middle" font-size="16" font-family="var(--mono)" fill="var(--ink-2)">${m===0?"0 h":(m>0?"+":"−")+Math.abs(m/60)+" h"}</text>`}
 s+=`<line x1="${xa}" x2="${xa}" y1="${P.t}" y2="${P.t+ih}" stroke="var(--real)" stroke-width="1.5" stroke-dasharray="5 5" opacity=".8"/>`;
 if(act===1&&D.t[i]>=0){
  const real=[],reported=[];for(let k=K0;k<=i;k++){if(pa[k]!==null&&pb[k]!==null){real.push(X(k).toFixed(1)+','+Y(pa[k]).toFixed(1));reported.unshift(X(k).toFixed(1)+','+Y(pb[k]).toFixed(1));}}
  if(real.length>1)s+='<polygon class="salinity-gap" points="'+real.concat(reported).join(' ')+'" fill="var(--real)" opacity=".12"/>';
 }
 s+=`<path d="${line(pa,lo,hi,P.l,iw,P.t,ih,i)}" fill="none" stroke="var(--real)" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`;
 s+=`<path d="${line(pb,lo,hi,P.l,iw,P.t,ih,i)}" fill="none" stroke="var(--screen)" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`;
 s+=`<line x1="${X(i)}" x2="${X(i)}" y1="${P.t}" y2="${P.t+ih}" stroke="var(--ink)" stroke-opacity=".28"/>`;
 // Fixed right-hand readouts; no connector lines that could resemble data.
 const ya=pa[i]===null?null:Y(pa[i]),yb=pb[i]===null?null:Y(pb[i]);

 // Explain the live gap without adding horizontal leaders or altering data traces.
 if(D.t[i]>=0&&ya!==null&&yb!==null){
  const bx=X(i)+9,top=Math.min(ya,yb),bottom=Math.max(ya,yb),mid=(top+bottom)/2;
  const explanation=act===3?'Unexplained power':'Hidden by the attacker';
  const words=act===3?['Unexplained','power']:['Hidden by the','attacker'];
  s+=`<g class="gap-annotation" role="img" aria-label="${explanation}"><path class="gap-bracket" d="M${bx-5},${top} H${bx} V${bottom} H${bx-5}" fill="none" stroke="var(--ink-2)" stroke-width="1.4" stroke-linejoin="round"/><text x="${bx-15}" y="${mid-3}" text-anchor="end" font-size="15" font-weight="500" fill="var(--ink)" stroke="var(--panel)" stroke-width="4" stroke-linejoin="round" paint-order="stroke"><tspan x="${bx-15}">${words[0]}</tspan><tspan x="${bx-15}" dy="19">${words[1]}</tspan></text></g>`;
 }
 const la=P.t+ih*.32,lb=P.t+ih*.70;
 const labelX=W-P.r+18;
 for(const [y,ly,c,v,lab] of [[ya,la,"var(--real)",pa[ri],A.la],[yb,lb,"var(--screen)",pb[ri],A.lb]]){if(y===null)continue;
  s+=`<circle cx="${X(i)}" cy="${y}" r="5" fill="${c}" stroke="var(--panel)" stroke-width="2"/>`+
  `<text class="point-value" x="${labelX}" y="${ly-2}" font-size="22" font-weight="600" font-family="var(--mono)" fill="${c}" stroke="var(--panel)" stroke-width="4" stroke-linejoin="round" paint-order="stroke">${fmt(v,A.dp)}</text>`+
  `<text class="point-label" x="${labelX}" y="${ly+19}" font-size="16" fill="var(--ink-2)" stroke="var(--panel)" stroke-width="3" stroke-linejoin="round" paint-order="stroke">${chartLabel(lab,labelX)}</text>`}
 $("chart").innerHTML=s;$("chart").setAttribute("aria-label",A.chart+": "+A.la+" "+fmt(pa[ri],A.dp)+" "+A.unit+", "+A.lb+" "+fmt(pb[ri],A.dp)+" "+A.unit+". "+Math.abs(D.t[ri])+" minutes "+(D.t[ri]<0?"before":"after")+" attack onset.");
 $("chartTitle").innerHTML=`${A.chart} <span style="color:var(--ink-3);font-weight:400">(${A.unit})</span>`;
 $("legend").innerHTML=`<span><s style="background:var(--real)"></s>${A.la}</span><span><s style="background:var(--screen)"></s>${A.lb}</span>`;
 const live=D.t[i]>=0;
 for(const k of ["ai","phys","energy"]){
  const row=$("R-"+k),chip=$("C-"+k),on=A.on[k];row.className="row";
  const g=$("G-"+k),gw=420,gh=46,cap=3,a=series(k);
  const gx=gw*K0/(N-1);let gs=`<rect x="${gx}" y="0" width="${gw-gx}" height="${gh}" fill="var(--attack)" opacity=".7"/>`;
  const ty=gh-gh*1/cap;gs+=`<line x1="0" x2="${gw}" y1="${ty}" y2="${ty}" stroke="var(--alarm)" stroke-dasharray="4 4" stroke-opacity=".7"/>`;
  if(on&&a){gs+=`<path d="${line(a.map(v=>v===null?null:Math.min(v,cap)),0,cap,0,gw,0,gh,i)}" fill="none" stroke="var(--ink-2)" stroke-width="1.8" stroke-linejoin="round"/>`}
  g.innerHTML=gs;
  if(!on){row.classList.add("off");chip.textContent="Off";$("W-"+k).textContent="Not used in this scenario.";continue}
  const v=a[i],alarm=live&&v!==null&&v>1;row.classList.add(alarm?"alarm":"quiet");
  chip.textContent=alarm?"Alarm":"Quiet";$("W-"+k).textContent=detectorDescription(k,live,alarm,v)}
 updateDetectionBanner(live,A);updateTakeaway();
 const m=D.t[i],h=Math.floor(Math.abs(m)/60),mm=String(Math.abs(m)%60).padStart(2,"0");
 $("clock").innerHTML=m<0?`<em>before attack</em> −${h}:${mm}`:`<em>attack</em> +${h}:${mm}`;
 $("scrub").value=i; updateReadouts(ri);
}
const detectionIcons={
 baseline:'<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 20 6v6c0 4-4 7-8 9-4-2-8-5-8-9V6Z"/><path d="m8 12 3 3 5-6"/></svg>',
 detected:'<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 22 20H2Z"/><path d="M12 9v5"/><circle cx="12" cy="17" r=".8" fill="currentColor" stroke="none"/></svg>',
 undetected:'<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 12S6 6 12 6s9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z"/><circle cx="12" cy="12" r="2.8"/><path d="M4 4 20 20" stroke="#f5ead4" stroke-width="4.5"/><path d="M4 4 20 20"/></svg>'
};
function updateDetectionBanner(live,A){
 const names={ai:'AI detector',phys:'Physics checks',energy:'Independent energy check'};
 const flagged=live?Object.keys(names).filter(k=>A.on[k]&&series(k)&&series(k)[i]!==null&&series(k)[i]>1):[];
 const state=!live?'baseline':flagged.length?'detected':'undetected';
 const title=!live?'Normal baseline':flagged.length?'Attack detected':'Attack active · not detected';
 const detail=!live?'Before the selected simulated attack.':flagged.length?(act===1&&flagged.includes('phys')?'Physics: salt passage conflicts with pressure and flow.':flagged.map(k=>names[k]).join(' + ')+' raised the alarm in this scenario.'):'Enabled checks currently remain quiet in this scenario.';
 const banner=$('detectionBanner');
 if(banner.dataset.state!==state){banner.dataset.state=state;banner.className='detection-banner '+state;$('detectionIcon').innerHTML=detectionIcons[state];}
 if($('detectionTitle').textContent!==title)$('detectionTitle').textContent=title;
 if($('detectionDetail').textContent!==detail)$('detectionDetail').textContent=detail;
}
function setAct(a){act=[1,2,3].includes(a)?a:1;a=act;$("actCounter").textContent="/ 0"+act;$("scenario").setAttribute("aria-labelledby","tab"+act);document.querySelectorAll(".step").forEach(b=>b.setAttribute("aria-selected",String(+b.dataset.act===a)));
 $("title").textContent=ACTS[a].title;$("line").textContent=ACTS[a].line;i=0;draw()}
function stop(){if(timer){clearInterval(timer);timer=null}$("playText").textContent="Play";
 $("playIcon").innerHTML='<path d="M2 1l9 5-9 5z" fill="currentColor"/>';$("play").setAttribute("aria-label","Play");draw()}
function play(){if(timer){stop();return}if(i>=N-1){i=0;draw();}$("playText").textContent="Pause";
 $("playIcon").innerHTML='<path d="M2 1h3v10H2zM7 1h3v10H7z" fill="currentColor"/>';$("play").setAttribute("aria-label","Pause");
 timer=setInterval(()=>{i=Math.min(N-1,i+1);draw(false);if(i>=N-1)stop()},20/Number($("speed").value))}
$("scrub").max=N-1;$("scrub").oninput=e=>{const nextIndex=+e.target.value;stop();i=nextIndex;draw()};
$("play").onclick=play;
$("energyScenarioLink").onclick=()=>{stop();setAct(3);$("tab3").focus()};
$("speed").onchange=()=>{if(timer){stop();play()}};
document.querySelectorAll(".step").forEach(b=>b.onclick=()=>{stop();setAct(+b.dataset.act)});
$("chart").addEventListener("pointerdown",e=>{stop();const point=$("chart").createSVGPoint();point.x=e.clientX;point.y=e.clientY;const x=point.matrixTransform($("chart").getScreenCTM().inverse()).x;
 i=Math.max(0,Math.min(N-1,Math.round((x-P.l)/(W-P.l-P.r)*(N-1))));draw()});
document.addEventListener("keydown",e=>{if(["INPUT","BUTTON","SUMMARY","SELECT","TEXTAREA"].includes(e.target.tagName))return;
 if(e.key===" "){e.preventDefault();play()}else if(/^[123]$/.test(e.key)){stop();setAct(+e.key)}});
function updateReadouts(ri=i){const A=displayAct();$("labelReal").textContent=A.la;$("labelScreen").textContent=A.lb;for(const [id,k] of [["valueReal",A.a],["valueScreen",A.b]]){const v=series(k)[ri];$(id).innerHTML=(v===null?"—":fmt(v,A.dp))+"<small>"+A.unit+"</small>";}const live=D.t[i]>=0;$("phase").textContent=live?"ATTACK ACTIVE":"NORMAL BASELINE";$("phase").className="signal-status"+(live?" attack":"");$("scrub").setAttribute("aria-valuetext",Math.abs(D.t[i])+" minutes "+(live?"after":"before")+" attack onset");}
$("jump").onclick=()=>{stop();i=K0;draw()};$("reset").onclick=()=>{stop();i=0;draw()};
document.querySelectorAll('.step').forEach((b,index)=>b.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();const next=(index+(e.key==='ArrowRight'?1:2))%3;stop();setAct(next+1);document.querySelectorAll('.step')[next].focus();}}));
const pct=v=>Math.round(v*100)+"%";
$("note").textContent=`Every line is computed by the TrustFlow simulator (seed 0), not drawn by hand. AI detector: an autoencoder on 17 process readings, quiet on normal data here (${(S.ai_false_alarm_unattacked_window*100).toFixed(1)}% of rows). Act 1: the AI flags ${pct(S.act1.ai_flags_raw_spoof)} of attack rows before concealment and ${pct(S.act1.ai_flags_concealed)} after; physics flags ${pct(S.act1.physics_flags_concealed)}, mainly ${S.act1.main_checks.join(" and ")}. Act 2: AI ${pct(S.act2.ai_flags)}, physics ${pct(S.act2.physics_flags)}. Act 3: energy check ${pct(S.act3.energy_flags)}. Scores are shown against each check's alarm threshold (dashed). The energy check has been tested in simulation only. These results describe this simulation run and these selected attacks; they do not establish real-plant performance or coverage of other attacks.`;
const h=location.hash.slice(1),hp=new URLSearchParams(h);
setAct(+(hp.get("act")||(h.match(/^act([123])$/)||[])[1]||1));
if(hp.get("t")){const v=hp.get("t");i=v==="end"?N-1:Math.max(0,D.t.indexOf(+v));draw()}

// Fitting is layout-only: simulation timing, values, and playback state are preserved.
let recordingFitFrame=0;
const RECORD_W=1520,RECORD_PANEL=470;   // recording view: lay out at one fixed size, then scale to the window
function clearRecordingFit(){const s=$('demoShell').style;['zoom','width','height','minHeight','margin'].forEach(k=>s[k]='');document.querySelector('main.wrap').style.removeProperty('--record-panel-height');}
function fitRecordingView(){
 clearRecordingFit();
 if(!document.body.classList.contains('recording')||window.innerWidth<=1050)return;
 const shell=$('demoShell'),main=document.querySelector('main.wrap');
 main.style.setProperty('--record-panel-height',RECORD_PANEL+'px');
 shell.style.height='auto';shell.style.minHeight='0';
 // Fit the height first, then give the layout the window's own shape: columns widen, nothing is stretched.
 let width=RECORD_W;shell.style.width=width+'px';
 for(let pass=0;pass<3;pass++){
  const scale=window.innerHeight/shell.getBoundingClientRect().height;
  width=Math.max(1240,Math.min(2100,window.innerWidth/scale));shell.style.width=width+'px';
 }
 const scale=Math.min(window.innerWidth/width,window.innerHeight/shell.getBoundingClientRect().height);
 shell.style.zoom=scale;shell.style.margin='0 auto';
}
function queueRecordingFit(){if(recordingFitFrame)return;recordingFitFrame=requestAnimationFrame(()=>{recordingFitFrame=0;fitRecordingView();draw(false)});}
function setRecordingView(enabled){
 document.body.classList.toggle('recording',enabled);
 $('recordingView').setAttribute('aria-pressed',String(enabled));
 $('recordingView').textContent=enabled?'⛶ Exit recording view':'⛶ Recording view';
 if(enabled){window.scrollTo(0,0);fitRecordingView();queueRecordingFit();}
 else{clearRecordingFit();draw();}
}
$('recordingView').onclick=()=>setRecordingView(!document.body.classList.contains('recording'));
window.addEventListener('resize',queueRecordingFit);
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&document.body.classList.contains('recording'))setRecordingView(false)});
if(document.fonts)document.fonts.ready.then(queueRecordingFit);
if(typeof ResizeObserver!=='undefined'){const layoutObserver=new ResizeObserver(queueRecordingFit);layoutObserver.observe($('demoShell'));layoutObserver.observe($('chart'));}



</script></body></html>
"""


if __name__ == "__main__":
    main()
