"""
Live demo view (STATUS item 2c): one self-contained HTML page that replays a
spoofed SWRO plant. Run from the repo root:

    .venv/bin/python src/make_demo.py  ->  results/swro_demo.html

Story: permeate TDS is spoofed 10% low from day 45. The reported value stays
inside assumed plant alarm limits (100-400 mg/L), so a limit check (the
stand-in for the ML detector here) stays quiet. The physics residual for permeate salt passage fires.

The reference is frozen at the clean-data trailing median (the onset view used
in eval_swro_invariants.py). A trailing reference would absorb the spoof after
about six hours, so the page shows the onset case only.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

from swro_sim import simulate
from invariants_swro import fit, residuals, quantities, trailing_reference, NAMES

ROOT = Path(__file__).resolve().parent.parent
TRAILING = dict(window_min=360, lag_min=60)
CAL_DAYS, SPOOF_DAY, SHOW = 30, 45.0, (40.0, 50.0)
SPOOF_FACTOR = 0.90
STEP = 5                                   # minutes per plotted point
SPOOF_TAG = "C_perm"
ALARM_LIMITS = (100.0, 400.0)              # mg/L, assumed plant alarm limits
INVARIANT = "ro_salt_perm"


def main():
    truth, rep, ev = simulate(days=60, seed=0)
    n_cal = int((ev.day < CAL_DAYS).sum())
    m = (ev.day < CAL_DAYS).values
    log = {"cip": np.flatnonzero(ev.cip.values), "ceb": np.flatnonzero(ev.ceb.values),
           "uf_phase": ev.uf_phase.values}
    log_cal = {"cip": log["cip"][log["cip"] < n_cal], "ceb": log["ceb"][log["ceb"] < n_cal],
               "uf_phase": log["uf_phase"][m]}
    cal = fit(rep[m], trailing=TRAILING, events=log_cal)

    # Frozen reference from the clean run, then spoof the reported tag from day 45.
    ref = trailing_reference(quantities(rep, cal["orp_k"]), **TRAILING, events=log)
    x = rep.copy()
    start = int(SPOOF_DAY * 1440)
    x.loc[start:, SPOOF_TAG] *= SPOOF_FACTOR
    R = residuals(x, cal, ref=ref, events=log).abs().numpy()
    R_max, R_arg = R.max(1), R.argmax(1)

    # Assumed fixed alarm limits for permeate TDS. Not from our data: a limit
    # check uses the plant's own alarms. The clean drift is too large for a
    # band fitted on days 0-30 (it flags 31% of clean held-out rows).
    lo, hi = ALARM_LIMITS
    a, b = (int(SHOW[0] * 1440), int(SHOW[1] * 1440))
    idx = np.arange(a, b, STEP)
    inv_i = NAMES.index(INVARIANT)

    data = dict(
        t=[round(float(v), 4) for v in ev.day.values[idx]],
        true=[round(float(v), 2) for v in truth[SPOOF_TAG].values[idx]],
        reported=[round(float(v), 2) for v in x[SPOOF_TAG].values[idx]],
        band=[round(lo, 2), round(hi, 2)],
        phys=[round(float(v), 3) for v in R[idx, inv_i]],
        phys_max=[round(float(v), 3) for v in R_max[idx]],
        phys_arg=[NAMES[i] for i in R_arg[idx]],
        spoof_day=SPOOF_DAY, factor=SPOOF_FACTOR,
        invariant=INVARIANT, tag=SPOOF_TAG, unit=truth.attrs.get("unit", "mg/L"),
    )
    # Sanity numbers printed for the record, not just the page.
    before = R[:start, inv_i] > 1
    after = R[start:, inv_i] > 1
    limit_quiet = bool(((x[SPOOF_TAG].values[start:] >= lo) & (x[SPOOF_TAG].values[start:] <= hi)).all())
    print(f"clean band {lo:.1f}-{hi:.1f} mg/L; spoofed reported value stays inside band: {limit_quiet}")
    print(f"{INVARIANT} violations before spoof: {before.mean():.4%}; after spoof: {after.mean():.1%}")

    html = TEMPLATE.replace("__DATA__", json.dumps(data))
    out = ROOT / "results" / "swro_demo.html"
    out.write_text(html, encoding="utf-8")
    print("wrote", out)


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TrustFlow live demo: SWRO spoof</title>
<style>
:root { --bg:#f7f7f5; --card:#ffffff; --ink:#1b1b1b; --muted:#5f6368; --line:#d0d0cc;
        --true:#1f6f5c; --rep:#c0392b; --band:#d9ebe6; --alarm:#c0392b; --ok:#1f6f5c; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg:#15171a; --card:#1e2226; --ink:#e8e8e6; --muted:#9aa0a6;
        --line:#3a3f45; --true:#5fbfa6; --rep:#ef6b5b; --band:#23403a; --alarm:#ef6b5b; --ok:#5fbfa6; }
}
:root[data-theme="dark"] { --bg:#15171a; --card:#1e2226; --ink:#e8e8e6; --muted:#9aa0a6;
        --line:#3a3f45; --true:#5fbfa6; --rep:#ef6b5b; --band:#23403a; --alarm:#ef6b5b; --ok:#5fbfa6; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.4 system-ui, Segoe UI, Arial, sans-serif; }
main { max-width:1100px; margin:0 auto; padding:16px; }
h1 { font-size:20px; margin:4px 0 2px; }
.sub { color:var(--muted); margin-bottom:12px; font-size:13px; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
@media (max-width:760px) { .grid { grid-template-columns:1fr; } }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; }
.card h2 { font-size:14px; margin:0 0 6px; }
.big { font-size:22px; font-weight:600; }
.status { font-size:13px; color:var(--muted); }
svg { width:100%; height:auto; display:block; }
.controls { display:flex; gap:10px; align-items:center; margin:12px 0; flex-wrap:wrap; }
button { background:var(--card); color:var(--ink); border:1px solid var(--line); border-radius:6px; padding:6px 12px; cursor:pointer; }
input[type=range] { flex:1; min-width:180px; }
.legend { font-size:12px; color:var(--muted); display:flex; gap:12px; flex-wrap:wrap; margin-top:4px; }
.sw { display:inline-block; width:12px; height:3px; vertical-align:middle; margin-right:4px; }
.note { font-size:12px; color:var(--muted); margin-top:10px; }
</style></head>
<body><main>
<h1>Permeate TDS is spoofed. The limit check stays quiet. The physics check fires.</h1>
<div class="sub">Simulated SWRO plant, one row every 5 minutes. Spoof starts day 45 and lowers the reported permeate TDS by 10%.</div>

<div class="grid">
  <div class="card">
    <h2>1. Plant: permeate TDS, true vs reported</h2>
    <svg id="pTrue" viewBox="0 0 560 240"></svg>
    <div class="legend"><span><span class="sw" style="background:var(--true)"></span>true (hidden from operators)</span>
      <span><span class="sw" style="background:var(--rep)"></span>reported to SCADA</span>
      <span><span class="sw" style="background:var(--band)"></span>normal range, clean data</span></div>
  </div>
  <div class="card">
    <h2>2. Detector (limit check, stand-in for the ML model)</h2>
    <div class="big" id="detState">QUIET</div>
    <div class="status" id="detMsg"></div>
    <svg id="pDet" viewBox="0 0 560 120" style="margin-top:6px"></svg>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>3. Physics check: permeate salt passage (ASTM D4516 normalised), residual / tolerance</h2>
    <div class="big" id="physState">OK</div>
    <div class="status" id="physMsg"></div>
    <svg id="pPhys" viewBox="0 0 1100 220" style="margin-top:6px"></svg>
    <div class="legend"><span><span class="sw" style="background:var(--ink)"></span>residual</span>
      <span><span class="sw" style="background:var(--alarm)"></span>alarm threshold (|r| = 1)</span></div>
  </div>
</div>

<div class="controls">
  <button id="play">Play</button>
  <input type="range" id="scrub" min="0" max="0" value="0">
  <span class="status" id="clock"></span>
</div>
<div class="note">Simulated data. Detector limits (100-400 mg/L) are assumed plant alarms, not fitted. Physics tolerance is calibrated on days 0 to 30 with 20% margin. The reference is frozen at the clean-data trailing median. A trailing reference would absorb the spoof after about six hours.</div>
</main>

<script>
const D = __DATA__;
const N = D.t.length;
const W = 560, H = 240, PAD = 40;

function xScale(i, w) { return PAD + (w - PAD - 10) * i / (N - 1); }
function yScale(v, lo, hi, h) { return h - 20 - (h - 40) * (v - lo) / (hi - lo); }

function line(vals, lo, hi, w, h, color, upto, dash) {
  let d = "";
  for (let i = 0; i <= upto; i++) d += (i ? "L" : "M") + xScale(i, w).toFixed(1) + "," + yScale(vals[i], lo, hi, h).toFixed(1);
  return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2" ${dash ? 'stroke-dasharray="4 3"' : ""}/>`;
}

function axes(w, h, lo, hi, ticks, unit) {
  let s = `<line x1="${PAD}" y1="${h-20}" x2="${w-10}" y2="${h-20}" stroke="var(--line)"/>`;
  for (const v of ticks) {
    const y = yScale(v, lo, hi, h);
    s += `<line x1="${PAD}" y1="${y}" x2="${w-10}" y2="${y}" stroke="var(--line)" stroke-dasharray="2 4" opacity=".6"/>`;
    s += `<text x="${PAD-6}" y="${y+4}" text-anchor="end" font-size="11" fill="var(--muted)">${v}</text>`;
  }
  s += `<text x="${PAD}" y="${h-4}" font-size="11" fill="var(--muted)">day ${D.t[0].toFixed(1)}</text>`;
  s += `<text x="${w-10}" y="${h-4}" text-anchor="end" font-size="11" fill="var(--muted)">day ${D.t[N-1].toFixed(1)}</text>`;
  return s;
}

function spoofMarker(w, h) {
  const i = D.t.findIndex(t => t >= D.spoof_day);
  if (i < 0) return "";
  const x = xScale(i, w);
  return `<line x1="${x}" y1="10" x2="${x}" y2="${h-20}" stroke="var(--alarm)" stroke-dasharray="3 3"/>` +
         `<text x="${x+4}" y="22" font-size="11" fill="var(--alarm)">spoof starts</text>`;
}

const [lo, hi] = [Math.min(...D.true, ...D.reported) - 10, Math.max(...D.true, ...D.reported) + 10];
const ticks = [];
for (let v = Math.ceil(lo/20)*20; v <= hi; v += 20) ticks.push(v);

function drawStatic() {
  // Panel 1: band fill, true, reported
  const b0 = yScale(D.band[1], lo, hi, H), b1 = yScale(D.band[0], lo, hi, H);
  let s = axes(W, H, lo, hi, ticks, "mg/L");
  s += `<rect x="${PAD}" y="${b0}" width="${W-PAD-10}" height="${b1-b0}" fill="var(--band)" opacity=".7"/>`;
  s += spoofMarker(W, H);
  document.getElementById("pTrue").innerHTML = s;
  // Panel 2: limit check value vs band
  const lo2 = D.band[0] - 30, hi2 = D.band[1] + 30;
  let s2 = axes(W, 120, lo2, hi2, [Math.round(D.band[0]), Math.round(D.band[1])], "mg/L");
  s2 += `<rect x="${PAD}" y="${yScale(D.band[1],lo2,hi2,120)}" width="${W-PAD-10}" height="${yScale(D.band[0],lo2,hi2,120)-yScale(D.band[1],lo2,hi2,120)}" fill="var(--band)" opacity=".7"/>`;
  s2 += spoofMarker(W, 120);
  document.getElementById("pDet").innerHTML = s2;
  // Panel 3: physics residual
  const hiP = Math.max(2.5, ...D.phys) + 0.2;
  let s3 = axes(1100, 220, 0, hiP, [0, 1, 2].filter(v => v <= hiP), "");
  const yA = yScale(1, 0, hiP, 220);
  s3 += `<line x1="${PAD}" y1="${yA}" x2="${1090}" y2="${yA}" stroke="var(--alarm)" stroke-dasharray="4 3"/>`;
  s3 += spoofMarker(1100, 220);
  document.getElementById("pPhys").innerHTML = s3;
  window._phys = {hiP};
}

function drawCursor(i) {
  const xs = xScale(i, W), x3 = xScale(i, 1100);
  const cur = (x, w, h) => `<line x1="${x}" y1="0" x2="${x}" y2="${h}" stroke="var(--ink)" opacity=".5"/>`;
  // Trace lines
  let p = line(D.true, lo, hi, W, H, "var(--true)", i, false) + line(D.reported, lo, hi, W, H, "var(--rep)", i, false);
  document.getElementById("pTrue").insertAdjacentHTML("beforeend", p + cur(xs, W, H));
  let q = line(D.reported, D.band[0] - 30, D.band[1] + 30, W, 120, "var(--rep)", i, false);
  document.getElementById("pDet").insertAdjacentHTML("beforeend", q + cur(xs, W, 120));
  const hiP = window._phys.hiP;
  document.getElementById("pPhys").insertAdjacentHTML("beforeend", line(D.phys, 0, hiP, 1100, 220, "var(--ink)", i, false) + cur(x3, 1100, 220));
}

function render(i) {
  // Redraw static parts then traces (simple and reliable at this size)
  drawStatic();
  drawCursor(i);
  const t = D.t[i];
  const outOfBand = D.reported[i] < D.band[0] || D.reported[i] > D.band[1];
  const detAlarm = outOfBand;
  document.getElementById("detState").textContent = detAlarm ? "ALARM" : "QUIET";
  document.getElementById("detState").style.color = detAlarm ? "var(--alarm)" : "var(--ok)";
  document.getElementById("detMsg").textContent = `Reported ${D.reported[i].toFixed(1)} ${D.unit || "mg/L"}, normal range ${D.band[0]}-${D.band[1]}`;
  const physAlarm = D.phys[i] > 1;
  document.getElementById("physState").textContent = physAlarm ? "ALARM" : "OK";
  document.getElementById("physState").style.color = physAlarm ? "var(--alarm)" : "var(--ok)";
  const firstAlarm = D.phys.findIndex((v, k) => v > 1 && D.t[k] >= D.spoof_day);
  let msg = `Residual ${D.phys[i].toFixed(2)} at day ${t.toFixed(2)}`;
  if (firstAlarm >= 0 && i >= firstAlarm) msg += `. First alarm at day ${D.t[firstAlarm].toFixed(2)}`;
  document.getElementById("physMsg").textContent = msg;
  document.getElementById("clock").textContent = `day ${t.toFixed(2)}`;
}

const scrub = document.getElementById("scrub");
scrub.max = N - 1;
scrub.value = Math.round(N * 0.2);
let timer = null;
document.getElementById("play").onclick = () => {
  if (timer) { clearInterval(timer); timer = null; document.getElementById("play").textContent = "Play"; return; }
  document.getElementById("play").textContent = "Pause";
  timer = setInterval(() => {
    let v = +scrub.value + 2;
    if (v >= N - 1) { v = 0; }
    scrub.value = v; render(v);
  }, 60);
};
scrub.oninput = () => render(+scrub.value);
render(+scrub.value);
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
