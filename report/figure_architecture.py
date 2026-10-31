"""Figure 1 (final report): how TrustFlow works, as a numbered story in plain words.

1 the plant, 2 readings to the control room, 3 where the attacker edits them,
4 the pump's electricity meter on its own network, 5 TrustFlow's three questions,
6 the alarm. Technical names are left to Table 2 and Section 5.2.
Run: .venv/bin/python report/figure_architecture.py  ->  report/fig_architecture.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

OUT = Path(__file__).resolve().parent / "fig_architecture.png"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#c4c4c4"
RED, BLUE, GREEN, AMBER = "#c0392b", "#1f5fa8", "#1d7a55", "#b9770e"
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(7.2, 3.9), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 150); ax.set_ylim(0, 80); ax.axis("off")


def box(x, y, w, h, ec=INK, lw=0.8, fc="white"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=2))


def text(x, y, s, size=6.8, color=INK, weight="normal", ha="left", va="center", style="normal", z=4):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, style=style,
            linespacing=1.3, zorder=z)


def arrow(x1, y1, x2, y2, color=INK, lw=0.9, ls="-", head=7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                 mutation_scale=head, shrinkA=0, shrinkB=0, zorder=3))


def step(x, y, n, color=INK):
    ax.add_patch(Circle((x, y), 2.3, fc=color, ec=color, zorder=5))
    text(x, y - 0.1, str(n), size=6.6, color="white", weight="bold", ha="center", z=6)


# 1 the plant
step(3.5, 74, 1)
text(7.5, 74, "The plant turns seawater into drinking water", size=7.2, weight="bold")
stages = [("Seawater\nfilter", 4), ("Chlorine\nadded & removed", 30), ("High-pressure\npump", 56), ("RO\nmembranes", 82)]
sw, sy, sh = 20, 57, 11
for name, x in stages:
    box(x, sy, sw, sh)
    text(x + sw / 2, sy + sh / 2, name, size=6.5, ha="center")
for (_, a), (_, b) in zip(stages[:-1], stages[1:]):
    arrow(a + sw, sy + sh / 2, b, sy + sh / 2, lw=1.1)
arrow(82 + sw, sy + sh / 2, 110, sy + sh / 2, lw=1.1)
text(111, sy + sh / 2, "drinking\nwater", size=6.4, color=MUTED)
ax.text(4, sy - 3.4, "Sensors on every stage measure flow, pressure, salt and chlorine.", fontsize=6.0, color=MUTED, va="top", zorder=4,
        bbox=dict(fc="white", ec="none", pad=0.6))

# 2 readings to the control room, 3 attacker
ny = 38
for x in (14, 40, 66, 92):
    ax.plot([x, x], [sy, ny], color=GRID, lw=0.7, zorder=1)
ax.plot([4, 104], [ny, ny], color=INK, lw=1.2, zorder=1)
step(3.5, ny + 6, 2)
text(7.5, ny + 6, "Readings go to the control room", size=7.0, weight="bold")
arrow(104, ny, 116.5, ny, lw=1.1)

ax.add_patch(Rectangle((17, ny - 2), 20, 4, fc=RED, ec=RED, lw=0, zorder=3))
text(27, ny, "ATTACKER", size=6.4, color="white", weight="bold", ha="center")
step(19.3, ny - 8, 3, RED)
text(23.3, ny - 8, "changes the readings here,\nso the screen looks normal", size=6.0, color=RED, va="center")

# 4 the electricity meter on its own network
step(3.5, 17, 4, AMBER)
text(7.5, 17, "The pump's electricity meter", size=7.0, weight="bold")
text(7.5, 12.4, "on a separate network the attacker cannot reach", size=6.0, color=MUTED)
box(60, 11, 26, 9, ec=AMBER, lw=1.0)
text(73, 15.5, "power meter", size=6.4, ha="center")
ax.plot([66, 66], [sy, 20], color=AMBER, lw=0.9, ls=(0, (2.2, 1.6)), zorder=1)
arrow(86, 15.5, 118.5, 15.5, color=AMBER, lw=1.1, ls=(0, (2.2, 1.6)))

# 5 TrustFlow's three questions, 6 alarm
tx = 117
ax.add_patch(Rectangle((tx, 0.8), 31, 51.2, fc="none", ec=BLUE, lw=1.1, zorder=1))
step(tx + 3.5, 48.3, 5, BLUE)
text(tx + 7.5, 48.3, "TrustFlow asks", size=7.0, weight="bold", color=BLUE)
qs = [("Does it look normal?", "AI detector", BLUE, 44.5, 8.5),
      ("Does it obey physics?", "water and salt balance,\nmeters agree", GREEN, 34.5, 10),
      ("Does the pump's power\nmatch its work?", "energy check, using\nthe separate meter", AMBER, 22.5, 13.5)]
for q, a, c, top, h in qs:
    box(tx + 2, top - h, 27, h, ec=c, lw=1.0)
    text(tx + 3.5, top - 1.4, q, size=6.2, weight="bold", va="top")
    text(tx + 3.5, top - 1.4 - 3.1 * (q.count("\n") + 1) - 0.4, a, size=5.4, color=MUTED, va="top")
step(tx + 3.5, 4.6, 6, RED)
text(tx + 7.5, 4.6, "Alarm: says which\ncheck failed", size=5.9, weight="bold")

fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.05)
print("saved", OUT)
