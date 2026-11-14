"""Довідкове зображення: де саме знаходяться точки калібрування finger_map.py.

Запуск:  .venv/bin/python landmarks_reference.py   ->  landmarks_reference.png
"""
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import MaxNLocator

from finger_map import LANDMARKS

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LAND, SEA, MARK = "#d6d5cf", "#eaf2fc", "#d03b3b"

plt.rcParams.update({"font.size": 10, "text.color": INK, "figure.facecolor": SURFACE,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})

land = json.load(open("data/ne_50m_land.geojson"))
rings = []
for f in land["features"]:
    g = f["geometry"]
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    rings += [p[0] for p in polys]  # лише зовнішні контури


def draw_land(ax, lw):
    ax.set_facecolor(SEA)
    for r in rings:
        ax.add_patch(Polygon(r, closed=True, fc=LAND, ec=INK2, lw=lw))


def marker(ax, lat, lon, n, size):
    ax.plot(lon, lat, "o", ms=size, mfc=MARK, mec="white", mew=1.5, zorder=5)
    ax.annotate(str(n), (lon, lat), xytext=(0, 0), textcoords="offset points", ha="center", va="center",
                color="white", fontsize=size * 0.75, fontweight="bold", zorder=6)


fig = plt.figure(figsize=(19, 15), layout="constrained")
COLS = 5
gs = fig.add_gridspec(3, COLS, height_ratios=[2.1, 1, 1])

# --- світова карта ---
ax = fig.add_subplot(gs[0, :])
draw_land(ax, 0.3)
ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.set_aspect("equal")
ax.set_xticks(range(-180, 181, 30)); ax.set_yticks(range(-90, 91, 30))
ax.grid(color="white", lw=0.8)
ZOOM = 9  # половина ширини фрагмента, градусів
for n, (_, ua, lat, lon) in enumerate(LANDMARKS, 1):
    ax.add_patch(Rectangle((lon - ZOOM, lat - ZOOM * 0.75), 2 * ZOOM, 1.5 * ZOOM, fill=False, ec=MARK, lw=0.8, ls="--"))
    marker(ax, lat, lon, n, 16)
ax.set_title(f"Точки калібрування: клацайте ЦІ місця на зображенні з камери (по порядку 1 -> {len(LANDMARKS)})",
             fontsize=14, fontweight="bold", loc="left")

# --- збільшені фрагменти ---
for n, (_, ua, lat, lon) in enumerate(LANDMARKS, 1):
    a = fig.add_subplot(gs[1 + (n - 1) // COLS, (n - 1) % COLS])
    draw_land(a, 0.7)
    a.set_xlim(lon - ZOOM, lon + ZOOM); a.set_ylim(lat - ZOOM * 0.75, lat + ZOOM * 0.75)
    a.set_aspect("equal", adjustable="box")
    a.xaxis.set_major_locator(MaxNLocator(5)); a.yaxis.set_major_locator(MaxNLocator(5))
    a.grid(color="white", lw=0.6)
    a.axhline(lat, color=MARK, lw=0.6, ls=":"); a.axvline(lon, color=MARK, lw=0.6, ls=":")
    marker(a, lat, lon, n, 18)
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    a.set_title(f"{n}. {ua}", fontsize=11, fontweight="bold", loc="left")
    a.text(0.02, 0.03, f"{abs(lat):.2f}°{ns}, {abs(lon):.2f}°{ew}", transform=a.transAxes,
           fontsize=9, color=INK2, bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))

fig.savefig("landmarks_reference.png", dpi=110)
print("saved landmarks_reference.png")
