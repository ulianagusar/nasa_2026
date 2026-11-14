"""Огляд шарів Earth Jukebox: data/temperature (GISTEMP) і data/co2_curve (Mauna Loa).

Запуск:  .venv/bin/python explore_jukebox_data.py   ->  jukebox_data_overview.png
"""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SURFACE, INK, INK2, MUTED, GRID, LAND = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#d6d5cf"
BLUE, BLUE_L, RED = "#256abf", "#86b6ef", "#d03b3b"
div = LinearSegmentedColormap.from_list("div", ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#7f1d1d"])
div.set_bad(LAND)
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE,
                     "figure.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11})


def load_temperature(path="data/temperature/temperature.json"):
    d = json.load(open(path))
    A = np.array([[[np.nan if v is None else v for v in row] for row in yr] for yr in d["data"]], float)
    return np.array(d["years"]), np.array(d["lat"]), np.array(d["lon"]), A, d


years, lat, lon, A, meta = load_temperature()
co2 = json.load(open("data/co2_curve/co2_curve.json"))

# глобальна середня, зважена на площу клітинки (cos широти)
w = np.cos(np.radians(lat))[None, :, None] * np.ones_like(A)
w[np.isnan(A)] = 0
glob = np.nansum(np.nan_to_num(A) * w, (1, 2)) / w.sum((1, 2))

fig = plt.figure(figsize=(15, 10), layout="constrained")
gs = fig.add_gridspec(2, 2)

# 1) карта аномалії за 2025 (останній повний рік)
ax = fig.add_subplot(gs[0, 0])
y = 2025
im = ax.imshow(A[years == y][0], origin="lower", extent=[-180, 180, -90, 90], cmap=div,
               norm=TwoSlopeNorm(0, -3, 3), aspect="auto")
ax.set_title(f"temperature: аномалія {y} року відносно 1951–1980")
ax.set_xlabel("довгота, °"); ax.set_ylabel("широта, °")
fig.colorbar(im, ax=ax, label="°C", shrink=0.85, extend="both")

# 2) карта 1900 — видно прогалини в даних
ax = fig.add_subplot(gs[0, 1])
y = 1900
im = ax.imshow(A[years == y][0], origin="lower", extent=[-180, 180, -90, 90], cmap=div,
               norm=TwoSlopeNorm(0, -3, 3), aspect="auto")
ax.set_title(f"temperature: {y} рік (сірим — немає вимірювань)")
ax.set_xlabel("довгота, °"); ax.set_ylabel("широта, °")
fig.colorbar(im, ax=ax, label="°C", shrink=0.85, extend="both")

# 3) глобальна аномалія по роках
ax = fig.add_subplot(gs[1, 0])
full = years <= 2025
ax.axhline(0, color=GRID, lw=1)
ax.plot(years[full], glob[full], color=RED, lw=2)
ax.plot(years[~full], glob[~full], "o", color=RED, mfc="white", ms=6)
ax.annotate("2026: неповний рік", (years[-1], glob[-1]), xytext=(-110, 12), textcoords="offset points",
            color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_title("temperature: середня по планеті (суша + океан)")
ax.set_ylabel("аномалія, °C"); ax.grid(axis="y", color=GRID, lw=0.6)

# 4) CO2: помісячно + річні
ax = fig.add_subplot(gs[1, 1])
mo = co2["monthly"]
t = np.array(mo["year"]) + (np.array(mo["month"]) - 0.5) / 12
ax.plot(t, mo["ppm"], color=BLUE_L, lw=1, label="щомісяця (видно сезонне «дихання»)")
ax.plot(np.array(co2["years"][:-1]) + 0.5, co2["values"][:-1], color=BLUE, lw=2, label="середнє за рік")
ax.set_title("co2_curve: CO₂ на Мауна-Лоа, 1958–2026")
ax.set_ylabel("ppm"); ax.grid(axis="y", color=GRID, lw=0.6); ax.legend(frameon=False, loc="upper left")

fig.suptitle("Шари Earth Jukebox: що лежить у data/temperature і data/co2_curve", fontsize=14, fontweight="bold")
fig.savefig("jukebox_data_overview.png", dpi=120)
print("saved jukebox_data_overview.png")
