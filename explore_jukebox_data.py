"""Overview of the Earth Jukebox layers: data/temperature (GISTEMP) and data/co2_curve (Mauna Loa).

Run:  .venv/bin/python explore_jukebox_data.py   ->  jukebox_data_overview.png
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

# global mean, weighted by cell area (cos of latitude)
w = np.cos(np.radians(lat))[None, :, None] * np.ones_like(A)
w[np.isnan(A)] = 0
glob = np.nansum(np.nan_to_num(A) * w, (1, 2)) / w.sum((1, 2))

fig = plt.figure(figsize=(15, 10), layout="constrained")
gs = fig.add_gridspec(2, 2)

# 1) anomaly map for 2025 (the last full year)
ax = fig.add_subplot(gs[0, 0])
y = 2025
im = ax.imshow(A[years == y][0], origin="lower", extent=[-180, 180, -90, 90], cmap=div,
               norm=TwoSlopeNorm(0, -3, 3), aspect="auto")
ax.set_title(f"temperature: {y} anomaly vs 1951–1980")
ax.set_xlabel("longitude, °"); ax.set_ylabel("latitude, °")
fig.colorbar(im, ax=ax, label="°C", shrink=0.85, extend="both")

# 2) map for 1900 - gaps in the data are visible
ax = fig.add_subplot(gs[0, 1])
y = 1900
im = ax.imshow(A[years == y][0], origin="lower", extent=[-180, 180, -90, 90], cmap=div,
               norm=TwoSlopeNorm(0, -3, 3), aspect="auto")
ax.set_title(f"temperature: year {y} (grey - no measurements)")
ax.set_xlabel("longitude, °"); ax.set_ylabel("latitude, °")
fig.colorbar(im, ax=ax, label="°C", shrink=0.85, extend="both")

# 3) global anomaly by year
ax = fig.add_subplot(gs[1, 0])
full = years <= 2025
ax.axhline(0, color=GRID, lw=1)
ax.plot(years[full], glob[full], color=RED, lw=2)
ax.plot(years[~full], glob[~full], "o", color=RED, mfc="white", ms=6)
ax.annotate("2026: partial year", (years[-1], glob[-1]), xytext=(-110, 12), textcoords="offset points",
            color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_title("temperature: planet-wide mean (land + ocean)")
ax.set_ylabel("anomaly, °C"); ax.grid(axis="y", color=GRID, lw=0.6)

# 4) CO2: monthly + yearly
ax = fig.add_subplot(gs[1, 1])
mo = co2["monthly"]
t = np.array(mo["year"]) + (np.array(mo["month"]) - 0.5) / 12
ax.plot(t, mo["ppm"], color=BLUE_L, lw=1, label="monthly (seasonal 'breathing' is visible)")
ax.plot(np.array(co2["years"][:-1]) + 0.5, co2["values"][:-1], color=BLUE, lw=2, label="yearly average")
ax.set_title("co2_curve: CO₂ at Mauna Loa, 1958–2026")
ax.set_ylabel("ppm"); ax.grid(axis="y", color=GRID, lw=0.6); ax.legend(frameon=False, loc="upper left")

fig.suptitle("Earth Jukebox layers: what is inside data/temperature and data/co2_curve", fontsize=14, fontweight="bold")
fig.savefig("jukebox_data_overview.png", dpi=120)
print("saved jukebox_data_overview.png")
