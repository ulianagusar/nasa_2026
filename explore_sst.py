"""Overview of NOAA ERSST v5 (sst.mnmean.nc): file structure + 4 basic visualizations.

Run:     .venv/bin/python explore_sst.py
Output:  sst_overview.png
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ---- colours (single-hue sequential, blue<->red diverging, grey zero) ----
SURFACE, INK, INK2, MUTED, GRID, LAND = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#d6d5cf"
BLUE, RED = "#256abf", "#d03b3b"
seq = LinearSegmentedColormap.from_list("seq_orange", ["#fdeee4", "#f6b48f", "#eb6834", "#b7461c", "#6e2509"])
div = LinearSegmentedColormap.from_list("div", ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#7f1d1d"])
seq.set_bad(LAND); div.set_bad(LAND)

plt.rcParams.update({
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 11,
})

ds = xr.open_dataset("sst.mnmean.nc", engine="h5netcdf")
# longitude 0..358 -> -180..178, so that Europe/Africa are in the middle of the map
ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
sst = ds.sst
last = sst.time.values[-1]
last_str = np.datetime_as_string(last, unit="M")

fig = plt.figure(figsize=(15, 10), layout="constrained")
gs = fig.add_gridspec(2, 2)
ext = [float(sst.lon.min()) - 1, float(sst.lon.max()) + 1, float(sst.lat.min()) - 1, float(sst.lat.max()) + 1]

# 1) map: temperature in the latest month
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(sst.sel(time=last).values, extent=ext, cmap=seq, vmin=-2, vmax=32, aspect="auto")
ax.set_title(f"Sea surface temperature, {last_str}")
ax.set_xlabel("longitude, °"); ax.set_ylabel("latitude, °")
fig.colorbar(im, ax=ax, label="°C", shrink=0.85)

# 2) map: anomaly - the last 10 full years vs 1971–2000
base = sst.sel(time=slice("1971", "2000")).mean("time")
recent = sst.sel(time=slice("2016", "2025")).mean("time")
anom = recent - base
ax = fig.add_subplot(gs[0, 1])
im = ax.imshow(anom.values, extent=ext, cmap=div, norm=TwoSlopeNorm(0, -1.5, 1.5), aspect="auto")
ax.set_title("How much warmer: 2016–2025 minus 1971–2000")
ax.set_xlabel("longitude, °"); ax.set_ylabel("latitude, °")
fig.colorbar(im, ax=ax, label="difference, °C", shrink=0.85, extend="both")

# 3) global mean (60°S–60°N, weighted by cell area cos(lat)), yearly values
band = sst.sel(lat=slice(60, -60))
w = np.cos(np.deg2rad(band.lat))
glob = band.weighted(w).mean(("lat", "lon"))
annual = glob.sel(time=slice("1854", "2025")).groupby("time.year").mean()
smooth = annual.rolling(year=10, center=True).mean()
ax = fig.add_subplot(gs[1, 0])
ax.plot(annual.year, annual, color="#86b6ef", lw=1.2, label="yearly value")
ax.plot(smooth.year, smooth, color=BLUE, lw=2.2, label="10-year average")
ax.set_title("Mean ocean temperature (60°S–60°N), 1854–2025")
ax.set_ylabel("°C"); ax.grid(axis="y", color=GRID, lw=0.6); ax.legend(frameon=False, loc="upper left")
y_last = float(annual[-1])
ax.annotate(f"{int(annual.year[-1])}: {y_last:.2f} °C", (annual.year[-1], y_last),
            xytext=(-110, -6), textcoords="offset points", color=INK2, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
y_first = float(annual[0])

# 4) a single point: Black Sea (44°N, 32°E) - seasonal cycle 1971–2000 vs 2023–2025
pt = sst.sel(lat=44, lon=32, method="nearest")
clim_old = pt.sel(time=slice("1971", "2000")).groupby("time.month").mean()
clim_new = pt.sel(time=slice("2023", "2025")).groupby("time.month").mean()
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ax = fig.add_subplot(gs[1, 1])
ax.plot(range(1, 13), clim_old, color="#5598e7", lw=2, marker="o", ms=5, label="1971–2000")
ax.plot(range(1, 13), clim_new, color=RED, lw=2, marker="o", ms=5, label="2023–2025")
ax.set_xticks(range(1, 13), months)
ax.set_title(f"Black Sea ({float(pt.lat):.0f}°N, {float(pt.lon):.0f}°E): seasonal cycle")
ax.set_ylabel("°C"); ax.grid(axis="y", color=GRID, lw=0.6); ax.legend(frameon=False, loc="upper left")
d_aug = float(clim_new.sel(month=8) - clim_old.sel(month=8))
ax.annotate(f"August: +{d_aug:.1f} °C", (8, float(clim_new.sel(month=8))),
            xytext=(12, 4), textcoords="offset points", color=INK2, fontsize=9)

fig.suptitle("NOAA ERSST v5 - what is inside sst.mnmean.nc", fontsize=14, fontweight="bold")
fig.savefig("sst_overview.png", dpi=130)
print("saved sst_overview.png")
print(f"Global mean: {y_first:.2f} °C (1854) -> {y_last:.2f} °C (2025)")
print(f"Black Sea, August: {float(clim_old.sel(month=8)):.1f} -> {float(clim_new.sel(month=8)):.1f} °C")
