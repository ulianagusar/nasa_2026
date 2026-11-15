"""Earth Jukebox data layers: any map -> value at a point -> colour and sound.

The program works with maps only (the value depends on the place). A layer has one interface:
    layer.sample(lat, lon, year) -> value, normalized value 0..1 (note pitch),
                                    "tension" (how far above/below the baseline: beating,
                                    pulsing), lines for the panel
    layer.grid(year)             -> 2D array for the map

Where layers come from (automatically, see discover_layers):
  * sst.mnmean.nc                       -> "sst": sea surface temperature (NOAA ERSST v5)
  * data/<name>/<name>.meta.json + .u8.bin (or .json with data)  -> map by years
  * data/<name>/jukebox.json (optional) -> title, colours, baseline for the sound
Folders without *.meta.json (e.g. curves such as data/co2_curve) are skipped.

How to add a new layer - see LAYERS.md. NetCDF converter:
    .venv/bin/python layers.py from-netcdf file.nc --var variable_name --name my_layer
    .venv/bin/python layers.py list
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
SST_FILE = HERE / "sst.mnmean.nc"
COAST_FILE = DATA_DIR / "ne_50m_land.geojson"

SURFACE, INK, INK2, MUTED, GRIDC, LANDC = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#d6d5cf"
WARM, COOL = "#d03b3b", "#256abf"
CMAPS = {
    "diverging": ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#7f1d1d"],   # blue - grey zero - red
    "sequential": ["#fdeee4", "#f6b48f", "#eb6834", "#b7461c", "#6e2509"],  # light -> dark orange
    "blues": ["#e8f1fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"],
    "greens": ["#eaf6e6", "#a9d99b", "#4fa54a", "#257a2c", "#0e4a19"],
}
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
             "October", "November", "December"]

# Built-in settings of known layers. Any key can be overridden in data/<name>/jukebox.json.
BUILTIN = {
    "temperature": {
        "title": "Temperature anomaly (land + ocean)",
        "units": "°C (anomaly vs 1951–1980)",
        "fmt": "{:+.2f}", "default_year": 2025, "tension_scale": 2.0,
        "note": "difference from the 1951–1980 average",
        "speech_note": "This is the difference from the 1951–1980 average.",
    },
    "ocean_sst_video": {
        "title": "Ocean temperature from NASA video (ECCO2)",
        "units": "°C (estimated from video colour)",
        "fmt": "{:.1f}", "cmap": "sequential", "default_frame": 0,
        "baseline": "mean", "base_label": "the yearly average at this spot", "tension_scale": 4.0,
        "missing_label": "land / no colour", "note": "estimated from video colour; frame dates are approximate",
        "speech_note": "The temperature is estimated from the colours of a NASA video.",
        "source": "NASA SVS 3912, ECCO2 model (MIT/JPL)",
    },
    "currents_speed": {
        "title": "Ocean surface current speed",
        "units": "m/s", "fmt": "{:.2f}", "cmap": "blues", "default_frame": 0,
        "source": "OSCAR (ESR / NASA PO.DAAC) via NOAA ERDDAP",
        "missing_label": "land / no data",
        "note": "monthly mean; OSCAR is an analysis of satellite data, not a direct measurement",
        "speech_note": "The faster the current, the rougher the water sounds.",
        "sound": "water", "sound_dir": "data/currents_sounds",
    },
    "earthquakes_maxmag": {
        "title": "Strongest earthquake of the week",
        "units": "magnitude", "fmt": "{:.1f}", "cmap": ["#fdeee4", "#f6b48f", "#eb6834", "#b7461c", "#6e2509"],
        "default_frame": 0, "missing_label": "no earthquakes",
        "note": "strongest earthquake (M 4.5+) in a 2° cell during the week", "source": "USGS Earthquake Catalog",
        "speech_note": "This is the strongest earthquake of that week near this place.",
        # felt on the trackpad: quakes within 500 km count, weaker with distance (most cells are empty)
        "haptic": {"radius_km": 500, "min": 0.2},
    },
    "ocean_flow_activity": {
        "title": "Ocean current activity from NASA video (0–1)",
        "units": "", "fmt": "{:.2f}", "cmap": "blues", "default_frame": 0,
        "missing_label": "land / no lines",
        "note": "visual density of current lines, not speed; frame dates are approximate",
        "speech_note": "This is how dense the current lines are in the video, from zero to one.",
        "source": "NASA SVS 3912 (derived from the visualization)",
    },
}
CFG_KEYS = ("title", "units", "source", "description", "note", "fmt", "vmin", "vmax", "cmap", "baseline",
            "base_label", "tension_scale", "default_year", "default_frame", "missing_label", "static_label",
            "partial_years", "prefer", "speech_note", "max_near_km", "sound", "sound_dir", "scale", "haptic")


def wrap180(lon):
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def fmt_coord(lat, lon):
    return f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def make_lut(cmap):
    """cmap: a name from CMAPS or a list of hex colours -> 256x3 table (RGB, uint8)."""
    stops = CMAPS.get(cmap, cmap) if isinstance(cmap, str) else cmap
    rgb = np.array([hex_rgb(c) for c in stops], float)
    x = np.linspace(0, 1, len(rgb))
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, x, rgb[:, k]) for k in range(3)], 1).astype(np.uint8)


_FONTS = {}


def get_fonts():
    if not _FONTS:
        from matplotlib import font_manager
        from PIL import ImageFont
        reg = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
        bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
        for k, (path, size) in {"xs": (reg, 13), "s": (reg, 16), "m": (bold, 19), "l": (bold, 17),
                                "xl": (bold, 46)}.items():
            _FONTS[k] = ImageFont.truetype(path, size)
    return _FONTS


def _unit_short(units):
    return units.split("(")[0].strip()


# ----------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------
class Layer:
    """Shared logic: formatting, normalization for note pitch, "tension" for beating/pulse."""

    def __init__(self, name, cfg):
        self.name, self.cfg = name, cfg
        self.title = cfg.get("title") or cfg.get("description") or name
        self.units = cfg.get("units", "")
        self.unit = _unit_short(self.units)
        self.source = cfg.get("source", "")
        self.note = cfg.get("note", "")
        self.fmt = cfg.get("fmt", "{:.2f}")
        self.vmin, self.vmax = float(cfg["vmin"]), float(cfg["vmax"])
        diverging = self.vmin < 0 < self.vmax
        self.cmap = cfg.get("cmap") or ("diverging" if diverging else "sequential")
        self.lut = make_lut(self.cmap)
        # "scale": "log" - colours and note pitch follow the logarithm (concentrations, aerosols)
        self.log = cfg.get("scale") == "log" and self.vmin > 0 and self.vmax > 0
        b = cfg.get("baseline", 0.0 if diverging else None)
        self.baseline = None if isinstance(b, str) else b      # "mean" is handled by GridLayer (grid of means)
        self.base_label = cfg.get("base_label", "the baseline")
        self.tension_scale = float(cfg.get("tension_scale") or
                                   (max(abs(self.vmin), abs(self.vmax)) if diverging else (self.vmax - self.vmin) / 2))
        self.missing_label = cfg.get("missing_label", "no data")
        self.partial_years = {int(y) for y in cfg.get("partial_years", [])}   # e.g. the current year
        # Time: a layer = a sequence of frames. times - the moment of each frame in years (2007.24 = end of March 2007),
        # labels - the frame label ("1990" or "28.03.2007"). None = a single moment.
        self.times, self.labels, self.has_dates = None, [], False

    # --- time: frame index t = 0..n_frames-1 ---
    @property
    def n_frames(self):
        return 1 if self.times is None else len(self.times)

    @property
    def dynamic(self):
        return self.n_frames > 1

    def default_frame(self):
        if not self.dynamic:
            return 0
        if self.cfg.get("default_frame") is not None:
            return self.clamp_frame(int(self.cfg["default_frame"]))
        y = self.cfg.get("default_year")
        if y is not None:
            cand = [i for i, t in enumerate(self.times) if int(t) == int(y)]
            if cand:
                return cand[-1]
        full = [i for i, t in enumerate(self.times) if int(t) not in self.partial_years]
        return full[-1] if full else self.n_frames - 1

    def clamp_frame(self, t):
        return self.default_frame() if t is None else int(np.clip(t, 0, self.n_frames - 1))

    def frame_at(self, when):
        """Frame nearest to the moment `when` (in years, e.g. 1990.5); to switch layers without losing time."""
        if not self.dynamic or when is None:
            return self.default_frame()
        return int(np.argmin(np.abs(np.asarray(self.times) - when)))

    def time_of(self, t):
        return None if not self.dynamic else float(self.times[self.clamp_frame(t)])

    def time_label(self, t):
        if not self.dynamic:
            return self.cfg.get("static_label", "")
        k = self.clamp_frame(t)
        lab = self.labels[k]
        return f"{lab} (partial year)" if int(self.times[k]) in self.partial_years else lab

    def short_label(self, t):
        """Short label for the time scale: "1990" or "03.2007"."""
        lab = self.labels[self.clamp_frame(t)]
        return lab[3:] if self.has_dates else lab

    # --- values ---
    def fmt_value(self, v, clipped=0):
        s = f"{self.fmt.format(v)} {self.unit}".strip()
        return ("≥ " if clipped > 0 else "≤ " if clipped < 0 else "") + s

    def norm(self, v):
        return float(self.norm_array(v))

    def norm_array(self, v):
        """Values -> 0..1 on the colour scale (NaN stays NaN); logarithmic for "scale": "log"."""
        v = np.asarray(v, np.float64)
        if self.log:
            with np.errstate(invalid="ignore", divide="ignore"):
                x = (np.log(np.maximum(v, self.vmin)) - np.log(self.vmin)) / (np.log(self.vmax) - np.log(self.vmin))
            return np.where(np.isnan(v), np.nan, np.clip(x, 0, 1))
        return np.clip((v - self.vmin) / (self.vmax - self.vmin), 0, 1)

    def tension(self, v, base):
        return 0.0 if base is None or np.isnan(base) else float((v - base) / self.tension_scale)

    def fmt_delta(self, d):
        dec = "".join(ch for ch in self.fmt.split(".")[-1] if ch.isdigit()) or "2"
        return f"{d:+.{dec}f} {self.unit}".strip()

    def _result(self, v, base, clipped=0, near_km=0.0, extra=()):
        lines = list(extra)
        # for anomaly layers the baseline = 0 is obvious, "normal/difference" lines are not needed
        trivial = getattr(self, "baseline_grid", None) is None and base == 0.0
        if base is not None and not np.isnan(base) and not trivial:
            d = v - base
            label = self.base_label[4:] if self.base_label.startswith("the ") else self.base_label
            lines.append((f"{label[0].upper() + label[1:]}: {self.fmt_value(base)}", "ink2"))
            lines.append((f"difference: {self.fmt_delta(d)}", "warm" if d > 0 else "cool" if d < 0 else "ink2"))
        if self.note:
            lines.append((self.note, "muted"))
        return {"value": float(v), "clipped": clipped, "norm": self.norm(v), "tension": self.tension(v, base),
                "missing": False, "near_km": near_km, "base": base, "lines": lines,
                "text": self.fmt_value(v, clipped)}

    def _missing(self):
        return {"value": None, "missing": True, "near_km": None, "lines": [], "text": self.missing_label,
                "norm": None, "tension": 0.0}


class GridLayer(Layer):
    """Map on a regular grid: data[frame][latitude ascending][longitude -180..180].

    times/labels: moment (in years) and label of each frame; None = a single moment.
    cfg["baseline"] = "mean" -> the baseline of each cell = its mean over all frames."""

    def __init__(self, name, cfg, lat, lon, data, times=None, labels=None, baseline_grid=None, clipped=None,
                 search_cells=3, has_dates=False):
        lat, lon, data = np.asarray(lat, float), np.asarray(lon, float), np.asarray(data, np.float32)
        if lat[0] > lat[-1]:                                   # latitudes: south to north
            lat, data = lat[::-1], data[:, ::-1]
            baseline_grid = None if baseline_grid is None else baseline_grid[::-1]
            clipped = None if clipped is None else clipped[:, ::-1]
        lon_w = wrap180(lon)                                   # longitudes: -180..180 ascending
        order = np.argsort(lon_w)
        lon, data = lon_w[order], data[:, :, order]
        baseline_grid = None if baseline_grid is None else baseline_grid[:, order]
        clipped = None if clipped is None else clipped[:, :, order]
        cfg = dict(cfg)
        if "vmin" not in cfg:
            cfg["vmin"], cfg["vmax"] = float(np.nanpercentile(data, 1)), float(np.nanpercentile(data, 99))
        if cfg.get("baseline") == "mean" and baseline_grid is None:
            import warnings
            with warnings.catch_warnings():                    # cells with no data in any frame
                warnings.simplefilter("ignore", RuntimeWarning)
                baseline_grid = np.nanmean(data, axis=0)
        super().__init__(name, cfg)
        self.lat, self.lon, self.data = lat, lon, data
        if times is not None and len(times) > 1:
            self.times = np.asarray(times, float)
            self.labels = list(labels) if labels is not None else [str(int(t)) for t in times]
            self.has_dates = has_dates
        self.baseline_grid = baseline_grid
        self.clipped = clipped
        self.search_cells = search_cells
        # beyond this distance the "nearest value" becomes misleading (e.g. ocean in the middle of Ethiopia)
        self.max_near_km = float(cfg.get("max_near_km", 300))

    def grid(self, t):
        return self.data[self.clamp_frame(t)]

    def extent(self):
        dlat = abs(self.lat[1] - self.lat[0]); dlon = abs(self.lon[1] - self.lon[0])
        return (self.lon[0] - dlon / 2, self.lon[-1] + dlon / 2, self.lat[0] - dlat / 2, self.lat[-1] + dlat / 2)

    def covers(self, lat, lon):
        """Is the point inside the area of the grid (always True for a whole-world grid)?"""
        x0, x1, y0, y1 = self.extent()
        world = x1 - x0 >= 359.0
        return y0 <= lat <= y1 and (world or x0 <= wrap180(lon) <= x1)

    def sample(self, lat, lon, t=None):
        if not self.covers(lat, lon):
            return self._missing()
        g = self.grid(t)
        k = self.clamp_frame(t)
        i = int(np.argmin(np.abs(self.lat - lat)))
        j = int(np.argmin(np.abs(wrap180(self.lon - lon))))
        near = 0.0
        if np.isnan(g[i, j]):                                   # no data -> nearest cell with data
            best, n = None, len(self.lon)
            for di in range(-self.search_cells, self.search_cells + 1):
                for dj in range(-self.search_cells, self.search_cells + 1):
                    ii, jj = i + di, (j + dj) % n
                    if 0 <= ii < len(self.lat) and not np.isnan(g[ii, jj]):
                        d = float(haversine_km(lat, lon, self.lat[ii], self.lon[jj]))
                        if best is None or d < best[0]:
                            best = (d, ii, jj)
            if best is None or best[0] > self.max_near_km:
                return self._missing()
            near, i, j = best
        base = self.baseline_grid[i, j] if self.baseline_grid is not None else self.baseline
        if base is not None and np.isnan(base):
            base = None
        clip = 0 if self.clipped is None else int(self.clipped[k, i, j])
        extra = [(f"{self.missing_label}: nearest value ~{near:.0f} km away", "ink2")] if near else []
        return self._result(float(g[i, j]), None if base is None else float(base), clip, near, extra)

    # --- loading from a folder in the Earth Jukebox format ---
    @classmethod
    def from_folder(cls, folder, overrides=None):
        folder = Path(folder)
        meta_path = next(folder.glob("*.meta.json"))
        name = meta_path.name[: -len(".meta.json")]
        meta = json.loads(meta_path.read_text())
        cfg = {k: meta[k] for k in CFG_KEYS if k in meta}
        cfg.update(BUILTIN.get(folder.name, {}))
        cfg.update(_folder_overrides(folder))
        cfg.update(overrides or {})
        years, dates, lat, lon = meta.get("years"), meta.get("dates"), meta["lat"], meta["lon"]
        n = len(dates) if dates else (len(years) if years else 1)
        shape = (n, len(lat), len(lon))
        if dates:     # frames with dates (e.g. weeks): years is then just the year of each frame
            times = [_decimal_year(d) for d in dates]
            labels = [f"{d[8:10]}.{d[5:7]}.{d[:4]}" for d in dates]
        elif years:
            times, labels = [y + 0.5 for y in years], [str(y) for y in years]
        else:
            times = labels = None
        u8, js = folder / f"{name}.u8.bin", folder / f"{name}.json"
        # The full .json is more accurate (no quantization, no clipping at vmin/vmax), so use it if it is
        # small enough for a quick start; otherwise the compact .u8.bin. Can be set in jukebox.json: "prefer".
        prefer = cfg.get("prefer") or ("json" if js.exists() and js.stat().st_size < 20_000_000 else "u8")
        clipped = None
        if u8.exists() and (prefer == "u8" or not js.exists()):
            b = np.fromfile(u8, np.uint8).reshape(shape)
            vmin, vmax = float(meta["vmin"]), float(meta["vmax"])
            if meta.get("scale") == "log" and vmin > 0:             # bytes spread over the logarithm
                vals = np.exp(np.log(vmin) + b.astype(np.float64) / 254 * (np.log(vmax) - np.log(vmin)))
            else:
                vals = vmin + b.astype(np.float32) / 254 * (vmax - vmin)
            data = np.where(b == 255, np.nan, vals).astype(np.float32)
            clipped = np.where(b == 254, 1, np.where(b == 0, -1, 0)).astype(np.int8)  # value at the edge of the scale
        else:
            raw = json.loads(js.read_text())["data"]
            data = np.array(raw, dtype=np.float32)                 # null -> NaN
            if data.ndim == 2:
                data = data[None]
        lay = cls(folder.name, cfg, lat, lon, data, times, labels, clipped=clipped, has_dates=bool(dates))
        lay.storage = "u8" if clipped is not None else "json"
        return lay


def _decimal_year(date):
    """'2007-03-28' -> 2007.236 (middle of the day as a fraction of the year)."""
    from datetime import date as _d
    y, m, d = int(date[:4]), int(date[5:7]), int(date[8:10])
    doy = (_d(y, m, d) - _d(y, 1, 1)).days
    return y + (doy + 0.5) / (366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365)


def load_sst_layer(path=SST_FILE, month=None):
    """Ocean from NetCDF (NOAA ERSST v5): latest month + the 1971-2000 normal of the same month for every cell."""
    import xarray as xr
    ds = xr.open_dataset(path, engine="h5netcdf")
    t = ds.time.values
    ti = len(t) - 1 if month is None else int(np.argmin(np.abs(t - np.datetime64(month + "-01"))))
    date = str(ds.time.values[ti])
    m = int(date[5:7])
    cur = ds.sst.isel(time=ti).values
    base = ds.sst.sel(time=slice("1971", "2000"))
    base = base.sel(time=base["time.month"] == m).mean("time").values
    lat, lon = ds.lat.values, ds.lon.values
    ds.close()
    cfg = {"title": "Sea surface temperature", "units": "°C", "source": "NOAA ERSST v5",
           "fmt": "{:.1f}", "vmin": -2.0, "vmax": 32.0, "cmap": "sequential",
           "base_label": "the 1971–2000 normal", "tension_scale": 2.5, "missing_label": "land",
           "static_label": f"{MONTHS_EN[m - 1]} {date[:4]}"}
    cfg.update(_folder_overrides(DATA_DIR / "sst"))            # e.g. a sound set chosen for this layer
    return GridLayer("sst", cfg, lat, lon, cur[None], baseline_grid=base)


def _folder_overrides(folder):
    p = Path(folder) / "jukebox.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ----------------------------------------------------------------------------
# Layer discovery
# ----------------------------------------------------------------------------
class LayerSpec:
    def __init__(self, name, title, kind, loader):
        self.name, self.title, self.kind, self._loader, self._layer = name, title, kind, loader, None

    def load(self):
        if self._layer is None:
            self._layer = self._loader()
        return self._layer


def discover_layers(month=None, data_dir=DATA_DIR, sst_file=SST_FILE):
    """All available layers. Nothing heavy is read until a layer is chosen (load())."""
    specs = []
    if Path(sst_file).exists():
        specs.append(LayerSpec("sst", "Sea surface temperature", "map",
                               lambda: load_sst_layer(sst_file, month)))
    for folder in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()) if Path(data_dir).exists() else []:
        ov = _folder_overrides(folder)
        metas = list(folder.glob("*.meta.json"))
        if metas:
            meta = json.loads(metas[0].read_text())
            title = ov.get("title") or BUILTIN.get(folder.name, {}).get("title") or meta.get("description") or folder.name
            specs.append(LayerSpec(folder.name, title, "map", lambda f=folder: GridLayer.from_folder(f)))
    return specs


# ----------------------------------------------------------------------------
# Window with the map of the selected layer
# ----------------------------------------------------------------------------
class LayerView:
    W, H = 1040, 610
    MAP = (64, 70, 1004, 540)   # x0, y0, x1, y1 of the map area

    def __init__(self):
        self.fonts = get_fonts()
        self.rings = []
        if COAST_FILE.exists():
            land = json.loads(COAST_FILE.read_text())
            for f in land["features"]:
                g = f["geometry"]
                for p in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
                    self.rings.append(np.asarray(p[0], float))
        self.cache = {}

    def _px(self, lat, lon, ext):
        x0, y0, x1, y1 = self.MAP
        return (x0 + (np.asarray(lon) - ext[0]) / (ext[1] - ext[0]) * (x1 - x0),
                y0 + (ext[3] - np.asarray(lat)) / (ext[3] - ext[2]) * (y1 - y0))

    def _header(self, d, layer, year):
        d.text((self.MAP[0], 14), f"{layer.title} · {layer.time_label(year)}", font=self.fonts["m"], fill=hex_rgb(INK))
        sub = " · ".join(s for s in (layer.units, layer.source) if s)
        d.text((self.MAP[0], 40), sub, font=self.fonts["xs"], fill=hex_rgb(MUTED))

    def _colorbar(self, d, img, layer, y):
        from PIL import Image
        w = 420
        x = (self.W - w) // 2
        strip = np.repeat(layer.lut[np.linspace(0, 255, w).astype(int)][None], 12, axis=0)
        img.paste(Image.fromarray(strip), (x, y))
        mid = np.sqrt(layer.vmin * layer.vmax) if layer.log else (layer.vmin + layer.vmax) / 2
        ticks = [layer.vmin, mid, layer.vmax]
        for v in ticks:
            px = x + int(layer.norm(v) * (w - 1))
            s = layer.fmt.format(v)
            d.text((px - d.textlength(s, font=self.fonts["xs"]) / 2, y + 16), s, font=self.fonts["xs"],
                   fill=hex_rgb(MUTED))
        d.text((x + w + 12, y - 2), layer.unit, font=self.fonts["xs"], fill=hex_rgb(INK2))

    def _base_map(self, layer, year):
        from PIL import Image, ImageDraw
        x0, y0, x1, y1 = self.MAP
        ext = layer.extent()
        g = layer.grid(year)[::-1]                             # north at the top
        nrm = layer.norm_array(g)
        rgb = layer.lut[np.nan_to_num(nrm * 255).astype(int)]
        rgb[np.isnan(g)] = hex_rgb(LANDC)
        rgb = cv2.resize(rgb, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        canvas = np.full((self.H, self.W, 3), hex_rgb(SURFACE), np.uint8)
        canvas[y0:y1, x0:x1] = rgb
        bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        segs = [np.int32(np.stack(self._px(r[:, 1], r[:, 0], ext), 1)) for r in self.rings]
        cv2.polylines(bgr, segs, True, (78, 81, 82), 1, cv2.LINE_AA)
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (217, 224, 225), 1)
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        self._header(d, layer, year)
        for lo in range(-120, 181, 60):
            px, _ = self._px(0, lo, ext)
            s = f"{abs(lo)}°{'E' if lo > 0 else 'W' if lo < 0 else ''}"
            d.text((px - d.textlength(s, font=self.fonts["xs"]) / 2, y1 + 4), s, font=self.fonts["xs"], fill=hex_rgb(MUTED))
        for la in range(-60, 61, 30):
            _, py = self._px(la, 0, ext)
            s = f"{abs(la)}°{'N' if la > 0 else 'S' if la < 0 else ''}"
            d.text((x0 - 8 - d.textlength(s, font=self.fonts["xs"]), py - 8), s, font=self.fonts["xs"], fill=hex_rgb(MUTED))
        self._colorbar(d, img, layer, y1 + 32)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), ext

    def render(self, layer, t, pos=None, trail=(), dwell=0.0, label=None):
        key = (layer.name, layer.clamp_frame(t))
        if key not in self.cache:
            self.cache[key] = self._base_map(layer, t)
        base, ext = self.cache[key]
        img = base.copy()
        pts = [tuple(int(v) for v in self._px(la, lo, ext)) for la, lo in trail]
        for (a, b), (c, e) in zip(pts, pts[1:]):
            if abs(c - a) < (self.MAP[2] - self.MAP[0]) / 4:   # not across the date line
                cv2.line(img, (a, b), (c, e), (120, 120, 120), 2, cv2.LINE_AA)
        if pos is not None:
            self._marker(img, tuple(int(v) for v in self._px(pos[0], pos[1], ext)), dwell, label)
        return img

    @staticmethod
    def _marker(img, p, dwell, label):
        cv2.circle(img, p, 11, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, 7, (20, 20, 20), -1, cv2.LINE_AA)
        if dwell > 0:
            col = (12, 163, 12) if dwell >= 1 else (52, 104, 235)
            cv2.ellipse(img, p, (18, 18), -90, 0, 360 * min(dwell, 1.0), col, 3, cv2.LINE_AA)
        if label:
            s = label.replace("°", "").replace("≥ ", ">").replace("≤ ", "<")
            (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            x = p[0] + 22 if p[0] + 22 + tw < img.shape[1] - 4 else p[0] - 22 - tw   # near the right edge - to the left
            for col, th in (((255, 255, 255), 4), ((20, 20, 20), 2)):
                cv2.putText(img, s, (x, p[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Export: any array/NetCDF -> layer folder in the Earth Jukebox format
# ----------------------------------------------------------------------------
def write_layer(out_dir, name, years, lat, lon, data, units="", title="", source="", vmin=None, vmax=None,
                diverging=False, extra_cfg=None, dates=None, scale="linear"):
    """data: [frame][latitude][longitude] (NaN = no data); years - the year of each frame (None = a single moment);
    dates - optional frame dates 'YYYY-MM-DD' (for monthly/weekly data).
    scale="log" - the bytes are spread evenly over the logarithm (vmin must be > 0).
    Writes <name>.meta.json, <name>.u8.bin, jukebox.json."""
    data = np.asarray(data, np.float32)
    if vmin is None or vmax is None:
        if diverging:
            m = float(np.nanpercentile(np.abs(data), 99)); vmin, vmax = -m, m
        else:
            vmin, vmax = float(np.nanpercentile(data, 1)), float(np.nanpercentile(data, 99))
    folder = Path(out_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    log = scale == "log" and vmin > 0
    if log:
        x = (np.log(np.clip(data, vmin, vmax)) - np.log(vmin)) / (np.log(vmax) - np.log(vmin))
    else:
        x = (np.clip(data, vmin, vmax) - vmin) / (vmax - vmin)
    b = np.where(np.isnan(data), 255, np.round(np.nan_to_num(x) * 254)).astype(np.uint8)
    b.tofile(folder / f"{name}.u8.bin")
    meta = {"layer": name, "description": title or name, "source": source, "units": units, "static": years is None,
            "years": None if years is None else [int(y) for y in years],
            "lat": [float(v) for v in lat], "lon": [float(v) for v in lon], "axis_order": "[year][lat][lon]",
            "vmin": float(vmin), "vmax": float(vmax), "null_meaning": "no data"}
    if log:
        meta["scale"] = "log"
    if dates is not None:
        meta["dates"] = [str(d)[:10] for d in dates]
    (folder / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    cfg = {"title": title or name, "units": units, "source": source}
    cfg.update(extra_cfg or {})
    (folder / "jukebox.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    return folder


def netcdf_to_layer(nc_path, var, name, out_dir=DATA_DIR, title="", units=None, source="", annual=True,
                    diverging=False, level=0):
    """NetCDF with a (time, lat, lon) variable -> layer. Time is averaged by year (annual=True)."""
    import xarray as xr
    try:
        ds = xr.open_dataset(nc_path, engine="h5netcdf")
    except Exception:
        ds = xr.open_dataset(nc_path)
    da = ds[var]
    ren = {}
    for d in da.dims:
        dl = d.lower()
        if dl in ("latitude", "y", "nav_lat"):
            ren[d] = "lat"
        elif dl in ("longitude", "x", "nav_lon"):
            ren[d] = "lon"
    da = da.rename(ren)
    for d in list(da.dims):                                    # extra dimensions (level, ensemble) -> first index
        if d not in ("time", "lat", "lon"):
            da = da.isel({d: level})
    years = dates = None
    if "time" in da.dims:
        if annual:
            da = da.groupby("time.year").mean("time")
            years = da["year"].values
            da = da.transpose("year", "lat", "lon")
        else:                                                  # every time step is a separate frame with a date
            da = da.transpose("time", "lat", "lon")
            dates = [str(t)[:10] for t in da["time"].values]
            years = [int(d[:4]) for d in dates]
        data = da.values
    else:
        data = da.transpose("lat", "lon").values[None]
    lat, lon = da["lat"].values.astype(float), wrap180(da["lon"].values.astype(float))
    order = np.argsort(lon)
    data, lon = data[:, :, order], lon[order]
    if lat[0] > lat[-1]:
        data, lat = data[:, ::-1], lat[::-1]
    units = units if units is not None else str(da.attrs.get("units", ""))
    return write_layer(out_dir, name, years, lat, lon, data, units=units,
                       title=title or str(da.attrs.get("long_name", name)),
                       source=source or str(ds.attrs.get("title", ds.attrs.get("dataset_title", ""))),
                       diverging=diverging, dates=dates)


def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show the layers found")
    p = sub.add_parser("from-netcdf", help="convert a NetCDF file into a layer in data/<name>/")
    p.add_argument("nc"); p.add_argument("--var", required=True); p.add_argument("--name", required=True)
    p.add_argument("--title", default=""); p.add_argument("--units"); p.add_argument("--source", default="")
    p.add_argument("--monthly", action="store_true", help="do not average by year: every time step is a separate frame with a date")
    p.add_argument("--diverging", action="store_true", help="symmetric scale around 0 (for anomalies)")
    p.add_argument("--out", default=str(DATA_DIR))
    a = ap.parse_args()
    if a.cmd == "list":
        for k, s in enumerate(discover_layers(), 1):
            lay = s.load()
            when = (lay.time_label(0) or "single moment") if not lay.dynamic else \
                f"{lay.labels[0]} … {lay.labels[-1]} ({lay.n_frames} frames)"
            print(f"{k}. {s.name:20s} {s.title} | {lay.units} | {when}")
    else:
        f = netcdf_to_layer(a.nc, a.var, a.name, a.out, a.title, a.units, a.source, not a.monthly, a.diverging)
        print(f"Done: {f}  (appears in finger_map.py automatically if --out = data/)")


if __name__ == "__main__":
    _cli()
