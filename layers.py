"""Шари даних Earth Jukebox: будь-яка карта чи крива -> значення в точці -> колір і звук.

Шар має один інтерфейс:
    layer.sample(lat, lon, year) -> значення, нормоване значення 0..1 (висота ноти),
                                    "напруга" (наскільки вище/нижче базового рівня: биття,
                                    пульсація), рядки для панелі
    layer.grid(year)             -> 2D-масив для карти (None для кривої)

Звідки беруться шари (автоматично, див. discover_layers):
  * sst.mnmean.nc                       -> "sst": температура поверхні океану (NOAA ERSST v5)
  * data/<назва>/<назва>.meta.json + .u8.bin (або .json з data)  -> карта за роками
  * data/<назва>/<назва>.json з years/values (без lat/lon)        -> крива (напр. CO2)
  * data/<назва>/jukebox.json (необов'язково) -> назва, кольори, базовий рівень для звуку

Як додати новий шар - див. LAYERS.md. Конвертер з NetCDF:
    .venv/bin/python layers.py from-netcdf файл.nc --var назва_змінної --name мій_шар
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
    "diverging": ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#7f1d1d"],   # синій - сірий нуль - червоний
    "sequential": ["#fdeee4", "#f6b48f", "#eb6834", "#b7461c", "#6e2509"],  # світлий -> темний помаранчевий
    "blues": ["#e8f1fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"],
    "greens": ["#eaf6e6", "#a9d99b", "#4fa54a", "#257a2c", "#0e4a19"],
}
MONTHS_UA = ["січень", "лютий", "березень", "квітень", "травень", "червень", "липень",
             "серпень", "вересень", "жовтень", "листопад", "грудень"]

# Вбудовані налаштування відомих шарів. Будь-який ключ можна перекрити в data/<назва>/jukebox.json.
BUILTIN = {
    "temperature": {
        "title": "Аномалія температури (суша + океан)",
        "fmt": "{:+.2f}", "default_year": 2025, "tension_scale": 2.0,
        "note": "відносно норми 1951–1980",
    },
    "co2_curve": {
        "title": "CO₂ в атмосфері",
        "fmt": "{:.1f}", "default_year": 2025, "cmap": "blues",
        "baseline": 280.0, "base_label": "доіндустріальний рівень", "tension_scale": 170.0,
        "note": "однаково для всієї планети",
    },
}
CFG_KEYS = ("title", "units", "source", "description", "note", "fmt", "vmin", "vmax", "cmap", "baseline",
            "base_label", "tension_scale", "default_year", "missing_label", "static_label")


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
    """cmap: назва з CMAPS або список hex-кольорів -> таблиця 256x3 (RGB, uint8)."""
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
# Шари
# ----------------------------------------------------------------------------
class Layer:
    """Спільна логіка: форматування, нормування для висоти ноти, "напруга" для биття/пульсу."""
    spatial = True

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
        self.baseline = cfg.get("baseline", 0.0 if diverging else None)
        self.base_label = cfg.get("base_label", "базовий рівень")
        self.tension_scale = float(cfg.get("tension_scale") or
                                   (max(abs(self.vmin), abs(self.vmax)) if diverging else (self.vmax - self.vmin) / 2))
        self.missing_label = cfg.get("missing_label", "немає даних")
        self.years = None
        self.partial_years = set()

    # --- час ---
    def default_year(self):
        if self.years is None:
            return None
        y = self.cfg.get("default_year")
        if y is not None and self.years[0] <= y <= self.years[-1]:
            return int(y)
        full = [y for y in self.years if y not in self.partial_years]
        return int(full[-1] if full else self.years[-1])

    def clamp_year(self, year):
        if self.years is None:
            return None
        if year is None:
            return self.default_year()
        return int(np.clip(year, self.years[0], self.years[-1]))

    def year_index(self, year):
        return 0 if self.years is None else int(np.argmin(np.abs(self.years - self.clamp_year(year))))

    def time_label(self, year):
        if self.years is None:
            return self.cfg.get("static_label", "")
        y = int(self.years[self.year_index(year)])
        return f"{y} (неповний рік)" if y in self.partial_years else str(y)

    # --- значення ---
    def fmt_value(self, v, clipped=0):
        s = f"{self.fmt.format(v)} {self.unit}".strip()
        return ("≥ " if clipped > 0 else "≤ " if clipped < 0 else "") + s

    def norm(self, v):
        return float(np.clip((v - self.vmin) / (self.vmax - self.vmin), 0, 1))

    def tension(self, v, base):
        return 0.0 if base is None or np.isnan(base) else float((v - base) / self.tension_scale)

    def fmt_delta(self, d):
        dec = "".join(ch for ch in self.fmt.split(".")[-1] if ch.isdigit()) or "2"
        return f"{d:+.{dec}f} {self.unit}".strip()

    def _result(self, v, base, clipped=0, near_km=0.0, extra=()):
        lines = list(extra)
        # для шарів-аномалій базовий рівень = 0 і так очевидний, рядки "норма/різниця" не потрібні
        trivial = getattr(self, "baseline_grid", None) is None and base == 0.0
        if base is not None and not np.isnan(base) and not trivial:
            d = v - base
            lines.append((f"{self.base_label}: {self.fmt_value(base)}", "ink2"))
            lines.append((f"різниця: {self.fmt_delta(d)}", "warm" if d > 0 else "cool" if d < 0 else "ink2"))
        if self.note:
            lines.append((self.note, "muted"))
        return {"value": float(v), "clipped": clipped, "norm": self.norm(v), "tension": self.tension(v, base),
                "missing": False, "near_km": near_km, "base": base, "lines": lines,
                "text": self.fmt_value(v, clipped)}

    def _missing(self):
        return {"value": None, "missing": True, "near_km": None, "lines": [], "text": self.missing_label,
                "norm": None, "tension": 0.0}


class GridLayer(Layer):
    """Карта на регулярній сітці: frames[рік][широта зростає][довгота -180..180]."""

    def __init__(self, name, cfg, lat, lon, frames, years=None, baseline_grid=None, clipped=None, search_cells=3):
        lat, lon, frames = np.asarray(lat, float), np.asarray(lon, float), np.asarray(frames, np.float32)
        if lat[0] > lat[-1]:                                   # широти: з півдня на північ
            lat, frames = lat[::-1], frames[:, ::-1]
            baseline_grid = None if baseline_grid is None else baseline_grid[::-1]
            clipped = None if clipped is None else clipped[:, ::-1]
        lon_w = wrap180(lon)                                   # довготи: -180..180 за зростанням
        order = np.argsort(lon_w)
        lon, frames = lon_w[order], frames[:, :, order]
        baseline_grid = None if baseline_grid is None else baseline_grid[:, order]
        clipped = None if clipped is None else clipped[:, :, order]
        cfg = dict(cfg)
        if "vmin" not in cfg:
            cfg["vmin"], cfg["vmax"] = float(np.nanpercentile(frames, 1)), float(np.nanpercentile(frames, 99))
        super().__init__(name, cfg)
        self.lat, self.lon, self.frames = lat, lon, frames
        self.years = None if years is None else np.asarray(years, int)
        self.baseline_grid = baseline_grid
        self.clipped = clipped
        self.search_cells = search_cells

    def grid(self, year):
        return self.frames[self.year_index(year)]

    def extent(self):
        dlat = abs(self.lat[1] - self.lat[0]); dlon = abs(self.lon[1] - self.lon[0])
        return (self.lon[0] - dlon / 2, self.lon[-1] + dlon / 2, self.lat[0] - dlat / 2, self.lat[-1] + dlat / 2)

    def sample(self, lat, lon, year=None):
        g = self.grid(year)
        k = self.year_index(year)
        i = int(np.argmin(np.abs(self.lat - lat)))
        j = int(np.argmin(np.abs(wrap180(self.lon - lon))))
        near = 0.0
        if np.isnan(g[i, j]):                                   # немає даних -> найближча клітинка з даними
            best, n = None, len(self.lon)
            for di in range(-self.search_cells, self.search_cells + 1):
                for dj in range(-self.search_cells, self.search_cells + 1):
                    ii, jj = i + di, (j + dj) % n
                    if 0 <= ii < len(self.lat) and not np.isnan(g[ii, jj]):
                        d = float(haversine_km(lat, lon, self.lat[ii], self.lon[jj]))
                        if best is None or d < best[0]:
                            best = (d, ii, jj)
            if best is None:
                return self._missing()
            near, i, j = best
        base = self.baseline_grid[i, j] if self.baseline_grid is not None else self.baseline
        clip = 0 if self.clipped is None else int(self.clipped[k, i, j])
        extra = [(f"{self.missing_label}: найближче значення ~{near:.0f} км", "ink2")] if near else []
        return self._result(float(g[i, j]), None if base is None else float(base), clip, near, extra)

    # --- завантаження з папки у форматі Earth Jukebox ---
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
        years, lat, lon = meta.get("years"), meta["lat"], meta["lon"]
        shape = (len(years) if years else 1, len(lat), len(lon))
        u8 = folder / f"{name}.u8.bin"
        clipped = None
        if u8.exists():
            b = np.fromfile(u8, np.uint8).reshape(shape)
            vmin, vmax = float(meta["vmin"]), float(meta["vmax"])
            frames = np.where(b == 255, np.nan, vmin + b.astype(np.float32) / 254 * (vmax - vmin)).astype(np.float32)
            clipped = np.where(b == 254, 1, np.where(b == 0, -1, 0)).astype(np.int8)  # значення на межі шкали
        else:
            data = json.loads((folder / f"{name}.json").read_text())["data"]
            frames = np.array([[[np.nan if v is None else v for v in row] for row in fr] for fr in
                               (data if years else [data])], np.float32)
        return cls(folder.name, cfg, lat, lon, frames, years, clipped=clipped)


class CurveLayer(Layer):
    """Одне значення на рік для всієї планети (напр. CO2). Місце пальця не важливе, важливий рік."""
    spatial = False

    def __init__(self, name, cfg, years, values, monthly=None):
        values = np.asarray(values, float)
        cfg = dict(cfg)
        if "vmin" not in cfg:
            pad = 0.02 * (values.max() - values.min())
            cfg["vmin"], cfg["vmax"] = float(values.min() - pad), float(values.max() + pad)
        super().__init__(name, cfg)
        self.years, self.values, self.monthly = np.asarray(years, int), values, monthly
        if monthly:
            ys = np.asarray(monthly["year"])
            self.partial_years = {int(y) for y in np.unique(ys) if (ys == y).sum() < 12}

    def grid(self, year):
        return None

    def sample(self, lat=0.0, lon=0.0, year=None):
        k = self.year_index(year)
        return self._result(float(self.values[k]), None if self.baseline is None else float(self.baseline))

    @classmethod
    def from_folder(cls, folder, overrides=None):
        folder = Path(folder)
        path = folder / f"{folder.name}.json"
        if not path.exists():
            path = next(p for p in folder.glob("*.json") if p.name != "jukebox.json")
        d = json.loads(path.read_text())
        cfg = {k: d[k] for k in CFG_KEYS if k in d}
        cfg.update(BUILTIN.get(folder.name, {}))
        cfg.update(_folder_overrides(folder))
        cfg.update(overrides or {})
        return cls(folder.name, cfg, d["years"], d["values"], d.get("monthly"))


def load_sst_layer(path=SST_FILE, month=None):
    """Океан з NetCDF (NOAA ERSST v5): останній місяць + норма того ж місяця 1971-2000 для кожної клітинки."""
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
    cfg = {"title": "Температура поверхні океану", "units": "°C", "source": "NOAA ERSST v5",
           "fmt": "{:.1f}", "vmin": -2.0, "vmax": 32.0, "cmap": "sequential",
           "base_label": "норма 1971–2000", "tension_scale": 2.5, "missing_label": "суша",
           "static_label": f"{MONTHS_UA[m - 1]} {date[:4]}"}
    return GridLayer("sst", cfg, lat, lon, cur[None], None, baseline_grid=base)


def _folder_overrides(folder):
    p = Path(folder) / "jukebox.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ----------------------------------------------------------------------------
# Автопошук шарів
# ----------------------------------------------------------------------------
class LayerSpec:
    def __init__(self, name, title, kind, loader):
        self.name, self.title, self.kind, self._loader, self._layer = name, title, kind, loader, None

    def load(self):
        if self._layer is None:
            self._layer = self._loader()
        return self._layer


def discover_layers(month=None, data_dir=DATA_DIR, sst_file=SST_FILE):
    """Усі доступні шари. Нічого важкого не читає, доки шар не обрано (load())."""
    specs = []
    if Path(sst_file).exists():
        specs.append(LayerSpec("sst", "Температура поверхні океану", "карта",
                               lambda: load_sst_layer(sst_file, month)))
    for folder in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()) if Path(data_dir).exists() else []:
        ov = _folder_overrides(folder)
        metas = list(folder.glob("*.meta.json"))
        if metas:
            meta = json.loads(metas[0].read_text())
            title = ov.get("title") or BUILTIN.get(folder.name, {}).get("title") or meta.get("description") or folder.name
            specs.append(LayerSpec(folder.name, title, "карта", lambda f=folder: GridLayer.from_folder(f)))
            continue
        jsons = [p for p in folder.glob("*.json") if p.name != "jukebox.json" and p.stat().st_size < 20_000_000]
        for p in jsons:
            d = json.loads(p.read_text())
            if isinstance(d, dict) and "years" in d and "values" in d and "lat" not in d:
                title = ov.get("title") or BUILTIN.get(folder.name, {}).get("title") or d.get("layer") or folder.name
                specs.append(LayerSpec(folder.name, title, "крива", lambda f=folder: CurveLayer.from_folder(f)))
                break
    return specs


# ----------------------------------------------------------------------------
# Вікно з картою (або кривою) обраного шару
# ----------------------------------------------------------------------------
class LayerView:
    W, H = 1040, 610
    MAP = (64, 70, 1004, 540)   # x0, y0, x1, y1 області карти

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
        ticks = [layer.vmin, (layer.vmin + layer.vmax) / 2, layer.vmax]
        for v in ticks:
            px = x + int((v - layer.vmin) / (layer.vmax - layer.vmin) * (w - 1))
            s = layer.fmt.format(v)
            d.text((px - d.textlength(s, font=self.fonts["xs"]) / 2, y + 16), s, font=self.fonts["xs"],
                   fill=hex_rgb(MUTED))
        d.text((x + w + 12, y - 2), layer.unit, font=self.fonts["xs"], fill=hex_rgb(INK2))

    def _base_map(self, layer, year):
        from PIL import Image, ImageDraw
        x0, y0, x1, y1 = self.MAP
        ext = layer.extent()
        g = layer.grid(year)[::-1]                             # північ зверху
        nrm = np.clip((g - layer.vmin) / (layer.vmax - layer.vmin), 0, 1)
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

    def _base_curve(self, layer):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(figsize=(self.W / 100, self.H / 100), dpi=100, facecolor=SURFACE)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.08, 0.1, 0.89, 0.76])
        ax.set_facecolor(SURFACE)
        if layer.monthly:
            mo = layer.monthly
            t = np.array(mo["year"]) + (np.array(mo["month"]) - 0.5) / 12
            ax.plot(t, mo[[k for k in mo if k not in ("year", "month")][0]], color=CMAPS["blues"][1], lw=1)
        full = np.array([y not in layer.partial_years for y in layer.years])
        ax.plot(layer.years[full] + 0.5, layer.values[full], color=CMAPS["blues"][3], lw=2)
        if layer.baseline is not None:
            ax.axhline(layer.baseline, color=GRIDC, lw=1, ls="--")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRIDC)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="y", color=GRIDC, lw=0.6)
        ax.set_ylabel(layer.unit, color=INK2)
        canvas.draw()
        img = cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        H = img.shape[0]
        tr = ax.transData

        def to_px(year, value):
            x, y = tr.transform((year + 0.5, value))
            return int(x), int(H - y)
        (_, ytop), (_, ybot) = tr.transform([(0, ax.get_ylim()[1]), (0, ax.get_ylim()[0])])
        return img, to_px, (int(H - ytop), int(H - ybot))

    def render(self, layer, year, pos=None, trail=(), dwell=0.0, label=None):
        if layer.spatial:
            key = (layer.name, layer.year_index(year))
            if key not in self.cache:
                self.cache[key] = self._base_map(layer, year)
            base, ext = self.cache[key]
            img = base.copy()
            pts = [tuple(int(v) for v in self._px(la, lo, ext)) for la, lo in trail]
            for (a, b), (c, e) in zip(pts, pts[1:]):
                if abs(c - a) < (self.MAP[2] - self.MAP[0]) / 4:   # не через лінію зміни дат
                    cv2.line(img, (a, b), (c, e), (120, 120, 120), 2, cv2.LINE_AA)
            if pos is not None:
                self._marker(img, tuple(int(v) for v in self._px(pos[0], pos[1], ext)), dwell, label)
            return img
        key = (layer.name, "curve")
        if key not in self.cache:
            from PIL import Image
            img, to_px, yr = self._base_curve(layer)
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            self.cache[key] = (pil, to_px, yr)
        pil, to_px, (ytop, ybot) = self.cache[key]
        from PIL import ImageDraw
        frame = pil.copy()
        d = ImageDraw.Draw(frame)
        d.rectangle([0, 0, self.W, 62], fill=hex_rgb(SURFACE))
        self._header(d, layer, year)
        img = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        k = layer.year_index(year)
        p = to_px(int(layer.years[k]), float(layer.values[k]))
        cv2.line(img, (p[0], ytop), (p[0], ybot), (200, 200, 200), 1, cv2.LINE_AA)
        self._marker(img, p, dwell, label)
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
            x = p[0] + 22 if p[0] + 22 + tw < img.shape[1] - 4 else p[0] - 22 - tw   # біля правого краю - ліворуч
            for col, th in (((255, 255, 255), 4), ((20, 20, 20), 2)):
                cv2.putText(img, s, (x, p[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Експорт: будь-який масив/NetCDF -> папка шару у форматі Earth Jukebox
# ----------------------------------------------------------------------------
def write_layer(out_dir, name, years, lat, lon, data, units="", title="", source="", vmin=None, vmax=None,
                diverging=False, extra_cfg=None):
    """data: [роки][широта][довгота] (NaN = немає даних). Пише <name>.meta.json, <name>.u8.bin, jukebox.json."""
    data = np.asarray(data, np.float32)
    if vmin is None or vmax is None:
        if diverging:
            m = float(np.nanpercentile(np.abs(data), 99)); vmin, vmax = -m, m
        else:
            vmin, vmax = float(np.nanpercentile(data, 1)), float(np.nanpercentile(data, 99))
    folder = Path(out_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    b = np.where(np.isnan(data), 255, np.round((np.clip(data, vmin, vmax) - vmin) / (vmax - vmin) * 254)).astype(np.uint8)
    b.tofile(folder / f"{name}.u8.bin")
    meta = {"layer": name, "description": title or name, "source": source, "units": units, "static": years is None,
            "years": None if years is None else [int(y) for y in years],
            "lat": [float(v) for v in lat], "lon": [float(v) for v in lon], "axis_order": "[year][lat][lon]",
            "vmin": float(vmin), "vmax": float(vmax), "null_meaning": "no data"}
    (folder / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    cfg = {"title": title or name, "units": units, "source": source}
    cfg.update(extra_cfg or {})
    (folder / "jukebox.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    return folder


def netcdf_to_layer(nc_path, var, name, out_dir=DATA_DIR, title="", units=None, source="", annual=True,
                    diverging=False, level=0):
    """NetCDF зі змінною (time, lat, lon) -> шар. Час усереднюється по роках (annual=True)."""
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
    for d in list(da.dims):                                    # зайві виміри (рівень, ансамбль) -> перший індекс
        if d not in ("time", "lat", "lon"):
            da = da.isel({d: level})
    years = None
    if "time" in da.dims:
        if annual:
            da = da.groupby("time.year").mean("time")
            years = da["year"].values
            da = da.transpose("year", "lat", "lon")
        else:
            da = da.transpose("time", "lat", "lon")
            years = np.arange(da.sizes["time"])
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
                       diverging=diverging)


def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="показати знайдені шари")
    p = sub.add_parser("from-netcdf", help="перетворити NetCDF на шар у data/<name>/")
    p.add_argument("nc"); p.add_argument("--var", required=True); p.add_argument("--name", required=True)
    p.add_argument("--title", default=""); p.add_argument("--units"); p.add_argument("--source", default="")
    p.add_argument("--monthly", action="store_true", help="не усереднювати по роках (кожен крок часу = 'рік' 0,1,2..)")
    p.add_argument("--diverging", action="store_true", help="симетрична шкала навколо 0 (для аномалій)")
    p.add_argument("--out", default=str(DATA_DIR))
    a = ap.parse_args()
    if a.cmd == "list":
        for k, s in enumerate(discover_layers(), 1):
            lay = s.load()
            yrs = "статичний" if lay.years is None else f"{lay.years[0]}–{lay.years[-1]}"
            print(f"{k}. {s.name:14s} [{s.kind}] {s.title} | {lay.units} | {yrs}")
    else:
        f = netcdf_to_layer(a.nc, a.var, a.name, a.out, a.title, a.units, a.source, not a.monthly, a.diverging)
        print(f"Готово: {f}  (з'явиться в finger_map.py автоматично, якщо --out = data/)")


if __name__ == "__main__":
    _cli()
