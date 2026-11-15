"""Spoken description of a point: where (country / sea / natural region), when (year or date), what the data says.

    geo = Geocoder()                          # offline, Natural Earth outlines with English names
    text = compose(layer, t, lat, lon, sample, geo.locate(lat, lon))
    Speaker().speak(text)                     # Kokoro neural voice (offline), else macOS `say`

Example: "Central Ethiopia. Eastern Africa. Natural region: Ethiopian Highlands. Year 2025.
Temperature anomaly: plus 1.28 degrees. This is the departure from the 1951 to 1980 average."

Data (in data/, downloaded once):
  ne_50m_admin_0_countries.geojson, ne_50m_geography_marine_polys.geojson,
  ne_50m_geography_regions_polys.geojson  - https://github.com/nvkelso/natural-earth-vector
"""
import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
NE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/{}.geojson"
NE_FILES = {"countries": "ne_50m_admin_0_countries", "marine": "ne_50m_geography_marine_polys",
            "regions": "ne_50m_geography_regions_polys"}

PHYSICAL = {"Desert", "Range/mtn", "Plateau", "Basin", "Plain", "Tundra", "Peninsula", "Pen/cape", "Delta",
            "Lowland", "Valley", "Wetlands", "Foothills"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
PARTS = {(0, 0): "central", (1, 0): "northern", (-1, 0): "southern", (0, 1): "eastern", (0, -1): "western",
         (1, 1): "north-eastern", (1, -1): "north-western", (-1, 1): "south-eastern", (-1, -1): "south-western"}
# Voices that sound natural, best first. Premium/Enhanced voices (if installed) are preferred automatically.
PREFERRED_VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Tessa", "Rishi"]


def _cap(s):
    """Upper-case only the first letter (str.capitalize() would lower-case month names)."""
    return s[:1].upper() + s[1:]


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _clean(name):
    name = (name or "").strip()
    return name.title() if name.isupper() else name             # "SAHEL" -> "Sahel"


def ensure_data(data_dir=DATA_DIR):
    """Download the Natural Earth outlines if missing. -> True if all files are present."""
    import urllib.request
    ok = True
    for f in NE_FILES.values():
        p = Path(data_dir) / f"{f}.geojson"
        if not p.exists():
            try:
                print(f"  downloading {p.name} ...")
                urllib.request.urlretrieve(NE_URL.format(f), p)
            except Exception as e:
                print(f"  could not download {p.name}: {e}")
                ok = False
    return ok


class _Shapes:
    """Polygons with names + fast point-in-polygon."""

    def __init__(self, path, name_keys, filt=None, extra=None):
        self.items = []
        if not Path(path).exists():
            return
        for f in json.loads(Path(path).read_text())["features"]:
            props = f["properties"]
            if filt and not filt(props):
                continue
            name = _clean(next((props.get(k) for k in name_keys if props.get(k)), ""))
            if not name:
                continue
            g = f["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            rings = [np.asarray(p[0], float) for p in polys]
            allpts = np.vstack(rings)
            area_of = lambda r: abs(np.dot(r[:, 0], np.roll(r[:, 1], 1)) - np.dot(r[:, 1], np.roll(r[:, 0], 1))) / 2
            biggest = max(rings, key=area_of)
            self.items.append({
                "name": name, "props": props, "area": sum(area_of(r) for r in rings),
                "bbox": (allpts[:, 0].min(), allpts[:, 1].min(), allpts[:, 0].max(), allpts[:, 1].max()),
                "paths": [(MplPath(r), (r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max())) for r in rings],
                "main_bbox": (biggest[:, 0].min(), biggest[:, 1].min(), biggest[:, 0].max(), biggest[:, 1].max()),
                "verts": allpts[:: max(1, len(allpts) // 400)],
                **(extra(props) if extra else {}),
            })

    def containing(self, lat, lon):
        hits = []
        for it in self.items:
            x0, y0, x1, y1 = it["bbox"]
            if x0 <= lon <= x1 and y0 <= lat <= y1:
                for path, (a, b, c, d) in it["paths"]:
                    if a <= lon <= c and b <= lat <= d and path.contains_point((lon, lat)):
                        hits.append(it)
                        break
        return sorted(hits, key=lambda it: it["area"])               # smallest (most specific) first

    def nearest(self, lat, lon, max_km):
        best = None
        pad = max_km / 111.0 + 1
        for it in self.items:
            x0, y0, x1, y1 = it["bbox"]
            if not (x0 - pad * 2 <= lon <= x1 + pad * 2 and y0 - pad <= lat <= y1 + pad):
                continue
            d = float(_haversine_km(lat, lon, it["verts"][:, 1], it["verts"][:, 0]).min())
            if d <= max_km and (best is None or d < best[0]):
                best = (d, it)
        return best


class Geocoder:
    def __init__(self, data_dir=DATA_DIR, download=True):
        if download:
            ensure_data(data_dir)
        d = Path(data_dir)
        self.countries = _Shapes(d / f"{NE_FILES['countries']}.geojson", ["NAME", "NAME_EN"],
                                 extra=lambda p: {"subregion": p.get("SUBREGION") or ""})
        self.marine = _Shapes(d / f"{NE_FILES['marine']}.geojson", ["name_en", "name"])
        self.regions = _Shapes(d / f"{NE_FILES['regions']}.geojson", ["NAME_EN", "NAME"],
                               filt=lambda p: p.get("FEATURECLA") in PHYSICAL)
        self.ready = bool(self.countries.items)

    @staticmethod
    def _part(it, lat, lon):
        """Which part of the country, by position in the main territory's box (large countries only)."""
        x0, y0, x1, y1 = it["main_bbox"]
        if (x1 - x0) < 5 and (y1 - y0) < 5:
            return ""
        if not (x0 <= lon <= x1 and y0 <= lat <= y1):
            return ""
        fx, fy = (lon - x0) / (x1 - x0), (lat - y0) / (y1 - y0)
        k = lambda f: 1 if f > 0.62 else -1 if f < 0.38 else 0
        return PARTS[(k(fy), k(fx))]

    def locate(self, lat, lon):
        out = {"lat": lat, "lon": lon, "country": None, "part": "", "subregion": "", "sea": None,
               "near_country": None, "near_km": None, "region": None}
        c = self.countries.containing(lat, lon)
        if c:
            it = c[0]
            out.update(country=it["name"], part=self._part(it, lat, lon), subregion=it["subregion"])
        else:
            m = self.marine.containing(lat, lon)
            if m:
                out["sea"] = m[0]["name"]
            n = self.countries.nearest(lat, lon, 700)
            if n:
                out["near_country"], out["near_km"] = n[1]["name"], n[0]
                out["subregion"] = n[1]["subregion"]
        r = self.regions.containing(lat, lon)
        if r:
            out["region"] = r[0]["name"]
        return out


def _spoken_coord(loc):
    lat, lon = loc["lat"], loc["lon"]
    return (f"{abs(lat):.0f} degrees {'north' if lat >= 0 else 'south'}, "
            f"{abs(lon):.0f} degrees {'east' if lon >= 0 else 'west'}")


def place_phrase(loc):
    """The place in plain English: "Central Ethiopia. Eastern Africa. Natural region: Ethiopian Highlands." """
    if loc is None:
        return "Position unknown."
    bits = []
    if loc["country"]:
        bits.append(f"{loc['part']} {loc['country']}".strip())
        if loc["subregion"] and loc["subregion"] != loc["country"] and "Seven seas" not in loc["subregion"]:
            bits.append(loc["subregion"])
    else:
        bits.append(loc["sea"] or "Open ocean")
        if loc["near_country"]:
            km = int(round(loc["near_km"] / 10) * 10) if loc["near_km"] >= 30 else None
            bits.append(f"nearest country: {loc['near_country']}" + (f", about {km} kilometers away" if km else ""))
    if loc["region"] and loc["region"] not in bits[0]:
        bits.append(f"natural region: {loc['region']}")
    return ". ".join(b[0].upper() + b[1:] for b in bits) + "."


# ----------------------------------------------------------------------------
# Numbers and units the way they should be spoken
# ----------------------------------------------------------------------------
def spoken_number(v, decimals, signed=False):
    s = f"{abs(v):.{decimals}f}"
    sign = ("plus " if v > 0 else "minus " if v < 0 else "") if signed else ("minus " if v < 0 else "")
    return sign + s


def spoken_quantity(v, unit, decimals, signed=False):
    fractional = decimals > 0 and round(abs(v), decimals) != int(round(abs(v), decimals))
    num = spoken_number(v, decimals if fractional else 0, signed)    # "21 degrees", not "21.0 degrees"
    if unit in ("°C", "°"):
        return f"{num} degree" if num.split()[-1] == "1" else f"{num} degrees"
    if unit == "ppm":
        return f"{num} parts per million"
    return f"{num} {unit}".strip()


def _decimals(fmt):
    m = re.search(r"\.(\d+)f", fmt)
    return int(m.group(1)) if m else 2


def spoken_time(layer, t):
    if not layer.dynamic:
        return _cap(layer.cfg.get("static_label") or "")
    lab = layer.labels[layer.clamp_frame(t)]
    if layer.has_dates:                                          # "28.03.2007" -> "March 28, 2007"
        d, m, y = lab.split(".")
        s = f"{MONTHS[int(m) - 1]} {int(d)}, {y}"
    else:
        s = f"Year {lab}"
    if int(layer.times[layer.clamp_frame(t)]) in layer.partial_years:
        s += ", partial data for this year"
    return s


def _spoken_years(text):
    return re.sub(r"(\d{4})\s*[–-]\s*(\d{4})", r"\1 to \2", text)


def compose(layer, t, lat, lon, sample, loc=None):
    """The full message: place. time. layer value. comparison with the baseline. layer note."""
    parts = [place_phrase(loc) if loc else f"Position: {_spoken_coord({'lat': lat, 'lon': lon})}."]
    when = spoken_time(layer, t)
    if when:
        parts.append(when + ".")
    title = spoken_title(layer)
    if sample is None or sample["missing"]:
        parts.append(f"{title}: no data at this spot.")
        return " ".join(parts)
    dec = _decimals(layer.fmt)
    signed = "+" in layer.fmt
    value = spoken_quantity(sample["value"], layer.unit, dec, signed)
    if sample.get("clipped", 0) > 0:
        value = "more than " + value
    elif sample.get("clipped", 0) < 0:
        value = "less than " + value
    parts.append(f"{title}: {value}.")
    base = sample.get("base")
    trivial = getattr(layer, "baseline_grid", None) is None and base == 0.0
    if base is not None and not trivial:
        diff = sample["value"] - base
        if abs(diff) < 0.5 * 10 ** -dec:
            parts.append(f"That is the same as {_spoken_years(layer.base_label)}.")
        else:
            parts.append(f"That is {spoken_quantity(abs(diff), layer.unit, dec)} "
                         f"{'above' if diff > 0 else 'below'} {_spoken_years(layer.base_label)}.")
    note = layer.cfg.get("speech_note")
    if note:
        parts.append(_spoken_years(note))
    if sample.get("near_km"):
        parts.append(f"Value taken from the nearest cell with data, "
                     f"about {int(round(sample['near_km'] / 10) * 10)} kilometers away.")
    return " ".join(parts)


# ----------------------------------------------------------------------------
# Announcements: list of maps, help, switching maps
# ----------------------------------------------------------------------------
def spoken_title(layer_or_title):
    title = layer_or_title if isinstance(layer_or_title, str) else layer_or_title.title
    return re.sub(r"\s*\([^)]*\)", "", title).strip()


def spoken_range(layer):
    """Time span of a layer: "from 1880 to 2026, 147 years" / "from March 2007 to February 2008, 48 weekly frames"."""
    if not layer.dynamic:
        return (layer.cfg.get("static_label") or "a single moment") + ", a single snapshot"
    a, b = layer.labels[0], layer.labels[-1]
    if layer.has_dates:
        _, m1, y1 = a.split("."); _, m2, y2 = b.split(".")
        step = (layer.times[-1] - layer.times[0]) / max(layer.n_frames - 1, 1) * 365
        kind = "weekly " if step < 10 else "monthly " if step < 40 else ""
        return f"from {MONTHS[int(m1) - 1]} {y1} to {MONTHS[int(m2) - 1]} {y2}, {layer.n_frames} {kind}frames"
    return f"from {a} to {b}, {layer.n_frames} years"


def layer_intro(idx, n, layer, surface="map"):
    """Short message when switching maps."""
    s = f"Map {idx} of {n}: {spoken_title(layer)}. {_cap(spoken_range(layer))}."
    if layer.dynamic:
        s += (" Move your left hand along the map to change the time." if surface in ("map", "vmap")
              else " Use comma and period to change the time.")
    return s


def layers_announcement(specs, current_idx, load=True):
    """All maps with numbers and time spans; which one is selected; how to switch."""
    n = len(specs)
    parts = [f"There {'is' if n == 1 else 'are'} {n} map{'s' if n != 1 else ''}."]
    for k, s in enumerate(specs, 1):
        lay = s.load() if load else None
        rng = f", {spoken_range(lay)}" if lay is not None else ""
        parts.append(f"Map {k}: {spoken_title(lay or s.title)}{rng}.")
    parts.append(f"Map {current_idx} is selected.")
    parts.append(f"To choose a map, press a number from 1 to {min(n, 9)}, or press L for the next map.")
    return " ".join(parts)


_SOUND_EXPLAINED = ("The pitch tells you the value at that spot: a higher sound means a higher value. "
                    "If the sound pulses, the value is above normal. "
                    "You hear the sound on the left when your finger is in the west, and on the right in the east.")


def help_announcement(keys, n_layers, surface="map"):
    """What the program does and what each key does. keys: [(key, how to say it, what it does)]."""
    if surface == "vmap":
        intro = ("Earth Jukebox, a sound map of the Earth. A world map is drawn on top of the camera picture. "
                 "Point at it with the index finger of your right hand. " + _SOUND_EXPLAINED + " "
                 "Hold your right finger still for three seconds, and I will tell you the place, the time, "
                 "and the value. Slide your left index finger along the map to change the time: "
                 "the left edge is the oldest data, the right edge is the newest. "
                 "To zoom, hold your left thumb and index finger out over a place, like on a phone screen: "
                 "spread them apart to zoom in, bring them together to zoom out. Fold your thumb to keep the zoom. "
                 "The map itself does not move: "
                 "to go somewhere else, zoom out first, then zoom in over the new place. "
                 "Press zero to see the whole world again.")
    elif surface == "virtual":
        intro = ("Earth Jukebox, a sound map of the Earth. A globe is drawn on top of the camera picture. "
                 "Point at it with the index finger of your right hand. " + _SOUND_EXPLAINED + " "
                 "To turn the globe, pinch your left thumb and index finger together, as if you were holding it, "
                 "and move your hand. Open your fingers to let go. "
                 "The poles stay in place, like on a globe stand: move sideways to spin it, "
                 "up or down to tilt it towards a pole. "
                 "Hold your right finger still for three seconds, and I will tell you the place, the time, "
                 "and the value. Change the time with the comma and period keys.")
    elif surface == "globe":
        intro = ("Earth Jukebox, a sound map of the Earth. The camera sees the globe and your right hand. "
                 "Point at any place on the globe with your right index finger. " + _SOUND_EXPLAINED + " "
                 "You can spin the globe and move it around, the program finds it by itself. "
                 "Hold your finger still for three seconds, and I will tell you the place, the time, and the value. "
                 "Change the time with the comma and period keys.")
    else:
        intro = ("Earth Jukebox, a sound map of the Earth. The camera sees a paper world map and your hands. "
                 "Move your right index finger over the map. " + _SOUND_EXPLAINED + " "
                 "Hold your right finger still for three seconds, and I will tell you the place, the time, "
                 "and the value. Slide your left index finger along the map to change the time: "
                 "the left edge is the oldest data, the right edge is the newest.")
    keys_txt = " ".join(f"{spoken[0].upper() + spoken[1:]}: {what}." for _, spoken, what in keys)
    return f"{intro} Keys. {keys_txt}".replace("{n}", str(min(n_layers, 9)))


# ----------------------------------------------------------------------------
# Voice
# ----------------------------------------------------------------------------
def english_voices():
    """Installed English voices of macOS `say`: [(name, locale)], best first."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    voices = []
    for line in out.splitlines():
        m = re.match(r"^(.+?)\s+(en_[A-Z]{2})\s", line)
        if m:
            voices.append((m.group(1).strip(), m.group(2)))

    def rank(v):
        name = v[0]
        if "Premium" in name:
            return 0
        if "Enhanced" in name:
            return 1
        base = name.split(" (")[0]
        return 2 + (PREFERRED_VOICES.index(base) if base in PREFERRED_VOICES and "(" not in name else 50)
    return sorted(voices, key=rank)


# ----------------------------------------------------------------------------
# Neural voice: Kokoro (82M parameters, Apache 2.0, runs offline on the CPU, ~3x faster than real time on an M2).
# Model files (not in git): models/kokoro-v1.0.onnx, models/voices-v1.0.bin - `python speech.py --download`.
# ----------------------------------------------------------------------------
KOKORO_MODEL = HERE / "models" / "kokoro-v1.0.onnx"
KOKORO_VOICES = HERE / "models" / "voices-v1.0.bin"
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_DEFAULT = "af_heart"          # the best rated English voice of Kokoro v1.0
TTS_CACHE = HERE / ".tts_cache"      # sentences already spoken (menu, help, layer names) play at once
_KOKORO_NAME = re.compile(r"^[ab][fm]_[a-z]+$")


def is_kokoro_voice(name):
    return bool(name) and bool(_KOKORO_NAME.match(str(name)))


def kokoro_available():
    if not (KOKORO_MODEL.exists() and KOKORO_VOICES.exists() and shutil.which("afplay")):
        return False
    try:
        import kokoro_onnx  # noqa: F401
        return True
    except Exception:
        return False


def download_kokoro():
    """Download the model files into models/ (~350 MB, once)."""
    import urllib.request
    KOKORO_MODEL.parent.mkdir(exist_ok=True)
    for dst in (KOKORO_MODEL, KOKORO_VOICES):
        if dst.exists():
            print(f"{dst.name}: already there")
            continue
        tmp = dst.with_suffix(dst.suffix + ".part")
        print(f"Downloading {dst.name} ...")
        urllib.request.urlretrieve(KOKORO_URL + dst.name, tmp)
        tmp.rename(dst)
    print("Done. The program now speaks with the Kokoro voice.")


class _Kokoro:
    """The model, loaded once per program (in the background) and shared by all Speakers."""
    _inst = None
    _make = threading.Lock()

    @classmethod
    def get(cls):
        with cls._make:
            if cls._inst is None:
                cls._inst = cls()
            return cls._inst

    def __init__(self):
        self.ready, self.error, self.kokoro, self.voices = threading.Event(), None, None, set()
        self.lock = threading.Lock()                        # one synthesis at a time
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            sess = ort.InferenceSession(str(KOKORO_MODEL), sess_options=opts, providers=["CPUExecutionProvider"])
            self.kokoro = Kokoro.from_session(sess, str(KOKORO_VOICES))
            self.voices = set(self.kokoro.get_voices())
            self.kokoro.create("Ready.", voice=KOKORO_DEFAULT, speed=1.0, lang="en-us")   # warm up
            _prune_cache()
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            print(f"  Kokoro voice not available ({self.error}); using the macOS voice.")
        self.ready.set()

    def synth(self, text, voice, speed):
        """-> (float32 samples, sample rate) or None."""
        self.ready.wait()
        if self.kokoro is None:
            return None
        if voice not in self.voices:
            voice = KOKORO_DEFAULT
        with self.lock:
            a, sr = self.kokoro.create(text, voice=voice, speed=speed, lang="en-gb" if voice[0] == "b" else "en-us")
        return np.asarray(a, np.float32), sr


def _prune_cache(keep=3000):
    if TTS_CACHE.exists():
        files = sorted(TTS_CACHE.glob("*.wav"), key=lambda f: f.stat().st_mtime)
        for f in files[:-keep]:
            f.unlink(missing_ok=True)


def _write_wav(path, a, sr):
    import wave
    tmp = Path(str(path) + ".part")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes())
    tmp.replace(path)


def split_sentences(text, short=25, long=280):
    """Text -> pieces spoken one after another: the first sentence starts while the next ones are made.
    Very short pieces are joined to the next one (each piece costs a little time), very long ones are cut at commas."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", str(text).strip()) if p.strip()]
    out = []
    for p in parts:
        while len(p) > long:
            cut = max(p.rfind(", ", 0, long), p.rfind("; ", 0, long))
            if cut < 40:
                break
            out.append(p[:cut + 1]); p = p[cut + 2:]
        if out and len(out) > 1 and len(out[-1]) < short:
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out


class Speaker:
    """Speech, non-blocking. The neural Kokoro voice when its model is in models/ (natural, offline),
    otherwise macOS `say` with the best installed English voice.
    voice: "auto" (Kokoro af_heart, else the best `say` voice), a Kokoro voice (af_heart, af_bella, bf_emma,
    am_michael, bm_george ...) or a macOS voice name (Samantha, Daniel ...). rate: words per minute (175 = normal)."""

    def __init__(self, enabled=True, voice="auto", rate=175):
        self.voice, self.rate, self.proc, self.engine = None, rate, None, None
        self.enabled = False
        self._gen, self._worker, self._worker_gen = 0, None, -1
        self._plock = threading.Lock()
        if not enabled:
            return
        auto = not voice or voice == "auto"
        if (auto or is_kokoro_voice(voice)) and kokoro_available():
            self.engine, self.voice, self.enabled = "kokoro", (KOKORO_DEFAULT if auto else voice), True
            _Kokoro.get()                                   # starts loading in the background
            return
        if is_kokoro_voice(voice):
            print("  Kokoro model not found (python speech.py --download); using the macOS voice.")
            voice = "auto"
        if not shutil.which("say"):
            print("  Voice off: the `say` command is not available (not macOS).")
            return
        voices = english_voices()
        names = [v[0] for v in voices]
        if voice and voice != "auto":
            if voice not in names:
                print(f"  Voice '{voice}' is not installed; using the best available English voice.")
            else:
                self.voice = voice
        if self.voice is None:
            if not voices:
                print("  Voice off: no English voice installed.")
                return
            self.voice = voices[0][0]
        self.engine, self.enabled = "say", True

    @property
    def speed(self):
        return float(np.clip(self.rate / 175.0, 0.6, 1.6))

    # --- Kokoro ---
    def _piece(self, text):
        """WAV file of one piece of text (from the cache, or made now)."""
        import hashlib
        key = hashlib.sha1(f"{self.voice}|{self.speed:.2f}|{text}".encode()).hexdigest()[:24]
        f = TTS_CACHE / f"{key}.wav"
        if f.exists():
            f.touch()
            return f
        res = _Kokoro.get().synth(text, self.voice, self.speed)
        if res is None:
            return None
        TTS_CACHE.mkdir(exist_ok=True)
        _write_wav(f, *res)
        return f

    def _play(self, text, gen):
        for piece in split_sentences(text):
            if gen != self._gen:
                return
            f = self._piece(piece)                          # made while the previous piece is playing
            if f is None:                                   # the model failed: say it with macOS
                if gen == self._gen and shutil.which("say"):
                    with self._plock:
                        self.proc = subprocess.Popen(["say", "-r", str(self.rate), text],
                                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            while self.proc is not None and self.proc.poll() is None:
                if gen != self._gen:
                    return
                time.sleep(0.01)
            with self._plock:
                if gen != self._gen:
                    return
                self.proc = subprocess.Popen(["afplay", str(f)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # --- the interface used by the program ---
    def speak(self, text):
        if not self.enabled or not str(text).strip():
            return
        self.stop()
        if self.engine == "say":
            self.proc = subprocess.Popen(["say", "-v", self.voice, "-r", str(self.rate), text],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        gen = self._gen
        self._worker_gen = gen
        self._worker = threading.Thread(target=self._play, args=(text, gen), daemon=True)
        self._worker.start()

    def speaking(self):
        busy = self._worker is not None and self._worker.is_alive() and self._worker_gen == self._gen
        return busy or (self.proc is not None and self.proc.poll() is None)

    def stop(self):
        with self._plock:
            self._gen += 1
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
            self.proc = None

    def to_file(self, text, path):
        """Render speech to a file (tests, playlist; .wav, or .aiff/.m4a via afconvert). -> True on success."""
        if not self.enabled:
            return False
        path = Path(path)
        if self.engine == "say":
            r = subprocess.run(["say", "-v", self.voice, "-r", str(self.rate), "-o", str(path), text],
                               capture_output=True, timeout=60)
            return r.returncode == 0 and path.exists()
        chunks, sr = [], 24000
        for piece in split_sentences(text):
            res = _Kokoro.get().synth(piece, self.voice, self.speed)
            if res is None:
                return False
            a, sr = res
            chunks += [a, np.zeros(int(0.12 * sr), np.float32)]
        if not chunks:
            return False
        wav = path if path.suffix.lower() == ".wav" else path.with_suffix(".tmp.wav")
        _write_wav(wav, np.concatenate(chunks), sr)
        if wav != path:
            fmt = {".aiff": ("AIFF", "BEI16"), ".aif": ("AIFF", "BEI16"), ".m4a": ("m4af", "aac")}.get(path.suffix.lower(), ("AIFF", "BEI16"))
            r = subprocess.run(["afconvert", "-f", fmt[0], "-d", fmt[1], str(wav), str(path)], capture_output=True)
            wav.unlink(missing_ok=True)
            if r.returncode != 0:
                return False
        return path.exists()


if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:
        download_kokoro()
    else:
        sp = Speaker(voice=sys.argv[1] if len(sys.argv) > 1 else "auto")
        print(f"Voice: {sp.voice} ({sp.engine})")
        sp.speak("Hello! This is Earth Jukebox. Point at the map and hold your finger still for three seconds.")
        while sp.speaking():
            time.sleep(0.1)
