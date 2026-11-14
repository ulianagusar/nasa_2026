"""Озвучення точки: де це (країна / море / природний регіон), коли (рік або дата) і що там у даних.

    geo = Geocoder()                          # офлайн, контури Natural Earth з українськими назвами
    text = compose(layer, t, lat, lon, sample, geo.locate(lat, lon))
    Speaker().speak(text)                     # macOS `say`, голос Lesya (uk_UA)

Приклад: "Ефіопія, північна частина. Східна Африка. 2025 рік. Аномалія температури: плюс 1,3 градуса.
Це відхилення від норми 1951–1980 років."

Дані (у data/, завантажуються один раз):
  ne_50m_admin_0_countries.geojson, ne_50m_geography_marine_polys.geojson,
  ne_50m_geography_regions_polys.geojson  - https://github.com/nvkelso/natural-earth-vector
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
NE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/{}.geojson"
NE_FILES = {"countries": "ne_50m_admin_0_countries", "marine": "ne_50m_geography_marine_polys",
            "regions": "ne_50m_geography_regions_polys"}

SUBREGIONS_UA = {
    "Eastern Africa": "Східна Африка", "Middle Africa": "Центральна Африка", "Northern Africa": "Північна Африка",
    "Southern Africa": "південь Африки", "Western Africa": "Західна Африка",
    "Caribbean": "Карибський регіон", "Central America": "Центральна Америка", "South America": "Південна Америка",
    "Northern America": "Північна Америка",
    "Central Asia": "Центральна Азія", "Eastern Asia": "Східна Азія", "South-Eastern Asia": "Південно-Східна Азія",
    "Southern Asia": "Південна Азія", "Western Asia": "Західна Азія",
    "Eastern Europe": "Східна Європа", "Northern Europe": "Північна Європа", "Southern Europe": "Південна Європа",
    "Western Europe": "Західна Європа",
    "Australia and New Zealand": "Австралія та Нова Зеландія", "Melanesia": "Меланезія", "Micronesia": "Мікронезія",
    "Polynesia": "Полінезія", "Antarctica": "Антарктида",
}
PHYSICAL = {"Desert", "Range/mtn", "Plateau", "Basin", "Plain", "Tundra", "Peninsula", "Pen/cape", "Delta",
            "Lowland", "Valley", "Wetlands", "Foothills"}
MONTHS_GEN = ["січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня", "жовтня",
              "листопада", "грудня"]
PARTS = {(0, 0): "центральна частина", (1, 0): "північна частина", (-1, 0): "південна частина",
         (0, 1): "східна частина", (0, -1): "західна частина", (1, 1): "північно-східна частина",
         (1, -1): "північно-західна частина", (-1, 1): "південно-східна частина", (-1, -1): "південно-західна частина"}


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _clean(name):
    name = (name or "").strip()
    return name.title() if name.isupper() else name             # "САХЕЛ" -> "Сахел"


def ensure_data(data_dir=DATA_DIR):
    """Завантажує контури Natural Earth, якщо їх ще немає. -> True, якщо всі файли на місці."""
    import urllib.request
    ok = True
    for f in NE_FILES.values():
        p = Path(data_dir) / f"{f}.geojson"
        if not p.exists():
            try:
                print(f"  завантаження {p.name} ...")
                urllib.request.urlretrieve(NE_URL.format(f), p)
            except Exception as e:
                print(f"  не вдалося завантажити {p.name}: {e}")
                ok = False
    return ok


class _Shapes:
    """Полігони з українськими назвами + швидка перевірка 'точка всередині'."""

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
            area = sum(abs(np.dot(r[:, 0], np.roll(r[:, 1], 1)) - np.dot(r[:, 1], np.roll(r[:, 0], 1))) / 2 for r in rings)
            biggest = max(rings, key=lambda r: abs(np.dot(r[:, 0], np.roll(r[:, 1], 1)) - np.dot(r[:, 1], np.roll(r[:, 0], 1))))
            self.items.append({
                "name": name, "props": props, "area": area,
                "bbox": (allpts[:, 0].min(), allpts[:, 1].min(), allpts[:, 0].max(), allpts[:, 1].max()),
                "paths": [(MplPath(r), (r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max())) for r in rings],
                "main_bbox": (biggest[:, 0].min(), biggest[:, 1].min(), biggest[:, 0].max(), biggest[:, 1].max()),
                "verts": allpts[:: max(1, len(allpts) // 400)],             # для відстані "поруч"
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
        return sorted(hits, key=lambda it: it["area"])               # найдрібніший (найконкретніший) перший

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
        self.countries = _Shapes(d / f"{NE_FILES['countries']}.geojson", ["NAME_UK", "NAME"],
                                 extra=lambda p: {"subregion": SUBREGIONS_UA.get(p.get("SUBREGION"), "")})
        self.marine = _Shapes(d / f"{NE_FILES['marine']}.geojson", ["name_uk", "name"])
        self.regions = _Shapes(d / f"{NE_FILES['regions']}.geojson", ["NAME_UK", "NAME"],
                               filt=lambda p: p.get("FEATURECLA") in PHYSICAL)
        self.ready = bool(self.countries.items)

    @staticmethod
    def _part(it, lat, lon):
        """Яка частина країни: за положенням точки в рамці основної території (для великих країн)."""
        x0, y0, x1, y1 = it["main_bbox"]
        if (x1 - x0) < 5 and (y1 - y0) < 5:                           # маленька країна - без "частин"
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


def place_phrase(loc):
    """Опис місця українською без відмінювання назв (щоб не помилятись у відмінках)."""
    if loc is None:
        return f"Координати {_spoken_coord(loc)}."
    bits = []
    if loc["country"]:
        bits.append(loc["country"] + (f", {loc['part']}" if loc["part"] else ""))
        if loc["subregion"] and loc["subregion"] != loc["country"]:
            bits.append(loc["subregion"])
    else:
        bits.append(loc["sea"] or "Відкритий океан")
        if loc["near_country"]:
            km = int(round(loc["near_km"] / 10) * 10) if loc["near_km"] >= 30 else None
            bits.append(f"найближча країна: {loc['near_country']}" + (f", приблизно {km} кілометрів" if km else ""))
    if loc["region"] and loc["region"] not in bits[0]:
        bits.append(f"природний регіон: {loc['region']}")
    return ". ".join(b[0].upper() + b[1:] for b in bits) + "."


def _spoken_coord(loc):
    lat, lon = loc["lat"], loc["lon"]
    return (f"{abs(lat):.0f} градусів {'північної' if lat >= 0 else 'південної'} широти, "
            f"{abs(lon):.0f} градусів {'східної' if lon >= 0 else 'західної'} довготи")


# ----------------------------------------------------------------------------
# Числа й одиниці так, як їх треба вимовляти
# ----------------------------------------------------------------------------
def _plural(n, one, few, many):
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def spoken_number(v, decimals, signed=False):
    s = f"{abs(v):.{decimals}f}".replace(".", ",")
    sign = ("плюс " if v > 0 else "мінус " if v < 0 else "") if signed else ("мінус " if v < 0 else "")
    return sign + s


def spoken_quantity(v, unit, decimals, signed=False):
    fractional = decimals > 0 and round(abs(v), decimals) != int(round(abs(v), decimals))
    num = spoken_number(v, decimals if fractional else 0, signed)    # "21 градус", а не "21,0 градус"
    if unit in ("°C", "°"):
        word = "градуса" if fractional else _plural(round(abs(v)), "градус", "градуси", "градусів")
        return f"{num} {word}"
    if unit == "ppm":
        return f"{num} мільйонних часток"
    return f"{num} {unit}".strip()


def _decimals(fmt):
    m = re.search(r"\.(\d+)f", fmt)
    return int(m.group(1)) if m else 2


def spoken_time(layer, t):
    if not layer.dynamic:
        return (layer.cfg.get("static_label") or "").capitalize()
    lab = layer.labels[layer.clamp_frame(t)]
    if layer.has_dates:                                          # "28.03.2007" -> "28 березня 2007 року"
        d, m, y = lab.split(".")
        s = f"{int(d)} {MONTHS_GEN[int(m) - 1]} {y} року"
    else:
        s = f"{lab} рік"
    if int(layer.times[layer.clamp_frame(t)]) in layer.partial_years:
        s += ", неповні дані за рік"
    return s


def _spoken_years(text):
    return re.sub(r"(\d{4})\s*[–-]\s*(\d{4})", r"\1–\2 років", text)


def compose(layer, t, lat, lon, sample, loc=None):
    """Повне повідомлення для озвучення: місце. час. значення. порівняння з нормою. примітка."""
    parts = [place_phrase(loc) if loc else f"Точка: {_spoken_coord({'lat': lat, 'lon': lon})}."]
    when = spoken_time(layer, t)
    if when:
        parts.append(when + ".")
    title = re.sub(r"\s*\([^)]*\)", "", layer.title).strip()      # без пояснень у дужках
    if sample is None or sample["missing"]:
        parts.append(f"{title}: даних у цій точці немає.")
        return " ".join(parts)
    dec = _decimals(layer.fmt)
    signed = "+" in layer.fmt
    value = spoken_quantity(sample["value"], layer.unit, dec, signed)
    if sample.get("clipped", 0) > 0:
        value = "понад " + value
    elif sample.get("clipped", 0) < 0:
        value = "менше ніж " + value
    parts.append(f"{title}: {value}.")
    base = sample.get("base")
    trivial = getattr(layer, "baseline_grid", None) is None and base == 0.0
    if base is not None and not trivial:
        diff = sample["value"] - base
        if abs(diff) < 0.5 * 10 ** -dec:
            parts.append(f"Це як {_spoken_years(layer.base_label)}.")
        else:
            parts.append(f"Це на {spoken_quantity(abs(diff), layer.unit, dec)} "
                         f"{'вище' if diff > 0 else 'нижче'}, ніж {_spoken_years(layer.base_label)}.")
    note = layer.cfg.get("speech_note")
    if note:
        parts.append(_spoken_years(note))
    if sample.get("near_km"):
        parts.append(f"Значення з найближчої клітинки з даними, за {int(round(sample['near_km'] / 10) * 10)} кілометрів.")
    return " ".join(parts)


# ----------------------------------------------------------------------------
# Оголошення: перелік карт, довідка, перемикання карти
# ----------------------------------------------------------------------------
MONTHS_NOM = ["січень", "лютий", "березень", "квітень", "травень", "червень", "липень", "серпень", "вересень",
              "жовтень", "листопад", "грудень"]


def spoken_title(layer_or_title):
    title = layer_or_title if isinstance(layer_or_title, str) else layer_or_title.title
    return re.sub(r"\s*\([^)]*\)", "", title).strip()


def spoken_range(layer):
    """Період даних шару: "з 1880 по 2026 рік, 147 років" / "з березня 2007 по лютий 2008 року, 48 тижневих кадрів"."""
    if not layer.dynamic:
        return (layer.cfg.get("static_label") or "один момент часу") + ", один момент часу"
    a, b = layer.labels[0], layer.labels[-1]
    if layer.has_dates:
        _, m1, y1 = a.split("."); _, m2, y2 = b.split(".")
        step = (layer.times[-1] - layer.times[0]) / max(layer.n_frames - 1, 1) * 365
        kind = "тижневих" if step < 10 else "місячних" if step < 40 else ""
        return (f"з {MONTHS_GEN[int(m1) - 1]} {y1} по {MONTHS_NOM[int(m2) - 1]} {y2} року, "
                f"{layer.n_frames} {kind} кадрів".replace("  ", " "))
    n = layer.n_frames
    return f"з {a} по {b} рік, {n} {_plural(n, 'рік', 'роки', 'років')}"


def layer_intro(idx, n, layer, surface="map"):
    """Коротко при перемиканні: "Карта 2 з 4: Активність течій з відео NASA. З березня 2007 по ...". """
    s = f"Карта {idx} з {n}: {spoken_title(layer)}. {spoken_range(layer).capitalize()}."
    if layer.dynamic:
        s += (" Час змінюйте лівою рукою вздовж карти." if surface == "map"
              else " Час змінюйте клавішами кома і крапка.")
    return s


def layers_announcement(specs, current_idx, load=True):
    """Перелік усіх карт з номерами і періодами; яка обрана зараз; як перемкнути."""
    n = len(specs)
    parts = [f"Доступно {n} {_plural(n, 'карта', 'карти', 'карт')}."]
    for k, s in enumerate(specs, 1):
        lay = s.load() if load else None
        rng = f", {spoken_range(lay)}" if lay is not None else ""
        parts.append(f"Карта {k}: {spoken_title(lay or s.title)}{rng}.")
    parts.append(f"Зараз обрано карту {current_idx}.")
    parts.append(f"Щоб обрати карту, натисніть цифру від 1 до {min(n, 9)}, або клавішу ел для наступної карти.")
    return " ".join(parts)


def help_announcement(keys, n_layers, surface="map"):
    """Що робить програма і які клавіші що роблять. keys: [(клавіша, як вимовити, що робить)]."""
    if surface == "globe":
        intro = ("Earth Jukebox, звукова карта Землі. Камера бачить глобус і вашу праву руку. "
                 "Показуйте вказівним пальцем правої руки на будь-яке місце глобуса: висота звуку показує "
                 "значення в цьому місці, вищий звук означає більше значення. Якщо звук пульсує, значення "
                 "вище за норму. Ліворуч звук чути, коли палець на заході, праворуч, коли на сході. "
                 "Глобус можна крутити і переставляти, програма сама його знаходить. "
                 "Затримайте палець на три секунди, і я назву місце, час і значення. "
                 "Час змінюйте клавішами: кома і крапка, крок назад і вперед.")
        keys_txt = " ".join(f"{spoken.capitalize()}: {what}." for _, spoken, what in keys)
        return f"{intro} Клавіші. {keys_txt}".replace("{n}", str(min(n_layers, 9)))
    intro = ("Earth Jukebox, звукова карта Землі. Камера бачить паперову карту світу і ваші руки. "
             "Правим вказівним пальцем водіть по карті: висота звуку показує значення в цьому місці, "
             "вищий звук означає більше значення. Якщо звук пульсує, значення вище за норму. "
             "Ліворуч звук чути, коли палець на заході, праворуч, коли на сході. "
             "Затримайте правий палець на три секунди, і я назву місце, час і значення. "
             "Лівим вказівним пальцем ведіть уздовж карти, щоб змінювати час: "
             "лівий край карти означає найдавніші дані, правий край найновіші.")
    keys_txt = " ".join(f"{spoken.capitalize()}: {what}." for _, spoken, what in keys)
    return f"{intro} Клавіші. {keys_txt}".replace("{n}", str(min(n_layers, 9)))


# ----------------------------------------------------------------------------
# Голос
# ----------------------------------------------------------------------------
class Speaker:
    """Озвучення через macOS `say` (голос Lesya, українська). Не блокує програму."""

    def __init__(self, enabled=True, voice="Lesya", rate=185):
        self.voice, self.rate, self.proc = voice, rate, None
        self.enabled = False
        if not enabled:
            return
        if not shutil.which("say"):
            print("  Голос вимкнено: немає команди `say` (не macOS).")
            return
        try:
            voices = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
        except Exception as e:
            print(f"  Голос вимкнено: {e}")
            return
        if not re.search(rf"^{re.escape(voice)}\s", voices, re.M):
            uk = re.findall(r"^(\S+)\s+uk_UA", voices, re.M)
            if not uk:
                print("  Голос вимкнено: немає українського голосу. Додайте Lesya: Системні параметри -> "
                      "Доступність -> Вміст мовлення -> Системний голос -> Керувати голосами.")
                return
            self.voice = uk[0]
        self.enabled = True

    def speak(self, text):
        if not self.enabled:
            return
        self.stop()
        self.proc = subprocess.Popen(["say", "-v", self.voice, "-r", str(self.rate), text],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def speaking(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.speaking():
            self.proc.terminate()
        self.proc = None

    def to_file(self, text, path):
        """Записати озвучення у файл (для тестів і плейлиста). -> True, якщо вдалося."""
        if not self.enabled:
            return False
        r = subprocess.run(["say", "-v", self.voice, "-r", str(self.rate), "-o", str(path), text],
                           capture_output=True, timeout=60)
        return r.returncode == 0 and Path(path).exists()
