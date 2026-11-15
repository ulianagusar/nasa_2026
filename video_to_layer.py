"""
Earth Jukebox: відео NASA -> шар даних у тому ж форматі
=======================================================
Для відео "Global Sea Surface Currents and Temperature" (NASA SVS 3912, модель ECCO2).
Відео це ВІЗУАЛІЗАЦІЯ: температура моря закодована КОЛЬОРОМ ліній течій.
Скрипт нарізає кадри, розшифровує колір назад у температуру і складає
сітку 2 градуси в тому самому форматі, що й решта шарів (див. earth_jukebox_data.py).

Потрібно: файл earth_jukebox_data.py лежить поруч (у Colab: завантажте обидва файли).

ВАЖЛИВО (чесно про точність):
  * Колір -> температура розшифровується ПРИБЛИЗНО. NASA описує шкалу словами:
    синій = 0 °C, зелений = 10-20 °C, жовтий ~ 25 °C, червоний = 32 °C.
    Скрипт переводить відтінок (hue) у температуру лінійно: 240° (синій) = 0 °C,
    0° (червоний) = 32 °C. Це збігається з описом NASA, але це оцінка, а не вимір.
  * Відео охоплює лише 2007-03-25 ... 2008-03-03 (близько року).
  * Течії (напрямок і швидкість) із відео витягти не можна. Є лише додатковий шар
    flow_activity: частка кольорових пікселів у клітинці. Це візуальний показник
    "скільки ліній намальовано", а не фізична швидкість.
  * Суша виключається маскою (потрібен cartopy, у Colab ставиться автоматично).
"""

import os
import sys
import subprocess

import numpy as np
import pandas as pd

from earth_jukebox_data import (LAT, LON, TARGET_STEP, OUT_DIR, export_layer,  # noqa: F401
                                preview, sample_at, zip_layers, download_layers, download_all)

try:
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"])
    import cv2


N_LAT, N_LON = len(LAT), len(LON)   # 90 x 180


# ---------------------------------------------------------------------------
# 1. Нарізка кадрів
# ---------------------------------------------------------------------------
def _count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:   # деякі webm не повідомляють кількість кадрів, рахуємо самі
        total = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            total += 1
    cap.release()
    return total


def iter_frames(video_path, n_frames=48, margin=0.01):
    """Віддає (номер_кадру, RGB-кадр) для n_frames кадрів, рівномірно по всьому відео."""
    total = _count_frames(video_path)
    if total < 2:
        raise ValueError("У відео не знайшла кадрів: %s" % video_path)
    lo, hi = int(total * margin), int(total * (1 - margin)) - 1
    wanted = sorted(set(np.linspace(lo, hi, n_frames).round().astype(int).tolist()))
    wanted_set = set(wanted)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while i <= wanted[-1]:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted_set:
            yield i, total, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


def get_frame(video_path, frame_index=None):
    """Один кадр для перегляду. За замовчуванням кадр з середини відео."""
    total = _count_frames(video_path)
    idx = total // 2 if frame_index is None else frame_index
    cap = cv2.VideoCapture(video_path)
    for _ in range(idx):
        cap.grab()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Не змогла прочитати кадр %d" % idx)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def show_frame(frame_rgb, box=None):
    """Показує кадр. box=(x0, y0, x1, y1) в пікселях малює рамку карти."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.imshow(frame_rgb)
    if box:
        x0, y0, x1, y1 = box
        plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "w-", lw=1.5)
    h, w = frame_rgb.shape[:2]
    plt.title("кадр %dx%d (співвідношення сторін %.2f, для карти світу має бути 2.00)" % (w, h, w / h))
    plt.show()


# ---------------------------------------------------------------------------
# 2. Маска суші
# ---------------------------------------------------------------------------
def land_mask_from_geometry(land_geom, subdiv=5, land_share=0.5):
    """True = клітинка 2x2 градуси переважно суша. Перевіряємо subdiv x subdiv точок."""
    try:
        import shapely
        contains_xy = shapely.contains_xy
    except (ImportError, AttributeError):
        from shapely.vectorized import contains as contains_xy
    offs = (np.arange(subdiv) + 0.5) / subdiv * TARGET_STEP - TARGET_STEP / 2
    mask = np.zeros((N_LAT, N_LON), dtype=bool)
    lon_g, lat_g = np.meshgrid(LON, LAT)
    inside = np.zeros((N_LAT, N_LON), dtype="float32")
    for dy in offs:
        for dx in offs:
            inside += contains_xy(land_geom, lon_g + dx, lat_g + dy).astype("float32")
    mask = (inside / (subdiv * subdiv)) > land_share
    return mask


def make_land_mask():
    """Маска суші з Natural Earth (через cartopy). Потрібен інтернет (у Colab є)."""
    try:
        import cartopy.io.shapereader as shpreader
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "cartopy"])
        import cartopy.io.shapereader as shpreader
    from shapely.ops import unary_union
    path = shpreader.natural_earth(resolution="110m", category="physical", name="land")
    geom = unary_union(list(shpreader.Reader(path).geometries()))
    return land_mask_from_geometry(geom)


# ---------------------------------------------------------------------------
# 3. Колір -> температура
# ---------------------------------------------------------------------------
def hue_to_temp(hue_deg, t_max=32.0):
    """Відтінок -> °C. 240 (синій)=0, 120 (зелений)=16, 60 (жовтий)=24, 0 (червоний)=32."""
    return np.clip(t_max * (240.0 - hue_deg) / 240.0, 0, t_max)


def decode_frame(frame_rgb, box=None, land_mask=None, k=8, min_valid=8,
                 min_sat=0.45, min_val=0.35):
    """Один кадр -> (температура [90][180], активність [90][180]).
    box = (x0, y0, x1, y1): де в кадрі лежить карта (за замовчуванням весь кадр).
    k = скільки пікселів на сторону клітинки ми аналізуємо (k*k на клітинку).
    min_valid = скільки "кольорових" пікселів потрібно в клітинці, щоб довіряти їй."""
    if box:
        x0, y0, x1, y1 = box
        frame_rgb = frame_rgb[y0:y1, x0:x1]
    img = cv2.resize(frame_rgb, (N_LON * k, N_LAT * k), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype("float32") * 2.0          # 0..358
    sat = hsv[..., 1].astype("float32") / 255.0
    val = hsv[..., 2].astype("float32") / 255.0
    colored = (sat >= min_sat) & (val >= min_val) & (hue <= 250.0)

    temp_px = np.where(colored, hue_to_temp(hue), np.nan).astype("float32")
    # [lat_рядок][k][lon_стовпець][k] -> [lat][lon][k*k]
    blocks = temp_px.reshape(N_LAT, k, N_LON, k).transpose(0, 2, 1, 3).reshape(N_LAT, N_LON, k * k)
    n_valid = np.sum(~np.isnan(blocks), axis=2)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            temp = np.nanmedian(blocks, axis=2)
    temp[n_valid < min_valid] = np.nan
    activity = (n_valid / float(k * k)).astype("float32")

    # у зображенні рядок 0 = північ, а наша сітка йде з півдня на північ
    temp = temp[::-1].copy()
    activity = activity[::-1].copy()

    if land_mask is not None:
        temp[land_mask] = np.nan
        activity[land_mask] = np.nan
    return temp.astype("float32"), activity


# ---------------------------------------------------------------------------
# 3b. Заповнення дірок і згладжування (щоб був рівний градієнт)
# ---------------------------------------------------------------------------
def fill_gaps(temp, ocean_mask=None, sigmas=(1.5, 3.0, 5.0), smooth=1.0,
              min_weight=0.08, lat_limit=70.0):
    """Заповнює дірки в океані за сусідніми значеннями (зважене усереднення,
    яке поступово розширюється) і трохи згладжує. Суша (ocean_mask=False) НЕ заповнюється
    і не впливає на сусідів. Клітинки північніше/південніше lat_limit не чіпаємо:
    там у відео майже нічого немає, і вигадувати значення було б нечесно.
    temp: [90][180] з NaN для дірок."""
    from scipy.ndimage import gaussian_filter
    modes = ("nearest", "wrap")   # по довготі карта замкнена
    temp = np.asarray(temp, dtype="float64")
    ocean = np.ones(temp.shape, bool) if ocean_mask is None else np.asarray(ocean_mask, bool)
    allowed = ocean & (np.abs(LAT)[:, None] <= lat_limit)

    known = np.isfinite(temp) & ocean
    out = np.where(known, temp, 0.0)
    for s in sigmas:
        w = known.astype("float64")
        num = gaussian_filter(out * w, s, mode=modes)
        den = gaussian_filter(w, s, mode=modes)
        est = np.where(den > min_weight, num / np.maximum(den, 1e-9), np.nan)
        new = (~known) & allowed & np.isfinite(est)
        out = np.where(new, est, out)
        known = known | new
    if smooth:
        w = known.astype("float64")
        sm = gaussian_filter(out * w, smooth, mode=modes) / np.maximum(
            gaussian_filter(w, smooth, mode=modes), 1e-9)
        out = np.where(known, sm, out)
    return np.where(known, out, np.nan).astype("float32")


def fill_saved_layer(name="ocean_sst_video", new_name=None, **kwargs):
    """Бере вже готовий шар з папки earth_jukebox_out, заповнює дірки і зберігає як новий
    шар (за замовчуванням <name>_smooth). Відео перераховувати НЕ треба."""
    import json
    with open(os.path.join(OUT_DIR, name + ".json")) as f:
        d = json.load(f)
    data = np.array(d["data"], dtype="float64")          # None -> NaN
    land = make_land_mask()
    before = np.mean(np.isfinite(data[:, ~land]))
    filled = np.stack([fill_gaps(fr, ocean_mask=~land, **kwargs) for fr in data])
    after = np.mean(np.isfinite(filled[:, ~land]))
    new_name = new_name or name + "_smooth"
    print("Океан із даними: було %.0f%%, стало %.0f%%" % (before * 100, after * 100))
    extra = {k: d[k] for k in ("dates", "time_note") if k in d}
    extra["interpolated_share"] = round(float(after - before), 3)
    extra["note"] = "Дірки в океані заповнені інтерполяцією за сусідніми клітинками і згладжені"
    return export_layer(
        new_name, filled, d["years"],
        units=d["units"],
        description=d["description"] + " (дірки заповнені інтерполяцією, згладжено)",
        source=d["source"], extra_meta=extra,
    )


# ---------------------------------------------------------------------------
# 4. Усе разом
# ---------------------------------------------------------------------------
def build_ocean_video_layer(video_path, n_frames=48, box=None,
                            start_date="2007-03-25", end_date="2008-03-03",
                            use_land_mask=True, with_activity=True, fill=True,
                            name="ocean_sst_video"):
    """Відео -> шар(и) у форматі Earth Jukebox.
    n_frames=48 це приблизно один кадр на тиждень (відео охоплює рік)."""
    land = make_land_mask() if use_land_mask else None
    t0, t1 = pd.Timestamp(start_date), pd.Timestamp(end_date)

    temps, acts, dates = [], [], []
    for idx, total, frame in iter_frames(video_path, n_frames):
        t, a = decode_frame(frame, box=box, land_mask=land)
        temps.append(t)
        acts.append(a)
        frac = idx / max(total - 1, 1)
        dates.append((t0 + (t1 - t0) * frac).strftime("%Y-%m-%d"))
        if len(temps) % 10 == 0:
            print("  оброблено кадрів:", len(temps))

    temps = np.stack(temps)   # [кадр][lat][lon]
    extra_fill = {}
    if fill:
        ocean = None if land is None else ~land
        before = np.mean(np.isfinite(temps[:, ocean])) if ocean is not None else np.mean(np.isfinite(temps))
        temps = np.stack([fill_gaps(t, ocean_mask=ocean) for t in temps])
        after = np.mean(np.isfinite(temps[:, ocean])) if ocean is not None else np.mean(np.isfinite(temps))
        print("Океан із даними: було %.0f%%, стало %.0f%%" % (before * 100, after * 100))
        extra_fill = {"interpolated_share": round(float(after - before), 3),
                      "note": "Дірки в океані заповнені інтерполяцією за сусідніми клітинками"}
    years = [int(d[:4]) for d in dates]
    extra = {"dates": dates,
             "time_note": "Дати кадрів оцінені рівномірно між %s і %s" % (start_date, end_date)}
    extra_sst = dict(extra, **extra_fill)

    meta = export_layer(
        name, temps, years,
        units="°C (оцінка за кольором)",
        description="Температура поверхні океану, розшифрована з кольору відео NASA SVS 3912 (ECCO2)",
        source="NASA SVS 3912 Global Sea Surface Currents and Temperature; модель ECCO2 (MIT/JPL); "
               "https://svs.gsfc.nasa.gov/3912/",
        extra_meta=extra_sst,
    )
    if with_activity:
        export_layer(
            name.replace("sst_video", "flow_activity") if "sst_video" in name else name + "_activity",
            np.stack(acts), years,
            units="частка (0-1), візуальний показник",
            description="Частка кольорових пікселів у клітинці (скільки ліній течій намальовано). "
                        "Це візуальний показник, а не фізична швидкість течії.",
            source="NASA SVS 3912 (розрахунок з візуалізації)",
            extra_meta=extra,
        )
    return meta


# ---------------------------------------------------------------------------
# ПРИКЛАД (кожен рядок в окремій клітинці Colab)
# ---------------------------------------------------------------------------
# !wget -q https://svs.gsfc.nasa.gov/vis/a000000/a003900/a003912/ecco2_sst_flow.mp4
# from video_to_layer import *
# fr = get_frame("ecco2_sst_flow.mp4"); show_frame(fr)       # подивитись кадр
# build_ocean_video_layer("ecco2_sst_flow.mp4")              # зробити шар (дірки вже заповнюються)
# fill_saved_layer("ocean_sst_video")   # АБО: заповнити дірки в уже готовому шарі, без перерахунку відео
# preview("ocean_sst_video")                                 # картинка-перевірка
# zip_layers()                                               # архіви
