"""Палець на паперовій карті -> температура океану (NOAA ERSST v5) у консоль.

Камера дивиться на карту світу. MediaPipe знаходить кінчик вказівного пальця.
Якщо палець тримається на одному місці 3 секунди, програма перетворює точку
камери в широту/довготу і друкує температуру поверхні океану.

Запуск:
    .venv/bin/python finger_map.py                 # калібрування (перший раз) + трекінг
    .venv/bin/python finger_map.py --recalibrate   # якщо камеру або карту зсунули
    .venv/bin/python finger_map.py --selftest      # перевірка математики без камери

Калібрування: клацніть мишкою на зображенні камери запропоновані точки
(південний край Африки, Гібралтар, ...). Мінімум 4, краще 6-8: тоді програма
сама визначить проєкцію карти (рівнокутна, Меркатор, Міллер, Робінсон).
"""
import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CALIB_FILE = HERE / "calibration.json"
MODEL_FILE = HERE / "models" / "hand_landmarker.task"
SST_FILE = HERE / "sst.mnmean.nc"
INDEX_TIP = 8  # MediaPipe: кінчик вказівного пальця

# Впізнавані точки для калібрування: (назва для вікна, назва для консолі, lat, lon)
LANDMARKS = [
    ("Cape Horn - S tip of South America", "мис Горн (південь Південної Америки)", -55.98, -67.27),
    ("Cape Agulhas - S tip of Africa", "мис Агульяс (південь Африки)", -34.83, 20.00),
    ("Strait of Gibraltar", "Гібралтарська протока", 35.95, -5.60),
    ("Cape Farewell - S tip of Greenland", "мис Фарвель (південь Гренландії)", 59.77, -43.90),
    ("Cape Comorin - S tip of India", "мис Коморин (південь Індії)", 8.08, 77.55),
    ("Cape York - N tip of Australia", "мис Йорк (північ Австралії)", -10.69, 142.53),
    ("Cape Lopatka - S tip of Kamchatka", "мис Лопатка (південь Камчатки)", 50.87, 156.67),
    ("Cabo San Lucas - S tip of Baja California", "Кабо-Сан-Лукас (південь Каліфорнії)", 22.88, -109.91),
]

# ----------------------------------------------------------------------------
# Картографічні проєкції: forward (lat, lon_rel) -> (x, y) та inverse
# lon_rel = довгота відносно центрального меридіана карти lon0, в [-180, 180)
# ----------------------------------------------------------------------------
_ROB_LAT = np.arange(0, 95, 5, dtype=float)
_ROB_X = np.array([1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600, 0.9427, 0.9216, 0.8962,
                   0.8679, 0.8350, 0.7986, 0.7597, 0.7186, 0.6732, 0.6213, 0.5722, 0.5322])
_ROB_Y = np.array([0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720, 0.4340, 0.4958, 0.5571,
                   0.6176, 0.6769, 0.7346, 0.7903, 0.8435, 0.8936, 0.9394, 0.9761, 1.0000])


def wrap180(lon):
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def fwd(proj, lat, lon_rel):
    lat = np.asarray(lat, float); lon_rel = np.asarray(lon_rel, float)
    phi, lam = np.radians(lat), np.radians(lon_rel)
    if proj == "equirect":
        return lon_rel, lat
    if proj == "mercator":
        return lam, np.log(np.tan(np.pi / 4 + np.clip(phi, -1.5, 1.5) / 2))
    if proj == "miller":
        return lam, 1.25 * np.log(np.tan(np.pi / 4 + 0.4 * phi))
    if proj == "robinson":
        a = np.abs(lat)
        return 0.8487 * np.interp(a, _ROB_LAT, _ROB_X) * lam, 1.3523 * np.sign(lat) * np.interp(a, _ROB_LAT, _ROB_Y)
    raise ValueError(proj)


def inv(proj, x, y):
    """(x, y) -> (lat, lon_rel, valid)"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if proj == "equirect":
        lat, lon = y, x
    elif proj == "mercator":
        lat, lon = np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2), np.degrees(x)
    elif proj == "miller":
        lat, lon = np.degrees(2.5 * np.arctan(np.exp(0.8 * y)) - 0.625 * np.pi), np.degrees(x)
    elif proj == "robinson":
        yy = np.clip(np.abs(y) / 1.3523, 0, 1)
        a = np.interp(yy, _ROB_Y, _ROB_LAT)
        lat = np.sign(y) * a
        lon = np.degrees(x / (0.8487 * np.interp(a, _ROB_LAT, _ROB_X)))
        valid = (np.abs(y) <= 1.3523) & (np.abs(lon) <= 180)
        return lat, lon, valid
    else:
        raise ValueError(proj)
    valid = (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    return lat, lon, valid


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


# ----------------------------------------------------------------------------
# Калібрування: точки камери <-> lat/lon  =>  гомографія камера -> площина проєкції
# ----------------------------------------------------------------------------
class MapCalibration:
    def __init__(self, proj, lon0, H):
        self.proj, self.lon0, self.H = proj, float(lon0), np.asarray(H, float)

    def cam_to_latlon(self, px, py):
        pt = cv2.perspectiveTransform(np.array([[[px, py]]], np.float64), self.H)[0, 0]
        lat, lon_rel, ok = inv(self.proj, pt[0], pt[1])
        return float(lat), float(wrap180(lon_rel + self.lon0)), bool(ok)

    def save(self, path, cam_pts, geo_pts):
        path.write_text(json.dumps({"projection": self.proj, "lon0": self.lon0, "H": self.H.tolist(),
                                    "camera_points": np.asarray(cam_pts).tolist(),
                                    "geo_points": np.asarray(geo_pts).tolist()}, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(path.read_text())
        return cls(d["projection"], d["lon0"], d["H"])


def fit_calibration(cam_pts, geo_pts, fallback_proj="equirect", fallback_lon0=0.0, verbose=True):
    """Підбирає проєкцію + центральний меридіан, що найкраще пояснюють клацнуті точки."""
    cam = np.asarray(cam_pts, np.float64)
    geo = np.asarray(geo_pts, np.float64)
    if len(cam) < 4:
        raise ValueError("Потрібно щонайменше 4 точки")
    candidates = []
    projs = ["equirect", "mercator", "miller", "robinson"]
    lon0s = sorted({0.0, 10.0, 150.0, float(fallback_lon0)})
    for proj in projs:
        for lon0 in lon0s:
            x, y = fwd(proj, geo[:, 0], wrap180(geo[:, 1] - lon0))
            H, _ = cv2.findHomography(cam, np.stack([x, y], 1), 0)
            if H is None:
                continue
            cal = MapCalibration(proj, lon0, H)
            errs = [haversine_km(*cal.cam_to_latlon(*c)[:2], g[0], g[1]) for c, g in zip(cam, geo)]
            candidates.append((float(np.mean(errs)), float(np.max(errs)), cal))
    candidates.sort(key=lambda c: c[0])
    if len(cam) == 4:
        # 4 точки гомографія проходить точно для будь-якої проєкції -> вибрати не можна
        best = next(c for c in candidates if c[2].proj == fallback_proj and c[2].lon0 == fallback_lon0)
        if verbose:
            print(f"  Лише 4 точки: проєкцію неможливо визначити автоматично, беру '{fallback_proj}'.")
            print("  Для автовизначення клацніть 6+ точок.")
        return best[2]
    if verbose:
        print("  Похибка калібрування (середня / максимальна, км):")
        seen = set()
        for mean_e, max_e, cal in candidates:
            if cal.proj in seen:
                continue
            seen.add(cal.proj)
            mark = "  <- обрано" if cal is candidates[0][2] else ""
            print(f"    {cal.proj:9s} (центр {cal.lon0:+.0f}°): {mean_e:7.0f} / {max_e:7.0f}{mark}")
    return candidates[0][2]


def run_calibration(cap, args):
    """Інтерактивне калібрування: клік мишкою на кожну запропоновану точку."""
    print("\n=== КАЛІБРУВАННЯ ===")
    print("У вікні камери клацайте мишкою точку, назва якої написана зверху.")
    print("  s - пропустити точку (якщо її не видно на карті), u - скасувати останню,")
    print("  Enter - завершити (мінімум 4 точки, краще 6+), q - вийти.\n")
    clicks, cam_pts, geo_pts = [], [], []
    win = "Calibration"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda ev, x, y, *_: clicks.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    idx = 0
    if idx < len(LANDMARKS):
        print(f"  [{idx + 1}/{len(LANDMARKS)}] {LANDMARKS[idx][1]}")
    while True:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("Не вдалося отримати кадр з камери")
        if clicks:
            if idx < len(LANDMARKS):
                c = clicks[-1]
                cam_pts.append(c); geo_pts.append(LANDMARKS[idx][2:])
                print(f"      клік {c} -> {LANDMARKS[idx][2]:+.2f}, {LANDMARKS[idx][3]:+.2f}")
                idx += 1
                if idx < len(LANDMARKS):
                    print(f"  [{idx + 1}/{len(LANDMARKS)}] {LANDMARKS[idx][1]}")
            clicks.clear()
        for k, (c, g) in enumerate(zip(cam_pts, geo_pts)):
            cv2.circle(frame, c, 6, (0, 0, 255), 2)
            cv2.putText(frame, str(k + 1), (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        title = (f"Click: {LANDMARKS[idx][0]}" if idx < len(LANDMARKS) else "All points done - press Enter")
        _banner(frame, [title, f"points: {len(cam_pts)}  |  s=skip  u=undo  Enter=finish  q=quit"])
        cv2.imshow(win, frame)
        key = cv2.waitKey(15) & 0xFF
        if key == ord("q"):
            raise SystemExit(0)
        if key == ord("s") and idx < len(LANDMARKS):
            print("      пропущено")
            idx += 1
            if idx < len(LANDMARKS):
                print(f"  [{idx + 1}/{len(LANDMARKS)}] {LANDMARKS[idx][1]}")
        if key == ord("u") and cam_pts:
            cam_pts.pop(); geo_pts.pop(); idx = max(0, idx - 1)
            print(f"      скасовано. Знову: [{idx + 1}/{len(LANDMARKS)}] {LANDMARKS[idx][1]}")
        if key in (13, 10):
            if len(cam_pts) < 4:
                print("  Потрібно щонайменше 4 точки.")
                continue
            break
    cv2.destroyWindow(win)
    cal = fit_calibration(cam_pts, geo_pts, args.projection, args.lon0)
    cal.save(CALIB_FILE, cam_pts, geo_pts)
    print(f"  Збережено в {CALIB_FILE.name}: проєкція {cal.proj}, центральний меридіан {cal.lon0:+.0f}°\n")
    return cal


# ----------------------------------------------------------------------------
# Дані температури
# ----------------------------------------------------------------------------
MONTHS_UA = ["січень", "лютий", "березень", "квітень", "травень", "червень", "липень",
             "серпень", "вересень", "жовтень", "листопад", "грудень"]


class SSTLookup:
    def __init__(self, path, month=None):
        import xarray as xr
        ds = xr.open_dataset(path, engine="h5netcdf")
        t = ds.time.values
        ti = len(t) - 1 if month is None else int(np.argmin(np.abs(t - np.datetime64(month + "-01"))))
        self.date = ds.time.values[ti]
        m = int(str(self.date)[5:7])
        self.label = f"{MONTHS_UA[m - 1]} {str(self.date)[:4]}"
        self.cur = ds.sst.isel(time=ti).values
        base = ds.sst.sel(time=slice("1971", "2000"))
        self.base = base.sel(time=base["time.month"] == m).mean("time").values
        self.lat = ds.lat.values.astype(float)
        self.lon = ds.lon.values.astype(float)  # 0..358
        ds.close()

    def at(self, lat, lon, search_cells=3):
        i = int(np.argmin(np.abs(self.lat - lat)))
        j = int(np.argmin(np.abs(wrap180(self.lon - lon))))
        if not np.isnan(self.cur[i, j]):
            return {"land": False, "t": float(self.cur[i, j]), "base": float(self.base[i, j]),
                    "cell": (self.lat[i], wrap180(self.lon[j]).item()), "dist_km": 0.0}
        best = None  # суша -> шукаємо найближчу клітинку з водою
        n = len(self.lon)
        for di in range(-search_cells, search_cells + 1):
            for dj in range(-search_cells, search_cells + 1):
                ii, jj = i + di, (j + dj) % n
                if 0 <= ii < len(self.lat) and not np.isnan(self.cur[ii, jj]):
                    d = float(haversine_km(lat, lon, self.lat[ii], self.lon[jj]))
                    if best is None or d < best[0]:
                        best = (d, ii, jj)
        if best is None:
            return {"land": True, "t": None}
        d, ii, jj = best
        return {"land": True, "t": float(self.cur[ii, jj]), "base": float(self.base[ii, jj]),
                "cell": (self.lat[ii], wrap180(self.lon[jj]).item()), "dist_km": d}


def fmt_coord(lat, lon):
    return f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"


def report(lat, lon, sst):
    r = sst.at(lat, lon)
    now = datetime.now().strftime("%H:%M:%S")
    head = f"[{now}] {fmt_coord(lat, lon)}"
    if r["t"] is None:
        return f"{head} -> суша, води поблизу немає"
    tail = f"{r['t']:.1f} °C ({sst.label})"
    if not np.isnan(r["base"]):
        tail += f" | норма 1971–2000: {r['base']:.1f} °C ({r['t'] - r['base']:+.1f} °C)"
    if r["land"]:
        return f"{head} -> суша. Найближча вода (~{r['dist_km']:.0f} км): {tail}"
    return f"{head} -> {tail}"


# ----------------------------------------------------------------------------
# Трекінг пальця
# ----------------------------------------------------------------------------
def _banner(frame, lines):
    h = 24 * len(lines) + 10
    cv2.rectangle(frame, (0, 0), (frame.shape[1], h), (0, 0, 0), -1)
    for k, s in enumerate(lines):
        cv2.putText(frame, s, (10, 24 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def make_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL_FILE)),
                                        running_mode=vision.RunningMode.VIDEO, num_hands=1,
                                        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
    return mp, vision.HandLandmarker.create_from_options(opts)


def run_tracking(cap, cal, sst, args):
    mp, landmarker = make_landmarker()
    print(f"=== ТРЕКІНГ === Дані: {sst.label}. Тримайте палець {args.dwell:.0f} с на місці.")
    print("  У вікні: q - вийти, c - перекалібрувати.\n")
    win = "Earth Map"
    smooth = None            # згладжена позиція кінчика пальця
    anchor, anchor_t = None, 0.0
    fired = False
    last_seen = 0.0
    t0 = time.monotonic()
    last_ts = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Камера перестала віддавати кадри"); break
        h, w = frame.shape[:2]
        radius = args.radius * max(w, h)
        now = time.monotonic()
        ts = int((now - t0) * 1000)
        ts = max(ts, last_ts + 1); last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        tip = None
        if res.hand_landmarks:
            lm = res.hand_landmarks[0][INDEX_TIP]
            tip = np.array([lm.x * w, lm.y * h])
            smooth = tip if smooth is None else 0.5 * smooth + 0.5 * tip
            last_seen = now
        elif now - last_seen > 0.4:   # руку загублено -> скидаємо фіксацію
            smooth, anchor, fired = None, None, False

        lines = [f"data: {sst.label_en}  |  projection: {cal.proj}  |  q=quit  c=recalibrate"]
        if smooth is not None:
            px, py = smooth
            lat, lon, inside = cal.cam_to_latlon(px, py)
            if anchor is None or np.hypot(*(smooth - anchor)) > radius:
                anchor, anchor_t, fired = smooth.copy(), now, False
            frac = min(1.0, (now - anchor_t) / args.dwell)
            if frac >= 1.0 and not fired:
                fired = True
                print(report(lat, lon, sst) if inside else
                      f"[{datetime.now():%H:%M:%S}] палець поза картою")
            color = (0, 200, 0) if fired else (0, 165, 255)
            c = (int(px), int(py))
            cv2.circle(frame, c, 6, color, -1)
            cv2.circle(frame, (int(anchor[0]), int(anchor[1])), int(radius), (200, 200, 200), 1)
            cv2.ellipse(frame, c, (22, 22), -90, 0, 360 * frac, color, 3)
            lines.append(f"{abs(lat):.1f}{'N' if lat >= 0 else 'S'}  {abs(lon):.1f}{'E' if lon >= 0 else 'W'}"
                         + ("" if inside else "  (off map)") + ("  FIXED" if fired else ""))
        else:
            lines.append("no hand")
        _banner(frame, lines)
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            cv2.destroyWindow(win)
            cal = run_calibration(cap, args)
    landmarker.close()


# ----------------------------------------------------------------------------
# Самоперевірка без камери: синтетична "карта" під перспективою
# ----------------------------------------------------------------------------
def selftest(args):
    rng = np.random.default_rng(1)
    geo = np.array([l[2:] for l in LANDMARKS])
    # перспективна "камера": карта 1000 px завширшки, трохи нахилена
    Hc = np.array([[0.95, 0.08, 120], [-0.03, 0.9, 80], [0.00004, 0.00012, 1.0]])
    ok_all = True
    for proj in ["equirect", "mercator", "miller", "robinson"]:
        for lon0 in [0.0, 150.0]:
            xmax = float(fwd(proj, 0, 180)[0])
            s = 1000 / (2 * xmax)
            def to_cam(lat, lon):
                x, y = fwd(proj, lat, wrap180(np.asarray(lon) - lon0))
                p = np.stack([s * x + 500, -s * y + 400], -1).reshape(-1, 1, 2).astype(np.float64)
                return cv2.perspectiveTransform(p, Hc).reshape(-1, 2)
            cam = to_cam(geo[:, 0], geo[:, 1]) + rng.normal(0, 1.5, (len(geo), 2))  # ~1.5 px неточність кліку
            cal = fit_calibration(cam, geo, verbose=False)
            tl = rng.uniform(-70, 70, 300); tn = rng.uniform(-180, 180, 300)
            tc = to_cam(tl, tn)
            errs = [haversine_km(*cal.cam_to_latlon(*c)[:2], a, b) for c, a, b in zip(tc, tl, tn)]
            good = cal.proj == proj
            ok_all &= good
            print(f"  карта {proj:9s} центр {lon0:+4.0f}° -> визначено {cal.proj:9s} центр {cal.lon0:+4.0f}° "
                  f"{'OK ' if good else 'FAIL'} | похибка 300 точок: медіана {np.median(errs):5.0f} км, "
                  f"95% {np.percentile(errs, 95):5.0f} км")
    sst = SSTLookup(SST_FILE, args.month)
    print(f"\n  Температура ({sst.label}):")
    for name, la, lo in [("Чорне море", 43.5, 34.0), ("Сахара (суша)", 23.0, 12.0),
                         ("Одеса (берег)", 46.48, 30.73), ("Екватор, Тихий океан", 0.0, -140.0),
                         ("Антарктида (суша)", -82.0, 40.0)]:
        print(f"    {name:22s} {report(la, lo, sst)}")
    print("\nSELFTEST", "PASSED" if ok_all else "FAILED")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="індекс камери (0 - вбудована)")
    ap.add_argument("--recalibrate", action="store_true", help="пройти калібрування заново")
    ap.add_argument("--dwell", type=float, default=3.0, help="скільки секунд тримати палець (3)")
    ap.add_argument("--radius", type=float, default=0.02,
                    help="допустиме тремтіння пальця, частка ширини кадру (0.02 = 2%%)")
    ap.add_argument("--month", help="місяць даних РРРР-ММ (за замовчуванням останній у файлі)")
    ap.add_argument("--projection", default="equirect", choices=["equirect", "mercator", "miller", "robinson"],
                    help="проєкція, якщо клацнуто лише 4 точки")
    ap.add_argument("--lon0", type=float, default=0.0, help="центральний меридіан карти, якщо 4 точки")
    ap.add_argument("--width", type=int, default=1280, help="ширина кадру камери")
    ap.add_argument("--selftest", action="store_true", help="перевірка без камери")
    args = ap.parse_args()

    if args.selftest:
        selftest(args); return

    print("Завантаження даних температури...")
    sst = SSTLookup(SST_FILE, args.month)
    m = int(str(sst.date)[5:7])
    sst.label_en = f"{str(sst.date)[:4]}-{m:02d}"

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.width * 9 / 16))
    if not cap.isOpened():
        raise SystemExit("Камера не відкрилась. На macOS дозвольте доступ: Системні параметри -> "
                         "Приватність і безпека -> Камера -> ваш термінал / VS Code.")
    try:
        if CALIB_FILE.exists() and not args.recalibrate:
            cal = MapCalibration.load(CALIB_FILE)
            print(f"Калібрування з {CALIB_FILE.name}: проєкція {cal.proj}. (--recalibrate щоб переробити)")
        else:
            cal = run_calibration(cap, args)
        run_tracking(cap, cal, sst, args)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
