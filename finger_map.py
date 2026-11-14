"""Палець на паперовій карті -> температура океану (NOAA ERSST v5).

Камера дивиться на роздруковану карту світу. Програма:
  * відстежує карту за її власним малюнком (ORB-точки + гомографія), без фізичних
    міток, тож карту можна зсувати й повертати на столі без повторного калібрування;
  * відстежує кінчик вказівного пальця (MediaPipe);
  * у панелі праворуч наживо показує координати й температуру під пальцем;
  * у вікні "Ocean Temperature" точка на карті температури повторює рух пальця;
  * сонифікація: температура під пальцем звучить наживо (висота ноти, тембр,
    биття/пульсація при потеплінні, стерео за довготою; m - вимкнути звук);
  * якщо палець тримається на місці 3 секунди, записує точку в консоль (і дзвіночок).

Запуск:
    .venv/bin/python finger_map.py                 # калібрування (перший раз) + трекінг
    .venv/bin/python finger_map.py --recalibrate   # пройти калібрування заново
    .venv/bin/python finger_map.py --selftest      # перевірка математики без камери

Калібрування (2 кроки):
  1) 4 кути карти (віртуально): клацніть їх мишкою або натисніть A, щоб прийняти
     кути, знайдені автоматично (зелена рамка). У цей момент програма запам'ятовує
     вигляд карти як еталон, тому рука не має бути над картою;
  2) 6-8 впізнаваних місць (мис Горн, Гібралтар, ...), див. landmarks_reference.png.
     За ними програма визначає проєкцію карти.
"""
import argparse
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CALIB_FILE = HERE / "calibration.json"
CALIB_VERSION = 3
TEMPLATE_FILE = HERE / "calibration_template.png"
MODEL_FILE = HERE / "models" / "hand_landmarker.task"
SST_FILE = HERE / "sst.mnmean.nc"
COAST_FILE = HERE / "data" / "ne_50m_land.geojson"
INDEX_TIP = 8  # MediaPipe: кінчик вказівного пальця
WIN_CAM, WIN_SST = "Earth Map", "Ocean Temperature"

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
    ("North Cape - N tip of Norway", "Нордкап (північ Норвегії)", 71.17, 25.78),
]
CORNER_NAMES = [("TOP-LEFT", "верхній лівий"), ("TOP-RIGHT", "верхній правий"),
                ("BOTTOM-RIGHT", "нижній правий"), ("BOTTOM-LEFT", "нижній лівий")]

# Кольори (RGB для PIL / matplotlib)
SURFACE, INK, INK2, MUTED, GRIDC, LANDC = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#d6d5cf"
WARM, COOL = "#d03b3b", "#256abf"
SST_VMIN, SST_VMAX = -2.0, 32.0


def sst_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("sst", ["#fdeee4", "#f6b48f", "#eb6834", "#b7461c", "#6e2509"])
    cm.set_bad(LANDC)
    return cm


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ----------------------------------------------------------------------------
# Картографічні проєкції: forward (lat, lon_rel) -> (x, y) та inverse
# lon_rel = довгота відносно центрального меридіана карти lon0, в [-180, 180)
# ----------------------------------------------------------------------------
_ROB_LAT = np.arange(0, 95, 5, dtype=float)
_ROB_X = np.array([1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600, 0.9427, 0.9216, 0.8962,
                   0.8679, 0.8350, 0.7986, 0.7597, 0.7186, 0.6732, 0.6213, 0.5722, 0.5322])
_ROB_Y = np.array([0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720, 0.4340, 0.4958, 0.5571,
                   0.6176, 0.6769, 0.7346, 0.7903, 0.8435, 0.8936, 0.9394, 0.9761, 1.0000])
PROJECTIONS = ["equirect", "mercator", "miller", "robinson"]


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
        a = np.interp(np.clip(np.abs(y) / 1.3523, 0, 1), _ROB_Y, _ROB_LAT)
        lat = np.sign(y) * a
        lon = np.degrees(x / (0.8487 * np.interp(a, _ROB_LAT, _ROB_X)))
        return lat, lon, (np.abs(y) <= 1.3523) & (np.abs(lon) <= 180)
    else:
        raise ValueError(proj)
    return lat, lon, (np.abs(lat) <= 90) & (np.abs(lon) <= 180)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


# ----------------------------------------------------------------------------
# Аркуш карти: 4 кути -> власні координати аркуша (u, v) у квадраті [0,1]x[0,1]
# Калібрування зберігається в (u, v), тому не залежить від положення аркуша на столі.
# ----------------------------------------------------------------------------
UNIT = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)  # TL, TR, BR, BL


def cam_to_sheet_H(corners):
    return cv2.getPerspectiveTransform(np.float32(corners), np.float32(UNIT))


def apply_H(H, pts):
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, np.asarray(H, np.float64)).reshape(-1, 2)


def quad_area(q):
    return abs(cv2.contourArea(np.float32(q)))


def order_quad(q):
    """Кути за годинниковою стрілкою (на екрані), починаючи з верхнього лівого."""
    q = np.asarray(q, float)
    c = q.mean(0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]
    return np.roll(q, -int(np.argmin(q.sum(1))), axis=0)


def align_quad(q, ref):
    """Нумерація кутів q, найближча до попередніх кутів ref (щоб поворот аркуша не плутав кути)."""
    q = order_quad(q)
    best = min(range(4), key=lambda r: np.linalg.norm(np.roll(q, -r, 0) - ref, axis=1).sum())
    return np.roll(q, -best, 0)


def _refine_quad(contour, quad):
    """Уточнення кутів: пряма через усі точки кожної сторони (fitLine), кут = перетин сусідніх прямих."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    lines = []
    for k in range(4):
        a, b = quad[k], quad[(k + 1) % 4]
        L = float(np.linalg.norm(b - a))
        n = (b - a) / L
        rel = pts - a
        t = rel @ n / L
        dist = np.abs(rel[:, 0] * n[1] - rel[:, 1] * n[0])
        sel = pts[(t > 0.12) & (t < 0.88) & (dist < max(3.0, 0.03 * L))]
        if len(sel) < 8:
            return quad
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
    out = []
    for k in range(4):  # кут k лежить між стороною k-1 і стороною k
        (p1, d1), (p2, d2) = lines[k - 1], lines[k]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-3:
            return quad
        s = np.linalg.solve(A, p2 - p1)
        out.append(p1 + s[0] * d1)
    out = np.array(out)
    return out if np.linalg.norm(out - quad, axis=1).max() < 0.05 * np.sqrt(quad_area(quad)) else quad


def detect_quads(frame, min_area_frac=0.06):
    """Кандидати на аркуш: великі опуклі чотирикутники (межі за Canny та Otsu)."""
    h, w = frame.shape[:2]
    s = min(1.0, 640.0 / max(h, w))
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    # close (а не dilate): закриває розриви в контурі, але не зсуває межу аркуша назовні
    edges = cv2.morphologyEx(cv2.Canny(gray, 40, 120), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    area_min = min_area_frac * small.shape[0] * small.shape[1]
    out = []
    for mask in (otsu, edges):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if cv2.contourArea(c) < area_min:
                continue
            peri = cv2.arcLength(c, True)
            for eps in (0.02, 0.035, 0.05):
                ap = cv2.approxPolyDP(c, eps * peri, True)
                if len(ap) == 4 and cv2.isContourConvex(ap):
                    q = _refine_quad(c, ap.reshape(4, 2).astype(float))
                    out.append(q / s)
                    break
    return out


class SheetTracker:
    """Тримає актуальні 4 кути аркуша. Якщо аркуш не знайдено (рука перекрила край),
    лишає останні відомі кути."""

    def __init__(self, ref_corners, frame_shape, static=False):
        self.corners = np.asarray(ref_corners, float)
        self.diag = float(np.hypot(*frame_shape[:2]))
        self.static = static
        self.locked = False   # перше захоплення після запуску: дозволяємо будь-яке зміщення
        self.status = "static" if static else "searching"

    def update(self, frame):
        if self.static:
            return self.corners
        ref_area = quad_area(self.corners)
        best = None
        for q in detect_quads(frame):
            q = align_quad(q, self.corners)
            d = float(np.mean(np.linalg.norm(q - self.corners, axis=1)))
            ratio = quad_area(q) / max(ref_area, 1.0)
            max_d = 0.25 * self.diag if self.locked else 10 * self.diag
            if 0.5 < ratio < 2.0 and d < max_d and (best is None or d < best[0]):
                best = (d, q)
        if best is None:
            self.status = "held" if self.locked else "searching"
            return self.corners
        d, q = best
        # дрібне тремтіння згладжуємо, справжній рух беремо одразу
        self.corners = 0.6 * self.corners + 0.4 * q if d < 0.008 * self.diag else q
        self.locked, self.status = True, "tracked"
        return self.corners


def make_template(frame, corners):
    """Еталон карти: ділянка кадру всередині 4 кутів, випрямлена в прямокутник."""
    c = np.asarray(corners, float)
    top, bottom = np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[3])
    left, right = np.linalg.norm(c[3] - c[0]), np.linalg.norm(c[2] - c[1])
    tw = int(max(400, (top + bottom) / 2))
    th = int(round(tw * (left + right) / max(top + bottom, 1.0)))
    G = cv2.getPerspectiveTransform(np.float32(c), np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]))
    return cv2.warpPerspective(frame, G, (tw, th))


class FeatureTracker:
    """Відстеження карти за її власним малюнком (без міток і без країв аркуша).

    Під час калібрування запам'ятовується еталон: випрямлене зображення карти.
    У кожному кадрі шукаються ORB-точки (вигини узбереж, острови, написи), зіставляються
    з точками еталона, і за сотнями збігів рахується гомографія еталон -> кадр (MAGSAC).
    Рука закриває частину точок, але решти вистачає.
    """
    WORK_W = 960     # ширина кадру для пошуку точок (швидкість/точність)
    MIN_INLIERS = 25

    def __init__(self, template, ref_corners, frame_shape):
        g = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.kt, self.dt = cv2.ORB_create(3000).detectAndCompute(g, None)
        if self.dt is None or len(self.kt) < 60:
            raise ValueError("На еталоні карти замало деталей для відстеження")
        self.tpts = np.float32([k.pt for k in self.kt])
        th, tw = g.shape
        self.tcorners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
        self.orb = cv2.ORB_create(2500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.corners = np.asarray(ref_corners, float)
        self.diag = float(np.hypot(*frame_shape[:2]))
        self.locked, self.status, self.inliers = False, "searching", 0

    def match(self, frame):
        """-> (кути карти в кадрі або None, кількість точок, що зійшлися)"""
        s = min(1.0, self.WORK_W / frame.shape[1])
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
        kf, df = self.orb.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), None)
        if df is None or len(kf) < self.MIN_INLIERS:
            return None, 0
        pairs = self.bf.knnMatch(self.dt, df, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if len(good) < self.MIN_INLIERS:
            return None, 0
        src = self.tpts[[m.queryIdx for m in good]]
        dst = np.float32([kf[m.trainIdx].pt for m in good]) / s
        H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 4.0)
        if H is None:
            return None, 0
        n = int(inl.sum())
        if n < self.MIN_INLIERS or n < 0.25 * len(good):
            return None, n
        return cv2.perspectiveTransform(self.tcorners, H).reshape(4, 2).astype(float), n

    def update(self, frame):
        q, self.inliers = self.match(frame)
        ok = q is not None and cv2.isContourConvex(np.float32(q))
        if ok and self.locked:
            ok = 0.4 < quad_area(q) / max(quad_area(self.corners), 1.0) < 2.5
        if not ok:
            self.status = "held" if self.locked else "searching"
            return self.corners
        d = float(np.linalg.norm(q - self.corners, axis=1).max())
        # субпіксельне тремтіння гасимо, справжній рух беремо одразу
        self.corners = 0.7 * self.corners + 0.3 * q if self.locked and d < 0.0015 * self.diag else q
        self.locked, self.status = True, "tracked"
        return self.corners


def make_tracker(mode, cal, frame_shape):
    if mode == "features":
        if cal.template is None:
            raise SystemExit("Немає еталона карти для трекінгу -> запустіть з --recalibrate")
        return FeatureTracker(cal.template, cal.ref_corners, frame_shape)
    return SheetTracker(cal.ref_corners, frame_shape, static=(mode == "static"))


class OneEuro:
    """Фільтр One Euro для точки пальця: сильно згладжує, коли палець стоїть,
    і майже без затримки, коли він рухається."""

    def __init__(self, min_cutoff=1.2, beta=0.015, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self.x is None:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return self.x
        dt = max(t - self.t, 1e-3)
        self.t = t
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * ((x - self.x) / dt - self.dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        return self.x

    def reset(self):
        self.x = None


# ----------------------------------------------------------------------------
# Калібрування: точки аркуша (u, v) <-> lat/lon  =>  гомографія аркуш -> площина проєкції
# ----------------------------------------------------------------------------
class MapCalibration:
    def __init__(self, proj, lon0, H, ref_corners=None):
        self.proj, self.lon0, self.H = proj, float(lon0), np.asarray(H, float)
        self.ref_corners = None if ref_corners is None else np.asarray(ref_corners, float)
        self.template = None                 # випрямлене зображення карти (еталон для трекінгу)
        self.sheet_pts, self.geo_pts = [], []

    def sheet_to_latlon(self, u, v):
        x, y = apply_H(self.H, [[u, v]])[0]
        lat, lon_rel, ok = inv(self.proj, x, y)
        return float(lat), float(wrap180(lon_rel + self.lon0)), bool(ok)

    def save(self, path):
        if self.template is not None:
            cv2.imwrite(str(TEMPLATE_FILE), self.template)
        path.write_text(json.dumps({
            "version": CALIB_VERSION, "projection": self.proj, "lon0": self.lon0, "H": self.H.tolist(),
            "ref_corners": self.ref_corners.tolist(), "template": TEMPLATE_FILE.name,
            "sheet_points": np.asarray(self.sheet_pts).tolist(),
            "geo_points": np.asarray(self.geo_pts).tolist()}, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(path.read_text())
        if d.get("version") != CALIB_VERSION:
            return None
        cal = cls(d["projection"], d["lon0"], d["H"], d["ref_corners"])
        cal.sheet_pts, cal.geo_pts = d.get("sheet_points", []), d.get("geo_points", [])
        tpath = path.parent / d.get("template", TEMPLATE_FILE.name)
        cal.template = cv2.imread(str(tpath)) if tpath.exists() else None
        return cal


def fit_calibration(sheet_pts, geo_pts, fallback_proj="equirect", fallback_lon0=0.0, verbose=True):
    """Підбирає проєкцію + центральний меридіан, що найкраще пояснюють клацнуті точки."""
    src = np.asarray(sheet_pts, np.float64)
    geo = np.asarray(geo_pts, np.float64)
    if len(src) < 4:
        raise ValueError("Потрібно щонайменше 4 точки")
    candidates = []
    for proj in PROJECTIONS:
        for lon0 in sorted({0.0, 10.0, 150.0, float(fallback_lon0)}):
            x, y = fwd(proj, geo[:, 0], wrap180(geo[:, 1] - lon0))
            H, _ = cv2.findHomography(src, np.stack([x, y], 1), 0)
            if H is None:
                continue
            cal = MapCalibration(proj, lon0, H)
            errs = [haversine_km(*cal.sheet_to_latlon(*p)[:2], g[0], g[1]) for p, g in zip(src, geo)]
            candidates.append((float(np.mean(errs)), float(np.max(errs)), cal))
    candidates.sort(key=lambda c: c[0])
    if len(src) == 4:
        best = next(c for c in candidates if c[2].proj == fallback_proj and c[2].lon0 == fallback_lon0)
        if verbose:
            print(f"  Лише 4 точки: проєкцію неможливо визначити автоматично, беру '{fallback_proj}'.")
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


def _banner(frame, lines, scale=1.0):
    fs = 0.6 / scale
    lh = int(26 / scale)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], lh * len(lines) + int(10 / scale)), (0, 0, 0), -1)
    for k, s in enumerate(lines):
        cv2.putText(frame, s, (int(10 / scale), lh * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                    max(1, int(1 / scale)), cv2.LINE_AA)


def _show(win, frame, scale):
    cv2.imshow(win, cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else frame)


def _read(cap):
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Не вдалося отримати кадр з камери")
    return frame


def pick_corners(cap, scale, win="Calibration"):
    """Крок 1: 4 кути аркуша (клік мишкою або A = прийняти знайдені автоматично)."""
    print("\n--- Крок 1: кути аркуша карти ---")
    print("Клацніть 4 кути АРКУША в порядку: верхній лівий -> верхній правий -> нижній правий -> нижній лівий")
    print("(верх = північ карти). Або натисніть A, якщо зелена рамка вже точно лягла на аркуш.")
    print("  u - скасувати останній клік, q - вийти.")
    clicks = []
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, lambda ev, x, y, *_: clicks.append((x / scale, y / scale))
                         if ev == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        frame = _read(cap)
        quads = detect_quads(frame)
        auto = order_quad(max(quads, key=quad_area)) if quads else None
        if auto is not None:
            cv2.polylines(frame, [np.int32(auto)], True, (0, 200, 0), 2)
        for k, c in enumerate(clicks[:4]):
            cv2.circle(frame, (int(c[0]), int(c[1])), 7, (0, 0, 255), 2)
            cv2.putText(frame, CORNER_NAMES[k][0], (int(c[0]) + 10, int(c[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        n = len(clicks)
        _banner(frame, [f"Step 1/2: click sheet corner {min(n + 1, 4)}/4: {CORNER_NAMES[min(n, 3)][0]} (north up)",
                        ("A = accept green auto-detected frame  |  " if auto is not None else "")
                        + "u=undo  q=quit"], scale)
        _show(win, frame, scale)
        key = cv2.waitKey(15) & 0xFF
        if key == ord("q"):
            raise SystemExit(0)
        if key == ord("u") and clicks:
            clicks.pop()
        if key == ord("a") and auto is not None:
            print("  Прийнято автоматично знайдені кути.")
            return auto
        if len(clicks) >= 4:
            corners = np.array(clicks[:4], float)
            diag = np.hypot(*frame.shape[:2])
            if auto is not None:
                auto_al = align_quad(auto, corners)
                if np.all(np.linalg.norm(auto_al - corners, axis=1) < 0.04 * diag):
                    print("  Кліки збіглися з автоматично знайденим аркушем, використовую точні кути детектора.")
                    return auto_al
            print("  Кути задано вручну. (Автотрекінг шукатиме аркуш поруч із ними.)")
            return corners


def run_calibration(cap, args, scale):
    print("\n=== КАЛІБРУВАННЯ === Під час калібрування карту не рухайте.")
    corners = pick_corners(cap, scale)
    template = capture_template(cap, corners)
    G = cam_to_sheet_H(corners)
    print("\n--- Крок 2: впізнавані місця (див. landmarks_reference.png) ---")
    print("Клацайте місце, назва якого написана зверху. s - пропустити, u - скасувати,")
    print("Enter - завершити (мінімум 4, краще 6+), q - вийти.\n")
    clicks, cam_pts, sheet_pts, geo_pts = [], [], [], []
    win = "Calibration"
    cv2.setMouseCallback(win, lambda ev, x, y, *_: clicks.append((x / scale, y / scale))
                         if ev == cv2.EVENT_LBUTTONDOWN else None)
    idx = 0
    print(f"  [{idx + 1}/{len(LANDMARKS)}] {LANDMARKS[idx][1]}")
    while True:
        frame = _read(cap)
        if clicks:
            if idx < len(LANDMARKS):
                c = clicks[-1]
                uv = apply_H(G, [c])[0]
                cam_pts.append(c); sheet_pts.append(uv); geo_pts.append(LANDMARKS[idx][2:])
                warn = "  (увага: поза аркушем!)" if not (-0.02 <= uv[0] <= 1.02 and -0.02 <= uv[1] <= 1.02) else ""
                print(f"      аркуш u={uv[0]:.3f} v={uv[1]:.3f} -> {LANDMARKS[idx][2]:+.2f}, {LANDMARKS[idx][3]:+.2f}{warn}")
                idx += 1
                if idx < len(LANDMARKS):
                    print(f"  [{idx + 1}/{len(LANDMARKS)}] {LANDMARKS[idx][1]}")
            clicks.clear()
        cv2.polylines(frame, [np.int32(corners)], True, (0, 200, 0), 2)
        for k, c in enumerate(cam_pts):
            cv2.circle(frame, (int(c[0]), int(c[1])), 7, (0, 0, 255), 2)
            cv2.putText(frame, str(k + 1), (int(c[0]) + 9, int(c[1]) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        title = f"Step 2/2: click {LANDMARKS[idx][0]}" if idx < len(LANDMARKS) else "All points done - press Enter"
        _banner(frame, [title, f"points: {len(cam_pts)}  |  s=skip  u=undo  Enter=finish  q=quit"], scale)
        _show(win, frame, scale)
        key = cv2.waitKey(15) & 0xFF
        if key == ord("q"):
            raise SystemExit(0)
        if key == ord("s") and idx < len(LANDMARKS):
            print("      пропущено")
            idx += 1
            if idx < len(LANDMARKS):
                print(f"  [{idx + 1}/{len(LANDMARKS)}] {LANDMARKS[idx][1]}")
        if key == ord("u") and cam_pts:
            cam_pts.pop(); sheet_pts.pop(); geo_pts.pop(); idx = max(0, idx - 1)
            print(f"      скасовано. Знову: [{idx + 1}/{len(LANDMARKS)}] {LANDMARKS[idx][1]}")
        if key in (13, 10):
            if len(cam_pts) < 4:
                print("  Потрібно щонайменше 4 точки.")
                continue
            break
    cv2.destroyWindow(win)
    cal = fit_calibration(sheet_pts, geo_pts, args.projection, args.lon0)
    cal.ref_corners, cal.template = corners, template
    cal.sheet_pts, cal.geo_pts = sheet_pts, geo_pts
    cal.save(CALIB_FILE)
    print(f"  Збережено в {CALIB_FILE.name} + {TEMPLATE_FILE.name}: проєкція {cal.proj}, "
          f"центральний меридіан {cal.lon0:+.0f}°\n")
    return cal


def capture_template(cap, corners):
    """Знімок карти як еталон для трекінгу за малюнком. Рука в цей момент не має бути над картою."""
    template = make_template(_read(cap), corners)
    n = len(cv2.ORB_create(3000).detect(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), None))
    print(f"  Еталон карти: {template.shape[1]}x{template.shape[0]} px, характерних точок: {n}")
    if n < 300:
        print("  Увага: мало деталей. Трекінг за малюнком може бути нестабільним "
              "(краще більше світла / камера ближче), або запустіть з --track contour.")
    return template


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
        self.label_en = f"{str(self.date)[:4]}-{m:02d}"
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
            return {"land": False, "t": float(self.cur[i, j]), "base": float(self.base[i, j]), "dist_km": 0.0}
        best = None  # суша -> найближча клітинка з водою
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
        return {"land": True, "t": float(self.cur[ii, jj]), "base": float(self.base[ii, jj]), "dist_km": d}


def fmt_coord(lat, lon):
    return f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"


def report(lat, lon, sst):
    r = sst.at(lat, lon)
    head = f"[{datetime.now():%H:%M:%S}] {fmt_coord(lat, lon)}"
    if r["t"] is None:
        return f"{head} -> суша, води поблизу немає"
    tail = f"{r['t']:.1f} °C ({sst.label})"
    if not np.isnan(r["base"]):
        tail += f" | норма 1971–2000: {r['base']:.1f} °C ({r['t'] - r['base']:+.1f} °C)"
    if r["land"]:
        return f"{head} -> суша. Найближча вода (~{r['dist_km']:.0f} км): {tail}"
    return f"{head} -> {tail}"


# ----------------------------------------------------------------------------
# Панель з живою температурою (PIL, щоб була кирилиця)
# ----------------------------------------------------------------------------
class Panel:
    def __init__(self, width, sst):
        from matplotlib import font_manager
        from PIL import ImageFont
        fp = font_manager.FontProperties(family="DejaVu Sans")
        regular = font_manager.findfont(fp)
        bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
        self.f_xs = ImageFont.truetype(regular, 13)
        self.f_s = ImageFont.truetype(regular, 16)
        self.f_m = ImageFont.truetype(bold, 20)
        self.f_xl = ImageFont.truetype(bold, 50)
        self.w, self.sst = width, sst
        lut = (sst_cmap()(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
        self.bar_w = width - 40
        self.bar = np.repeat(lut[np.linspace(0, 255, self.bar_w).astype(int)][None], 14, axis=0)

    def render(self, height, info):
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (self.w, height), hex_rgb(SURFACE))
        d = ImageDraw.Draw(img)
        x, y = 20, 16
        d.text((x, y), "ПІД ПАЛЬЦЕМ", font=self.f_xs, fill=hex_rgb(MUTED)); y += 22
        if info.get("state") == "ok":
            d.text((x, y), fmt_coord(info["lat"], info["lon"]), font=self.f_m, fill=hex_rgb(INK)); y += 34
            r = info["r"]
            if r["t"] is None:
                d.text((x, y), "суша", font=self.f_xl, fill=hex_rgb(INK2)); y += 66
                d.text((x, y), "поблизу немає даних про воду", font=self.f_s, fill=hex_rgb(INK2)); y += 26
            else:
                d.text((x, y), f"{r['t']:.1f} °C", font=self.f_xl, fill=hex_rgb(INK)); y += 66
                if r["land"]:
                    d.text((x, y), f"суша: найближча вода ~{r['dist_km']:.0f} км", font=self.f_s,
                           fill=hex_rgb(INK2)); y += 24
                if not np.isnan(r["base"]):
                    an = r["t"] - r["base"]
                    d.text((x, y), f"норма 1971–2000: {r['base']:.1f} °C", font=self.f_s, fill=hex_rgb(INK2)); y += 24
                    d.text((x, y), f"аномалія: {an:+.1f} °C", font=self.f_m,
                           fill=hex_rgb(WARM if an > 0.05 else COOL if an < -0.05 else INK2)); y += 30
        else:
            msg = {"nohand": "руку не видно", "offmap": "палець поза картою"}.get(info.get("state"), "")
            d.text((x, y), msg, font=self.f_m, fill=hex_rgb(INK2)); y += 34 + 66 + 78
        # шкала температури з маркером
        y = max(y + 8, 230)
        img.paste(Image.fromarray(self.bar), (x, y))
        r = info.get("r")
        if info.get("state") == "ok" and r and r["t"] is not None:
            px = x + int(np.clip((r["t"] - SST_VMIN) / (SST_VMAX - SST_VMIN), 0, 1) * (self.bar_w - 1))
            d.polygon([(px - 6, y - 9), (px + 6, y - 9), (px, y - 1)], fill=hex_rgb(INK))
        for v in (SST_VMIN, 15, SST_VMAX):
            px = x + int((v - SST_VMIN) / (SST_VMAX - SST_VMIN) * (self.bar_w - 1))
            d.text((px - 10, y + 17), f"{v:.0f}°", font=self.f_xs, fill=hex_rgb(MUTED))
        y += 46
        d.line([(x, y), (self.w - 20, y)], fill=hex_rgb(GRIDC)); y += 10
        d.text((x, y), f"Дані: {self.sst.label}, NOAA ERSST v5", font=self.f_xs, fill=hex_rgb(INK2)); y += 20
        st = {"tracked": "знайдено", "held": "утримуються (карту не видно)", "searching": "пошук карти...",
              "static": "фіксовані"}[info.get("sheet", "static")]
        if info.get("inliers") and info.get("sheet") == "tracked":
            st += f" ({info['inliers']} точок)"
        d.text((x, y), f"Кути карти: {st}", font=self.f_xs, fill=hex_rgb(INK2)); y += 20
        snd = "вимкнено (m)" if info.get("muted") else (f"нота {info['note']}" if info.get("note") else "тиша")
        d.text((x, y), f"Звук: {snd}", font=self.f_xs, fill=hex_rgb(INK2)); y += 26
        # прогрес фіксації
        frac = info.get("dwell", 0.0)
        d.text((x, y), "Фіксація в консоль (3 с)", font=self.f_xs, fill=hex_rgb(INK2)); y += 18
        d.rectangle([x, y, self.w - 20, y + 8], fill=hex_rgb(GRIDC))
        if frac > 0:
            d.rectangle([x, y, x + int((self.w - 40) * frac), y + 8], fill=hex_rgb("#0ca30c" if frac >= 1 else "#eb6834"))
        y += 22
        if info.get("last_fix"):
            d.text((x, y), "Остання фіксація:", font=self.f_xs, fill=hex_rgb(MUTED)); y += 18
            for line in _wrap(info["last_fix"], 38)[:4]:
                d.text((x, y), line, font=self.f_xs, fill=hex_rgb(INK)); y += 17
        d.text((x, height - 24), "q вихід · c калібр. · k кути · m звук", font=self.f_xs, fill=hex_rgb(MUTED))
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _wrap(text, n):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur] if cur else lines


# ----------------------------------------------------------------------------
# Вікно з картою температури, де точка повторює рух пальця
# ----------------------------------------------------------------------------
class SSTMapView:
    def __init__(self, sst, width_in=10.0):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.collections import LineCollection
        from matplotlib.figure import Figure
        fig = Figure(figsize=(width_in, width_in * 0.56), dpi=100, facecolor=SURFACE)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.04, 0.17, 0.92, 0.73])
        data = np.roll(sst.cur, -int(np.argmin(np.abs(sst.lon - 180))), axis=1)  # lon 0..358 -> -180..178
        im = ax.imshow(np.ma.masked_invalid(data), extent=[-181, 179, -89, 89], cmap=sst_cmap(),
                       vmin=SST_VMIN, vmax=SST_VMAX, interpolation="nearest")
        if COAST_FILE.exists():
            land = json.loads(COAST_FILE.read_text())
            segs = []
            for f in land["features"]:
                g = f["geometry"]
                for p in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
                    segs.append(np.asarray(p[0]))
            ax.add_collection(LineCollection(segs, colors=INK2, linewidths=0.4))
        ax.set_facecolor(LANDC)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.set_aspect("equal")
        ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-90, 91, 30))
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRIDC)
        ax.set_title(f"Температура поверхні океану, {sst.label} (NOAA ERSST v5)",
                     fontsize=13, fontweight="bold", color=INK, loc="left")
        cb = fig.colorbar(im, cax=fig.add_axes([0.3, 0.07, 0.4, 0.03]), orientation="horizontal")
        cb.set_label("°C", color=INK2); cb.ax.tick_params(colors=MUTED, labelsize=8); cb.outline.set_visible(False)
        canvas.draw()
        self.base = cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
        H = self.base.shape[0]
        (x0, y0), (x1, y1) = ax.transData.transform([(-180, 90), (180, -90)])
        self.x0, self.y0, self.x1, self.y1 = x0, H - y0, x1, H - y1

    def to_px(self, lat, lon):
        return (int(round(self.x0 + (lon + 180) / 360 * (self.x1 - self.x0))),
                int(round(self.y0 + (90 - lat) / 180 * (self.y1 - self.y0))))

    def render(self, lat=None, lon=None, trail=(), dwell=0.0, temp=None):
        img = self.base.copy()
        pts = list(trail)
        for (la1, lo1), (la2, lo2) in zip(pts, pts[1:]):
            if abs(lo2 - lo1) < 90:  # не малюємо лінію через лінію зміни дат
                cv2.line(img, self.to_px(la1, lo1), self.to_px(la2, lo2), (120, 120, 120), 2, cv2.LINE_AA)
        if lat is not None:
            p = self.to_px(lat, lon)
            cv2.circle(img, p, 11, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, p, 7, (20, 20, 20), -1, cv2.LINE_AA)
            if dwell > 0:
                col = (12, 163, 12) if dwell >= 1 else (52, 104, 235)
                cv2.ellipse(img, p, (18, 18), -90, 0, 360 * min(dwell, 1.0), col, 3, cv2.LINE_AA)
            label = "land" if temp is None else f"{temp:.1f} C"
            cv2.putText(img, label, (p[0] + 22, p[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(img, label, (p[0] + 22, p[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        return img


# ----------------------------------------------------------------------------
# Трекінг пальця
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Сонифікація: температура -> звук у реальному часі
# ----------------------------------------------------------------------------
PENTATONIC = [0, 2, 4, 7, 9]           # мажорна пентатоніка: будь-які дві ноти звучать злагоджено
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def temp_to_midi(t, lo=SST_VMIN, hi=SST_VMAX, midi_lo=48, midi_hi=76):
    """Температура -> нота пентатоніки між C3 (холодно) і E5 (тепло)."""
    scale = [m for m in range(midi_lo, midi_hi + 1) if (m - midi_lo) % 12 in PENTATONIC]
    k = int(round(np.clip((t - lo) / (hi - lo), 0, 1) * (len(scale) - 1)))
    return scale[k]


def midi_to_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def midi_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


class Sonifier:
    """Адитивний синтезатор, параметри якого задають дані:
      температура  -> висота ноти (пентатоніка) і яскравість тембру (кількість гармонік);
      аномалія     -> биття (другий голос розстроєний) і пульсація-"серцебиття" при потеплінні;
      довгота      -> стерео-панорама (захід ліворуч, схід праворуч);
      суша         -> тихий шум; фіксація -> дзвіночок.
    Усі зміни плавні (портаменто/згладжування), щоб не було клацань.
    render() працює і без звукової карти (для запису WAV і тестів)."""
    SR = 44100
    N_H = 6
    TAU = {"freq": 0.06, "amp": 0.08, "noise": 0.15, "detune": 0.3, "bright": 0.3,
           "prate": 0.4, "pdepth": 0.3, "pan": 0.15}

    def __init__(self, enabled=True, volume=0.8):
        self.target = {"freq": 220.0, "amp": 0.0, "noise": 0.0, "detune": 0.0, "bright": 1.8,
                       "prate": 1.0, "pdepth": 0.0, "pan": 0.0}
        self.cur = dict(self.target)
        self.ph1 = np.zeros(self.N_H); self.ph2 = np.zeros(self.N_H); self.pph = 0.0
        self.chime_f, self.chime_t = None, 0.0
        self.noise_zi = np.zeros(1)
        self.rng = np.random.default_rng()
        self.volume, self.muted, self.stream = volume, False, None
        self.note = None
        if enabled:
            try:
                import sounddevice as sd
                self.stream = sd.OutputStream(samplerate=self.SR, channels=2, blocksize=512,
                                              callback=self._callback, latency="low")
                self.stream.start()
            except Exception as e:  # немає звукової карти тощо — працюємо без звуку
                print(f"  Звук вимкнено: {e}")

    # --- керування з головного потоку ---
    def water(self, temp, anomaly, lon, quiet=False):
        m = temp_to_midi(temp)
        self.note = m
        tnorm = float(np.clip((temp - SST_VMIN) / (SST_VMAX - SST_VMIN), 0, 1))
        an = 0.0 if anomaly is None or np.isnan(anomaly) else float(anomaly)
        self.target.update(
            freq=midi_to_hz(m),
            amp=0.45 if quiet else 1.0,
            noise=0.15 if quiet else 0.0,
            bright=2.4 - 1.5 * tnorm,                       # менший показник = більше гармонік = яскравіше
            detune=float(np.clip(abs(an) * 2.0, 0, 8)),     # Гц биття між двома голосами
            prate=float(np.clip(1.0 + max(an, 0) * 2.5, 1, 7)),
            pdepth=float(np.clip(max(an, 0) / 2.5, 0, 0.75)),
            pan=float(np.clip(lon / 180.0, -1, 1)))

    def land(self, lon):
        self.note = None
        self.target.update(amp=0.0, noise=0.35, pan=float(np.clip(lon / 180.0, -1, 1)))

    def silence(self):
        self.note = None
        self.target.update(amp=0.0, noise=0.0)

    def chime(self):
        self.chime_f, self.chime_t = 2 * self.target["freq"], 0.0

    def toggle_mute(self):
        self.muted = not self.muted
        return self.muted

    def close(self):
        if self.stream is not None:
            self.stream.stop(); self.stream.close()

    # --- синтез ---
    def _ramp(self, key, n):
        a = self.cur[key]
        b = a + (self.target[key] - a) * (1 - np.exp(-n / (self.SR * self.TAU[key])))
        self.cur[key] = b
        return np.linspace(a, b, n, endpoint=False)

    def render(self, n):
        from scipy.signal import lfilter
        f, amp, noise = self._ramp("freq", n), self._ramp("amp", n), self._ramp("noise", n)
        det, bright = self._ramp("detune", n), self._ramp("bright", n)
        prate, pdepth, pan = self._ramp("prate", n), self._ramp("pdepth", n), self._ramp("pan", n)
        h = np.arange(1, self.N_H + 1)[:, None]
        w = h ** (-bright[None, :])                                   # спад гармонік
        w /= w.sum(0, keepdims=True)
        inc1 = 2 * np.pi * h * f[None, :] / self.SR
        inc2 = 2 * np.pi * h * (f + det)[None, :] / self.SR
        p1 = self.ph1[:, None] + np.cumsum(inc1, 1)
        p2 = self.ph2[:, None] + np.cumsum(inc2, 1)
        self.ph1, self.ph2 = p1[:, -1] % (2 * np.pi), p2[:, -1] % (2 * np.pi)
        tone = 0.5 * ((w * np.sin(p1)).sum(0) + (w * np.sin(p2)).sum(0))
        pp = self.pph + np.cumsum(2 * np.pi * prate / self.SR)
        self.pph = pp[-1] % (2 * np.pi)
        tone *= 1 - pdepth * (0.5 + 0.5 * np.sin(pp))                 # пульсація-"серцебиття"
        brown, self.noise_zi = lfilter([0.02], [1, -0.98], self.rng.normal(0, 1, n), zi=self.noise_zi)
        sig = 0.6 * amp * tone + 0.8 * noise * brown
        if self.chime_f is not None:                                  # дзвіночок фіксації
            t = self.chime_t + np.arange(n) / self.SR
            env = np.exp(-t / 0.35)
            sig += 0.18 * env * (np.sin(2 * np.pi * self.chime_f * t) + 0.4 * np.sin(2 * np.pi * 2.76 * self.chime_f * t))
            self.chime_t = t[-1] + 1 / self.SR
            if self.chime_t > 2.0:
                self.chime_f = None
        sig = np.tanh(1.2 * sig) / 1.2 * self.volume
        theta = (pan + 1) * np.pi / 4                                 # рівнопотужна панорама
        return np.stack([sig * np.cos(theta), sig * np.sin(theta)], 1).astype(np.float32)

    def _callback(self, outdata, frames, time_info, status):
        out = self.render(frames)
        outdata[:] = 0 if self.muted else out


def write_wav(path, stereo, sr=Sonifier.SR):
    import wave
    data = (np.clip(stereo, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def make_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL_FILE)),
                                        running_mode=vision.RunningMode.VIDEO, num_hands=1,
                                        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
    return mp, vision.HandLandmarker.create_from_options(opts)


def run_tracking(cap, cal, sst, args, scale):
    mp, landmarker = make_landmarker()
    frame = _read(cap)
    tracker = make_tracker(args.track, cal, frame.shape)
    finger = OneEuro()
    son = Sonifier(enabled=not args.no_sound, volume=args.volume)
    if son.stream is not None:
        print("Звук: висота = температура (пентатоніка), яскравість = тепло, биття і пульсація = наскільки")
        print("      тепліше норми 1971–2000, ліво/право = довгота, шум = суша, дзвіночок = фіксація. m - вимкнути звук.")
    panel = Panel(330, sst)
    view = SSTMapView(sst)
    print(f"=== ТРЕКІНГ === Дані: {sst.label}. Панель і карта оновлюються наживо;")
    print(f"    якщо тримати палець {args.dwell:.0f} с на місці, точка запишеться сюди в консоль.")
    print("    У вікні: q - вийти, c - перекалібрувати все, k - заново позначити кути аркуша.\n")
    cv2.namedWindow(WIN_CAM, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WIN_SST, cv2.WINDOW_NORMAL)
    placed = False
    smooth, anchor, anchor_t, fired, last_seen = None, None, 0.0, False, 0.0
    trail = deque(maxlen=40)
    last_fix = ""
    t0, last_ts = time.monotonic(), -1
    while True:
        frame = _read(cap)
        h, w = frame.shape[:2]
        now = time.monotonic()
        ts = max(int((now - t0) * 1000), last_ts + 1); last_ts = ts

        corners = tracker.update(frame)
        G = cam_to_sheet_H(corners)

        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        if res.hand_landmarks:
            lm = res.hand_landmarks[0][INDEX_TIP]
            tip = np.array([lm.x * w, lm.y * h])
            smooth = finger(tip, now).copy()
            last_seen = now
        elif now - last_seen > 0.4:  # руку загублено
            smooth, anchor, fired = None, None, False
            finger.reset()

        info = {"state": "nohand", "sheet": tracker.status, "dwell": 0.0, "last_fix": last_fix,
                "inliers": getattr(tracker, "inliers", None)}
        lat = lon = temp = None
        if smooth is not None:
            u, v = apply_H(G, [smooth])[0]
            on_sheet = -0.02 <= u <= 1.02 and -0.02 <= v <= 1.02
            lat, lon, ok = cal.sheet_to_latlon(u, v)
            inside = ok and on_sheet
            radius = args.radius * max(w, h)
            if anchor is None or np.hypot(*(smooth - anchor)) > radius:
                anchor, anchor_t, fired = smooth.copy(), now, False
            frac = min(1.0, (now - anchor_t) / args.dwell)
            info["dwell"] = frac
            if inside:
                r = sst.at(lat, lon)
                temp = r["t"] if not r["land"] else (None if r["t"] is None else r["t"])
                info.update(state="ok", lat=lat, lon=lon, r=r)
                trail.append((lat, lon))
                if frac >= 1.0 and not fired:
                    fired = True
                    last_fix = report(lat, lon, sst)
                    print(last_fix)
                    son.chime()
            else:
                info["state"] = "offmap"
                lat = lon = None
                if frac >= 1.0 and not fired:
                    fired = True
                    print(f"[{datetime.now():%H:%M:%S}] палець поза картою")
            col = (12, 163, 12) if fired else (52, 104, 235)
            c = (int(smooth[0]), int(smooth[1]))
            cv2.circle(frame, c, 7, col, -1)
            cv2.ellipse(frame, c, (24, 24), -90, 0, 360 * frac, col, 3)

        # ---- звук ----
        if info["state"] == "ok":
            r = info["r"]
            if r["t"] is None:
                son.land(info["lon"])
            else:
                son.water(r["t"], r["t"] - r["base"], info["lon"], quiet=r["land"])
        else:
            son.silence()
        info["note"] = None if son.note is None else f"{midi_name(son.note)} · {midi_to_hz(son.note):.0f} Гц"
        info["muted"] = son.muted or son.stream is None

        qcol = {"tracked": (0, 200, 0), "held": (0, 200, 255), "searching": (0, 0, 255), "static": (200, 200, 200)}
        cv2.polylines(frame, [np.int32(corners)], True, qcol[tracker.status], 2)
        for k, p in enumerate(corners):
            cv2.circle(frame, (int(p[0]), int(p[1])), 6, qcol[tracker.status], -1)

        cam = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(WIN_CAM, np.hstack([cam, panel.render(cam.shape[0], info)]))
        sst_img = view.render(lat, lon, trail, info["dwell"] if lat is not None else 0.0, temp)
        cv2.imshow(WIN_SST, sst_img)
        if not placed:
            cv2.moveWindow(WIN_CAM, 0, 0)
            cv2.resizeWindow(WIN_SST, int(sst_img.shape[1] * 0.8), int(sst_img.shape[0] * 0.8))
            cv2.moveWindow(WIN_SST, 0, cam.shape[0] + 40)
            placed = True

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            print("Звук вимкнено" if son.toggle_mute() else "Звук увімкнено")
        if key == ord("c"):
            son.silence()
            cal = run_calibration(cap, args, scale)
            tracker = make_tracker(args.track, cal, frame.shape)
        if key == ord("k"):
            # ті самі 4 кути заново (напр. змінилось освітлення) -> новий еталон, калібрування місць лишається
            corners = pick_corners(cap, scale)
            cal.template = capture_template(cap, corners)
            cv2.destroyWindow("Calibration")
            cal.ref_corners = corners
            cal.save(CALIB_FILE)
            tracker = make_tracker(args.track, cal, frame.shape)
            tracker.locked = True
    son.close()
    landmarker.close()


# ----------------------------------------------------------------------------
# Самоперевірка без камери
# ----------------------------------------------------------------------------
def _synthetic_map_texture(w=900, h=450, seed=11):
    """Схожа на карту текстура для тесту: 'материки' з контурами, написи, сітка."""
    g = np.random.default_rng(seed)
    img = np.full((h, w, 3), (235, 220, 190), np.uint8)
    for x in range(0, w, 50):
        cv2.line(img, (x, 0), (x, h), (215, 200, 170), 1)
    for y in range(0, h, 50):
        cv2.line(img, (0, y), (w, y), (215, 200, 170), 1)
    for _ in range(16):
        c = g.uniform([0, 0], [w, h])
        n = int(g.integers(10, 18))
        ang = np.sort(g.uniform(0, 2 * np.pi, n))
        rad = g.uniform(25, 95, n)
        pts = np.int32(c + np.c_[np.cos(ang) * rad, np.sin(ang) * rad])
        cv2.fillPoly(img, [pts], tuple(int(v) for v in g.integers(90, 200, 3)))
        cv2.polylines(img, [pts], True, (60, 60, 60), 1, cv2.LINE_AA)
    words = ["OCEAN", "SEA", "BAY", "CAPE", "GULF", "ISLAND", "STRAIT", "RIVER"]
    for _ in range(30):
        p = (int(g.integers(0, w - 60)), int(g.integers(12, h)))
        cv2.putText(img, words[int(g.integers(len(words)))], p, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    return img


def selftest(args):
    rng = np.random.default_rng(1)
    geo = np.array([l[2:] for l in LANDMARKS])
    # Аркуш у кадрі 1280x720 під час калібрування, і після того, як його зсунули й повернули на 20°
    cal_corners = np.array([[250, 120], [1050, 140], [1080, 620], [220, 600]], float)
    c, (hw, hh), a = np.array([700, 380]), (330, 220), np.radians(20)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    moved_corners = (np.array([[-hw, -hh], [hw, -hh], [hw * 1.03, hh], [-hw * 0.97, hh]]) @ R.T) + c
    def scene(corners_px, hand=False):
        """Синтетичний кадр: стіл з шумом, світлий аркуш з 'материками', опційно рука через край."""
        g = np.random.default_rng(7)  # однаковий "малюнок" карти для обох положень
        Hc = cv2.getPerspectiveTransform(np.float32(UNIT), np.float32(corners_px))
        img = cv2.add(np.full((720, 1280, 3), 70, np.uint8), g.integers(0, 25, (720, 1280, 3), dtype=np.uint8))
        cv2.fillConvexPoly(img, np.int32(np.round(corners_px)), (235, 235, 230), cv2.LINE_AA)
        for _ in range(40):
            p = apply_H(Hc, g.uniform(0.1, 0.9, (1, 2)))[0]
            cv2.circle(img, (int(p[0]), int(p[1])), int(g.integers(10, 40)), tuple(int(v) for v in g.integers(60, 200, 3)), -1)
        if hand:
            mid = ((corners_px[2] + corners_px[3]) / 2).astype(int)
            cv2.line(img, tuple(mid), (mid[0] + 40, 719), (90, 140, 200), 90)
        return img, Hc

    img1, Hc1 = scene(cal_corners)
    img2, Hc2 = scene(moved_corners)
    t1 = SheetTracker(cal_corners, img1.shape); c1 = t1.update(img1).copy()
    t2 = SheetTracker(c1, img2.shape); c2 = t2.update(img2).copy()
    e1 = np.linalg.norm(c1 - cal_corners, axis=1).max()
    e2 = np.linalg.norm(c2 - moved_corners, axis=1).max()
    ok_all = t1.status == t2.status == "tracked" and max(e1, e2) < 3
    print("1) Режим contour (краї аркуша), синтетичні кадри:")
    print(f"   калібрувальне положення: {t1.status}, похибка кутів макс {e1:.2f} px")
    print(f"   зсунуто+повернуто на 20°: {t2.status}, похибка кутів макс {e2:.2f} px  {'OK' if ok_all else 'FAIL'}")
    hand_img, _ = scene(moved_corners, hand=True)
    before = t2.corners.copy()
    t2.update(hand_img)
    drift = np.linalg.norm(t2.corners - before, axis=1).max()
    print(f"   рука перекрила край: {t2.status}, зсув кутів {drift:.2f} px  {'OK' if drift < 3 else 'FAIL'}")
    ok_all &= drift < 3

    # ---- режим features: віртуальні кути (кліки) + трекінг за малюнком карти ----
    tex = _synthetic_map_texture()
    th_, tw_ = tex.shape[:2]

    def scene_tex(corners_px, hand=False, seed=3, sensor_noise=0.0):
        g = np.random.default_rng(seed)
        Hc = cv2.getPerspectiveTransform(np.float32(UNIT), np.float32(corners_px))
        Ht = Hc @ np.diag([1 / tw_, 1 / th_, 1.0])  # пікселі текстури -> одиничний квадрат -> кадр
        img = cv2.add(np.full((720, 1280, 3), 200, np.uint8), g.integers(0, 25, (720, 1280, 3), dtype=np.uint8))
        warped = cv2.warpPerspective(tex, Ht, (1280, 720))
        mask = cv2.warpPerspective(np.full((th_, tw_), 255, np.uint8), Ht, (1280, 720))
        img[mask > 0] = warped[mask > 0]
        if hand:
            mid = apply_H(Hc, [[0.55, 0.5]])[0].astype(int)
            cv2.line(img, tuple(mid), (mid[0] + 60, 719), (90, 140, 200), 90)
            cv2.circle(img, tuple(mid), 45, (90, 140, 200), -1)
        if sensor_noise:
            img = np.clip(img + g.normal(0, sensor_noise, img.shape), 0, 255).astype(np.uint8)
        return img, Hc

    print("\n2) Режим features (за малюнком карти; стіл світлий, як папір, щоб краї НЕ допомагали):")
    f1, fH1 = scene_tex(cal_corners)
    clicked = cal_corners + rng.normal(0, 1.5, (4, 2))      # віртуальні кути: кліки мишкою ±1.5 px
    ft = FeatureTracker(make_template(f1, clicked), clicked, f1.shape)
    f2, fH2 = scene_tex(moved_corners, hand=True)
    truth2 = apply_H(fH2 @ np.linalg.inv(fH1), clicked)     # де мають опинитись ті самі кути після переміщення
    fc2 = ft.update(f2).copy()
    ef = np.linalg.norm(fc2 - truth2, axis=1).max()
    ok_f = ft.status == "tracked" and ef < 3
    print(f"   зсунуто+повернуто на 20° + рука над картою: {ft.status}, {ft.inliers} точок, "
          f"похибка кутів макс {ef:.2f} px  {'OK' if ok_f else 'FAIL'}")
    jit = []
    for k in range(8):  # нерухома карта, але кожен кадр з новим шумом сенсора
        fk, _ = scene_tex(moved_corners, hand=True, seed=100 + k, sensor_noise=6)
        jit.append(ft.update(fk).copy())
    jitter = float(np.linalg.norm(np.array(jit) - np.mean(jit, 0), axis=2).max())
    ok_j = jitter < 1.0
    print(f"   тремтіння кутів на нерухомій карті (8 кадрів із шумом): макс {jitter:.2f} px  {'OK' if ok_j else 'FAIL'}")
    t0_ = time.perf_counter()
    for _ in range(5):
        ft.match(f2)
    print(f"   швидкість: {(time.perf_counter() - t0_) / 5 * 1000:.0f} мс/кадр")
    ok_all &= ok_f and ok_j
    det1, det2, Hc1, Hc2 = clicked, fc2, fH1, fH2

    print("\n3) Калібрування -> карту зсунули/повернули -> координати (режим features, кліки ±1.5 px):")
    for proj in PROJECTIONS:
        for lon0 in [0.0, 150.0]:
            xmax = float(fwd(proj, 0, 180)[0])
            ymax = float(fwd(proj, 85 if proj in ("mercator", "miller") else 90, 0)[1])
            m = 0.05  # білі поля аркуша

            def to_sheet(lat, lon):
                x, y = fwd(proj, lat, wrap180(np.asarray(lon) - lon0))
                return np.stack([m + (1 - 2 * m) * (x + xmax) / (2 * xmax),
                                 m + (1 - 2 * m) * (ymax - y) / (2 * ymax)], -1)
            # калібрування: віртуальні кути, кліки місць з неточністю ~1.5 px
            cam_land = apply_H(Hc1, to_sheet(geo[:, 0], geo[:, 1])) + rng.normal(0, 1.5, (len(geo), 2))
            cal = fit_calibration(apply_H(cam_to_sheet_H(det1), cam_land), geo, verbose=False)
            # карту перемістили: кути з трекера за малюнком, палець у 300 випадкових точках
            G2 = cam_to_sheet_H(det2)
            tl, tn = rng.uniform(-70, 70, 300), rng.uniform(-180, 180, 300)
            sheet = apply_H(G2, apply_H(Hc2, to_sheet(tl, tn)))
            errs = [haversine_km(*cal.sheet_to_latlon(*s)[:2], a_, b_) for s, a_, b_ in zip(sheet, tl, tn)]
            good = cal.proj == proj and np.median(errs) < 150
            ok_all &= good
            print(f"   {proj:9s} центр {lon0:+4.0f}° -> {cal.proj:9s} {'OK ' if good else 'FAIL'} | "
                  f"похибка: медіана {np.median(errs):4.0f} км, 95% {np.percentile(errs, 95):4.0f} км")
    hand = f2
    tracker = ft

    sst = SSTLookup(SST_FILE, args.month)
    print(f"\n4) Температура ({sst.label}):")
    for name, la, lo in [("Чорне море", 43.5, 34.0), ("Сахара (суша)", 23.0, 12.0),
                         ("Одеса (берег)", 46.48, 30.73), ("Екватор, Тихий океан", 0.0, -140.0)]:
        print(f"   {name:22s} {report(la, lo, sst)}")

    print("\n5) Сонифікація (синтез без звукової карти):")
    sr = Sonifier.SR
    son = Sonifier(enabled=False)
    r = sst.at(46.48, 30.73)                                   # Одеса: 18.7 °C, аномалія +2.4
    son.water(r["t"], 0.0, 30.73)
    son.render(sr // 2)                                         # дати параметрам усталитись
    a = son.render(sr).mean(1)
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    peak = np.fft.rfftfreq(len(a), 1 / sr)[np.argmax(spec)]
    want = midi_to_hz(temp_to_midi(r["t"]))
    ok_pitch = abs(peak - want) / want < 0.01
    print(f"   {r['t']:.1f} °C -> нота {midi_name(temp_to_midi(r['t']))} ({want:.1f} Гц), "
          f"виміряно FFT {peak:.1f} Гц  {'OK' if ok_pitch else 'FAIL'}")
    lo_note, hi_note = midi_name(temp_to_midi(SST_VMIN)), midi_name(temp_to_midi(SST_VMAX))
    print(f"   діапазон: {SST_VMIN:.0f} °C -> {lo_note}, {SST_VMAX:.0f} °C -> {hi_note}")

    def pulse_ratio(anom):
        s = Sonifier(enabled=False)
        s.water(r["t"], anom, 30.73); s.render(sr)
        x = s.render(2 * sr).mean(1)
        rms = np.sqrt((x[: len(x) // 882 * 882].reshape(-1, 882) ** 2).mean(1))   # вікна по 20 мс
        return rms.max() / max(rms.min(), 1e-9)
    p0, p2 = pulse_ratio(0.0), pulse_ratio(2.4)
    ok_pulse = p0 < 1.25 and p2 > 1.8
    print(f"   пульсація гучності: аномалія 0 -> x{p0:.2f}, аномалія +2.4 °C -> x{p2:.2f}  {'OK' if ok_pulse else 'FAIL'}")
    s = Sonifier(enabled=False)
    s.water(20.0, 0.0, -150.0); s.render(sr // 2)
    st_ = s.render(sr)
    lr = np.sqrt((st_[:, 0] ** 2).mean() / (st_[:, 1] ** 2).mean())
    ok_pan = lr > 3
    print(f"   панорама: довгота 150°W -> ліво/право гучність x{lr:.1f}  {'OK' if ok_pan else 'FAIL'}")
    ok_all &= ok_pitch and ok_pulse and ok_pan

    # демо-подорож: Антарктичні води -> екватор Тихого океану -> Атлантика -> Середземне -> Чорне море
    route = [(-62, -150), (-30, -140), (0, -140), (15, -60), (40, -40), (55, -20), (36, 10), (43, 34)]
    seconds, blk = 16.0, 512
    son = Sonifier(enabled=False)
    out_chunks, nblk = [], int(seconds * sr / blk)
    seg = np.linspace(0, len(route) - 1, nblk)
    for k in range(nblk):
        i = min(int(seg[k]), len(route) - 2); f = seg[k] - i
        la = route[i][0] + f * (route[i + 1][0] - route[i][0])
        lo = route[i][1] + f * (route[i + 1][1] - route[i][1])
        rr = sst.at(la, lo)
        if rr["t"] is None:
            son.land(lo)
        else:
            son.water(rr["t"], rr["t"] - rr["base"], lo, quiet=rr["land"])
        if k == nblk - int(1.2 * sr / blk):
            son.chime()
        out_chunks.append(son.render(blk))
    wav = np.concatenate(out_chunks)
    ok_clip = np.abs(wav).max() < 0.99
    print(f"   демо-подорож {seconds:.0f} с, пік амплітуди {np.abs(wav).max():.2f} (без перевантаження: "
          f"{'OK' if ok_clip else 'FAIL'})")
    ok_all &= ok_clip

    print("\n6) Рендер панелі та карти температури:")
    view, panel = SSTMapView(sst), Panel(330, sst)
    trail = [(43.5 + k * 0.3, 20 + k) for k in range(15)]
    r = sst.at(46.48, 30.73)
    img_map = view.render(46.48, 30.73, trail, 0.6, r["t"])
    img_panel = panel.render(540, {"state": "ok", "lat": 46.48, "lon": 30.73, "r": r, "sheet": "tracked",
                                   "dwell": 0.6, "last_fix": report(43.5, 34.0, sst)})
    out = Path(args.selftest_out)
    out.mkdir(parents=True, exist_ok=True)
    write_wav(out / "sonification_demo.wav", wav)
    cv2.imwrite(str(out / "selftest_map.png"), img_map)
    cv2.imwrite(str(out / "selftest_panel.png"), img_panel)
    cv2.imwrite(str(out / "selftest_sheet.png"), cv2.polylines(hand, [np.int32(tracker.corners)], True, (0, 200, 0), 3))
    print(f"   збережено в {out}/")
    print("\nSELFTEST", "PASSED" if ok_all else "FAILED")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="індекс камери (0 - вбудована)")
    ap.add_argument("--recalibrate", action="store_true", help="пройти калібрування заново")
    ap.add_argument("--dwell", type=float, default=3.0, help="скільки секунд тримати палець для запису в консоль")
    ap.add_argument("--radius", type=float, default=0.02,
                    help="допустиме тремтіння пальця, частка ширини кадру (0.02 = 2%%)")
    ap.add_argument("--month", help="місяць даних РРРР-ММ (за замовчуванням останній у файлі)")
    ap.add_argument("--projection", default="equirect", choices=PROJECTIONS,
                    help="проєкція, якщо клацнуто лише 4 місця")
    ap.add_argument("--lon0", type=float, default=0.0, help="центральний меридіан карти, якщо 4 місця")
    ap.add_argument("--track", default="features", choices=["features", "contour", "static"],
                    help="як відстежувати карту: features - за малюнком карти (за замовчуванням), "
                         "contour - за краями аркуша, static - карта нерухома")
    ap.add_argument("--no-sound", action="store_true", help="без сонифікації")
    ap.add_argument("--volume", type=float, default=0.8, help="гучність 0..1")
    ap.add_argument("--width", type=int, default=1280, help="ширина кадру камери")
    ap.add_argument("--view", type=int, default=860, help="ширина відображення камери у вікні, px")
    ap.add_argument("--selftest", action="store_true", help="перевірка без камери")
    ap.add_argument("--selftest-out", default=str(HERE / "selftest_out"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.selftest:
        selftest(args); return

    print("Завантаження даних температури...")
    sst = SSTLookup(SST_FILE, args.month)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.width * 9 / 16))
    if not cap.isOpened():
        raise SystemExit("Камера не відкрилась. На macOS дозвольте доступ: Системні параметри -> "
                         "Приватність і безпека -> Камера -> ваш термінал / VS Code.")
    try:
        frame = _read(cap)
        scale = min(1.0, args.view / frame.shape[1])
        cal = MapCalibration.load(CALIB_FILE) if CALIB_FILE.exists() and not args.recalibrate else None
        if cal is None:
            if CALIB_FILE.exists() and not args.recalibrate:
                print("Старий формат calibration.json (без еталона карти), потрібне нове калібрування.")
            cal = run_calibration(cap, args, scale)
        else:
            print(f"Калібрування з {CALIB_FILE.name}: проєкція {cal.proj}. (--recalibrate щоб переробити)")
        run_tracking(cap, cal, sst, args, scale)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
