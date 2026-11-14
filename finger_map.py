"""Палець на паперовій карті -> значення обраного шару-карти (температура океану,
аномалія температури, ... - див. LAYERS.md) -> панель, карта і звук.

Дві руки: ПРАВИЙ вказівний палець водить по карті, ЛІВИЙ вказівний палець задає рік -
його положення вздовж карти (лівий край аркуша = перший рік, правий = останній).
Шари: клавіші l / 1-9; рік також , . (±1), < > (±10), p (відтворення); h - поміняти руки.
Фіксація (правий палець 3 с на місці) -> голос (Lesya) озвучує місце (країна / море / регіон),
час і значення шару (speech.py); v - вимкнути голос, r - повторити.

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
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import layers as L
import speech as SP
import globe as GB

HERE = Path(__file__).resolve().parent
CALIB_FILE = HERE / "calibration.json"
CALIB_VERSION = 3
TEMPLATE_FILE = HERE / "calibration_template.png"
MODEL_FILE = HERE / "models" / "hand_landmarker.task"
INDEX_TIP = 8  # MediaPipe: кінчик вказівного пальця
WIN_CAM, WIN_SST = "Earth Map", "Data Layer"

# Клавіші у вікні: (як показати, як вимовити, що робить). З цього одного списку будуються
# і довідка в консолі, і голосова довідка (клавіша ?), тож вони не розходяться.
KEYS = [
    ("?", "знак питання", "ця довідка; натисніть ще раз, щоб зупинити"),
    ("i", "латинська і", "перелік карт"),
    ("1-9", "цифри від 1 до {n}", "вибрати карту за номером"),
    ("l", "клавіша ел", "наступна карта"),
    (", .", "кома і крапка", "крок у часі назад і вперед"),
    ("< >", "шифт з комою і шифт з крапкою", "десять кроків у часі"),
    ("p", "клавіша пе", "автоматичне програвання часу, ще раз пауза"),
    ("r", "клавіша ер", "повторити останнє повідомлення"),
    ("m", "клавіша ем", "вимкнути або увімкнути звук"),
    ("v", "клавіша ве", "вимкнути або увімкнути голос"),
    ("h", "клавіша аш", "поміняти ролі рук, якщо програма їх плутає"),
    ("c", "клавіша це", "калібрування карти заново"),
    ("k", "клавіша ка", "заново позначити кути карти"),
    ("q", "клавіша кю", "вихід"),
]
# Глобус: часу лівою рукою немає, тож "кома/крапка/p" - головне керування часом; c/k означають інше.
KEYS_GLOBE = [k for k in KEYS if k[0] not in ("c", "k")] + [
    ("c", "клавіша це", "калібрування глобуса заново"),
    ("k", "клавіша ка", "заново вивчити кольори і запам'ятати поточний бік глобуса"),
]

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
# Глобус (--surface globe): калібрування і допоміжне. Математика - у globe.py
# ----------------------------------------------------------------------------
# На глобусі одночасно видно лише півкулю, тож для нього більше знайомих місць по всьому світу.
GLOBE_EXTRA_LANDMARKS = [
    ("Cape Guardafui - Horn of Africa", "мис Гвардафуй (Африканський Ріг)", 11.83, 51.28),
    ("Cape Verde - Dakar, W tip of Africa", "Зелений Мис, Дакар (захід Африки)", 14.72, -17.53),
    ("Cabo Sao Roque - E tip of Brazil", "мис Сан-Роке (схід Бразилії)", -5.48, -35.26),
    ("Singapore - S tip of Malay peninsula", "Сінгапур (південь Малаккського півострова)", 1.26, 103.82),
    ("Cape Reinga - N tip of New Zealand", "мис Реінга (північ Нової Зеландії)", -34.43, 172.68),
    ("S tip of Tasmania", "південь Тасманії", -43.64, 146.83),
    ("Cape Prince of Wales - Bering Strait", "Берингова протока (Аляска)", 65.6, -168.09),
    ("S tip of Florida", "південь Флориди", 25.12, -81.09),
]
GLOBE_LANDMARKS = LANDMARKS + GLOBE_EXTRA_LANDMARKS


class GlobeSurface:
    """Те, що run_tracking потребує від глобуса: модель (піксель -> lat/lon), відстеження повороту, накладання."""

    def __init__(self, model, tracker):
        self.model, self.tracker = model, tracker
        self.overlay = GB.GlobeOverlay(model)


def hands_mask(res, w, h):
    """Маска рук (і передпліччя) для відстеження глобуса: ці пікселі не порівнюються з картою суші."""
    mask = np.zeros((h, w), np.uint8)
    for lms in (res.hand_landmarks or []):
        pts = np.array([[p.x * w, p.y * h] for p in lms])
        wrist, mid = pts[0], pts[9]
        arm = wrist + 3.0 * (wrist - mid)                       # передпліччя: продовжуємо від зап'ястя
        hull = cv2.convexHull(np.int32(np.vstack([pts, arm[None]])))
        cv2.fillConvexPoly(mask, hull, 255)
    if mask.any():
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    return mask


def pick_globe_circle(cap, scale, win="Calibration"):
    """Крок 1: обрис глобуса (A = прийняти знайдене автоматично коло, або 3 кліки по краю кулі)."""
    print("\n--- Крок 1: обрис глобуса ---")
    print("Натисніть A, якщо зелене коло точно лягло по краю кулі, або клацніть 3 точки на краю глобуса.")
    print("  u - скасувати клік, q - вийти.")
    clicks = []
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, lambda ev, x, y, *_: clicks.append((x / scale, y / scale))
                         if ev == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        frame = _read(cap)
        auto = GB.detect_globe_circle(frame)
        if auto is not None:
            cv2.circle(frame, (int(auto[0]), int(auto[1])), int(auto[2]), (0, 200, 0), 2, cv2.LINE_AA)
        for c in clicks[:3]:
            cv2.circle(frame, (int(c[0]), int(c[1])), 7, (0, 0, 255), 2)
        _banner(frame, [f"Step 1/2: globe outline - click 3 points on the edge ({len(clicks)}/3)",
                        ("A = accept green circle  |  " if auto is not None else "") + "u=undo  q=quit"], scale)
        _show(win, frame, scale)
        key = cv2.waitKey(15) & 0xFF
        if key == ord("q"):
            raise SystemExit(0)
        if key == ord("u") and clicks:
            clicks.pop()
        if key == ord("a") and auto is not None:
            print(f"  Прийнято коло: центр ({auto[0]:.0f}, {auto[1]:.0f}), радіус {auto[2]:.0f} px")
            return auto
        if len(clicks) >= 3:
            c = GB.circle_from_3pts(*clicks[:3])
            if c is None:
                print("  Точки на одній прямій, клацніть ще раз.")
                clicks.clear()
                continue
            print(f"  Коло з 3 точок: центр ({c[0]:.0f}, {c[1]:.0f}), радіус {c[2]:.0f} px")
            return c


def run_globe_calibration(cap, args, scale):
    print("\n=== КАЛІБРУВАННЯ ГЛОБУСА === Під час калібрування глобус не крутіть, руку приберіть.")
    circle = pick_globe_circle(cap, scale)
    frame = _read(cap)
    H_, W_ = frame.shape[:2]
    print("\n--- Крок 2: знайомі місця на видимому боці глобуса ---")
    print("Клацайте місце, назва якого написана зверху. s - пропустити (місце на зворотному боці),")
    print("u - скасувати, Enter - завершити (мінімум 3, краще 5+), q - вийти.")
    print("Після 3 місць програма підкаже, де шукати наступне (жовте коло), і сама пропустить невидимі.\n")
    win = "Calibration"
    clicks, cam_pts, geo_pts = [], [], []
    cv2.setMouseCallback(win, lambda ev, x, y, *_: clicks.append((x / scale, y / scale))
                         if ev == cv2.EVENT_LBUTTONDOWN else None)
    idx, prelim, announced = 0, None, -1
    while True:
        frame = _read(cap)
        # після 3 точок: попередня орієнтація -> підказка і автопропуск місць на зворотному боці
        while prelim is not None and idx < len(GLOBE_LANDMARKS):
            x, y, vis = prelim.project_latlon(*GLOBE_LANDMARKS[idx][2:])
            _, _, fac = prelim.geom.hit(np.array([x]), np.array([y]))
            if bool(vis) and fac[0] > 0.2:
                break
            print(f"  [{idx + 1}/{len(GLOBE_LANDMARKS)}] {GLOBE_LANDMARKS[idx][1]} - на зворотному боці, пропущено")
            idx += 1
        if idx < len(GLOBE_LANDMARKS) and announced != idx:
            print(f"  [{idx + 1}/{len(GLOBE_LANDMARKS)}] {GLOBE_LANDMARKS[idx][1]}")
            announced = idx
        if clicks:
            c = clicks[-1]
            clicks.clear()
            if idx < len(GLOBE_LANDMARKS):
                if (c[0] - circle[0]) ** 2 + (c[1] - circle[1]) ** 2 > circle[2] ** 2:
                    print("      клік поза глобусом - спробуйте ще раз")
                else:
                    cam_pts.append(c); geo_pts.append(GLOBE_LANDMARKS[idx][2:])
                    print(f"      клік ({c[0]:.0f}, {c[1]:.0f}) -> {GLOBE_LANDMARKS[idx][2]:+.2f}, {GLOBE_LANDMARKS[idx][3]:+.2f}")
                    idx += 1
                    if len(cam_pts) >= 3:
                        try:
                            g_, R_, e_ = GB.solve_pose(cam_pts, geo_pts, circle[:2], circle[2], (W_, H_))
                            prelim = GB.GlobeModel(g_, R_)
                        except ValueError:
                            prelim = None
        cv2.circle(frame, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 200, 0), 2, cv2.LINE_AA)
        for k, c in enumerate(cam_pts):
            cv2.circle(frame, (int(c[0]), int(c[1])), 7, (0, 0, 255), 2)
            cv2.putText(frame, str(k + 1), (int(c[0]) + 9, int(c[1]) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if prelim is not None and idx < len(GLOBE_LANDMARKS):
            x, y, _ = prelim.project_latlon(*GLOBE_LANDMARKS[idx][2:])
            cv2.circle(frame, (int(x), int(y)), 22, (0, 220, 255), 2, cv2.LINE_AA)
        title = (f"Step 2/2: click {GLOBE_LANDMARKS[idx][0]}" if idx < len(GLOBE_LANDMARKS)
                 else "All points done - press Enter")
        _banner(frame, [title, f"points: {len(cam_pts)}  |  s=skip  u=undo  Enter=finish  q=quit"], scale)
        _show(win, frame, scale)
        key = cv2.waitKey(15) & 0xFF
        if key == ord("q"):
            raise SystemExit(0)
        if key == ord("s") and idx < len(GLOBE_LANDMARKS):
            print("      пропущено")
            idx += 1
        if key == ord("u") and cam_pts:
            cam_pts.pop(); geo_pts.pop(); idx = max(0, idx - 1); announced = -1
            prelim = None
        if key in (13, 10):
            if len(cam_pts) < 3:
                print("  Потрібно щонайменше 3 місця.")
                continue
            break
    geom, R0, err = GB.solve_pose(cam_pts, geo_pts, circle[:2], circle[2], (W_, H_))
    model = GB.GlobeModel(geom, R0)
    tracker = GB.GlobeTracker(model)
    frame = _read(cap)
    tracker.learn_colors(frame, 0.0)
    tracker.memory.add_from_frame(frame, model, tracker.orb, min_facing=0.6)
    fov = np.degrees(2 * np.arctan(W_ / 2 / geom.f))
    print(f"  Орієнтацію знайдено: поле зору камери {fov:.0f}°, середня похибка місць {err:.1f}° (~{err * 111:.0f} км)")
    globe_tour(cap, tracker, scale, win)
    cv2.destroyWindow(win)
    GB.save_globe(GB.GLOBE_CALIB, model, tracker, {"clicks": cam_pts, "latlons": geo_pts})
    print(f"  Збережено: {GB.GLOBE_CALIB.name} (калібрування) і {GB.GLOBE_MEMORY.name} "
          f"(пам'ять глобуса, {len(tracker.memory)} точок)\n")
    return GlobeSurface(model, tracker)


def globe_tour(cap, tracker, scale, win="Calibration", target=0.95):
    """Крок 3: глобус на місці, його повільно обертають на повний оберт - програма запам'ятовує всі боки.
    Кут визначається за контурами суші (без накопичення похибки), руку з порівняння вирізаємо."""
    print("\n--- Крок 3: запам'ятати глобус ---")
    print("Повільно поверніть глобус на повний оберт навколо осі (підставку не рухайте).")
    print("Програма запам'ятовує кожен бік, щоб потім знаходити глобус будь-де в кадрі.")
    print("  Enter - завершити раніше, s - пропустити (тоді глобус не можна буде переставляти), q - вийти.")
    mp, landmarker = make_landmarker()
    t0, last_ts, shown = time.monotonic(), -1, -1
    overlay = GB.GlobeOverlay(tracker.model)
    try:
        while True:
            frame = _read(cap)
            h, w = frame.shape[:2]
            ts = max(int((time.monotonic() - t0) * 1000), last_ts + 1); last_ts = ts
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
            theta, cov = tracker.tour_step(frame, hands_mask(res, w, h) if res.hand_landmarks else None)
            overlay.draw(frame, tracker.status)
            pct = int(cov * 100)
            if pct // 10 != shown // 10:
                print(f"  вивчено {pct}% глобуса (точок у пам'яті: {len(tracker.memory)})")
                shown = pct
            bw = int(w * 0.6)
            cv2.rectangle(frame, (int(w * 0.2), h - 40), (int(w * 0.2) + bw, h - 20), (200, 200, 200), 1)
            cv2.rectangle(frame, (int(w * 0.2), h - 40), (int(w * 0.2 + bw * cov), h - 20), (0, 200, 0), -1)
            _banner(frame, [f"Step 3/3: slowly rotate the globe one full turn - learned {pct}%",
                            f"angle {theta:.0f} deg  |  Enter=finish  s=skip  q=quit"], scale)
            _show(win, frame, scale)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise SystemExit(0)
            if key in (13, 10, ord("s")) or cov >= target:
                if cov >= target:
                    print(f"  Глобус вивчено ({pct}%).")
                elif key == ord("s"):
                    print("  Тур пропущено: глобус розпізнається лише з того боку, що був на калібруванні.")
                return
    finally:
        landmarker.close()


# ----------------------------------------------------------------------------
# Звіт у консоль і панель з живим значенням шару (PIL, щоб була кирилиця)
# ----------------------------------------------------------------------------
ROLE_COLORS = {"ink": INK, "ink2": INK2, "muted": MUTED, "warm": WARM, "cool": COOL}


def report(layer, t, lat, lon, s):
    head = f"[{datetime.now():%H:%M:%S}] {L.fmt_coord(lat, lon)} · {layer.title}, {layer.time_label(t)}"
    if s is None or s["missing"]:
        return f"{head} -> {layer.missing_label}"
    extra = [t for t, role in s["lines"] if role in ("ink2", "warm", "cool")]
    return f"{head} -> {s['text']}" + (" | " + " | ".join(extra) if extra else "")


def _wrap_px(d, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = f"{cur} {w}".strip()
        if cur and d.textlength(test, font=font) > width:
            lines.append(cur); cur = w
        else:
            cur = test
    return lines + [cur] if cur else lines


class Panel:
    def __init__(self, width):
        self.w = width
        self.f = L.get_fonts()
        self.bar_w = width - 40
        self._bars = {}

    def _bar(self, layer):
        if layer.name not in self._bars:
            self._bars[layer.name] = np.repeat(layer.lut[np.linspace(0, 255, self.bar_w).astype(int)][None], 14, axis=0)
        return self._bars[layer.name]

    def _text(self, d, x, y, text, font, role, width, lh, max_lines=2):
        for line in _wrap_px(d, text, font, width)[:max_lines]:
            d.text((x, y), line, font=font, fill=L.hex_rgb(ROLE_COLORS[role])); y += lh
        return y

    def render(self, height, info):
        from PIL import Image, ImageDraw
        layer, s = info["layer"], info.get("sample")
        img = Image.new("RGB", (self.w, height), L.hex_rgb(SURFACE))
        d = ImageDraw.Draw(img)
        x, y, W = 20, 14, self.w - 40
        col = lambda role: L.hex_rgb(ROLE_COLORS[role])
        n = info["n_layers"]
        d.text((x, y), f"ШАР {info['layer_idx']}/{n}  ·  l або 1–{min(n, 9)}: змінити", font=self.f["xs"],
               fill=col("muted")); y += 20
        y = self._text(d, x, y, layer.title, self.f["l"], "ink", W, 22)
        if layer.dynamic:
            mode = ("   ✋ ліва рука" if info.get("time_hand") else "   ▶ відтворення" if info.get("playing") else "")
            k = layer.clamp_frame(info["t"])
            d.text((x, y), f"{'дата' if layer.has_dates else 'рік'} {layer.time_label(k)}" + mode, font=self.f["s"],
                   fill=col("cool" if info.get("time_hand") else "ink2")); y += 22
            how = "ліва рука вздовж карти" if info.get("surface") != "globe" else "період (клавіші)"
            d.text((x, y), f"{how}: {layer.short_label(0)} ← → {layer.short_label(layer.n_frames - 1)}",
                   font=self.f["xs"], fill=col("muted")); y += 17
            d.text((x, y), f"кадр {k + 1}/{layer.n_frames}   , . ±1   < > ±10   p ▶/❚❚", font=self.f["xs"],
                   fill=col("muted")); y += 20
        else:
            d.text((x, y), layer.time_label(None) + " (один момент часу)", font=self.f["s"], fill=col("ink2")); y += 22
        y += 4
        d.line([(x, y), (self.w - 20, y)], fill=L.hex_rgb(GRIDC)); y += 10

        d.text((x, y), "ПІД ПРАВИМ ПАЛЬЦЕМ", font=self.f["xs"], fill=col("muted")); y += 20
        if info.get("pos"):
            d.text((x, y), L.fmt_coord(*info["pos"]), font=self.f["m"], fill=col("ink")); y += 28
            if info.get("edge"):
                d.text((x, y), "біля краю глобуса: точність нижча", font=self.f["xs"], fill=col("muted")); y += 18
        if s is not None:
            if s["missing"]:
                d.text((x, y), layer.missing_label, font=self.f["xl"], fill=col("ink2")); y += 60
            else:
                big = self.f["xl"] if d.textlength(s["text"], font=self.f["xl"]) <= W else self.f["m"]
                d.text((x, y), s["text"], font=big, fill=col("ink")); y += 60 if big is self.f["xl"] else 32
                for t, role in s["lines"]:
                    small = role == "muted"
                    y = self._text(d, x, y, t, self.f["xs"] if small else self.f["s"], role, W, 18 if small else 22)
        else:
            msg = {"nohand": "руку не видно", "offmap": "палець поза картою"}.get(info.get("state"), "")
            d.text((x, y), msg, font=self.f["m"], fill=col("ink2")); y += 40

        y += 12                                                # шкала шару з маркером
        img.paste(Image.fromarray(self._bar(layer)), (x, y))
        if s is not None and not s["missing"]:
            px = x + int(s["norm"] * (self.bar_w - 1))
            d.polygon([(px - 6, y - 9), (px + 6, y - 9), (px, y - 1)], fill=col("ink"))
        for v in (layer.vmin, (layer.vmin + layer.vmax) / 2, layer.vmax):
            px = x + int((v - layer.vmin) / (layer.vmax - layer.vmin) * (self.bar_w - 1))
            t = layer.fmt.format(v)
            tx = min(max(px - d.textlength(t, font=self.f["xs"]) / 2, x), x + self.bar_w - d.textlength(t, font=self.f["xs"]))
            d.text((tx, y + 17), t, font=self.f["xs"], fill=col("muted"))
        y += 42
        d.line([(x, y), (self.w - 20, y)], fill=L.hex_rgb(GRIDC)); y += 10

        if layer.source:
            y = self._text(d, x, y, f"Дані: {layer.source}", self.f["xs"], "ink2", W, 17)
        st = {"tracked": "знайдено", "held": "утримуються (карту не видно)", "searching": "пошук карти...",
              "static": "фіксовані"}[info.get("sheet", "static")]
        if info.get("inliers") and info.get("sheet") == "tracked":
            st += f" ({info['inliers']} точок)"
        gi = info.get("globe")
        if gi is not None and info.get("sheet") == "tracked":
            st = (f"знайдено ({gi['inliers']} точок)" if gi["mode"] == "features" else "за контурами суші")
            st += f" · вивчено {gi['coverage'] * 100:.0f}%"
        d.text((x, y), f"{'Глобус' if info.get('surface') == 'globe' else 'Карта'}: {st}", font=self.f["xs"],
               fill=col("ink2")); y += 18
        snd = "вимкнено (m)" if info.get("muted") else (f"нота {info['note']}" if info.get("note") else "тиша")
        d.text((x, y), f"Звук: {snd}", font=self.f["xs"], fill=col("ink2")); y += 18
        vo = info.get("voice")
        d.text((x, y), "Голос: " + ("говорить…" if info.get("speaking") else vo or "вимкнено (v)") +
               ("   r повторити" if vo else ""), font=self.f["xs"], fill=col("ink2")); y += 24
        frac = info.get("dwell", 0.0)
        d.text((x, y), "Фіксація в консоль (3 с)", font=self.f["xs"], fill=col("ink2")); y += 18
        d.rectangle([x, y, self.w - 20, y + 8], fill=L.hex_rgb(GRIDC))
        if frac > 0:
            d.rectangle([x, y, x + int(W * frac), y + 8], fill=L.hex_rgb("#0ca30c" if frac >= 1 else "#eb6834"))
        y += 20
        if info.get("last_fix"):
            d.text((x, y), "Остання фіксація:", font=self.f["xs"], fill=col("muted")); y += 18
            self._text(d, x, y, info["last_fix"], self.f["xs"], "ink", W, 17, max_lines=4)
        d.text((x, height - 42), "? довідка · i карти · r повтор · v голос", font=self.f["xs"], fill=col("muted"))
        d.text((x, height - 24), "q вихід · m звук · h руки · c/k калібр.", font=self.f["xs"], fill=col("muted"))
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------------
# Сонифікація: значення будь-якого шару -> звук у реальному часі
# ----------------------------------------------------------------------------
PENTATONIC = [0, 2, 4, 7, 9]           # мажорна пентатоніка: будь-які дві ноти звучать злагоджено
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def norm_to_midi(norm, midi_lo=48, midi_hi=76):
    """Нормоване значення шару 0..1 (vmin..vmax) -> нота пентатоніки між C3 (низько) і E5 (високо)."""
    scale = [m for m in range(midi_lo, midi_hi + 1) if (m - midi_lo) % 12 in PENTATONIC]
    return scale[int(round(np.clip(norm, 0, 1) * (len(scale) - 1)))]


def midi_to_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def midi_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


class Sonifier:
    """Адитивний синтезатор, параметри якого задають дані будь-якого шару:
      значення (norm 0..1)  -> висота ноти (пентатоніка) і яскравість тембру (кількість гармонік);
      напруга (tension)     -> наскільки значення вище/нижче базового рівня шару (норми):
                               биття (другий голос розстроєний) і пульсація-"серцебиття", коли вище;
      довгота               -> стерео-панорама (захід ліворуч, схід праворуч);
      немає даних           -> тихий шум; фіксація -> дзвіночок.
    Усі зміни плавні (портаменто/згладжування), щоб не було клацань.
    render() працює і без звукової карти (для запису WAV і тестів)."""
    SR = 44100
    N_H = 6
    TAU = {"freq": 0.06, "amp": 0.08, "noise": 0.15, "detune": 0.3, "bright": 0.3,
           "prate": 0.4, "pdepth": 0.3, "pan": 0.15, "duck": 0.12}

    def __init__(self, enabled=True, volume=0.8):
        self.target = {"freq": 220.0, "amp": 0.0, "noise": 0.0, "detune": 0.0, "bright": 1.8,
                       "prate": 1.0, "pdepth": 0.0, "pan": 0.0, "duck": 1.0}
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
    def play(self, norm, tension, lon, quiet=False):
        """norm: 0..1 у шкалі шару; tension: (значення - базовий рівень) / масштаб шару, ~1 = сильне відхилення."""
        m = norm_to_midi(norm)
        self.note = m
        tn = 0.0 if tension is None or np.isnan(tension) else float(tension)
        self.target.update(
            freq=midi_to_hz(m),
            amp=0.45 if quiet else 1.0,
            noise=0.15 if quiet else 0.0,
            bright=2.4 - 1.5 * float(np.clip(norm, 0, 1)),  # менший показник = більше гармонік = яскравіше
            detune=float(np.clip(abs(tn) * 5.0, 0, 8)),     # Гц биття між двома голосами
            prate=float(np.clip(1.0 + max(tn, 0) * 6.0, 1, 7)),
            pdepth=float(np.clip(max(tn, 0), 0, 0.75)),
            pan=float(np.clip(lon / 180.0, -1, 1)))

    def missing(self, lon):
        self.note = None
        self.target.update(amp=0.0, noise=0.35, pan=float(np.clip(lon / 180.0, -1, 1)))

    def silence(self):
        self.note = None
        self.target.update(amp=0.0, noise=0.0)

    def chime(self):
        self.chime_f, self.chime_t = 2 * self.target["freq"], 0.0

    def set_duck(self, on):
        """Притишити синтезатор (напр. поки звучить голос), щоб було чути слова."""
        self.target["duck"] = 0.2 if on else 1.0

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
        duck = self._ramp("duck", n)
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
        sig = np.tanh(1.2 * sig) / 1.2 * self.volume * duck
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
                                        running_mode=vision.RunningMode.VIDEO, num_hands=2,
                                        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)
    return mp, vision.HandLandmarker.create_from_options(opts)


def hand_roles(res, swap=False):
    """Результат MediaPipe -> {"map": кисть правої руки, "time": кисть лівої}.

    MediaPipe позначає руки так, ніби кадр дзеркальний (селфі-камера). Наш кадр звичайний,
    тому мітки "Left"/"Right" міняємо місцями; swap=True (клавіша h) міняє ще раз.
    Якщо обидві руки отримали однакову мітку, розводимо за положенням: лівіша в кадрі = ліва."""
    hands = []
    for lms, hd in zip(res.hand_landmarks, res.handedness):
        label = hd[0].category_name
        if not swap:
            label = "Right" if label == "Left" else "Left"
        hands.append([label, lms])
    if len(hands) == 2 and hands[0][0] == hands[1][0]:
        hands.sort(key=lambda h: h[1][0].x)                    # 0 = зап'ястя
        hands[0][0], hands[1][0] = "Left", "Right"
    return {("map" if label == "Right" else "time"): lms for label, lms in hands}


def frame_from_u(layer, u, current, hysteresis=0.6):
    """Положення лівого пальця вздовж карти (u: 0 = лівий край аркуша, 1 = правий) -> кадр шару
    (рік, тиждень, ... - скільки кадрів є). Гістерезис: кадр змінюється, лише коли палець відійшов
    більш ніж на 0.6 кадру, щоб не мерехтіло."""
    last = layer.n_frames - 1
    cont = float(np.clip(u, 0, 1)) * last
    if current is None or abs(cont - current) > hysteresis:
        return int(np.clip(round(cont), 0, last))
    return current


def _timeline_ticks(layer, max_labels=7):
    """Кадри з позначками на шкалі: круглі роки для річних шарів, рівномірно - для шарів з датами."""
    n = layer.n_frames
    if not layer.has_dates:
        y0, y1 = int(layer.times[0]), int(layer.times[-1])
        step = next(s for s in (1, 2, 5, 10, 20, 25, 50, 100) if (y1 - y0) / s <= max_labels * 2)
        ticks = [layer.frame_at(y + 0.5) for y in range((y0 // step + 1) * step, y1, step)]
        return [(k, layer.short_label(k), (int(layer.times[k]) // step) % 2 == 0) for k in ticks]
    ticks = np.unique(np.round(np.linspace(0, n - 1, max_labels)).astype(int))
    return [(int(k), layer.short_label(int(k)), True) for k in ticks]


def draw_timeline(frame, G, layer, t, active):
    """Шкала часу вздовж нижнього краю аркуша (там, де її "читає" ліва рука)."""
    Ginv = np.linalg.inv(G)
    last = max(layer.n_frames - 1, 1)
    col = (191, 106, 37) if active else (170, 170, 170)        # BGR: синій, коли ліва рука керує
    a, b = apply_H(Ginv, [[0.0, 1.04], [1.0, 1.04]])
    cv2.line(frame, tuple(np.int32(a)), tuple(np.int32(b)), col, 2, cv2.LINE_AA)
    for k, lab, show in _timeline_ticks(layer):
        p = apply_H(Ginv, [[k / last, 1.04]])[0]
        q = apply_H(Ginv, [[k / last, 1.07]])[0]
        cv2.line(frame, tuple(np.int32(p)), tuple(np.int32(q)), col, 1, cv2.LINE_AA)
        if show:
            cv2.putText(frame, lab, (int(q[0]) - 4 * len(lab), int(q[1]) + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col,
                        1, cv2.LINE_AA)
    k = layer.clamp_frame(t)
    c = apply_H(Ginv, [[k / last, 1.04]])[0]
    cv2.circle(frame, tuple(np.int32(c)), 8, col, -1, cv2.LINE_AA)
    lab = layer.labels[k]
    cv2.putText(frame, lab, (int(c[0]) - 7 * len(lab), int(c[1]) - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2,
                cv2.LINE_AA)


def run_tracking(cap, cal, specs, args, scale, globe=None):
    mp, landmarker = make_landmarker()
    frame = _read(cap)
    tracker = make_tracker(args.track, cal, frame.shape) if globe is None else globe.tracker
    keys = KEYS if globe is None else KEYS_GLOBE
    surface = "map" if globe is None else "globe"
    off_word = "палець поза картою" if globe is None else "палець поза глобусом"
    finger, time_finger = OneEuro(), OneEuro(min_cutoff=0.8)
    son = Sonifier(enabled=not args.no_sound, volume=args.volume)
    geo = SP.Geocoder()                                         # країни / моря / регіони з українськими назвами
    voice = SP.Speaker(enabled=not args.no_voice, voice=args.voice, rate=args.voice_rate)
    last_spoken = ""
    if voice.enabled:
        print(f"Голос: {voice.voice}. Після фіксації озвучується місце, час і значення (v - вимкнути, r - повторити).")
    panel, view = Panel(340), L.LayerView()
    names = [s.name for s in specs]
    li = names.index(args.layer) if args.layer in names else 0
    layer = specs[li].load()
    t = layer.frame_at(None if args.year is None else args.year + 0.5)   # індекс кадру шару
    playing, last_step = False, 0.0
    swap = args.swap_hands

    print("=== ТРЕКІНГ === Шари даних:")
    for k, s in enumerate(specs, 1):
        print(f"   {k}. {s.title}" + ("   <- зараз" if k - 1 == li else ""))
    if globe is None:
        print("   ПРАВА рука, вказівний палець: водити по карті.")
        print("   ЛІВА рука, вказівний палець: час - вести вздовж карти (лівий край = перший кадр, правий = останній).")
    else:
        print("   ПРАВА рука, вказівний палець: показувати на глобус. Глобус можна крутити, переставляти й нахиляти:")
        print(f"   програма знаходить його за запам'ятованими точками (пам'ять: {len(getattr(tracker, 'memory', []))} точок).")
        print("   Час - клавішами: , . (крок), < > (10 кроків), p (програвання).")
    print("   Клавіші у вікні:")
    for disp, _, what in keys:
        print(f"   {disp:>5s}  {what.replace('{n}', str(min(len(specs), 9)))}")
    print(f"   Правий палець {args.dwell:.0f} с на місці -> запис сюди.")
    if son.stream is not None:
        print("   Звук: висота = значення шару (пентатоніка), биття і пульсація = наскільки вище базового")
        print("   рівня (норми), ліво/право = довгота, шум = немає даних, дзвіночок = фіксація.\n")

    cv2.namedWindow(WIN_CAM, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WIN_SST, cv2.WINDOW_NORMAL)
    placed = False
    smooth, anchor, anchor_t, fired, last_seen = None, None, 0.0, False, 0.0
    tsmooth, time_seen = None, 0.0
    trail = deque(maxlen=40)
    last_fix = ""
    t0, last_ts = time.monotonic(), -1

    def switch(new_li):
        nonlocal li, layer, t
        li = new_li % len(specs)
        print(f"Шар {li + 1}/{len(specs)}: {specs[li].title} (завантаження...)")
        when = layer.time_of(t)                                # той самий момент часу в новому шарі
        layer = specs[li].load()
        t = layer.frame_at(when)
        trail.clear()
        intro = SP.layer_intro(li + 1, len(specs), layer, surface)
        print(f"   🔊 {intro}")
        voice.speak(intro)

    def announce(text):
        nonlocal last_spoken
        if voice.speaking() and last_spoken == text:            # повторне натискання під час мовлення = стоп
            voice.stop()
            print("🔇 зупинено")
            return
        last_spoken = text
        print(f"🔊 {text}")
        voice.speak(text)
        if not voice.enabled:
            print("   (голос вимкнено: v - увімкнути)")

    if voice.enabled:
        voice.speak(f"Earth Jukebox готовий. Обрано карту {li + 1} з {len(specs)}: {SP.spoken_title(layer)}. "
                    "Натисніть знак питання, щоб почути довідку, або латинську і, щоб почути перелік карт.")

    while True:
        frame = _read(cap)
        h, w = frame.shape[:2]
        now = time.monotonic()
        ts = max(int((now - t0) * 1000), last_ts + 1); last_ts = ts

        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        roles = hand_roles(res, swap) if res.hand_landmarks else {}
        if globe is None:
            corners = tracker.update(frame)
            G = cam_to_sheet_H(corners)
        else:                                                  # поворот глобуса; руку не порівнюємо з картою суші
            tracker.update(frame, hands_mask(res, w, h) if res.hand_landmarks else None)

        # ---- права рука: точка на карті ----
        if "map" in roles:
            lm = roles["map"][INDEX_TIP]
            smooth = finger(np.array([lm.x * w, lm.y * h]), now).copy()
            last_seen = now
        elif now - last_seen > 0.4:  # руку загублено
            smooth, anchor, fired = None, None, False
            finger.reset()

        # ---- ліва рука: час ----
        time_hand = False
        if "time" in roles and globe is None:
            lm = roles["time"][INDEX_TIP]
            tsmooth = time_finger(np.array([lm.x * w, lm.y * h]), now).copy()
            time_seen = now
        elif now - time_seen > 0.4:
            tsmooth = None
            time_finger.reset()
        if tsmooth is not None:
            tu, tv = apply_H(G, [tsmooth])[0]
            near_sheet = -0.15 <= tu <= 1.15 and -0.3 <= tv <= 1.4
            if layer.dynamic and near_sheet:
                t = frame_from_u(layer, tu, t)
                time_hand, playing = True, False
            c = (int(tsmooth[0]), int(tsmooth[1]))
            cv2.circle(frame, c, 9, (191, 106, 37) if time_hand else (150, 150, 150), -1, cv2.LINE_AA)
            cv2.putText(frame, "TIME", (c[0] + 12, c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (191, 106, 37) if time_hand else (150, 150, 150), 2, cv2.LINE_AA)

        if playing and layer.dynamic and now - last_step >= args.step:
            last_step = now
            t = t + 1 if t < layer.n_frames - 1 else 0

        info = {"state": "nohand", "sheet": tracker.status, "dwell": 0.0, "last_fix": last_fix,
                "inliers": getattr(tracker, "inliers", None), "layer": layer, "layer_idx": li + 1,
                "n_layers": len(specs), "t": t, "playing": playing, "time_hand": time_hand,
                "voice": voice.voice if voice.enabled else None, "speaking": voice.speaking(), "surface": surface,
                "globe": None if globe is None else {"mode": getattr(tracker, "mode", "spin"),
                                                     "inliers": getattr(tracker, "inliers", 0),
                                                     "coverage": tracker.coverage() if hasattr(tracker, "coverage") else 0}}
        lat = lon = None
        frac = 0.0
        if smooth is not None:
            if globe is None:
                u, v = apply_H(G, [smooth])[0]
                on_sheet = -0.02 <= u <= 1.02 and -0.02 <= v <= 1.02
                lat, lon, ok = cal.sheet_to_latlon(u, v)
                ok = ok and on_sheet
            else:
                lat, lon, hit, facing = globe.model.to_latlon(*smooth)
                ok = hit and facing > 0.12                     # самий край кулі - ненадійно
                info["edge"] = ok and facing < 0.35
            if anchor is None or np.hypot(*(smooth - anchor)) > args.radius * max(w, h):
                anchor, anchor_t, fired = smooth.copy(), now, False
            frac = min(1.0, (now - anchor_t) / args.dwell)
            info["dwell"] = frac
            if ok:
                info.update(state="ok", pos=(lat, lon))
                trail.append((lat, lon))
            else:
                info["state"] = "offmap"
                lat = lon = None
            col = (12, 163, 12) if fired else (52, 104, 235)
            c = (int(smooth[0]), int(smooth[1]))
            cv2.circle(frame, c, 7, col, -1)
            cv2.ellipse(frame, c, (24, 24), -90, 0, 360 * frac, col, 3)
            cv2.putText(frame, "MAP", (c[0] + 26, c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

        s = layer.sample(lat, lon, t) if lat is not None else None
        info["sample"] = s

        if smooth is not None and frac >= 1.0 and not fired:
            fired = True
            if s is not None:
                last_fix = report(layer, t, lat, lon, s)
                last_spoken = SP.compose(layer, t, lat, lon, s, geo.locate(lat, lon) if geo.ready else None)
                print(last_fix)
                print(f"   🔊 {last_spoken}")
                son.chime()
                voice.speak(last_spoken)
            else:
                print(f"[{datetime.now():%H:%M:%S}] {off_word}")
                voice.speak(off_word.capitalize() + ".")
        son.set_duck(voice.speaking())

        # ---- звук ----
        if s is None:
            son.silence()
        elif s["missing"]:
            son.missing(lon)
        else:
            son.play(s["norm"], s["tension"], lon, quiet=bool(s["near_km"]))
        info["note"] = None if son.note is None else f"{midi_name(son.note)} · {midi_to_hz(son.note):.0f} Гц"
        info["muted"] = son.muted or son.stream is None

        if globe is None:
            qcol = {"tracked": (0, 200, 0), "held": (0, 200, 255), "searching": (0, 0, 255), "static": (200, 200, 200)}
            cv2.polylines(frame, [np.int32(corners)], True, qcol[tracker.status], 2)
            for p in corners:
                cv2.circle(frame, (int(p[0]), int(p[1])), 6, qcol[tracker.status], -1)
            if layer.dynamic:
                draw_timeline(frame, G, layer, t, time_hand)
        else:                                                  # обрис, сітка і берегові лінії: видно, чи збіглося
            globe.overlay.draw(frame, tracker.status)

        cam = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ph = max(cam.shape[0], 700)
        if cam.shape[0] < ph:                                  # панель вища за камеру -> доповнюємо фоном
            cam = np.vstack([cam, np.full((ph - cam.shape[0], cam.shape[1], 3), (251, 252, 252), np.uint8)])
        cv2.imshow(WIN_CAM, np.hstack([cam, panel.render(ph, info)]))
        label = None if s is None else (layer.missing_label if s["missing"] else s["text"])
        pos = (lat, lon) if lat is not None else None
        map_img = view.render(layer, t, pos, trail, frac if pos is not None else 0.0, label)
        cv2.imshow(WIN_SST, map_img)
        if not placed:
            cv2.moveWindow(WIN_CAM, 0, 0)
            cv2.resizeWindow(WIN_SST, int(map_img.shape[1] * 0.85), int(map_img.shape[0] * 0.85))
            cv2.moveWindow(WIN_SST, 0, ph + 40)
            placed = True

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            print("Звук вимкнено" if son.toggle_mute() else "Звук увімкнено")
        if key == ord("v"):
            if voice.enabled:
                voice.stop()
                voice.enabled = False
                print("Голос вимкнено")
            elif not args.no_voice or voice.voice:
                voice = SP.Speaker(enabled=True, voice=args.voice, rate=args.voice_rate)
                print("Голос увімкнено" if voice.enabled else "Голос недоступний")
        if key == ord("r") and last_spoken and voice.enabled:
            voice.speak(last_spoken)
        if key == ord("i"):
            son.silence()
            announce(SP.layers_announcement(specs, li + 1))
        if key in (ord("?"), ord("/")):
            son.silence()
            announce(SP.help_announcement(keys, len(specs), surface))
        if key == ord("h"):
            swap = not swap
            print("Руки поміняно місцями: тепер " + ("ліва водить по карті, права керує часом" if swap
                                                     else "права водить по карті, ліва керує часом")
                  + " (за мітками MediaPipe).")
        if key == ord("l"):
            switch(li + 1)
        if ord("1") <= key <= ord("9") and key - ord("1") < len(specs):
            switch(key - ord("1"))
        if layer.dynamic:
            step = {ord(","): -1, ord("."): 1, ord("<"): -10, ord(">"): 10}.get(key)
            if step:
                t = layer.clamp_frame(t + step)
            if key == ord("p"):
                playing = not playing
                last_step = now
                print("Відтворення кадрів: " + ("▶" if playing else "пауза"))
        if key == ord("c") and globe is not None:
            son.silence()
            globe = run_globe_calibration(cap, args, scale)
            tracker = globe.tracker
        elif key == ord("c"):
            son.silence()
            cal = run_calibration(cap, args, scale)
            tracker = make_tracker(args.track, cal, frame.shape)
        if key == ord("k") and globe is not None:
            # освітлення змінилось -> заново вивчити кольори "вода/суша" при поточному повороті
            hm = hands_mask(res, w, h) if res.hand_landmarks else None
            tracker.learn_colors(frame, None, hand_mask=hm)
            if hasattr(tracker, "memory"):
                tracker.memory.add_from_frame(frame, globe.model, tracker.orb, hm, min_facing=0.6)
            GB.save_globe(GB.GLOBE_CALIB, globe.model, tracker)
            print("Кольори глобуса оновлено, поточний бік дозапам'ятовано")
        elif key == ord("k"):
            # ті самі 4 кути заново (напр. змінилось освітлення) -> новий еталон, калібрування місць лишається
            son.silence()
            corners = pick_corners(cap, scale)
            cal.template = capture_template(cap, corners)
            cv2.destroyWindow("Calibration")
            cal.ref_corners = corners
            cal.save(CALIB_FILE)
            tracker = make_tracker(args.track, cal, frame.shape)
            tracker.locked = True
    voice.stop()
    son.close()
    landmarker.close()
    if globe is not None:
        GB.save_globe(GB.GLOBE_CALIB, globe.model, tracker)       # останнє положення + дозапам'ятовані боки


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

    out = Path(args.selftest_out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- шари даних ----
    specs = L.discover_layers(args.month)
    by = {s.name: s.load() for s in specs}
    print(f"\n4) Шари даних ({len(specs)}), значення біля Одеси (46.5°N, 30.7°E):")
    for s in specs:
        lay = by[s.name]
        k_ = lay.default_frame()
        smp = lay.sample(46.48, 30.73, k_)
        frames_ = f"{lay.n_frames} кадр." if lay.dynamic else "1 момент"
        txt = "немає даних поблизу" if smp["missing"] else f"{smp['text']}  (висота {smp['norm']:.2f}, напруга {smp['tension']:+.2f})"
        print(f"   {s.name:20s} {frames_:>10s}, {lay.time_label(k_)}: {txt}")
    if "sst" in by:                       # звірка з NetCDF напряму
        import xarray as xr
        ds = xr.open_dataset(L.SST_FILE, engine="h5netcdf")
        raw = float(ds.sst.sel(lat=46.48, lon=30.73, method="nearest").isel(time=-1)) if args.month is None else None
        ds.close()
        if raw is not None:
            ok = abs(by["sst"].sample(46.48, 30.73)["value"] - raw) < 1e-3
            ok_all &= ok
            print(f"   sst == NetCDF напряму ({raw:.2f}): {'OK' if ok else 'FAIL'}")
    if "temperature" in by:               # звірка зі стисненого .u8.bin з повним .json
        d = json.loads((L.DATA_DIR / "temperature" / "temperature.json").read_text())
        yi = d["years"].index(2025)
        i = int(np.argmin(np.abs(np.array(d["lat"]) - 46.48))); j = int(np.argmin(np.abs(np.array(d["lon"]) - 30.73)))
        raw = d["data"][yi][i][j]
        got = by["temperature"].sample(46.48, 30.73, by["temperature"].frame_at(2025.5))["value"]
        ok = abs(got - raw) < 0.011
        ok_all &= ok
        print(f"   temperature 2025 == temperature.json ({raw:+.3f} vs {got:+.3f}): {'OK' if ok else 'FAIL'}")
    for name_ in ("ocean_sst_video", "ocean_flow_activity"):   # кадри з датами: звірка з повним .json
        if name_ not in by:
            continue
        lay = by[name_]
        d = json.loads((L.DATA_DIR / name_ / f"{name_}.json").read_text())
        i = int(np.argmin(np.abs(np.array(d["lat"]) - 35.0))); j = int(np.argmin(np.abs(np.array(d["lon"]) + 70.0)))
        step_ = (lay.vmax - lay.vmin) / 254
        errs = [abs(lay.sample(35.0, -70.0, k_)["value"] - d["data"][k_][i][j]) for k_ in (0, lay.n_frames - 1)]
        ok = (lay.n_frames == len(d["dates"]) and lay.has_dates and lay.labels[0] == "28.03.2007"
              and max(errs) <= step_ * 0.51)
        ok_all &= ok
        print(f"   {name_}: {lay.n_frames} кадрів {lay.labels[0]}…{lay.labels[-1]}, Гольфстрім кадр 1/{lay.n_frames}: "
              f"{lay.sample(35.0, -70.0, 0)['text']}, макс. похибка vs json {max(errs):.3f} (крок {step_:.3f}) "
              f"{'OK' if ok else 'FAIL'}")
    if "temperature" in by and "ocean_sst_video" in by:      # перемикання шару зберігає момент часу
        te, ov = by["temperature"], by["ocean_sst_video"]
        k_ = ov.frame_at(te.time_of(te.frame_at(2007.5)))
        ok = int(ov.times[k_]) == 2007 and ov.frame_at(te.time_of(te.frame_at(1950.5))) == 0
        ok_all &= ok
        print(f"   temperature 2007 -> ocean_sst_video: кадр {ov.labels[k_]}; 1950 -> найближчий {ov.labels[0]} "
              f"{'OK' if ok else 'FAIL'}")
    ok = "co2_curve" not in by
    ok_all &= ok
    print(f"   лише карти (криві на кшталт co2_curve пропущено): {sorted(by)} {'OK' if ok else 'FAIL'}")

    # ---- дві руки: права = карта, ліва = час ----
    print("\n4b) Дві руки:")
    from types import SimpleNamespace as NS  # noqa: F811

    def fake(*hands):   # hands: (мітка MediaPipe, x зап'ястя)
        lms = [[NS(x=x, y=0.5)] + [NS(x=x, y=0.5)] * 20 for _, x in hands]
        hd = [[NS(category_name=lab, score=0.9)] for lab, _ in hands]
        return NS(hand_landmarks=lms, handedness=hd)
    r1 = hand_roles(fake(("Left", 0.7), ("Right", 0.2)))        # кадр не дзеркальний -> мітки навпаки
    r2 = hand_roles(fake(("Left", 0.7), ("Right", 0.2)), swap=True)
    r3 = hand_roles(fake(("Left", 0.2), ("Left", 0.7)))         # однакові мітки -> за положенням
    ok_h = (r1["map"][0].x == 0.7 and r1["time"][0].x == 0.2 and r2["map"][0].x == 0.2
            and r3["time"][0].x == 0.2 and r3["map"][0].x == 0.7)
    print(f"   розподіл ролей (мітки MediaPipe, клавіша h, однакові мітки): {'OK' if ok_h else 'FAIL'}")
    for tl in [l_ for l_ in by.values() if l_.dynamic]:
        last = tl.n_frames - 1
        ks = [frame_from_u(tl, u_, None) for u_ in (0.0, 0.5, 1.0, -0.3, 1.4)]
        mid = round(0.5 * last)
        hyst = frame_from_u(tl, (mid + 0.4) / last, mid)        # зсув на 0.4 кадру -> кадр не змінюється
        moved = frame_from_u(tl, (mid + 1.0) / last, mid)       # зсув на 1 кадр -> змінюється
        ok_y = ks == [0, mid, last, 0, last] and hyst == mid and moved == mid + 1
        print(f"   {tl.name:20s} лівий палець -> кадр: край {tl.labels[ks[0]]}, середина {tl.labels[ks[1]]}, "
              f"край {tl.labels[ks[2]]} (усі {tl.n_frames}); гістерезис  {'OK' if ok_y else 'FAIL'}")
        tl_img = f2.copy()
        draw_timeline(tl_img, cam_to_sheet_H(ft.corners), tl, mid, True)
        cv2.imwrite(str(out / f"selftest_timeline_{tl.name}.png"), tl_img)
        ok_h &= ok_y
    ok_all &= ok_h

    print("\n5) Новий шар з NetCDF (конвертер layers.py from-netcdf), перевірка на sst.mnmean.nc:")
    if L.SST_FILE.exists():
        import xarray as xr
        folder = L.netcdf_to_layer(L.SST_FILE, "sst", "sst_annual_test", out / "layers", title="SST, середнє за рік")
        lay = L.GridLayer.from_folder(folder)
        ds = xr.open_dataset(L.SST_FILE, engine="h5netcdf")
        ann = ds.sst.sel(time=slice("2020", "2020")).mean("time").sel(lat=-30.0, lon=200.0, method="nearest")
        raw = float(ann); ds.close()
        got = lay.sample(-30.0, -160.0, lay.frame_at(2020.5))["value"]
        step = (lay.vmax - lay.vmin) / 254
        ok = abs(got - raw) <= step
        ok_all &= ok
        print(f"   створено {folder.name}/ ({lay.n_frames} кадрів {lay.labels[0]}–{lay.labels[-1]}); "
              f"2020, 30°S 160°W: {got:.2f} vs NetCDF {raw:.2f} (крок {step:.3f}) {'OK' if ok else 'FAIL'}")
        found = [s.name for s in L.discover_layers(data_dir=out / "layers", sst_file=out / "none.nc")]
        ok = found == ["sst_annual_test"]
        ok_all &= ok
        print(f"   автопошук бачить новий шар: {found} {'OK' if ok else 'FAIL'}")

    # ---- звук ----
    print("\n5b) Озвучення точки (географія + текст + голос):")
    geo = SP.Geocoder()
    expect = [((9.0, 39.5), "Ефіопія"), ((50.45, 30.52), "Україна"), ((45.8, 31.0), "Чорне море"),
              ((-28.0, 30.5), "Південно-Африканська Республіка"), ((0.0, -140.0), "Тихий океан"),
              ((28.0, 85.0), "Гімалаї")]
    ok_g = True
    for (la, lo), want in expect:
        ph = SP.place_phrase(geo.locate(la, lo))
        good = want in ph
        ok_g &= good
        print(f"   {la:+6.1f},{lo:+7.1f} -> {ph}  {'OK' if good else 'FAIL'}")
    plur = [SP.spoken_quantity(v, "°C", 1) for v in (1.0, 2.0, 5.0, 21.0, 22.4)]
    ok_p = plur == ["1 градус", "2 градуси", "5 градусів", "21 градус", "22,4 градуса"]
    print(f"   відмінки: {plur} {'OK' if ok_p else 'FAIL'}")
    msgs = []
    if "temperature" in by:
        te = by["temperature"]; k_ = te.frame_at(2025.5)
        msgs.append(SP.compose(te, k_, 9.0, 39.5, te.sample(9.0, 39.5, k_), geo.locate(9.0, 39.5)))
    if "sst" in by:
        msgs.append(SP.compose(by["sst"], 0, 45.8, 31.0, by["sst"].sample(45.8, 31.0), geo.locate(45.8, 31.0)))
    if "ocean_sst_video" in by:   # океанський шар посеред суходолу: чесно "даних немає", а не океан за 600 км
        ov = by["ocean_sst_video"]
        m_ = SP.compose(ov, 0, 9.0, 39.5, ov.sample(9.0, 39.5, 0), geo.locate(9.0, 39.5))
        ok_g &= "даних у цій точці немає" in m_
        msgs.append(m_)
    for m_ in msgs:
        print(f"   🔊 {m_}")
    ann = SP.layers_announcement(specs, 1)
    hlp = SP.help_announcement(KEYS, len(specs))
    intro = SP.layer_intro(2, len(specs), by[specs[1].name]) if len(specs) > 1 else ""
    ok_a = f"Доступно {len(specs)}" in ann and all(SP.spoken_title(by[s.name]) in ann for s in specs)
    nn = str(min(len(specs), 9))
    ok_k = all(sp_.replace("{n}", nn).lower() in hlp.lower() for _, sp_, _ in KEYS) and "{n}" not in hlp
    print(f"   🔊 клавіша i, перелік карт: {ann}  {'OK' if ok_a else 'FAIL'}")
    print(f"   🔊 перемикання карти: {intro}")
    print(f"   🔊 клавіша ?, довідка ({len(hlp)} символів, названо всі {len(KEYS)} клавіш {'OK' if ok_k else 'FAIL'}):")
    print("      " + hlp)
    ok_g &= ok_a and ok_k
    sp = SP.Speaker(enabled=True)
    if sp.enabled:
        import subprocess
        for fname, txt in (("layers_announcement.aiff", ann), ("help_announcement.aiff", hlp)):
            if sp.to_file(txt, out / fname):
                info_ = subprocess.run(["afinfo", str(out / fname)], capture_output=True, text=True).stdout
                dur = re.search(r"estimated duration: ([\d.]+)", info_)
                print(f"   записано {fname}: {float(dur.group(1)) if dur else 0:.0f} с звучання")
    if sp.enabled and msgs:
        aiff = out / "speech_test.aiff"
        ok_v = sp.to_file(msgs[0], aiff) and aiff.stat().st_size > 10000
        print(f"   голос {sp.voice}: записано {aiff.name} ({aiff.stat().st_size // 1024} КБ) {'OK' if ok_v else 'FAIL'}")
        ok_g &= ok_v
    else:
        print("   голос недоступний на цій машині (тест запису пропущено)")
    s_ = Sonifier(enabled=False)
    s_.play(0.5, 0.0, 0.0); s_.render(Sonifier.SR // 2)
    loud = np.abs(s_.render(Sonifier.SR // 4)).max()
    s_.set_duck(True); s_.render(Sonifier.SR // 2)
    quiet = np.abs(s_.render(Sonifier.SR // 4)).max()
    ok_d = quiet < 0.3 * loud
    print(f"   приглушення синтезатора під час голосу: {loud:.2f} -> {quiet:.2f} {'OK' if ok_d else 'FAIL'}")
    ok_all &= ok_g and ok_p and ok_d

    print("\n5c) Глобус (синтетичний кадр камери):")
    Wg, Hg = 1280, 720

    def look(lat0, lon0, tilt):                                 # орієнтація: точка (lat0, lon0) дивиться на камеру
        fwd = GB.latlon_to_vec(lat0, lon0); north = np.array([0, 0, 1.0])
        up = north - fwd * (north @ fwd); up /= np.linalg.norm(up); right = np.cross(up, fwd)
        a = np.radians(tilt)
        return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]) @ np.stack([right, -up, -fwd])
    geom_t = GB.GlobeGeometry((660, 370), 260, GB.fov_to_f(62, Wg), (Wg, Hg))
    R0_t = look(25, 15, 8)
    g0 = GB.render_synthetic(geom_t, R0_t, (Hg, Wg, 3))
    u_, v_, r_ = GB.detect_globe_circle(g0)
    ok_c = np.hypot(u_ - 660, v_ - 370) < 4 and abs(r_ - 260) < 4
    print(f"   обрис: центр ({u_:.0f},{v_:.0f}) r={r_:.0f} (істина 660,370 r=260) {'OK' if ok_c else 'FAIL'}")
    mt = GB.GlobeModel(geom_t, R0_t)
    clicks_, ll_ = [], []
    for lm_ in GLOBE_LANDMARKS:
        x_, y_, vis_ = mt.project_latlon(*lm_[2:])
        _, _, fac_ = geom_t.hit(np.array([x_]), np.array([y_]))
        if vis_ and fac_[0] > 0.25:
            clicks_.append((x_ + rng.normal(0, 1.5), y_ + rng.normal(0, 1.5))); ll_.append(lm_[2:])
    geom_g, R0_g, err_g = GB.solve_pose(clicks_, ll_, (u_, v_), r_, (Wg, Hg))
    fov_g = np.degrees(2 * np.arctan(Wg / 2 / geom_g.f))
    ang_g = np.degrees(np.arccos(np.clip((np.trace(R0_g.T @ R0_t) - 1) / 2, -1, 1)))
    ok_o = abs(fov_g - 62) <= 3 and ang_g < 2
    print(f"   калібрування: {len(clicks_)} видимих місць, поле зору {fov_g:.0f}° (істина 62°), "
          f"похибка орієнтації {ang_g:.2f}° {'OK' if ok_o else 'FAIL'}")
    gs = GlobeSurface(GB.GlobeModel(geom_g, R0_g), None)
    gs.tracker = GB.SpinTracker(gs.model)
    gs.tracker.learn_colors(g0, 0.0)
    errs_th, errs_km, times_ = [], [], []
    for th_t in (0, 37, 120, -75, 181):
        Rt = R0_t @ GB.rot_z(th_t)
        img_g = GB.render_synthetic(geom_t, Rt, (Hg, Wg, 3), seed=abs(th_t) % 97)
        # фальшива рука MediaPipe над глобусом: маска має її вирізати
        fake = NS(hand_landmarks=[[NS(x=(700 + 8 * k) / Wg, y=(380 + 12 * k) / Hg) for k in range(21)]])
        hm = hands_mask(fake, Wg, Hg)
        cv2.line(img_g, (700, 380), (900, 719), (90, 140, 200), 80)
        gs.model.theta, gs.tracker.locked = 0.0, False
        t0_ = time.perf_counter(); th_ = gs.tracker.update(img_g, hm); times_.append((time.perf_counter() - t0_) * 1000)
        errs_th.append(abs(GB.angdiff(th_, th_t % 360)))
        mtt = GB.GlobeModel(geom_t, Rt)
        for _ in range(150):
            la_, lo_ = rng.uniform(-80, 80), rng.uniform(-180, 180)
            x_, y_, vis_ = mtt.project_latlon(la_, lo_)
            _, _, fac_ = geom_t.hit(np.array([x_]), np.array([y_]))
            if vis_ and fac_[0] > 0.3:
                la2, lo2, okk, _ = gs.model.to_latlon(x_, y_)
                errs_km.append(L.haversine_km(la_, lo_, la2, lo2))
        if th_t == 37:
            ov_img = img_g.copy(); gs.overlay.draw(ov_img, gs.tracker.status)
            cv2.imwrite(str(out / "selftest_globe.png"), ov_img)
    ok_s = max(errs_th) < 2.0                                  # менше за одну клітинку даних (2°)
    ok_f = np.median(errs_km) < 60
    print(f"   поворот глобуса 0/37/120/-75/181° з рукою: макс. похибка {max(errs_th):.2f}°, "
          f"{np.mean(times_):.0f} мс/кадр {'OK' if ok_s else 'FAIL'}")
    print(f"   палець -> координати: медіана {np.median(errs_km):.0f} км, 95% {np.percentile(errs_km, 95):.0f} км "
          f"({len(errs_km)} точок) {'OK' if ok_f else 'FAIL'}")
    # --- запам'ятати глобус (тур) і знаходити його будь-де: переставлений, нахилений, з будь-якого боку
    tex_g = GB.synthetic_texture()
    gA_ = GB.GlobeGeometry((660, 370), 260, GB.fov_to_f(62, Wg), (Wg, Hg))
    imgA_ = GB.render_synthetic(gA_, R0_t, (Hg, Wg, 3), texture=tex_g)
    mdl = GB.GlobeModel(GB.GlobeGeometry((661, 368), 260, GB.fov_to_f(62, Wg), (Wg, Hg)), R0_t @ GB.rot_z(0.5))
    gtr = GB.GlobeTracker(mdl)
    gtr.learn_colors(imgA_, 0.0)
    gtr.memory.add_from_frame(imgA_, mdl, gtr.orb, min_facing=0.6)
    for k_ in range(1, 37):
        _, cov_ = gtr.tour_step(GB.render_synthetic(gA_, R0_t @ GB.rot_z(10 * k_), (Hg, Wg, 3), seed=k_, texture=tex_g),
                                now=float(k_))
    ok_t = cov_ >= 0.9
    print(f"   тур (повний оберт кроками по 10°): вивчено {cov_ * 100:.0f}%, пам'ять {len(gtr.memory)} точок "
          f"{'OK' if ok_t else 'FAIL'}")
    tilt_ = cv2.Rodrigues(np.array([0.05, 0.0, np.radians(12)]))[0]
    gB_ = GB.GlobeGeometry((470, 330), 300, GB.fov_to_f(62, Wg), (Wg, Hg))
    found, errs_o, times_o = 0, [], []
    for th_ in range(0, 360, 30):
        Rb = tilt_ @ R0_t @ GB.rot_z(th_)
        fr_ = GB.render_synthetic(gB_, Rb, (Hg, Wg, 3), seed=100 + th_, texture=tex_g)
        gtr.locked = False
        if hasattr(gtr, "_last_circle"):
            del gtr._last_circle
        t0_ = time.perf_counter(); gtr.update(fr_, now=1000.0 + th_); times_o.append((time.perf_counter() - t0_) * 1000)
        e_ = np.degrees(np.arccos(np.clip((np.trace(mdl.R().T @ Rb) - 1) / 2, -1, 1)))
        found += gtr.status == "tracked" and e_ < 5
        errs_o.append(e_)
    ok_m = found >= 11 and np.median(errs_o) < 1.5
    print(f"   глобус переставили, наблизили, нахилили на 12°: розпізнано з нуля {found}/12 боків, "
          f"похибка орієнтації медіана {np.median(errs_o):.2f}°, {np.mean(times_o):.0f} мс/кадр {'OK' if ok_m else 'FAIL'}")
    ov2 = fr_.copy(); GB.GlobeOverlay(mdl).draw(ov2, gtr.status); cv2.imwrite(str(out / "selftest_globe_moved.png"), ov2)
    ok_all &= ok_t and ok_m

    hg = SP.help_announcement(KEYS_GLOBE, len(specs), "globe")
    ok_hg = "глобус" in hg and "Лівим" not in hg and "кома і крапка" in hg
    print(f"   довідка для глобуса: про глобус, без лівої руки, час клавішами {'OK' if ok_hg else 'FAIL'}")
    ok_all &= ok_c and ok_o and ok_s and ok_f and ok_hg

    print("\n6) Сонифікація (синтез без звукової карти):")
    sr = Sonifier.SR
    base_layer = by.get("sst") or next(iter(by.values()))
    smp = base_layer.sample(46.48, 30.73, base_layer.default_frame())
    son = Sonifier(enabled=False)
    son.play(smp["norm"], 0.0, 30.73)
    son.render(sr // 2)                                         # дати параметрам усталитись
    a = son.render(sr).mean(1)
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    peak = np.fft.rfftfreq(len(a), 1 / sr)[np.argmax(spec)]
    want = midi_to_hz(norm_to_midi(smp["norm"]))
    ok_pitch = abs(peak - want) / want < 0.01
    print(f"   {base_layer.name}: {smp['text']} -> нота {midi_name(norm_to_midi(smp['norm']))} ({want:.1f} Гц), "
          f"виміряно FFT {peak:.1f} Гц  {'OK' if ok_pitch else 'FAIL'}")
    print(f"   діапазон будь-якого шару: vmin -> {midi_name(norm_to_midi(0))}, vmax -> {midi_name(norm_to_midi(1))}")

    def pulse_ratio(tension):
        s_ = Sonifier(enabled=False)
        s_.play(smp["norm"], tension, 30.73); s_.render(sr)
        x_ = s_.render(2 * sr).mean(1)
        rms = np.sqrt((x_[: len(x_) // 882 * 882].reshape(-1, 882) ** 2).mean(1))   # вікна по 20 мс
        return rms.max() / max(rms.min(), 1e-9)
    p0, p1 = pulse_ratio(0.0), pulse_ratio(smp["tension"] if smp["tension"] > 0.5 else 0.96)
    ok_pulse = p0 < 1.25 and p1 > 1.8
    print(f"   пульсація гучності: напруга 0 -> x{p0:.2f}, напруга {max(smp['tension'], 0.96):+.2f} -> x{p1:.2f}  "
          f"{'OK' if ok_pulse else 'FAIL'}")
    s_ = Sonifier(enabled=False)
    s_.play(0.5, 0.0, -150.0); s_.render(sr // 2)
    st_ = s_.render(sr)
    lr = np.sqrt((st_[:, 0] ** 2).mean() / (st_[:, 1] ** 2).mean())
    ok_pan = lr > 3
    print(f"   панорама: довгота 150°W -> ліво/право гучність x{lr:.1f}  {'OK' if ok_pan else 'FAIL'}")
    ok_all &= ok_pitch and ok_pulse and ok_pan

    def render_route(layer, points, seconds, blk=512):
        """points: [(lat, lon, момент у роках)]; лінійно між ними. -> стерео-масив"""
        son_ = Sonifier(enabled=False)
        nblk = int(seconds * sr / blk)
        seg = np.linspace(0, len(points) - 1, nblk)
        chunks = []
        for k in range(nblk):
            i = min(int(seg[k]), len(points) - 2); f = seg[k] - i
            la, lo, yr = [points[i][q] + f * (points[i + 1][q] - points[i][q]) for q in range(3)]
            r_ = layer.sample(la, lo, layer.frame_at(yr + 0.5) if layer.dynamic else None)
            if r_["missing"]:
                son_.missing(lo)
            else:
                son_.play(r_["norm"], r_["tension"], lo, quiet=bool(r_["near_km"]))
            if k == nblk - int(1.2 * sr / blk):
                son_.chime()
            chunks.append(son_.render(blk))
        return np.concatenate(chunks)

    demos = []
    if "sst" in by:   # подорож океаном: Антарктика -> екватор Тихого -> Атлантика -> Середземне -> Чорне море
        route = [(-62, -150, 0), (-30, -140, 0), (0, -140, 0), (15, -60, 0), (40, -40, 0), (55, -20, 0), (36, 10, 0),
                 (43, 34, 0)]
        demos.append(("sonification_demo.wav", "подорож океаном (sst)", render_route(by["sst"], route, 16.0)))
    if "temperature" in by:   # машина часу: одна точка (Одеса), 1880 -> 2025
        demos.append(("sonification_odesa_1880_2025.wav", "Одеса 1880→2025 (temperature)",
                      render_route(by["temperature"], [(46.48, 30.73, 1880), (46.48, 30.73, 2025)], 20.0)))
    for fname, title, wav in demos:
        ok = np.abs(wav).max() < 0.99
        ok_all &= ok
        write_wav(out / fname, wav)
        print(f"   {title}: {len(wav) / sr:.0f} с, пік {np.abs(wav).max():.2f} {'OK' if ok else 'FAIL'} -> {fname}")

    print("\n7) Рендер панелі та вікна для кожного шару:")
    view, panel = L.LayerView(), Panel(340)
    for k, s in enumerate(specs, 1):
        lay = by[s.name]
        y = lay.default_frame()
        smp = lay.sample(35.0, -70.0, y)                         # Гольфстрім: є дані в усіх шарах океану
        trail = [(30 + q * 0.6, -80 + q) for q in range(15)]
        cv2.imwrite(str(out / f"selftest_view_{s.name}.png"), view.render(lay, y, (35.0, -70.0), trail, 0.6, smp["text"]))
        cv2.imwrite(str(out / f"selftest_panel_{s.name}.png"), panel.render(700, {
            "layer": lay, "layer_idx": k, "n_layers": len(specs), "t": y, "playing": False, "state": "ok",
            "pos": (35.0, -70.0), "sample": smp, "sheet": "tracked", "inliers": 640, "dwell": 0.6,
            "note": None if smp["missing"] else midi_name(norm_to_midi(smp["norm"])),
            "last_fix": report(lay, y, 35.0, -70.0, smp)}))
    cv2.imwrite(str(out / "selftest_sheet.png"), cv2.polylines(hand, [np.int32(tracker.corners)], True, (0, 200, 0), 3))
    print(f"   збережено в {out}/")
    print("\nSELFTEST", "PASSED" if ok_all else "FAILED")


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="індекс камери (0 - вбудована)")
    ap.add_argument("--recalibrate", action="store_true", help="пройти калібрування заново")
    ap.add_argument("--layer", help="шар даних на старті (назва: sst, temperature, ...; див. layers.py list)")
    ap.add_argument("--swap-hands", action="store_true",
                    help="поміняти ролі рук (якщо програма плутає ліву і праву; у вікні - клавіша h)")
    ap.add_argument("--year", type=int, help="рік на старті: береться найближчий кадр шару")
    ap.add_argument("--step", type=float, default=0.3, help="секунд на рік у режимі відтворення (p)")
    ap.add_argument("--dwell", type=float, default=3.0, help="скільки секунд тримати палець для запису в консоль")
    ap.add_argument("--radius", type=float, default=0.02,
                    help="допустиме тремтіння пальця, частка ширини кадру (0.02 = 2%%)")
    ap.add_argument("--month", help="місяць для шару sst РРРР-ММ (за замовчуванням останній у файлі)")
    ap.add_argument("--projection", default="equirect", choices=PROJECTIONS,
                    help="проєкція, якщо клацнуто лише 4 місця")
    ap.add_argument("--lon0", type=float, default=0.0, help="центральний меридіан карти, якщо 4 місця")
    ap.add_argument("--surface", default="map", choices=["map", "globe"],
                    help="map - паперова карта (як раніше), globe - глобус на підставці з віссю")
    ap.add_argument("--track", default="features", choices=["features", "contour", "static"],
                    help="як відстежувати карту: features - за малюнком карти (за замовчуванням), "
                         "contour - за краями аркуша, static - карта нерухома")
    ap.add_argument("--no-sound", action="store_true", help="без сонифікації")
    ap.add_argument("--no-voice", action="store_true", help="не озвучувати точку голосом після фіксації")
    ap.add_argument("--voice", default="Lesya", help="голос macOS для озвучення (за замовчуванням Lesya, українська)")
    ap.add_argument("--voice-rate", type=int, default=185, help="швидкість мовлення, слів за хвилину")
    ap.add_argument("--volume", type=float, default=0.8, help="гучність 0..1")
    ap.add_argument("--width", type=int, default=1280, help="ширина кадру камери")
    ap.add_argument("--view", type=int, default=860, help="ширина відображення камери у вікні, px")
    ap.add_argument("--selftest", action="store_true", help="перевірка без камери")
    ap.add_argument("--selftest-out", default=str(HERE / "selftest_out"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.selftest:
        selftest(args); return

    specs = L.discover_layers(args.month)
    if not specs:
        raise SystemExit("Не знайдено жодного шару даних (sst.mnmean.nc або data/<шар>/). Див. LAYERS.md")
    if args.layer and args.layer not in [s.name for s in specs]:
        raise SystemExit(f"Немає шару '{args.layer}'. Є: {', '.join(s.name for s in specs)}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.width * 9 / 16))
    if not cap.isOpened():
        raise SystemExit("Камера не відкрилась. На macOS дозвольте доступ: Системні параметри -> "
                         "Приватність і безпека -> Камера -> ваш термінал / VS Code.")
    try:
        frame = _read(cap)
        scale = min(1.0, args.view / frame.shape[1])
        if args.surface == "globe":
            loaded = None if args.recalibrate else GB.load_globe()
            if loaded and tuple(loaded[0].geom.to_dict()["frame_wh"]) != (frame.shape[1], frame.shape[0]):
                print("Розмір кадру камери змінився з часу калібрування глобуса - потрібне нове калібрування.")
                loaded = None
            if loaded:
                print(f"Калібрування глобуса з {GB.GLOBE_CALIB.name}. (--recalibrate щоб переробити)")
            globe = GlobeSurface(*loaded) if loaded else run_globe_calibration(cap, args, scale)
            run_tracking(cap, None, specs, args, scale, globe=globe)
            return
        cal = MapCalibration.load(CALIB_FILE) if CALIB_FILE.exists() and not args.recalibrate else None
        if cal is None:
            if CALIB_FILE.exists() and not args.recalibrate:
                print("Старий формат calibration.json (без еталона карти), потрібне нове калібрування.")
            cal = run_calibration(cap, args, scale)
        else:
            print(f"Калібрування з {CALIB_FILE.name}: проєкція {cal.proj}. (--recalibrate щоб переробити)")
        run_tracking(cap, cal, specs, args, scale)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
