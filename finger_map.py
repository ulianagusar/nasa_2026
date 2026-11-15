"""Finger on a paper map (or a globe) -> value of the selected map layer (ocean temperature,
temperature anomaly, ... - see LAYERS.md) -> panel, map window, sound and voice.

Two hands: the RIGHT index finger moves over the map, the LEFT index finger sets the year -
by its position along the map (left edge of the sheet = first year, right edge = last).
Layers: keys l / 1-9; time also , . (±1), < > (±10), p (play); h - swap hands.
Announcement (right finger still for 3 s) -> a voice (best English voice) tells the place (country / sea / region),
the time and the layer value (speech.py); v - voice off, r - repeat.

The camera looks at a printed world map. The program:
  * tracks the map by its own picture (ORB points + homography), without physical
    markers, so the map can be moved and rotated on the table without recalibrating;
  * tracks the index fingertip (MediaPipe);
  * shows the coordinates and the value under the finger live in the panel on the right;
  * in the "Data Layer" window a dot on the layer map follows the finger;
  * sonification: the value under the finger is heard live (note pitch, timbre,
    beating/pulsing when warmer than normal, stereo by longitude; m - sound off);
  * if the finger stays still for 3 seconds, the point is announced and printed to the console (with a chime).

Run:
    .venv/bin/python finger_map.py                 # start menu: virtual map / paper map / physical globe / virtual globe
    .venv/bin/python finger_map.py --surface vmap  # skip the menu (map, globe, virtual, vmap)
    .venv/bin/python finger_map.py --recalibrate   # calibrate again
    .venv/bin/python finger_map.py --selftest      # check the maths without a camera

Calibration (2 steps):
  1) the 4 corners of the map (virtually): click them with the mouse or press A to accept
     the corners found automatically (green frame). At this moment the program remembers
     what the map looks like as a template, so keep your hand away from the map;
  2) 6-8 recognizable places (Cape Horn, Gibraltar, ...), see landmarks_reference.png.
     From them the program works out the map projection.
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
import virtual_map as VM

HERE = Path(__file__).resolve().parent
CALIB_FILE = HERE / "calibration.json"
CALIB_VERSION = 3
TEMPLATE_FILE = HERE / "calibration_template.png"
MODEL_FILE = HERE / "models" / "hand_landmarker.task"
INDEX_TIP = 8  # MediaPipe: index fingertip
WIN_CAM, WIN_SST = "Earth Map", "Data Layer"

# Keys in the window: (how to show it, how to say it, what it does). Both the console help and
# the spoken help (the ? key) are built from this one list, so they never disagree.
KEYS = [
    ("?", "question mark", "this help; press it again to stop"),
    ("i", "the I key", "list the maps"),
    ("1-9", "numbers 1 to {n}", "choose a map"),
    ("l", "the L key", "next map"),
    (", .", "comma and period", "step back or forward in time"),
    ("< >", "shift comma and shift period", "ten steps in time"),
    ("p", "the P key", "play through time; press again to pause"),
    ("r", "the R key", "repeat the last message"),
    ("m", "the M key", "sound on or off"),
    ("v", "the V key", "voice on or off"),
    ("h", "the H key", "swap hands if the program mixes them up"),
    ("c", "the C key", "recalibrate the map"),
    ("k", "the K key", "mark the map corners again"),
    ("b", "the B key", "back to the start menu"),
    ("q", "the Q key", "quit"),
]
# Virtual globe: the left hand rotates it, so time is on the keys; no calibration, but "north up" and size.
# Virtual map: the left index finger sets the time (as on paper), left thumb + index spread zooms; no calibration.
KEYS_VMAP = [k for k in KEYS if k[0] not in ("c", "k")] + [
    ("+ -", "plus and minus", "zoom in or out around the centre"),
    ("0", "zero", "zoom out to the whole world"),
]
KEYS_VIRTUAL = [k for k in KEYS if k[0] not in ("c", "k")] + [
    ("n", "the N key", "turn the globe north up"),
    ("+ -", "plus and minus", "make the globe bigger or smaller"),
]
# Globe: no left-hand time control, so "comma/period/p" are the main time controls; c/k mean something else.
KEYS_GLOBE = [k for k in KEYS if k[0] not in ("c", "k")] + [
    ("c", "the C key", "recalibrate the globe"),
    ("k", "the K key", "relearn the colours and remember the side of the globe you see now"),
]

# Recognizable calibration points: (name for the window, name for the console, lat, lon)
LANDMARKS = [
    ("Cape Horn - S tip of South America", "Cape Horn (southern tip of South America)", -55.98, -67.27),
    ("Cape Agulhas - S tip of Africa", "Cape Agulhas (southern tip of Africa)", -34.83, 20.00),
    ("Strait of Gibraltar", "Strait of Gibraltar", 35.95, -5.60),
    ("Cape Farewell - S tip of Greenland", "Cape Farewell (southern tip of Greenland)", 59.77, -43.90),
    ("Cape Comorin - S tip of India", "Cape Comorin (southern tip of India)", 8.08, 77.55),
    ("Cape York - N tip of Australia", "Cape York (northern tip of Australia)", -10.69, 142.53),
    ("Cape Lopatka - S tip of Kamchatka", "Cape Lopatka (southern tip of Kamchatka)", 50.87, 156.67),
    ("Cabo San Lucas - S tip of Baja California", "Cabo San Lucas (southern tip of Baja California)", 22.88, -109.91),
    ("North Cape - N tip of Norway", "North Cape (northern tip of Norway)", 71.17, 25.78),
]
CORNER_NAMES = [("TOP-LEFT", "top-left"), ("TOP-RIGHT", "top-right"),
                ("BOTTOM-RIGHT", "bottom-right"), ("BOTTOM-LEFT", "bottom-left")]

# Colours (RGB for PIL / matplotlib)
SURFACE, INK, INK2, MUTED, GRIDC, LANDC = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#d6d5cf"
WARM, COOL = "#d03b3b", "#256abf"


# ----------------------------------------------------------------------------
# Map projections: forward (lat, lon_rel) -> (x, y) and inverse
# lon_rel = longitude relative to the map's central meridian lon0, in [-180, 180)
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
# Map sheet: 4 corners -> the sheet's own coordinates (u, v) in the square [0,1]x[0,1]
# The calibration is stored in (u, v), so it does not depend on where the sheet lies on the table.
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
    """Corners clockwise (on screen), starting from the top-left one."""
    q = np.asarray(q, float)
    c = q.mean(0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]
    return np.roll(q, -int(np.argmin(q.sum(1))), axis=0)


def align_quad(q, ref):
    """Numbering of the corners q closest to the previous corners ref (so a rotated sheet does not mix up corners)."""
    q = order_quad(q)
    best = min(range(4), key=lambda r: np.linalg.norm(np.roll(q, -r, 0) - ref, axis=1).sum())
    return np.roll(q, -best, 0)


def _refine_quad(contour, quad):
    """Corner refinement: a line through all points of each side (fitLine), corner = intersection of neighbouring lines."""
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
    for k in range(4):  # corner k lies between side k-1 and side k
        (p1, d1), (p2, d2) = lines[k - 1], lines[k]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-3:
            return quad
        s = np.linalg.solve(A, p2 - p1)
        out.append(p1 + s[0] * d1)
    out = np.array(out)
    return out if np.linalg.norm(out - quad, axis=1).max() < 0.05 * np.sqrt(quad_area(quad)) else quad


def detect_quads(frame, min_area_frac=0.06):
    """Sheet candidates: large convex quadrilaterals (edges from Canny and Otsu)."""
    h, w = frame.shape[:2]
    s = min(1.0, 640.0 / max(h, w))
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    # close (not dilate): closes gaps in the contour but does not push the sheet edge outwards
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
    """Keeps the current 4 corners of the sheet. If the sheet is not found (a hand covers its edge),
    keeps the last known corners."""

    def __init__(self, ref_corners, frame_shape, static=False):
        self.corners = np.asarray(ref_corners, float)
        self.diag = float(np.hypot(*frame_shape[:2]))
        self.static = static
        self.locked = False   # first lock after start: any displacement is allowed
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
        # smooth small jitter, take real movement immediately
        self.corners = 0.6 * self.corners + 0.4 * q if d < 0.008 * self.diag else q
        self.locked, self.status = True, "tracked"
        return self.corners


def make_template(frame, corners):
    """Map template: the part of the frame inside the 4 corners, rectified into a rectangle."""
    c = np.asarray(corners, float)
    top, bottom = np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[2] - c[3])
    left, right = np.linalg.norm(c[3] - c[0]), np.linalg.norm(c[2] - c[1])
    tw = int(max(400, (top + bottom) / 2))
    th = int(round(tw * (left + right) / max(top + bottom, 1.0)))
    G = cv2.getPerspectiveTransform(np.float32(c), np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]))
    return cv2.warpPerspective(frame, G, (tw, th))


class FeatureTracker:
    """Tracking the map by its own picture (no markers and no sheet edges needed).

    During calibration a template is remembered: the rectified image of the map.
    In every frame ORB points are found (coast bends, islands, labels), matched
    with the template points, and from hundreds of matches a template -> frame homography is computed (MAGSAC).
    The hand covers some of the points, but the rest are enough.
    """
    WORK_W = 960     # frame width for point detection (speed/accuracy)
    MIN_INLIERS = 25

    def __init__(self, template, ref_corners, frame_shape):
        g = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.kt, self.dt = cv2.ORB_create(3000).detectAndCompute(g, None)
        if self.dt is None or len(self.kt) < 60:
            raise ValueError("The map template has too few details for tracking")
        self.tpts = np.float32([k.pt for k in self.kt])
        th, tw = g.shape
        self.tcorners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
        self.orb = cv2.ORB_create(2500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.corners = np.asarray(ref_corners, float)
        self.diag = float(np.hypot(*frame_shape[:2]))
        self.locked, self.status, self.inliers = False, "searching", 0

    def match(self, frame):
        """-> (map corners in the frame or None, number of matching points)"""
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
        # suppress sub-pixel jitter, take real movement immediately
        self.corners = 0.7 * self.corners + 0.3 * q if self.locked and d < 0.0015 * self.diag else q
        self.locked, self.status = True, "tracked"
        return self.corners


def make_tracker(mode, cal, frame_shape):
    if mode == "features":
        if cal.template is None:
            raise SystemExit("No map template for tracking -> run with --recalibrate")
        return FeatureTracker(cal.template, cal.ref_corners, frame_shape)
    return SheetTracker(cal.ref_corners, frame_shape, static=(mode == "static"))


class OneEuro:
    """One Euro filter for the fingertip: smooths strongly while the finger is still,
    and adds almost no lag while it moves."""

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
# Calibration: sheet points (u, v) <-> lat/lon  =>  homography sheet -> projection plane
# ----------------------------------------------------------------------------
class MapCalibration:
    def __init__(self, proj, lon0, H, ref_corners=None):
        self.proj, self.lon0, self.H = proj, float(lon0), np.asarray(H, float)
        self.ref_corners = None if ref_corners is None else np.asarray(ref_corners, float)
        self.template = None                 # rectified map image (template for tracking)
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
    """Finds the projection + central meridian that best explain the clicked points."""
    src = np.asarray(sheet_pts, np.float64)
    geo = np.asarray(geo_pts, np.float64)
    if len(src) < 4:
        raise ValueError("At least 4 points are needed")
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
            print(f"  Only 4 points: the projection cannot be detected automatically, using '{fallback_proj}'.")
        return best[2]
    if verbose:
        print("  Calibration error (mean / max, km):")
        seen = set()
        for mean_e, max_e, cal in candidates:
            if cal.proj in seen:
                continue
            seen.add(cal.proj)
            mark = "  <- chosen" if cal is candidates[0][2] else ""
            print(f"    {cal.proj:9s} (centre {cal.lon0:+.0f}°): {mean_e:7.0f} / {max_e:7.0f}{mark}")
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
        raise SystemExit("Could not read a frame from the camera")
    return frame


def pick_corners(cap, scale, win="Calibration"):
    """Step 1: the 4 corners of the sheet (mouse clicks, or A = accept the automatically found ones)."""
    print("\n--- Step 1: corners of the map sheet ---")
    print("Click the 4 corners of the SHEET in this order: top-left -> top-right -> bottom-right -> bottom-left")
    print("(top = north of the map). Or press A if the green frame already fits the sheet exactly.")
    print("  u - undo the last click, q - quit.")
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
            print("  Accepted the automatically found corners.")
            return auto
        if len(clicks) >= 4:
            corners = np.array(clicks[:4], float)
            diag = np.hypot(*frame.shape[:2])
            if auto is not None:
                auto_al = align_quad(auto, corners)
                if np.all(np.linalg.norm(auto_al - corners, axis=1) < 0.04 * diag):
                    print("  The clicks match the automatically found sheet, using the detector's precise corners.")
                    return auto_al
            print("  Corners set manually. (Auto-tracking will look for the sheet near them.)")
            return corners


def run_calibration(cap, args, scale):
    print("\n=== CALIBRATION === Do not move the map during calibration.")
    corners = pick_corners(cap, scale)
    template = capture_template(cap, corners)
    G = cam_to_sheet_H(corners)
    print("\n--- Step 2: recognizable places (see landmarks_reference.png) ---")
    print("Click the place whose name is shown at the top. s - skip, u - undo,")
    print("Enter - finish (at least 4, better 6+), q - quit.\n")
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
                warn = "  (warning: outside the sheet!)" if not (-0.02 <= uv[0] <= 1.02 and -0.02 <= uv[1] <= 1.02) else ""
                print(f"      sheet u={uv[0]:.3f} v={uv[1]:.3f} -> {LANDMARKS[idx][2]:+.2f}, {LANDMARKS[idx][3]:+.2f}{warn}")
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
            print("      skipped")
            idx += 1
            if idx < len(LANDMARKS):
                print(f"  [{idx + 1}/{len(LANDMARKS)}] {LANDMARKS[idx][1]}")
        if key == ord("u") and cam_pts:
            cam_pts.pop(); sheet_pts.pop(); geo_pts.pop(); idx = max(0, idx - 1)
            print(f"      undone. Again: [{idx + 1}/{len(LANDMARKS)}] {LANDMARKS[idx][1]}")
        if key in (13, 10):
            if len(cam_pts) < 4:
                print("  At least 4 points are needed.")
                continue
            break
    cv2.destroyWindow(win)
    cal = fit_calibration(sheet_pts, geo_pts, args.projection, args.lon0)
    cal.ref_corners, cal.template = corners, template
    cal.sheet_pts, cal.geo_pts = sheet_pts, geo_pts
    cal.save(CALIB_FILE)
    print(f"  Saved to {CALIB_FILE.name} + {TEMPLATE_FILE.name}: projection {cal.proj}, "
          f"central meridian {cal.lon0:+.0f}°\n")
    return cal


def capture_template(cap, corners):
    """Snapshot of the map as the template for picture-based tracking. Keep your hand away from the map at this moment."""
    template = make_template(_read(cap), corners)
    n = len(cv2.ORB_create(3000).detect(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), None))
    print(f"  Map template: {template.shape[1]}x{template.shape[0]} px, feature points: {n}")
    if n < 300:
        print("  Warning: few details. Picture-based tracking may be unstable "
              "(more light / camera closer helps), or run with --track contour.")
    return template


# ----------------------------------------------------------------------------
# Globe (--surface globe): calibration and helpers. The maths is in globe.py
# ----------------------------------------------------------------------------
# Only one hemisphere of a globe is visible at a time, so there are more known places all over the world.
GLOBE_EXTRA_LANDMARKS = [
    ("Cape Guardafui - Horn of Africa", "Cape Guardafui (Horn of Africa)", 11.83, 51.28),
    ("Cape Verde - Dakar, W tip of Africa", "Cape Verde, Dakar (western tip of Africa)", 14.72, -17.53),
    ("Cabo Sao Roque - E tip of Brazil", "Cabo Sao Roque (eastern tip of Brazil)", -5.48, -35.26),
    ("Singapore - S tip of Malay peninsula", "Singapore (southern tip of the Malay Peninsula)", 1.26, 103.82),
    ("Cape Reinga - N tip of New Zealand", "Cape Reinga (northern tip of New Zealand)", -34.43, 172.68),
    ("S tip of Tasmania", "southern tip of Tasmania", -43.64, 146.83),
    ("Cape Prince of Wales - Bering Strait", "Bering Strait (Alaska)", 65.6, -168.09),
    ("S tip of Florida", "southern tip of Florida", 25.12, -81.09),
]
GLOBE_LANDMARKS = LANDMARKS + GLOBE_EXTRA_LANDMARKS


class GlobeSurface:
    """What run_tracking needs from a globe: the model (pixel -> lat/lon), rotation tracking, overlay."""

    def __init__(self, model, tracker):
        self.model, self.tracker = model, tracker
        self.overlay = GB.GlobeOverlay(model)

    def to_latlon(self, px, py):
        return self.model.to_latlon(px, py)


def hands_mask(res, w, h):
    """Mask of the hands (and forearm) for globe tracking: these pixels are not compared with the land map."""
    mask = np.zeros((h, w), np.uint8)
    for lms in (res.hand_landmarks or []):
        pts = np.array([[p.x * w, p.y * h] for p in lms])
        wrist, mid = pts[0], pts[9]
        arm = wrist + 3.0 * (wrist - mid)                       # forearm: extend from the wrist
        hull = cv2.convexHull(np.int32(np.vstack([pts, arm[None]])))
        cv2.fillConvexPoly(mask, hull, 255)
    if mask.any():
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    return mask


def pick_globe_circle(cap, scale, win="Calibration"):
    """Step 1: globe outline (A = accept the automatically found circle, or 3 clicks on the edge of the sphere)."""
    print("\n--- Step 1: globe outline ---")
    print("Press A if the green circle fits the edge of the sphere exactly, or click 3 points on the edge of the globe.")
    print("  u - undo a click, q - quit.")
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
            print(f"  Circle accepted: centre ({auto[0]:.0f}, {auto[1]:.0f}), radius {auto[2]:.0f} px")
            return auto
        if len(clicks) >= 3:
            c = GB.circle_from_3pts(*clicks[:3])
            if c is None:
                print("  The points are on one line, click again.")
                clicks.clear()
                continue
            print(f"  Circle from 3 points: centre ({c[0]:.0f}, {c[1]:.0f}), radius {c[2]:.0f} px")
            return c


def run_globe_calibration(cap, args, scale):
    print("\n=== GLOBE CALIBRATION === Do not spin the globe during calibration, keep your hand away.")
    circle = pick_globe_circle(cap, scale)
    frame = _read(cap)
    H_, W_ = frame.shape[:2]
    print("\n--- Step 2: known places on the visible side of the globe ---")
    print("Click the place whose name is shown at the top. s - skip (the place is on the far side),")
    print("u - undo, Enter - finish (at least 3, better 5+), q - quit.")
    print("After 3 places the program shows where to look for the next one (yellow circle) and skips hidden ones.\n")
    win = "Calibration"
    clicks, cam_pts, geo_pts = [], [], []
    cv2.setMouseCallback(win, lambda ev, x, y, *_: clicks.append((x / scale, y / scale))
                         if ev == cv2.EVENT_LBUTTONDOWN else None)
    idx, prelim, announced = 0, None, -1
    while True:
        frame = _read(cap)
        # after 3 points: preliminary orientation -> hint and auto-skip of places on the far side
        while prelim is not None and idx < len(GLOBE_LANDMARKS):
            x, y, vis = prelim.project_latlon(*GLOBE_LANDMARKS[idx][2:])
            _, _, fac = prelim.geom.hit(np.array([x]), np.array([y]))
            if bool(vis) and fac[0] > 0.2:
                break
            print(f"  [{idx + 1}/{len(GLOBE_LANDMARKS)}] {GLOBE_LANDMARKS[idx][1]} - on the far side, skipped")
            idx += 1
        if idx < len(GLOBE_LANDMARKS) and announced != idx:
            print(f"  [{idx + 1}/{len(GLOBE_LANDMARKS)}] {GLOBE_LANDMARKS[idx][1]}")
            announced = idx
        if clicks:
            c = clicks[-1]
            clicks.clear()
            if idx < len(GLOBE_LANDMARKS):
                if (c[0] - circle[0]) ** 2 + (c[1] - circle[1]) ** 2 > circle[2] ** 2:
                    print("      click outside the globe - try again")
                else:
                    cam_pts.append(c); geo_pts.append(GLOBE_LANDMARKS[idx][2:])
                    print(f"      click ({c[0]:.0f}, {c[1]:.0f}) -> {GLOBE_LANDMARKS[idx][2]:+.2f}, {GLOBE_LANDMARKS[idx][3]:+.2f}")
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
            print("      skipped")
            idx += 1
        if key == ord("u") and cam_pts:
            cam_pts.pop(); geo_pts.pop(); idx = max(0, idx - 1); announced = -1
            prelim = None
        if key in (13, 10):
            if len(cam_pts) < 3:
                print("  At least 3 places are needed.")
                continue
            break
    geom, R0, err = GB.solve_pose(cam_pts, geo_pts, circle[:2], circle[2], (W_, H_))
    model = GB.GlobeModel(geom, R0)
    tracker = GB.GlobeTracker(model)
    frame = _read(cap)
    tracker.learn_colors(frame, 0.0)
    tracker.memory.add_from_frame(frame, model, tracker.orb, min_facing=0.6)
    fov = np.degrees(2 * np.arctan(W_ / 2 / geom.f))
    print(f"  Orientation found: camera field of view {fov:.0f}°, mean error of the places {err:.1f}° (~{err * 111:.0f} km)")
    globe_tour(cap, tracker, scale, win)
    cv2.destroyWindow(win)
    GB.save_globe(GB.GLOBE_CALIB, model, tracker, {"clicks": cam_pts, "latlons": geo_pts})
    print(f"  Saved: {GB.GLOBE_CALIB.name} (calibration) and {GB.GLOBE_MEMORY.name} "
          f"(globe memory, {len(tracker.memory)} points)\n")
    return GlobeSurface(model, tracker)


def globe_tour(cap, tracker, scale, win="Calibration", target=0.95):
    """Step 3: the globe stays in place and is slowly turned one full revolution - the program remembers every side.
    The angle comes from the land outlines (no accumulating error), the hand is cut out of the comparison."""
    print("\n--- Step 3: remember the globe ---")
    print("Slowly turn the globe one full revolution around its axis (do not move the stand).")
    print("The program remembers every side so that it can later find the globe anywhere in the frame.")
    print("  Enter - finish early, s - skip (then the globe cannot be moved around), q - quit.")
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
                print(f"  learned {pct}% of the globe (points in memory: {len(tracker.memory)})")
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
                    print(f"  Globe learned ({pct}%).")
                elif key == ord("s"):
                    print("  Tour skipped: the globe is recognized only from the side seen during calibration.")
                return
    finally:
        landmarker.close()


# ----------------------------------------------------------------------------
# Console report and the on-screen header (map name + year)
# ----------------------------------------------------------------------------
def report(layer, t, lat, lon, s):
    head = f"[{datetime.now():%H:%M:%S}] {L.fmt_coord(lat, lon)} · {layer.title}, {layer.time_label(t)}"
    if s is None or s["missing"]:
        return f"{head} -> {layer.missing_label}"
    extra = [t for t, role in s["lines"] if role in ("ink2", "warm", "cool")]
    return f"{head} -> {s['text']}" + (" | " + " | ".join(extra) if extra else "")


def draw_header(img, layer, t):
    """Small band at the top of the camera picture: the map name and the year (or date) - nothing else."""
    from PIL import Image, ImageDraw
    f = L.get_fonts()
    if layer.dynamic:
        when = ("Date " if layer.has_dates else "Year ") + layer.time_label(t)
    else:
        when = layer.time_label(None)
    hb = 62
    band = (img[:hb].astype(np.float32) * 0.35).astype(np.uint8)          # darkened strip under the text
    pil = Image.fromarray(cv2.cvtColor(band, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.text((14, 8), layer.title, font=f["l"], fill=(255, 255, 255))
    d.text((14, 33), when, font=f["s"], fill=(225, 225, 225))
    img[:hb] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return img


# ----------------------------------------------------------------------------
# Sonification: the value of any layer -> real-time sound
# ----------------------------------------------------------------------------
PENTATONIC = [0, 2, 4, 7, 9]           # major pentatonic: any two notes sound consonant
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def norm_to_midi(norm, midi_lo=48, midi_hi=76):
    """Normalized layer value 0..1 (vmin..vmax) -> pentatonic note between C3 (low) and E5 (high)."""
    scale = [m for m in range(midi_lo, midi_hi + 1) if (m - midi_lo) % 12 in PENTATONIC]
    return scale[int(round(np.clip(norm, 0, 1) * (len(scale) - 1)))]


def midi_to_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def midi_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


class Sonifier:
    """Additive synthesizer whose parameters are driven by the data of any layer:
      value (norm 0..1)     -> note pitch (pentatonic) and timbre brightness (number of harmonics);
      tension               -> how far the value is above/below the layer's baseline (normal):
                               beating (second voice detuned) and a pulsing "heartbeat" when above;
      longitude             -> stereo panorama (west on the left, east on the right);
      no data               -> soft noise; announcement -> chime.
    All changes are smooth (portamento/smoothing), so there are no clicks.
    render() also works without a sound card (for writing WAV files and tests)."""
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
            except Exception as e:  # no sound card etc. - run without sound
                print(f"  Sound off: {e}")

    # --- control from the main thread ---
    def play(self, norm, tension, lon, quiet=False):
        """norm: 0..1 on the layer scale; tension: (value - baseline) / layer scale, ~1 = strong deviation."""
        m = norm_to_midi(norm)
        self.note = m
        tn = 0.0 if tension is None or np.isnan(tension) else float(tension)
        self.target.update(
            freq=midi_to_hz(m),
            amp=0.45 if quiet else 1.0,
            noise=0.15 if quiet else 0.0,
            bright=2.4 - 1.5 * float(np.clip(norm, 0, 1)),  # lower exponent = more harmonics = brighter
            detune=float(np.clip(abs(tn) * 5.0, 0, 8)),     # Hz of beating between the two voices
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
        """Lower the synthesizer (e.g. while the voice is speaking) so the words can be heard."""
        self.target["duck"] = 0.2 if on else 1.0

    def toggle_mute(self):
        self.muted = not self.muted
        return self.muted

    def close(self):
        if self.stream is not None:
            self.stream.stop(); self.stream.close()

    # --- synthesis ---
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
        w = h ** (-bright[None, :])                                   # harmonic roll-off
        w /= w.sum(0, keepdims=True)
        inc1 = 2 * np.pi * h * f[None, :] / self.SR
        inc2 = 2 * np.pi * h * (f + det)[None, :] / self.SR
        p1 = self.ph1[:, None] + np.cumsum(inc1, 1)
        p2 = self.ph2[:, None] + np.cumsum(inc2, 1)
        self.ph1, self.ph2 = p1[:, -1] % (2 * np.pi), p2[:, -1] % (2 * np.pi)
        tone = 0.5 * ((w * np.sin(p1)).sum(0) + (w * np.sin(p2)).sum(0))
        pp = self.pph + np.cumsum(2 * np.pi * prate / self.SR)
        self.pph = pp[-1] % (2 * np.pi)
        tone *= 1 - pdepth * (0.5 + 0.5 * np.sin(pp))                 # pulsing "heartbeat"
        brown, self.noise_zi = lfilter([0.02], [1, -0.98], self.rng.normal(0, 1, n), zi=self.noise_zi)
        sig = 0.6 * amp * tone + 0.8 * noise * brown
        if self.chime_f is not None:                                  # announcement chime
            t = self.chime_t + np.arange(n) / self.SR
            env = np.exp(-t / 0.35)
            sig += 0.18 * env * (np.sin(2 * np.pi * self.chime_f * t) + 0.4 * np.sin(2 * np.pi * 2.76 * self.chime_f * t))
            self.chime_t = t[-1] + 1 / self.SR
            if self.chime_t > 2.0:
                self.chime_f = None
        sig = np.tanh(1.2 * sig) / 1.2 * self.volume * duck
        theta = (pan + 1) * np.pi / 4                                 # equal-power panning
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
    """MediaPipe result -> {"map": landmarks of the right hand, "time": landmarks of the left hand}.

    MediaPipe labels the hands as if the frame were mirrored (selfie camera). Our frame is not mirrored,
    so the "Left"/"Right" labels are swapped; swap=True (the h key) swaps them back.
    If both hands got the same label, they are told apart by position: the one further left in the frame = left."""
    hands = []
    for lms, hd in zip(res.hand_landmarks, res.handedness):
        label = hd[0].category_name
        if not swap:
            label = "Right" if label == "Left" else "Left"
        hands.append([label, lms])
    if len(hands) == 2 and hands[0][0] == hands[1][0]:
        hands.sort(key=lambda h: h[1][0].x)                    # 0 = wrist
        hands[0][0], hands[1][0] = "Left", "Right"
    return {("map" if label == "Right" else "time"): lms for label, lms in hands}


def frame_from_u(layer, u, current, hysteresis=0.6):
    """Position of the left finger along the map (u: 0 = left edge of the sheet, 1 = right) -> frame of the layer
    (year, week, ... - however many frames there are). Hysteresis: the frame changes only when the finger moved
    more than 0.6 of a frame, so it does not flicker."""
    last = layer.n_frames - 1
    cont = float(np.clip(u, 0, 1)) * last
    if current is None or abs(cont - current) > hysteresis:
        return int(np.clip(round(cont), 0, last))
    return current


def _timeline_ticks(layer, max_labels=7):
    """Frames with tick marks on the scale: round years for yearly layers, evenly spaced for dated layers."""
    n = layer.n_frames
    if not layer.has_dates:
        y0, y1 = int(layer.times[0]), int(layer.times[-1])
        step = next(s for s in (1, 2, 5, 10, 20, 25, 50, 100) if (y1 - y0) / s <= max_labels * 2)
        ticks = [layer.frame_at(y + 0.5) for y in range((y0 // step + 1) * step, y1, step)]
        return [(k, layer.short_label(k), (int(layer.times[k]) // step) % 2 == 0) for k in ticks]
    ticks = np.unique(np.round(np.linspace(0, n - 1, max_labels)).astype(int))
    return [(int(k), layer.short_label(int(k)), True) for k in ticks]


def draw_timeline(frame, G, layer, t, active):
    """Time scale along the bottom edge of the sheet (where the left hand "reads" it)."""
    Ginv = np.linalg.inv(G)
    last = max(layer.n_frames - 1, 1)
    col = (191, 106, 37) if active else (170, 170, 170)        # BGR: blue when the left hand is in control
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


class SpinMeter:
    """Globe rotation speed (°/s) over the last ~1 s - not between neighbouring frames,
    so that jitter of the orientation estimate (~0.3° per frame) does not look like rotation."""

    def __init__(self, window=1.0):
        self.window, self.hist, self.speed = window, deque(), 0.0

    def update(self, now, R):
        self.hist.append((now, np.array(R, float)))
        while len(self.hist) > 2 and now - self.hist[1][0] >= self.window:
            self.hist.popleft()
        t_old, R_old = self.hist[0]
        if now - t_old >= 0.4:
            a = np.degrees(np.arccos(np.clip((np.trace(R_old.T @ self.hist[-1][1]) - 1) / 2, -1, 1)))
            self.speed = a / (now - t_old)
        return self.speed


class Fixation:
    """Announcing a point: the finger must "stand still" for 3 s both on the screen and on the Earth.
    The countdown restarts if (1) the finger moved more than radius_px, (2) the point of the Earth under the finger
    moved more than radius_px corresponds to at the current scale (the map/globe was moved or
    turned under a still finger), or (3) the globe rotates faster than spin_limit °/s."""

    def __init__(self, dwell, radius_px, spin_limit=3.0):
        self.dwell, self.radius_px, self.spin_limit = dwell, radius_px, spin_limit
        self.reset()

    def reset(self):
        self.anchor, self.t0, self.fired, self.ll, self.tol = None, 0.0, False, None, None

    def update(self, now, px, ok, lat, lon, latlon_at, spin_speed=0.0):
        """-> share of the countdown 0..1. latlon_at(x, y) -> (lat, lon, ok) to convert the tolerance into km."""
        px = np.asarray(px, float)
        moved_geo = ok and self.ll is not None and self.tol is not None and \
            float(L.haversine_km(lat, lon, *self.ll)) > self.tol
        if (self.anchor is None or np.hypot(*(px - self.anchor)) > self.radius_px or moved_geo
                or spin_speed > self.spin_limit):
            self.anchor, self.t0, self.fired = px.copy(), now, False
            self.ll, self.tol = ((lat, lon) if ok else None), None
            if ok:
                la2, lo2, ok2 = latlon_at(px[0] + self.radius_px, px[1])
                if not ok2:
                    la2, lo2, ok2 = latlon_at(px[0] - self.radius_px, px[1])
                self.tol = max(float(L.haversine_km(lat, lon, la2, lo2)), 30.0) if ok2 else 300.0
        return min(1.0, (now - self.t0) / self.dwell)

    def should_fire(self, frac):
        """True exactly once, when the countdown is complete."""
        if frac >= 1.0 and not self.fired:
            self.fired = True
            return True
        return False


def run_tracking(cap, cal, specs, args, scale, globe=None):
    mp, landmarker = make_landmarker()
    frame = _read(cap)
    virtual = isinstance(globe, GB.VirtualGlobe)
    vmap = isinstance(globe, VM.VirtualMap)
    drawn = virtual or vmap                    # drawn surfaces: mirrored frame, no calibration
    tracker = make_tracker(args.track, cal, frame.shape) if globe is None else (globe if drawn else globe.tracker)
    keys = KEYS if globe is None else (KEYS_VMAP if vmap else KEYS_VIRTUAL if virtual else KEYS_GLOBE)
    surface = "map" if globe is None else ("vmap" if vmap else "virtual" if virtual else "globe")
    off_word = "finger is off the map" if globe is None or vmap else "finger is off the globe"
    zooming = False
    finger, time_finger = OneEuro(), OneEuro(min_cutoff=0.8)
    son = Sonifier(enabled=not args.no_sound, volume=args.volume)
    geo = SP.Geocoder()                                         # countries / seas / regions with English names
    voice = SP.Speaker(enabled=not args.no_voice, voice=args.voice, rate=args.voice_rate)
    last_spoken = ""
    if voice.enabled:
        print(f"Voice: {voice.voice}. Hold still for 3 s to hear the place, time and value (v - off, r - repeat).")
    view = L.LayerView()
    names = [s.name for s in specs]
    li = names.index(args.layer) if args.layer in names else 0
    layer = specs[li].load()
    t = layer.frame_at(None if args.year is None else args.year + 0.5)   # frame index of the layer
    playing, last_step = False, 0.0
    swap = args.swap_hands
    if drawn:              # mirrored frame: MediaPipe's hand labels are then correct, do not swap them
        swap = not swap

    print("=== TRACKING === Data layers:")
    for k, s in enumerate(specs, 1):
        print(f"   {k}. {s.title}" + ("   <- current" if k - 1 == li else ""))
    if globe is None:
        print("   RIGHT index finger: move over the map.")
        print("   LEFT index finger: time - slide along the map (left edge = first frame, right edge = last).")
    elif vmap:
        print("   Virtual map over the camera picture (mirrored, like a video call). The map does not move.")
        print("   RIGHT index finger: point at the map.")
        print("   LEFT index finger alone: time - slide along the map (left edge = first frame, right edge = last).")
        print("   LEFT thumb + index out like an 'L' over a place: SPREAD them to zoom in, BRING TOGETHER to zoom out")
        print("   (like on a phone screen); fold the thumb to keep the zoom; 0 = whole world.")
    elif virtual:
        print("   Virtual globe over the camera picture (mirrored, like a video call).")
        print("   RIGHT index finger: point at the globe.")
        print("   LEFT hand: pinch thumb and index finger and move to turn the globe; open the fingers to let go.")
        print("   Time - keys: , . (step), < > (10 steps), p (play).")
    else:
        print("   RIGHT index finger: point at the globe. You can spin, move and tilt the globe:")
        print(f"   it is found by its remembered points (memory: {len(getattr(tracker, 'memory', []))} points).")
        print("   Time - keys: , . (step), < > (10 steps), p (play).")
    print("   Keys (in the window):")
    for disp, _, what in keys:
        print(f"   {disp:>5s}  {what.replace('{n}', str(min(len(specs), 9)))}")
    print(f"   Hold the right finger still for {args.dwell:.0f} s -> announcement (also printed here).")
    if son.stream is not None:
        print("   Sound: pitch = layer value (pentatonic), beating and pulsing = how far above the baseline,")
        print("   left/right = longitude, noise = no data, chime = announcement.\n")

    cv2.namedWindow(WIN_CAM, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WIN_SST, cv2.WINDOW_NORMAL)
    placed = False
    result = "quit"
    smooth, last_seen = None, 0.0
    fix = Fixation(args.dwell, args.radius * max(frame.shape[1], frame.shape[0]))
    spin, spin_speed = SpinMeter(), 0.0        # globe rotation speed (°/s)

    def latlon_at(px, py):
        """Latitude/longitude under a pixel on the current surface (map or globe)."""
        if globe is None:
            uu, vv = apply_H(G, [[px, py]])[0]
            la, lo, okk = cal.sheet_to_latlon(uu, vv)
            return la, lo, okk and -0.02 <= uu <= 1.02 and -0.02 <= vv <= 1.02
        la, lo, hit_, fac_ = globe.to_latlon(px, py)
        return la, lo, hit_ and fac_ > 0.12
    tsmooth, time_seen = None, 0.0
    trail = deque(maxlen=40)
    last_fix = ""
    t0, last_ts = time.monotonic(), -1

    def switch(new_li):
        nonlocal li, layer, t
        li = new_li % len(specs)
        print(f"Layer {li + 1}/{len(specs)}: {specs[li].title} (loading...)")
        when = layer.time_of(t)                                # the same moment in time in the new layer
        layer = specs[li].load()
        t = layer.frame_at(when)
        trail.clear()
        intro = SP.layer_intro(li + 1, len(specs), layer, surface)
        print(f"   🔊 {intro}")
        voice.speak(intro)

    def announce(text):
        nonlocal last_spoken
        if voice.speaking() and last_spoken == text:            # pressing again while speaking = stop
            voice.stop()
            print("🔇 stopped")
            return
        last_spoken = text
        print(f"🔊 {text}")
        voice.speak(text)
        if not voice.enabled:
            print("   (voice is off: press v to turn it on)")

    if voice.enabled:
        voice.speak(f"Earth Jukebox is ready. Map {li + 1} of {len(specs)}: {SP.spoken_title(layer)}. "
                    "Press the question mark key for help, or the I key to hear the list of maps.")

    while True:
        frame = _read(cap)
        if drawn:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.monotonic()
        ts = max(int((now - t0) * 1000), last_ts + 1); last_ts = ts

        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        roles = hand_roles(res, swap) if res.hand_landmarks else {}
        if globe is None:
            corners = tracker.update(frame)
            G = cam_to_sheet_H(corners)
        elif vmap:                                             # left thumb+index spread = zoom; map drawn first,
            zooming = globe.handle_left(roles.get("time"), w, h, now)   # finger marks go on top of it
            G = globe.sheet_H()
            globe.draw(frame, layer, t)
        elif virtual:                                          # left hand: pinch and move rotate the globe
            globe.handle_left(roles.get("time"), w, h, now)
        else:                                                  # globe rotation; the hand is not compared with the land map
            tracker.update(frame, hands_mask(res, w, h) if res.hand_landmarks else None)

        # ---- right hand: point on the map ----
        if "map" in roles:
            lm = roles["map"][INDEX_TIP]
            smooth = finger(np.array([lm.x * w, lm.y * h]), now).copy()
            last_seen = now
        elif now - last_seen > 0.4:  # hand lost
            smooth = None
            fix.reset()
            finger.reset()

        # ---- left hand: time ----
        time_hand = False
        if "time" in roles and (globe is None or (vmap and not zooming)):
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
                "zoom": globe.zoom if vmap else None,
                "globe": None if globe is None or drawn else {"mode": getattr(tracker, "mode", "spin"),
                                                     "inliers": getattr(tracker, "inliers", 0),
                                                     "coverage": tracker.coverage() if hasattr(tracker, "coverage") else 0}}
        lat = lon = None
        frac = 0.0
        # globe rotation speed: while it is being turned, the point under the finger changes - no announcement
        # (measured over ~1 s, not between neighbouring frames: jitter of the orientation estimate then does not add up)
        if globe is not None and not vmap:
            spin_speed = spin.update(now, globe.R if virtual else globe.model.R())
        if smooth is not None:
            if globe is None:
                u, v = apply_H(G, [smooth])[0]
                on_sheet = -0.02 <= u <= 1.02 and -0.02 <= v <= 1.02
                lat, lon, ok = cal.sheet_to_latlon(u, v)
                ok = ok and on_sheet
            else:
                lat, lon, hit, facing = globe.to_latlon(*smooth)
                ok = hit and facing > 0.12                     # the very edge of the sphere - unreliable
                info["edge"] = ok and facing < 0.35
            frac = fix.update(now, smooth, ok, lat, lon, latlon_at, spin_speed)
            info["dwell"] = frac
            if ok:
                info.update(state="ok", pos=(lat, lon))
                trail.append((lat, lon))
            else:
                info["state"] = "offmap"
                lat = lon = None
            col = (12, 163, 12) if fix.fired else (52, 104, 235)
            c = (int(smooth[0]), int(smooth[1]))
            cv2.circle(frame, c, 7, col, -1)
            cv2.ellipse(frame, c, (24, 24), -90, 0, 360 * frac, col, 3)
            cv2.putText(frame, "MAP", (c[0] + 26, c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

        s = layer.sample(lat, lon, t) if lat is not None else None
        info["sample"] = s

        if smooth is not None and fix.should_fire(frac):
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

        # ---- sound ----
        if s is None:
            son.silence()
        elif s["missing"]:
            son.missing(lon)
        else:
            son.play(s["norm"], s["tension"], lon, quiet=bool(s["near_km"]))
        info["note"] = None if son.note is None else f"{midi_name(son.note)} · {midi_to_hz(son.note):.0f} Hz"
        info["muted"] = son.muted or son.stream is None

        if globe is None:
            qcol = {"tracked": (0, 200, 0), "held": (0, 200, 255), "searching": (0, 0, 255), "static": (200, 200, 200)}
            cv2.polylines(frame, [np.int32(corners)], True, qcol[tracker.status], 2)
            for p in corners:
                cv2.circle(frame, (int(p[0]), int(p[1])), 6, qcol[tracker.status], -1)
            if layer.dynamic:
                draw_timeline(frame, G, layer, t, time_hand)
        elif vmap:                                             # the map itself is already drawn above
            if layer.dynamic:
                draw_timeline(frame, G, layer, t, time_hand)
        elif virtual:                                          # sphere with the layer data, coastlines, pinch
            globe.draw(frame, layer, t)
        else:                                                  # outline, grid and coastlines: shows whether it matches
            globe.overlay.draw(frame, tracker.status)

        cam = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imshow(WIN_CAM, draw_header(cam, layer, t))       # only the map name and the year on top
        ph = cam.shape[0]
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
        if key in (ord("b"), 8, 127, 27):                      # b / Backspace / Esc: back to the start menu
            result = "menu"
            print("Back to the start menu")
            break
        if key == ord("m"):
            print("Sound off" if son.toggle_mute() else "Sound on")
        if key == ord("v"):
            if voice.enabled:
                voice.stop()
                voice.enabled = False
                print("Voice off")
            elif not args.no_voice or voice.voice:
                voice = SP.Speaker(enabled=True, voice=args.voice, rate=args.voice_rate)
                print("Voice on" if voice.enabled else "Voice not available")
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
            print("Hands swapped (by MediaPipe labels).")
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
                print("Playback: " + ("▶" if playing else "paused"))
        if virtual and key == ord("n"):
            globe.north_up()
        if virtual and key in (ord("+"), ord("=")):
            globe.resize(1.1)
        if virtual and key in (ord("-"), ord("_")):
            globe.resize(1 / 1.1)
        if vmap and key in (ord("+"), ord("=")):
            globe.zoom_center(1.5)
        if vmap and key in (ord("-"), ord("_")):
            globe.zoom_center(1 / 1.5)
        if vmap and key == ord("0"):
            globe.reset()
        if drawn and key in (ord("c"), ord("k")):
            pass                                               # drawn surfaces need no calibration
        elif key == ord("c") and globe is not None:
            son.silence()
            globe = run_globe_calibration(cap, args, scale)
            tracker = globe.tracker
        elif key == ord("c"):
            son.silence()
            cal = run_calibration(cap, args, scale)
            tracker = make_tracker(args.track, cal, frame.shape)
        if key == ord("k") and globe is not None and not drawn:
            # the lighting changed -> relearn the "water/land" colours at the current rotation
            hm = hands_mask(res, w, h) if res.hand_landmarks else None
            tracker.learn_colors(frame, None, hand_mask=hm)
            if hasattr(tracker, "memory"):
                tracker.memory.add_from_frame(frame, globe.model, tracker.orb, hm, min_facing=0.6)
            GB.save_globe(GB.GLOBE_CALIB, globe.model, tracker)
            print("Globe colours relearned, current side remembered")
        elif key == ord("k") and not drawn:
            # the same 4 corners again (e.g. the lighting changed) -> new template, the place calibration is kept
            son.silence()
            corners = pick_corners(cap, scale)
            cal.template = capture_template(cap, corners)
            cv2.destroyWindow("Calibration")
            cal.ref_corners = corners
            cal.save(CALIB_FILE)
            tracker = make_tracker(args.track, cal, frame.shape)
            tracker.locked = True
    voice.stop()                                               # the menu announces itself
    son.close()
    landmarker.close()
    if globe is not None and not drawn:
        GB.save_globe(GB.GLOBE_CALIB, globe.model, tracker)       # last pose + newly remembered sides
    for win in (WIN_CAM, WIN_SST):
        try:
            cv2.destroyWindow(win)
        except cv2.error:
            pass
    cv2.waitKey(1)
    return result


# ----------------------------------------------------------------------------
# Start menu: choose the surface (virtual map / paper map / physical globe / virtual globe)
# ----------------------------------------------------------------------------
WIN_MENU = "Earth Jukebox"
MENU_W, MENU_H = 560, 400
# (surface, title on screen, what it needs - only for the voice) - in the order shown and spoken
MENU = [
    ("vmap", "Virtual map", "just the camera"),
    ("map", "Paper map", "a printed world map on the table"),
    ("globe", "Physical globe", "a globe on a stand with blue oceans"),
    ("virtual", "Virtual globe", "just the camera"),
]


def _menu_rows():
    """Row rectangles (x0, y0, x1, y1) of the four options."""
    top, row, pad = 40, 80, 30
    return [(pad, top + k * row, MENU_W - pad, top + k * row + row - 10) for k in range(len(MENU))]


def render_menu(selected):
    """The menu picture (BGR): only the numbers and the option names. selected=None - nothing highlighted
    (a row is tinted only after the arrow keys were used)."""
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager
    bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    font = ImageFont.truetype(bold, 34)
    img = Image.new("RGB", (MENU_W, MENU_H), L.hex_rgb(SURFACE))
    d = ImageDraw.Draw(img)
    for k, ((_, title, _), (x0, y0, x1, y1)) in enumerate(zip(MENU, _menu_rows())):
        if k == selected:
            d.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=L.hex_rgb("#dbe8f8"))
        d.text((x0 + 22, y0 + 16), f"{k + 1}", font=font, fill=L.hex_rgb("#256abf" if k == selected else INK2))
        d.text((x0 + 80, y0 + 16), title, font=font, fill=L.hex_rgb(INK))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def menu_hit(x, y):
    """Which option is at (x, y) in the menu picture; None if none."""
    for k, (x0, y0, x1, y1) in enumerate(_menu_rows()):
        if x0 <= x <= x1 and y0 <= y <= y1:
            return k
    return None


def menu_announcement():
    parts = ["Earth Jukebox, start menu. Choose how you want to explore the Earth."]
    for k, (_, title, needs) in enumerate(MENU, 1):
        parts.append(f"Press {k} for {title.lower()}, {needs}.")
    parts.append("Use the arrow keys and Enter, or press Q to quit.")
    return " ".join(parts)


# arrow keys as returned by cv2.waitKeyEx (macOS, Windows, Linux)
_UP = {63232, 2490368, 65362, 63234, 2424832, 65361}      # up or left
_DOWN = {63233, 2621440, 65364, 63235, 2555904, 65363}    # down or right


def choose_surface(args):
    """Show the start menu, return the chosen surface ("vmap", "map", "globe", "virtual") or None to quit."""
    voice = SP.Speaker(enabled=not args.no_voice, voice=args.voice, rate=args.voice_rate)
    voice.speak(menu_announcement())
    print("=== START MENU === " + "  ".join(f"{k}. {m[1]}" for k, m in enumerate(MENU, 1)) + "   (q - quit)")
    selected, clicked = None, []                                # nothing highlighted until the arrows are used
    cv2.namedWindow(WIN_MENU, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN_MENU, lambda ev, x, y, *_: clicked.append(menu_hit(x, y))
                         if ev == cv2.EVENT_LBUTTONDOWN else None)
    try:
        while True:
            cv2.imshow(WIN_MENU, render_menu(selected))
            key = cv2.waitKeyEx(30)
            choice = None
            if clicked:
                choice = clicked.pop()
            elif key in (ord("q"), 27):
                voice.stop()
                return None
            elif 0 <= key < 256 and ord("1") <= key <= ord("0") + len(MENU):
                choice = key - ord("1")
            elif key in (13, 10, 32) and selected is not None:
                choice = selected
            elif key in _UP or key in _DOWN:
                if selected is None:
                    selected = 0 if key in _DOWN else len(MENU) - 1
                else:
                    selected = (selected + (1 if key in _DOWN else -1)) % len(MENU)
                voice.speak(f"{selected + 1}, {MENU[selected][1]}.")       # say the highlighted option
            if choice is not None:
                surf, title = MENU[choice][0], MENU[choice][1]
                print(f"Mode: {title}")
                voice.speak(f"{title}.")
                while voice.speaking():                                   # let it finish before the camera starts
                    cv2.waitKey(20)
                return surf
    finally:
        cv2.destroyWindow(WIN_MENU)
        cv2.waitKey(1)


# ----------------------------------------------------------------------------
# Self-test without a camera
# ----------------------------------------------------------------------------
def _synthetic_map_texture(w=900, h=450, seed=11):
    """Map-like texture for the test: outlined 'continents', labels, grid."""
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
    # The sheet in a 1280x720 frame during calibration, and after it has been moved and rotated by 20°
    cal_corners = np.array([[250, 120], [1050, 140], [1080, 620], [220, 600]], float)
    c, (hw, hh), a = np.array([700, 380]), (330, 220), np.radians(20)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    moved_corners = (np.array([[-hw, -hh], [hw, -hh], [hw * 1.03, hh], [-hw * 0.97, hh]]) @ R.T) + c
    def scene(corners_px, hand=False):
        """Synthetic frame: noisy table, light sheet with 'continents', optionally a hand across the edge."""
        g = np.random.default_rng(7)  # the same map "picture" for both positions
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
    print("1) Contour mode (sheet edges), synthetic frames:")
    print(f"   calibration position: {t1.status}, max corner error {e1:.2f} px")
    print(f"   moved+rotated by 20°: {t2.status}, max corner error {e2:.2f} px  {'OK' if ok_all else 'FAIL'}")
    hand_img, _ = scene(moved_corners, hand=True)
    before = t2.corners.copy()
    t2.update(hand_img)
    drift = np.linalg.norm(t2.corners - before, axis=1).max()
    print(f"   hand over the edge: {t2.status}, corner shift {drift:.2f} px  {'OK' if drift < 3 else 'FAIL'}")
    ok_all &= drift < 3

    # ---- features mode: virtual corners (clicks) + tracking by the map picture ----
    tex = _synthetic_map_texture()
    th_, tw_ = tex.shape[:2]

    def scene_tex(corners_px, hand=False, seed=3, sensor_noise=0.0):
        g = np.random.default_rng(seed)
        Hc = cv2.getPerspectiveTransform(np.float32(UNIT), np.float32(corners_px))
        Ht = Hc @ np.diag([1 / tw_, 1 / th_, 1.0])  # texture pixels -> unit square -> frame
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

    print("\n2) Features mode (by the map picture; the table is as light as the paper, so the edges do NOT help):")
    f1, fH1 = scene_tex(cal_corners)
    clicked = cal_corners + rng.normal(0, 1.5, (4, 2))      # virtual corners: mouse clicks ±1.5 px
    ft = FeatureTracker(make_template(f1, clicked), clicked, f1.shape)
    f2, fH2 = scene_tex(moved_corners, hand=True)
    truth2 = apply_H(fH2 @ np.linalg.inv(fH1), clicked)     # where the same corners must be after the move
    fc2 = ft.update(f2).copy()
    ef = np.linalg.norm(fc2 - truth2, axis=1).max()
    ok_f = ft.status == "tracked" and ef < 3
    print(f"   moved+rotated by 20° + hand over the map: {ft.status}, {ft.inliers} points, "
          f"max corner error {ef:.2f} px  {'OK' if ok_f else 'FAIL'}")
    jit = []
    for k in range(8):  # map does not move, but every frame has new sensor noise
        fk, _ = scene_tex(moved_corners, hand=True, seed=100 + k, sensor_noise=6)
        jit.append(ft.update(fk).copy())
    jitter = float(np.linalg.norm(np.array(jit) - np.mean(jit, 0), axis=2).max())
    ok_j = jitter < 1.0
    print(f"   corner jitter on a still map (8 noisy frames): max {jitter:.2f} px  {'OK' if ok_j else 'FAIL'}")
    t0_ = time.perf_counter()
    for _ in range(5):
        ft.match(f2)
    print(f"   speed: {(time.perf_counter() - t0_) / 5 * 1000:.0f} ms/frame")
    ok_all &= ok_f and ok_j
    det1, det2, Hc1, Hc2 = clicked, fc2, fH1, fH2

    print("\n3) Calibration -> map moved/rotated -> coordinates (features mode, clicks ±1.5 px):")
    for proj in PROJECTIONS:
        for lon0 in [0.0, 150.0]:
            xmax = float(fwd(proj, 0, 180)[0])
            ymax = float(fwd(proj, 85 if proj in ("mercator", "miller") else 90, 0)[1])
            m = 0.05  # white margins of the sheet

            def to_sheet(lat, lon):
                x, y = fwd(proj, lat, wrap180(np.asarray(lon) - lon0))
                return np.stack([m + (1 - 2 * m) * (x + xmax) / (2 * xmax),
                                 m + (1 - 2 * m) * (ymax - y) / (2 * ymax)], -1)
            # calibration: virtual corners, place clicks with ~1.5 px inaccuracy
            cam_land = apply_H(Hc1, to_sheet(geo[:, 0], geo[:, 1])) + rng.normal(0, 1.5, (len(geo), 2))
            cal = fit_calibration(apply_H(cam_to_sheet_H(det1), cam_land), geo, verbose=False)
            # the map was moved: corners from the picture tracker, finger at 300 random points
            G2 = cam_to_sheet_H(det2)
            tl, tn = rng.uniform(-70, 70, 300), rng.uniform(-180, 180, 300)
            sheet = apply_H(G2, apply_H(Hc2, to_sheet(tl, tn)))
            errs = [haversine_km(*cal.sheet_to_latlon(*s)[:2], a_, b_) for s, a_, b_ in zip(sheet, tl, tn)]
            good = cal.proj == proj and np.median(errs) < 150
            ok_all &= good
            print(f"   {proj:9s} centre {lon0:+4.0f}° -> {cal.proj:9s} {'OK ' if good else 'FAIL'} | "
                  f"error: median {np.median(errs):4.0f} km, 95% {np.percentile(errs, 95):4.0f} km")
    hand = f2
    tracker = ft

    out = Path(args.selftest_out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- data layers ----
    specs = L.discover_layers(args.month)
    by = {s.name: s.load() for s in specs}
    print(f"\n4) Data layers ({len(specs)}), values near Odesa (46.5°N, 30.7°E):")
    for s in specs:
        lay = by[s.name]
        k_ = lay.default_frame()
        smp = lay.sample(46.48, 30.73, k_)
        frames_ = f"{lay.n_frames} frames" if lay.dynamic else "1 moment"
        txt = "no data nearby" if smp["missing"] else f"{smp['text']}  (pitch {smp['norm']:.2f}, tension {smp['tension']:+.2f})"
        print(f"   {s.name:20s} {frames_:>10s}, {lay.time_label(k_)}: {txt}")
    if "sst" in by:                       # compare with the NetCDF directly
        import xarray as xr
        ds = xr.open_dataset(L.SST_FILE, engine="h5netcdf")
        raw = float(ds.sst.sel(lat=46.48, lon=30.73, method="nearest").isel(time=-1)) if args.month is None else None
        ds.close()
        if raw is not None:
            ok = abs(by["sst"].sample(46.48, 30.73)["value"] - raw) < 1e-3
            ok_all &= ok
            print(f"   sst == NetCDF directly ({raw:.2f}): {'OK' if ok else 'FAIL'}")
    if "temperature" in by:               # compare the compressed .u8.bin with the full .json
        d = json.loads((L.DATA_DIR / "temperature" / "temperature.json").read_text())
        yi = d["years"].index(2025)
        i = int(np.argmin(np.abs(np.array(d["lat"]) - 46.48))); j = int(np.argmin(np.abs(np.array(d["lon"]) - 30.73)))
        raw = d["data"][yi][i][j]
        got = by["temperature"].sample(46.48, 30.73, by["temperature"].frame_at(2025.5))["value"]
        ok = abs(got - raw) < 0.011
        ok_all &= ok
        print(f"   temperature 2025 == temperature.json ({raw:+.3f} vs {got:+.3f}): {'OK' if ok else 'FAIL'}")
    for name_ in ("ocean_sst_video", "ocean_flow_activity"):   # dated frames: compare with the full .json
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
        print(f"   {name_}: {lay.n_frames} frames {lay.labels[0]}…{lay.labels[-1]}, Gulf Stream frame 1/{lay.n_frames}: "
              f"{lay.sample(35.0, -70.0, 0)['text']}, max error vs json {max(errs):.3f} (step {step_:.3f}) "
              f"{'OK' if ok else 'FAIL'}")
    if "temperature" in by and "ocean_sst_video" in by:      # switching layers keeps the moment in time
        te, ov = by["temperature"], by["ocean_sst_video"]
        k_ = ov.frame_at(te.time_of(te.frame_at(2007.5)))
        ok = int(ov.times[k_]) == 2007 and ov.frame_at(te.time_of(te.frame_at(1950.5))) == 0
        ok_all &= ok
        print(f"   temperature 2007 -> ocean_sst_video: frame {ov.labels[k_]}; 1950 -> nearest {ov.labels[0]} "
              f"{'OK' if ok else 'FAIL'}")
    ok = "co2_curve" not in by
    ok_all &= ok
    print(f"   maps only (curves such as co2_curve are skipped): {sorted(by)} {'OK' if ok else 'FAIL'}")

    # ---- two hands: right = map, left = time ----
    print("\n4b) Two hands:")
    from types import SimpleNamespace as NS  # noqa: F811

    def fake(*hands):   # hands: (MediaPipe label, wrist x)
        lms = [[NS(x=x, y=0.5)] + [NS(x=x, y=0.5)] * 20 for _, x in hands]
        hd = [[NS(category_name=lab, score=0.9)] for lab, _ in hands]
        return NS(hand_landmarks=lms, handedness=hd)
    r1 = hand_roles(fake(("Left", 0.7), ("Right", 0.2)))        # frame not mirrored -> labels swapped
    r2 = hand_roles(fake(("Left", 0.7), ("Right", 0.2)), swap=True)
    r3 = hand_roles(fake(("Left", 0.2), ("Left", 0.7)))         # same labels -> by position
    ok_h = (r1["map"][0].x == 0.7 and r1["time"][0].x == 0.2 and r2["map"][0].x == 0.2
            and r3["time"][0].x == 0.2 and r3["map"][0].x == 0.7)
    print(f"   role assignment (MediaPipe labels, the h key, same labels): {'OK' if ok_h else 'FAIL'}")
    for tl in [l_ for l_ in by.values() if l_.dynamic]:
        last = tl.n_frames - 1
        ks = [frame_from_u(tl, u_, None) for u_ in (0.0, 0.5, 1.0, -0.3, 1.4)]
        mid = round(0.5 * last)
        hyst = frame_from_u(tl, (mid + 0.4) / last, mid)        # shift by 0.4 frame -> the frame does not change
        moved = frame_from_u(tl, (mid + 1.0) / last, mid)       # shift by 1 frame -> it changes
        ok_y = ks == [0, mid, last, 0, last] and hyst == mid and moved == mid + 1
        print(f"   {tl.name:20s} left finger -> frame: edge {tl.labels[ks[0]]}, middle {tl.labels[ks[1]]}, "
              f"edge {tl.labels[ks[2]]} (all {tl.n_frames}); hysteresis  {'OK' if ok_y else 'FAIL'}")
        tl_img = f2.copy()
        draw_timeline(tl_img, cam_to_sheet_H(ft.corners), tl, mid, True)
        cv2.imwrite(str(out / f"selftest_timeline_{tl.name}.png"), tl_img)
        ok_h &= ok_y
    ok_all &= ok_h

    print("\n5) New layer from NetCDF (the layers.py from-netcdf converter), checked on sst.mnmean.nc:")
    if L.SST_FILE.exists():
        import xarray as xr
        folder = L.netcdf_to_layer(L.SST_FILE, "sst", "sst_annual_test", out / "layers", title="SST, yearly mean")
        lay = L.GridLayer.from_folder(folder)
        ds = xr.open_dataset(L.SST_FILE, engine="h5netcdf")
        ann = ds.sst.sel(time=slice("2020", "2020")).mean("time").sel(lat=-30.0, lon=200.0, method="nearest")
        raw = float(ann); ds.close()
        got = lay.sample(-30.0, -160.0, lay.frame_at(2020.5))["value"]
        step = (lay.vmax - lay.vmin) / 254
        ok = abs(got - raw) <= step
        ok_all &= ok
        print(f"   created {folder.name}/ ({lay.n_frames} frames {lay.labels[0]}–{lay.labels[-1]}); "
              f"2020, 30°S 160°W: {got:.2f} vs NetCDF {raw:.2f} (step {step:.3f}) {'OK' if ok else 'FAIL'}")
        found = [s.name for s in L.discover_layers(data_dir=out / "layers", sst_file=out / "none.nc")]
        ok = found == ["sst_annual_test"]
        ok_all &= ok
        print(f"   discovery sees the new layer: {found} {'OK' if ok else 'FAIL'}")

    # ---- sound ----
    print("\n5b) Announcing a point (geography + text + voice):")
    geo = SP.Geocoder()
    expect = [((9.0, 39.5), "Ethiopia"), ((50.45, 30.52), "Ukraine"), ((45.8, 31.0), "Black Sea"),
              ((-28.0, 30.5), "South Africa"), ((0.0, -140.0), "Pacific Ocean"),
              ((28.0, 85.0), "Himalayas")]
    ok_g = True
    for (la, lo), want in expect:
        ph = SP.place_phrase(geo.locate(la, lo))
        good = want in ph
        ok_g &= good
        print(f"   {la:+6.1f},{lo:+7.1f} -> {ph}  {'OK' if good else 'FAIL'}")
    plur = [SP.spoken_quantity(v, "°C", 1) for v in (1.0, 2.0, 5.0, 21.0, 22.4)]
    ok_p = plur == ["1 degree", "2 degrees", "5 degrees", "21 degrees", "22.4 degrees"]
    print(f"   units: {plur} {'OK' if ok_p else 'FAIL'}")
    msgs = []
    if "temperature" in by:
        te = by["temperature"]; k_ = te.frame_at(2025.5)
        msgs.append(SP.compose(te, k_, 9.0, 39.5, te.sample(9.0, 39.5, k_), geo.locate(9.0, 39.5)))
    if "sst" in by:
        msgs.append(SP.compose(by["sst"], 0, 45.8, 31.0, by["sst"].sample(45.8, 31.0), geo.locate(45.8, 31.0)))
    if "ocean_sst_video" in by:   # ocean layer in the middle of land: honestly "no data", not ocean 600 km away
        ov = by["ocean_sst_video"]
        m_ = SP.compose(ov, 0, 9.0, 39.5, ov.sample(9.0, 39.5, 0), geo.locate(9.0, 39.5))
        ok_g &= "no data at this spot" in m_
        msgs.append(m_)
    for m_ in msgs:
        print(f"   🔊 {m_}")
    ann = SP.layers_announcement(specs, 1)
    hlp = SP.help_announcement(KEYS, len(specs))
    intro = SP.layer_intro(2, len(specs), by[specs[1].name]) if len(specs) > 1 else ""
    ok_a = f"There are {len(specs)} maps" in ann and all(SP.spoken_title(by[s.name]) in ann for s in specs)
    nn = str(min(len(specs), 9))
    ok_k = all(sp_.replace("{n}", nn).lower() in hlp.lower() for _, sp_, _ in KEYS) and "{n}" not in hlp
    print(f"   🔊 I key, list of maps: {ann}  {'OK' if ok_a else 'FAIL'}")
    print(f"   🔊 switching maps: {intro}")
    print(f"   🔊 ? key, help ({len(hlp)} characters, all {len(KEYS)} keys named {'OK' if ok_k else 'FAIL'}):")
    print("      " + hlp)
    ok_g &= ok_a and ok_k
    sp = SP.Speaker(enabled=True)
    if sp.enabled:
        import subprocess
        for fname, txt in (("layers_announcement.aiff", ann), ("help_announcement.aiff", hlp)):
            if sp.to_file(txt, out / fname):
                info_ = subprocess.run(["afinfo", str(out / fname)], capture_output=True, text=True).stdout
                dur = re.search(r"estimated duration: ([\d.]+)", info_)
                print(f"   wrote {fname}: {float(dur.group(1)) if dur else 0:.0f} s of speech")
    if sp.enabled and msgs:
        aiff = out / "speech_test.aiff"
        ok_v = sp.to_file(msgs[0], aiff) and aiff.stat().st_size > 10000
        print(f"   voice {sp.voice}: wrote {aiff.name} ({aiff.stat().st_size // 1024} KB) {'OK' if ok_v else 'FAIL'}")
        ok_g &= ok_v
    else:
        print("   no voice on this machine (recording test skipped)")
    s_ = Sonifier(enabled=False)
    s_.play(0.5, 0.0, 0.0); s_.render(Sonifier.SR // 2)
    loud = np.abs(s_.render(Sonifier.SR // 4)).max()
    s_.set_duck(True); s_.render(Sonifier.SR // 2)
    quiet = np.abs(s_.render(Sonifier.SR // 4)).max()
    ok_d = quiet < 0.3 * loud
    print(f"   synthesizer ducking while the voice speaks: {loud:.2f} -> {quiet:.2f} {'OK' if ok_d else 'FAIL'}")
    ok_all &= ok_g and ok_p and ok_d

    print("\n5c) Globe (synthetic camera frame):")
    Wg, Hg = 1280, 720

    def look(lat0, lon0, tilt):                                 # orientation: the point (lat0, lon0) faces the camera
        fwd = GB.latlon_to_vec(lat0, lon0); north = np.array([0, 0, 1.0])
        up = north - fwd * (north @ fwd); up /= np.linalg.norm(up); right = np.cross(up, fwd)
        a = np.radians(tilt)
        return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]) @ np.stack([right, -up, -fwd])
    geom_t = GB.GlobeGeometry((660, 370), 260, GB.fov_to_f(62, Wg), (Wg, Hg))
    R0_t = look(25, 15, 8)
    g0 = GB.render_synthetic(geom_t, R0_t, (Hg, Wg, 3))
    u_, v_, r_ = GB.detect_globe_circle(g0)
    ok_c = np.hypot(u_ - 660, v_ - 370) < 4 and abs(r_ - 260) < 4
    print(f"   outline: centre ({u_:.0f},{v_:.0f}) r={r_:.0f} (truth 660,370 r=260) {'OK' if ok_c else 'FAIL'}")
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
    print(f"   calibration: {len(clicks_)} visible places, field of view {fov_g:.0f}° (truth 62°), "
          f"orientation error {ang_g:.2f}° {'OK' if ok_o else 'FAIL'}")
    gs = GlobeSurface(GB.GlobeModel(geom_g, R0_g), None)
    gs.tracker = GB.SpinTracker(gs.model)
    gs.tracker.learn_colors(g0, 0.0)
    errs_th, errs_km, times_ = [], [], []
    for th_t in (0, 37, 120, -75, 181):
        Rt = R0_t @ GB.rot_z(th_t)
        img_g = GB.render_synthetic(geom_t, Rt, (Hg, Wg, 3), seed=abs(th_t) % 97)
        # fake MediaPipe hand over the globe: the mask must cut it out
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
    ok_s = max(errs_th) < 2.0                                  # less than one data cell (2°)
    ok_f = np.median(errs_km) < 60
    print(f"   globe rotation 0/37/120/-75/181° with a hand: max error {max(errs_th):.2f}°, "
          f"{np.mean(times_):.0f} ms/frame {'OK' if ok_s else 'FAIL'}")
    print(f"   finger -> coordinates: median {np.median(errs_km):.0f} km, 95% {np.percentile(errs_km, 95):.0f} km "
          f"({len(errs_km)} points) {'OK' if ok_f else 'FAIL'}")
    # --- remember the globe (tour) and find it anywhere: moved, tilted, from any side
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
    print(f"   tour (one full revolution in 10° steps): learned {cov_ * 100:.0f}%, memory {len(gtr.memory)} points "
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
    print(f"   globe moved, brought closer, tilted by 12°: recognized from scratch {found}/12 sides, "
          f"median orientation error {np.median(errs_o):.2f}°, {np.mean(times_o):.0f} ms/frame {'OK' if ok_m else 'FAIL'}")
    ov2 = fr_.copy(); GB.GlobeOverlay(mdl).draw(ov2, gtr.status); cv2.imwrite(str(out / "selftest_globe_moved.png"), ov2)
    ok_all &= ok_t and ok_m

    hg = SP.help_announcement(KEYS_GLOBE, len(specs), "globe")
    ok_hg = "globe" in hg and "left index finger" not in hg and "comma and period" in hg
    print(f"   globe help: about the globe, no left hand, time on the keys {'OK' if ok_hg else 'FAIL'}")
    ok_all &= ok_c and ok_o and ok_s and ok_f and ok_hg

    print("\n5d) Virtual globe (over the camera picture, rotated with a left-hand pinch):")
    vg = GB.VirtualGlobe((1280, 720))
    c_ = vg.to_latlon(vg.u0, vg.v0)
    up_ = vg.to_latlon(vg.u0, vg.v0 - vg.r * 0.5)
    rt_ = vg.to_latlon(vg.u0 + vg.r * 0.5, vg.v0)
    ok_v = abs(c_[0] - 20) < 1e-6 and abs(c_[1] - 15) < 1e-6 and up_[0] > c_[0] and rt_[1] > c_[1]
    print(f"   centre {c_[0]:.0f}°N {c_[1]:.0f}°E; above -> further north, right -> further east {'OK' if ok_v else 'FAIL'}")
    left_ = vg.to_latlon(vg.u0 - vg.r * 0.5, vg.v0)
    vg.rotate_drag(vg.r * np.arcsin(0.5), 0)
    c2_ = vg.to_latlon(vg.u0, vg.v0)
    ok_d = abs(c2_[0] - left_[0]) < 0.1 and abs(c2_[1] - left_[1]) < 0.1
    print(f"   dragging right: what was on the left is now in the centre {'OK' if ok_d else 'FAIL'}")
    vg.north_up()

    def vhand(pinch, cx):
        pts = [NS(x=cx / 1280, y=0.7) for _ in range(21)]
        pts[0] = NS(x=cx / 1280, y=0.85); pts[9] = NS(x=cx / 1280, y=0.70)
        pts[4] = NS(x=(cx - (5 if pinch else 60)) / 1280, y=0.6); pts[8] = NS(x=(cx + (5 if pinch else 60)) / 1280, y=0.6)
        return pts
    l0 = vg.to_latlon(vg.u0, vg.v0)[1]
    for k_ in range(10):
        vg.handle_left(vhand(False, 300 + 20 * k_), 1280, 720, k_ * 0.033)   # hand without a pinch: no rotation
    l1 = vg.to_latlon(vg.u0, vg.v0)[1]
    for k_ in range(10):
        vg.handle_left(vhand(True, 300 + 20 * k_), 1280, 720, 1 + k_ * 0.033)
    l2 = vg.to_latlon(vg.u0, vg.v0)[1]
    vg.handle_left(vhand(False, 500), 1280, 720, 1.4)
    for k_ in range(5):
        vg.handle_left(None, 1280, 720, 1.45 + k_ * 0.05)
    l3 = vg.to_latlon(vg.u0, vg.v0)[1]
    ok_p = abs(l1 - l0) < 1e-6 and l2 < l1 - 20 and l3 < l2 - 5
    print(f"   no rotation without a pinch; pinch and move -> {l1:.0f}° -> {l2:.0f}°; inertia -> {l3:.0f}° {'OK' if ok_p else 'FAIL'}")
    two = NS(hand_landmarks=[[NS(x=x_, y=0.5)] * 21 for x_ in (0.2, 0.7)],
             handedness=[[NS(category_name=c_, score=0.9)] for c_ in ("Left", "Right")])
    r_m = hand_roles(two, swap=True)                            # mirrored frame: labels are not swapped
    ok_r = r_m["time"][0].x == 0.2 and r_m["map"][0].x == 0.7
    print(f"   mirrored frame: left hand = rotate, right hand = point {'OK' if ok_r else 'FAIL'}")
    tl_ = next((l_ for l_ in by.values() if l_.dynamic), next(iter(by.values())))
    fr_v = np.full((720, 1280, 3), 110, np.uint8)
    t0_ = time.perf_counter()
    for _ in range(5):
        f_ = fr_v.copy(); vg.draw(f_, tl_, tl_.default_frame())
    dt_v = (time.perf_counter() - t0_) / 5 * 1000
    cv2.imwrite(str(out / "selftest_virtual_globe.png"), f_)
    hv = SP.help_announcement(KEYS_VIRTUAL, len(specs), "virtual")
    ok_hv = "thumb and index finger" in hv and "The N key" in hv
    print(f"   drawing {dt_v:.0f} ms/frame; help mentions the pinch and the n key {'OK' if ok_hv else 'FAIL'}")
    ok_all &= ok_v and ok_d and ok_p and ok_r and ok_hv

    print("\n5e) No announcement while the globe is being turned under a still finger:")
    vgf = GB.VirtualGlobe((1280, 720))
    px_ = np.array([vgf.u0 + 30, vgf.v0 - 20])
    latlon_v = lambda x, y: (lambda r: (r[0], r[1], r[2]))(vgf.to_latlon(x, y))

    def run_fix(frames, rotate, jitter=0.0, seed=0):
        """frames frames of 1/30 s each; rotate(k) -> degrees of rotation per frame. -> when it fired"""
        rg = np.random.default_rng(seed)
        fx, sm = Fixation(3.0, 0.02 * 1280), SpinMeter()
        fires = []
        for k in range(frames):
            now = k / 30
            vgf.rotate_drag(vgf.r * np.radians(rotate(k)), 0)
            R_seen = vgf.R @ GB._rodrigues(rg.normal(0, np.radians(jitter), 3)) if jitter else vgf.R
            sp = sm.update(now, R_seen)
            la, lo, okk, _ = vgf.to_latlon(*px_)
            if fx.should_fire(fx.update(now, px_, okk, la, lo, latlon_v, sp)):
                fires.append(round(now, 1))
        return fires
    f1 = run_fix(150, lambda k: 10 / 30)                         # 5 s turning at 10°/s
    f2 = run_fix(150, lambda k: 2.5 / 30)                        # 5 s slowly at 2.5°/s (below the 3°/s speed limit)
    f3 = run_fix(240, lambda k: 10 / 30 if k < 90 else 0)        # 3 s turning, then the finger stays still
    f4 = run_fix(150, lambda k: 0, jitter=0.3)                   # still physical globe with a jittery estimate
    ok_fx = not f1 and not f2 and len(f3) == 1 and f3[0] >= 6.0 and len(f4) == 1 and f4[0] <= 3.3
    print(f"   turning at 10°/s: fired {len(f1)} times; slowly at 2.5°/s (caught by the geographic check): {len(f2)}")
    print(f"   turning for 3 s, then stop: fired at {f3} s (expected ≈ 3 + 3 = 6 s)")
    print(f"   still globe with orientation jitter 0.3°/frame: fired at {f4} s  {'OK' if ok_fx else 'FAIL'}")
    ok_all &= ok_fx

    print("\n5f) Virtual map (fixed map, left thumb+index spread = zoom, left index alone = time):")
    vm = VM.VirtualMap((1280, 720))
    c1, c2 = vm.to_latlon(vm.x0, vm.y0), vm.to_latlon(vm.x0 + vm.mw, vm.y0 + vm.mh)
    ok_c = abs(c1[0] - 90) < 1e-6 and abs(c1[1] + 180) < 1e-6 and abs(c2[0] + 90) < 1e-6 and abs(c2[1] - 180) < 1e-6
    print(f"   whole world: corners {c1[0]:.0f},{c1[1]:.0f} and {c2[0]:.0f},{c2[1]:.0f} {'OK' if ok_c else 'FAIL'}")
    nx_, ny_ = [float(v) for v in vm.to_px(40.7, -74.0)]
    vm.zoom_about(nx_, ny_, 4.0)
    keep = vm.to_latlon(nx_, ny_)
    ok_z = abs(keep[0] - 40.7) < 1e-6 and abs(keep[1] + 74.0) < 1e-6 and abs(vm.lon_span - 90) < 1e-9
    print(f"   zoom x4 over New York: the same place stays under the fingers {'OK' if ok_z else 'FAIL'}")
    vm.reset()
    vm.zoom_about(vm.x0 + 2, vm.y0 + 2, 6.0)                    # zoom at the very corner: view stays inside the world
    ok_e = vm.lon_min >= -180 - 1e-9 and vm.lat_max <= 90 + 1e-9
    print(f"   zoom at the map corner stays inside the world {'OK' if ok_e else 'FAIL'}")
    vm.reset()

    def mhand(cx, cy, spread_px, thumb_out=True):
        """Fake left hand: fingertips at the top around (cx, cy), palm below (palm size 90 px).
        thumb_out=False - thumb folded against the middle finger (pointing pose = time control)."""
        pts = [NS(x=cx / 1280, y=(cy + 60) / 720) for _ in range(21)]
        pts[0] = NS(x=cx / 1280, y=(cy + 170) / 720)            # wrist
        pts[9] = NS(x=cx / 1280, y=(cy + 80) / 720)             # middle-finger base
        pts[10] = NS(x=cx / 1280, y=(cy + 70) / 720)            # middle-finger PIP (finger folded)
        pts[8] = NS(x=(cx + spread_px / 2) / 1280, y=cy / 720)  # index tip
        pts[4] = (NS(x=(cx - spread_px / 2) / 1280, y=cy / 720) if thumb_out
                  else NS(x=(cx + 5) / 1280, y=(cy + 65) / 720))  # thumb tip
        return pts
    ax_, ay_ = [float(v) for v in vm.to_px(5.0, 20.0)]           # over Africa
    clock = iter(np.arange(0, 100, 1 / 30))
    zs = [vm.zoom]
    for _ in range(5):                                          # "L" held only ~0.15 s: too short, no zoom
        vm.handle_left(mhand(ax_, ay_, 27), 1280, 720, next(clock))
    for _ in range(5):
        vm.handle_left(mhand(ax_, ay_, 81, thumb_out=False), 1280, 720, next(clock))
    zs.append(vm.zoom)
    for _ in range(10):                                         # "L" over Africa, fingers close (0.3 palm)
        vm.handle_left(mhand(ax_, ay_, 27), 1280, 720, next(clock))
    for sp_ in np.linspace(27, 81, 12):                         # spread the fingers to 0.9 palm = zoom in
        vm.handle_left(mhand(ax_, ay_, sp_), 1280, 720, next(clock))
    zs.append(vm.zoom)
    anchor_ok = np.allclose(vm.to_latlon(ax_, ay_)[:2], (5.0, 20.0), atol=1e-6)
    for k_ in range(5):                                         # fold the thumb, move around: zoom is kept
        vm.handle_left(mhand(ax_ + 40 * k_, ay_ + 60, 27, thumb_out=False), 1280, 720, next(clock))
    zs.append(vm.zoom)
    for _ in range(10):                                         # "L" again with fingers wide apart ...
        vm.handle_left(mhand(ax_, ay_, 81), 1280, 720, next(clock))
    for sp_ in np.linspace(81, 27, 12):                         # ... and bring them together = zoom out
        vm.handle_left(mhand(ax_, ay_, sp_), 1280, 720, next(clock))
    zs.append(vm.zoom)
    expect_in = (81 / 27) ** VM.VirtualMap.ZOOM_POWER
    ok_g = (zs[1] == zs[0] and abs(zs[2] - expect_in) < 0.05 and anchor_ok and zs[3] == zs[2]
            and abs(zs[4] - 1.0) < 0.05)
    print(f"   brief 'L' (<0.25 s): zoom stays {zs[1]:.1f}; 'L' over Africa + spread fingers x3: "
          f"{zs[0]:.1f} -> {zs[2]:.1f} (Africa stays put); thumb folded: stays {zs[3]:.1f}; "
          f"fingers together: {zs[4]:.1f} {'OK' if ok_g else 'FAIL'}")
    vm.handle_left(None, 1280, 720)
    tl_ = next((l_ for l_ in by.values() if l_.dynamic), None)
    ok_tm = True
    if tl_ is not None:                                          # time: the left index along the map, as on paper
        Gv = vm.sheet_H()
        u_l = apply_H(Gv, [[vm.x0 + 1, vm.y0 + vm.mh / 2]])[0][0]
        u_r = apply_H(Gv, [[vm.x0 + vm.mw - 1, vm.y0 + vm.mh / 2]])[0][0]
        k_l, k_r = frame_from_u(tl_, u_l, None), frame_from_u(tl_, u_r, None)
        ok_tm = k_l == 0 and k_r == tl_.n_frames - 1
        print(f"   left index along the map -> time: left edge {tl_.labels[k_l]}, right edge {tl_.labels[k_r]} "
              f"{'OK' if ok_tm else 'FAIL'}")
    fxv = Fixation(3.0, 0.02 * 1280)                             # zooming under a still finger: no announcement
    vm.reset()
    pxv = np.array([vm.x0 + vm.mw * 0.55, vm.y0 + vm.mh * 0.45])
    latlon_vm = lambda x, y: vm.to_latlon(x, y)[:3]
    fired_v = []
    for k_ in range(240):                                       # 8 s: zooming for 3 s, then 5 s still
        now_ = k_ / 30
        if k_ < 90:
            vm.zoom_about(vm.x0 + vm.mw * 0.2, vm.y0 + vm.mh * 0.3, 1.0 + k_ * 0.05)   # zooming for 3 s elsewhere
        la_, lo_, okv = latlon_vm(*pxv)
        if fxv.should_fire(fxv.update(now_, pxv, okv, la_, lo_, latlon_vm, 0.0)):
            fired_v.append(round(now_, 1))
    ok_fv = len(fired_v) == 1 and fired_v[0] >= 5.0
    print(f"   zooming for 3 s under a still finger, then still: announced at {fired_v} s "
          f"(expected ≈ 3 + 3 s) {'OK' if ok_fv else 'FAIL'}")
    fr_m = np.full((720, 1280, 3), 110, np.uint8)
    vm.reset()
    t0_ = time.perf_counter(); vm.draw(fr_m, tl_ or next(iter(by.values())), 0); dt_m = (time.perf_counter() - t0_) * 1000
    vm.zoom_about(*[float(v) for v in vm.to_px(5.0, 20.0)], 4.0)
    fr_m2 = np.full((720, 1280, 3), 110, np.uint8); vm.draw(fr_m2, tl_ or next(iter(by.values())), 0)
    cv2.imwrite(str(out / "selftest_virtual_map.png"), np.vstack([fr_m, fr_m2]))
    hm_ = SP.help_announcement(KEYS_VMAP, len(specs), "vmap")
    ok_hm = "spread them apart" in hm_ and "left index finger along the map" in hm_ and "Zero" in hm_
    print(f"   drawing {dt_m:.0f} ms when the view changes; help mentions zoom, time and the zero key "
          f"{'OK' if ok_hm else 'FAIL'}")
    ok_all &= ok_c and ok_z and ok_e and ok_g and ok_tm and ok_fv and ok_hm

    print("\n6) Sonification (synthesis without a sound card):")
    sr = Sonifier.SR
    base_layer = by.get("sst") or next(iter(by.values()))
    smp = base_layer.sample(46.48, 30.73, base_layer.default_frame())
    son = Sonifier(enabled=False)
    son.play(smp["norm"], 0.0, 30.73)
    son.render(sr // 2)                                         # let the parameters settle
    a = son.render(sr).mean(1)
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    peak = np.fft.rfftfreq(len(a), 1 / sr)[np.argmax(spec)]
    want = midi_to_hz(norm_to_midi(smp["norm"]))
    ok_pitch = abs(peak - want) / want < 0.01
    print(f"   {base_layer.name}: {smp['text']} -> note {midi_name(norm_to_midi(smp['norm']))} ({want:.1f} Hz), "
          f"measured by FFT {peak:.1f} Hz  {'OK' if ok_pitch else 'FAIL'}")
    print(f"   range of any layer: vmin -> {midi_name(norm_to_midi(0))}, vmax -> {midi_name(norm_to_midi(1))}")

    def pulse_ratio(tension):
        s_ = Sonifier(enabled=False)
        s_.play(smp["norm"], tension, 30.73); s_.render(sr)
        x_ = s_.render(2 * sr).mean(1)
        rms = np.sqrt((x_[: len(x_) // 882 * 882].reshape(-1, 882) ** 2).mean(1))   # 20 ms windows
        return rms.max() / max(rms.min(), 1e-9)
    p0, p1 = pulse_ratio(0.0), pulse_ratio(smp["tension"] if smp["tension"] > 0.5 else 0.96)
    ok_pulse = p0 < 1.25 and p1 > 1.8
    print(f"   loudness pulsing: tension 0 -> x{p0:.2f}, tension {max(smp['tension'], 0.96):+.2f} -> x{p1:.2f}  "
          f"{'OK' if ok_pulse else 'FAIL'}")
    s_ = Sonifier(enabled=False)
    s_.play(0.5, 0.0, -150.0); s_.render(sr // 2)
    st_ = s_.render(sr)
    lr = np.sqrt((st_[:, 0] ** 2).mean() / (st_[:, 1] ** 2).mean())
    ok_pan = lr > 3
    print(f"   panning: longitude 150°W -> left/right loudness x{lr:.1f}  {'OK' if ok_pan else 'FAIL'}")
    ok_all &= ok_pitch and ok_pulse and ok_pan

    def render_route(layer, points, seconds, blk=512):
        """points: [(lat, lon, moment in years)]; linear between them. -> stereo array"""
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
    if "sst" in by:   # ocean journey: Antarctic -> equatorial Pacific -> Atlantic -> Mediterranean -> Black Sea
        route = [(-62, -150, 0), (-30, -140, 0), (0, -140, 0), (15, -60, 0), (40, -40, 0), (55, -20, 0), (36, 10, 0),
                 (43, 34, 0)]
        demos.append(("sonification_demo.wav", "ocean journey (sst)", render_route(by["sst"], route, 16.0)))
    if "temperature" in by:   # time machine: one point (Odesa), 1880 -> 2025
        demos.append(("sonification_odesa_1880_2025.wav", "Odesa 1880→2025 (temperature)",
                      render_route(by["temperature"], [(46.48, 30.73, 1880), (46.48, 30.73, 2025)], 20.0)))
    for fname, title, wav in demos:
        ok = np.abs(wav).max() < 0.99
        ok_all &= ok
        write_wav(out / fname, wav)
        print(f"   {title}: {len(wav) / sr:.0f} s, peak {np.abs(wav).max():.2f} {'OK' if ok else 'FAIL'} -> {fname}")

    print("\n6b) Start menu:")
    rows = _menu_rows()
    hits = [menu_hit((x0 + x1) / 2, (y0 + y1) / 2) for x0, y0, x1, y1 in rows]
    ok_mh = hits == [0, 1, 2, 3] and menu_hit(2, 2) is None
    ann_m = menu_announcement()
    ok_ma = all(f"Press {k} for {m[1].lower()}" in ann_m for k, m in enumerate(MENU, 1))
    order = [m[1] for m in MENU]
    ok_mo = order == ["Virtual map", "Paper map", "Physical globe", "Virtual globe"]
    ok_kb = all(any(k_[0] == "b" for k_ in ks) for ks in (KEYS, KEYS_GLOBE, KEYS_VIRTUAL, KEYS_VMAP))
    cv2.imwrite(str(out / "selftest_menu.png"), render_menu(None))
    print(f"   options {order}; clicks on rows -> {hits}; announcement names all 4; "
          f"'b' = back to the menu in every mode {'OK' if ok_mh and ok_ma and ok_mo and ok_kb else 'FAIL'}")
    ok_all &= ok_mh and ok_ma and ok_mo and ok_kb

    print("\n7) Rendering the header and the window for every layer:")
    view = L.LayerView()
    for k, s in enumerate(specs, 1):
        lay = by[s.name]
        y = lay.default_frame()
        smp = lay.sample(35.0, -70.0, y)                         # Gulf Stream: data in every ocean layer
        trail = [(30 + q * 0.6, -80 + q) for q in range(15)]
        cv2.imwrite(str(out / f"selftest_view_{s.name}.png"), view.render(lay, y, (35.0, -70.0), trail, 0.6, smp["text"]))
        cv2.imwrite(str(out / f"selftest_header_{s.name}.png"),
                    draw_header(np.full((484, 860, 3), 120, np.uint8), lay, y))
    cv2.imwrite(str(out / "selftest_sheet.png"), cv2.polylines(hand, [np.int32(tracker.corners)], True, (0, 200, 0), 3))
    print(f"   saved to {out}/")
    print("\nSELFTEST", "PASSED" if ok_all else "FAILED")


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def run_session(cap, specs, args, scale, frame, surface):
    """One session in the chosen mode (calibration if needed + tracking). -> "menu" or "quit"."""
    if surface == "vmap":                                      # nothing to calibrate: the map is drawn
        return run_tracking(cap, None, specs, args, scale, globe=VM.VirtualMap((frame.shape[1], frame.shape[0])))
    if surface == "virtual":                                   # nothing to calibrate: the globe is drawn
        return run_tracking(cap, None, specs, args, scale, globe=GB.VirtualGlobe((frame.shape[1], frame.shape[0])))
    if surface == "globe":
        loaded = None if args.recalibrate else GB.load_globe()
        if loaded and tuple(loaded[0].geom.to_dict()["frame_wh"]) != (frame.shape[1], frame.shape[0]):
            print("The camera frame size changed since the globe calibration - recalibration needed.")
            loaded = None
        if loaded:
            print(f"Globe calibration from {GB.GLOBE_CALIB.name}. (--recalibrate to redo it)")
        globe = GlobeSurface(*loaded) if loaded else run_globe_calibration(cap, args, scale)
        return run_tracking(cap, None, specs, args, scale, globe=globe)
    cal = MapCalibration.load(CALIB_FILE) if CALIB_FILE.exists() and not args.recalibrate else None
    if cal is None:
        if CALIB_FILE.exists() and not args.recalibrate:
            print("Old calibration.json format (no map template), recalibration needed.")
        cal = run_calibration(cap, args, scale)
    else:
        print(f"Calibration from {CALIB_FILE.name}: projection {cal.proj}. (--recalibrate to redo it)")
    return run_tracking(cap, cal, specs, args, scale)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="camera index (0 - built-in)")
    ap.add_argument("--recalibrate", action="store_true", help="calibrate again")
    ap.add_argument("--layer", help="data layer at start (name: sst, temperature, ...; see layers.py list)")
    ap.add_argument("--swap-hands", action="store_true",
                    help="swap the hand roles (if the program mixes up left and right; in the window - the h key)")
    ap.add_argument("--year", type=int, help="year at start: the nearest frame of the layer is used")
    ap.add_argument("--step", type=float, default=0.3, help="seconds per frame in playback mode (p)")
    ap.add_argument("--dwell", type=float, default=3.0, help="how many seconds to hold the finger still for an announcement")
    ap.add_argument("--radius", type=float, default=0.02,
                    help="allowed finger jitter, as a share of the frame width (0.02 = 2%%)")
    ap.add_argument("--month", help="month for the sst layer YYYY-MM (default: the latest in the file)")
    ap.add_argument("--projection", default="equirect", choices=PROJECTIONS,
                    help="projection, if only 4 places were clicked")
    ap.add_argument("--lon0", type=float, default=0.0, help="central meridian of the map, if only 4 places")
    ap.add_argument("--surface", default=None, choices=["map", "globe", "virtual", "vmap"],
                    help="skip the start menu: map - paper map, globe - globe on a stand with an axis, "
                         "virtual - virtual globe over the camera picture (rotated with a left-hand pinch), "
                         "vmap - virtual flat map over the camera picture (left thumb+index spread = zoom)")
    ap.add_argument("--track", default="features", choices=["features", "contour", "static"],
                    help="how to track the map: features - by the map picture (default), "
                         "contour - by the sheet edges, static - the map does not move")
    ap.add_argument("--no-sound", action="store_true", help="no sonification")
    ap.add_argument("--no-voice", action="store_true", help="do not announce the point by voice")
    ap.add_argument("--voice", default="auto",
                    help="macOS voice (default: auto = best installed English voice, e.g. Samantha; "
                         "Premium/Enhanced voices are picked automatically once installed)")
    ap.add_argument("--voice-rate", type=int, default=175, help="speaking rate, words per minute")
    ap.add_argument("--volume", type=float, default=0.8, help="volume 0..1")
    ap.add_argument("--width", type=int, default=1280, help="camera frame width")
    ap.add_argument("--view", type=int, default=860, help="width of the camera picture in the window, px")
    ap.add_argument("--selftest", action="store_true", help="check without a camera")
    ap.add_argument("--selftest-out", default=str(HERE / "selftest_out"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.selftest:
        selftest(args); return

    specs = L.discover_layers(args.month)
    if not specs:
        raise SystemExit("No data layer found (sst.mnmean.nc or data/<layer>/). See LAYERS.md")
    if args.layer and args.layer not in [s.name for s in specs]:
        raise SystemExit(f"No layer '{args.layer}'. Available: {', '.join(s.name for s in specs)}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.width * 9 / 16))
    if not cap.isOpened():
        raise SystemExit("The camera did not open. On macOS allow access: System Settings -> "
                         "Privacy & Security -> Camera -> your terminal / VS Code.")
    try:
        frame = _read(cap)
        scale = min(1.0, args.view / frame.shape[1])
        surface = args.surface                                 # --surface skips the menu the first time
        while True:
            if surface is None:                                # start menu: choose the mode
                surface = choose_surface(args)
                if surface is None:
                    return
            if run_session(cap, specs, args, scale, frame, surface) != "menu":
                return
            surface = None                                     # b / Backspace / Esc -> back to the menu
            args.recalibrate = False                           # recalibrate at most once per start
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
