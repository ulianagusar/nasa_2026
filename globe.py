"""Globe on a stand with an axis: finger pixel -> latitude/longitude; tracking of the globe's rotation.

Model:
  * pinhole camera: focal length f (pixels), principal point = frame centre;
  * sphere of radius 1, its centre C is found from the sphere outline in the frame (circle centre and radius);
  * globe orientation R(θ) = R0 · Rz(θ): R0 - from calibration with known places (Kabsch method),
    θ - rotation angle about the polar axis. The stand holds the axis, so only θ is searched each frame.

Tracking θ: a "water/land" mask from the frame colours (the colour model is learned from your globe
during calibration) is compared with the Natural Earth land outlines for all angles 0..360°.
Pixels under the hand are ignored. No markers and no accumulating error.
"""
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
LAND_FILE = HERE / "data" / "ne_50m_land.geojson"
GLOBE_CALIB = HERE / "globe_calibration.json"
GLOBE_VERSION = 1
RES = 0.25                       # land raster resolution, degrees


# ----------------------------------------------------------------------------
# Vectors and rotations
# ----------------------------------------------------------------------------
def latlon_to_vec(lat, lon):
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def vec_to_latlon(g):
    g = np.asarray(g, float)
    return np.degrees(np.arcsin(np.clip(g[..., 2], -1, 1))), np.degrees(np.arctan2(g[..., 1], g[..., 0]))


def rot_z(theta_deg):
    c, s = np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def kabsch(a, b):
    """R that minimizes Σ|R a_i - b_i|² (a, b: N×3)."""
    H = np.asarray(a).T @ np.asarray(b)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def angdiff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------------
# Camera + sphere geometry
# ----------------------------------------------------------------------------
class GlobeGeometry:
    def __init__(self, center_px, radius_px, f, frame_wh):
        self.u0, self.v0 = float(center_px[0]), float(center_px[1])
        self.r, self.f = float(radius_px), float(f)
        self.W, self.H = int(frame_wh[0]), int(frame_wh[1])
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        d = np.array([(self.u0 - self.cx) / self.f, (self.v0 - self.cy) / self.f, 1.0])
        d /= np.linalg.norm(d)
        alpha = np.arctan(self.r / self.f)                    # angular radius of the sphere
        self.C = d / np.sin(alpha)                             # sphere centre (radius 1) in camera coordinates

    @classmethod
    def from_center(cls, C, f, frame_wh):
        """Geometry from a known 3D sphere centre (e.g. from PnP), not from a circle in the frame."""
        g = cls((0, 0), 1, f, frame_wh)
        g.set_center(C)
        return g

    def set_center(self, C):
        self.C = np.asarray(C, float).copy()
        self.u0 = self.f * self.C[0] / self.C[2] + self.cx
        self.v0 = self.f * self.C[1] / self.C[2] + self.cy
        self.r = self.f * np.tan(np.arcsin(min(1.0, 1.0 / np.linalg.norm(self.C))))

    def K(self):
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])

    def rays(self, px, py):
        d = np.stack([(np.asarray(px, float) - self.cx) / self.f, (np.asarray(py, float) - self.cy) / self.f,
                      np.ones(np.shape(px))], -1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def hit(self, px, py):
        """Pixel -> (P: unit normal in camera coordinates, ok: the ray hits the sphere, facing: cos of the view angle)."""
        d = self.rays(px, py)
        b = d @ self.C
        disc = b * b - (self.C @ self.C - 1.0)
        ok = disc >= 0
        t = b - np.sqrt(np.maximum(disc, 0))
        P = t[..., None] * d - self.C
        facing = -(P * d).sum(-1)                              # 1 = looking straight on, 0 = edge of the sphere
        return P, ok, facing

    def project(self, P):
        """Normals P (N×3, camera coordinates) -> pixels and visibility (front hemisphere)."""
        S = self.C + P
        vis = (P * S).sum(-1) < 0
        return self.f * S[..., 0] / S[..., 2] + self.cx, self.f * S[..., 1] / S[..., 2] + self.cy, vis

    def to_dict(self):
        return {"center": [self.u0, self.v0], "radius": self.r, "f": self.f, "frame_wh": [self.W, self.H]}


def fov_to_f(fov_deg, width):
    return (width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)


def solve_pose(clicks_px, latlons, center, radius, frame_wh, f=None, default_fov=65.0):
    """Known places (clicks + lat/lon) -> (geometry, R0, mean error in degrees).
    If f is not given and there are ≥ 4 points, the focal length is chosen by the smallest error (field of view 30..100°)."""
    clicks = np.asarray(clicks_px, float)
    G = latlon_to_vec(np.asarray(latlons)[:, 0], np.asarray(latlons)[:, 1])
    if f is not None:
        cands = [float(f)]
    elif len(clicks) >= 4:
        cands = [fov_to_f(v, frame_wh[0]) for v in np.arange(30, 101, 1.0)]
    else:
        cands = [fov_to_f(default_fov, frame_wh[0])]
    best = None
    for fc in cands:
        geom = GlobeGeometry(center, radius, fc, frame_wh)
        P, ok, _ = geom.hit(clicks[:, 0], clicks[:, 1])
        if ok.sum() < 3:
            continue
        R = kabsch(G[ok], P[ok])
        err = np.degrees(np.arccos(np.clip((P[ok] * (G[ok] @ R.T)).sum(1), -1, 1)))
        if best is None or err.mean() < best[2]:
            best = (geom, R, float(err.mean()), err)
    if best is None:
        raise ValueError("Too few clicks inside the globe (need ≥ 3)")
    return best[0], best[1], best[2]


# ----------------------------------------------------------------------------
# Land raster (Natural Earth) and outlines for the overlay
# ----------------------------------------------------------------------------
_LAND = {}


def land_rings(path=LAND_FILE):
    land = json.loads(Path(path).read_text())
    rings = []
    for f in land["features"]:
        g = f["geometry"]
        for p in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            rings.append(np.asarray(p[0], float))
    return rings


def land_raster(res=RES, path=LAND_FILE):
    """bool[180/res, 360/res]: row 0 = 90°N, column 0 = 180°W."""
    key = (res, str(path))
    if key not in _LAND:
        h, w = int(round(180 / res)), int(round(360 / res))
        img = np.zeros((h, w), np.uint8)
        polys = [np.int32(np.stack([(r[:, 0] + 180) / res, (90 - r[:, 1]) / res], 1)) for r in land_rings(path)]
        cv2.fillPoly(img, polys, 1)
        _LAND[key] = img.astype(bool)
    return _LAND[key]


# ----------------------------------------------------------------------------
# Globe model: orientation + finger conversion
# ----------------------------------------------------------------------------
class GlobeModel:
    def __init__(self, geom, R0, theta=0.0):
        self.geom, self.R0, self.theta = geom, np.asarray(R0, float), float(theta)

    def R(self, theta=None):
        return self.R0 @ rot_z(self.theta if theta is None else theta)

    def to_latlon(self, px, py):
        """-> (lat, lon, ok, facing). ok=False if the finger is outside the sphere."""
        P, ok, facing = self.geom.hit(np.array([px]), np.array([py]))
        g = P @ self.R()                                        # = (R^T P^T)^T
        lat, lon = vec_to_latlon(g)
        return float(lat[0]), float(lon[0]), bool(ok[0]), float(facing[0])

    def project_latlon(self, lat, lon, theta=None):
        P = latlon_to_vec(lat, lon) @ self.R(theta).T
        return self.geom.project(P)


# ----------------------------------------------------------------------------
# Tracking the rotation angle
# ----------------------------------------------------------------------------
class SpinTracker:
    COARSE = np.arange(0, 360, 1.0)

    def __init__(self, model, n_target=2500, inner=0.9, hist_w=None, hist_l=None):
        self.model = model
        g = model.geom
        step = max(2.0, 2 * inner * g.r / np.sqrt(n_target * 4 / np.pi))
        xs = np.arange(g.u0 - inner * g.r, g.u0 + inner * g.r, step)
        ys = np.arange(g.v0 - inner * g.r, g.v0 + inner * g.r, step)
        X, Y = np.meshgrid(xs, ys)
        inside = (X - g.u0) ** 2 + (Y - g.v0) ** 2 <= (inner * g.r) ** 2
        inside &= (X >= 0) & (Y >= 0) & (X < g.W) & (Y < g.H)
        self.px, self.py = X[inside].astype(int), Y[inside].astype(int)
        P, ok, facing = g.hit(self.px, self.py)
        self.px, self.py, P, facing = self.px[ok], self.py[ok], P[ok], facing[ok]
        self.w = np.clip(facing, 0.2, 1.0)                     # near the edge of the sphere - lower weight
        self.P = P
        self.land = land_raster()
        self.nc = self.land.shape[1]
        self.rebase(model.R0)
        self.hist_w = None if hist_w is None else np.asarray(hist_w, float)
        self.hist_l = None if hist_l is None else np.asarray(hist_l, float)
        self.status, self.score, self.locked = "searching", 0.0, False

    def rebase(self, R0):
        """New base orientation (θ is measured from it); the frame points stay the same."""
        self.base = np.asarray(R0, float).copy()
        lat, lonq = vec_to_latlon(self.P @ self.base)          # coordinates in the globe-axis frame (θ = 0)
        self.lat_i = np.clip(((90 - lat) / RES).astype(int), 0, self.land.shape[0] - 1)
        self.lon_f = (lonq + 180) / RES

    # --- colour model: P(water | hue, saturation) ---
    def _hs(self, frame):
        pix = frame[self.py, self.px].reshape(-1, 1, 3)
        hsv = cv2.cvtColor(pix, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        return np.clip(hsv[:, 0] // 6, 0, 29).astype(int), np.clip(hsv[:, 1] // 8, 0, 31).astype(int)

    def pred_land(self, theta):
        idx = np.round(self.lon_f - theta / RES).astype(int) % self.nc
        return self.land[self.lat_i, idx]

    def learn_colors(self, frame, theta=0.0, valid=None):
        h, s = self._hs(frame)
        land = self.pred_land(theta)
        v = np.ones(len(h), bool) if valid is None else valid
        self.hist_w = np.zeros((30, 32)); self.hist_l = np.zeros((30, 32))
        np.add.at(self.hist_w, (h[v & ~land], s[v & ~land]), 1)
        np.add.at(self.hist_l, (h[v & land], s[v & land]), 1)
        k = cv2.getGaussianKernel(5, 1.0); k = k @ k.T             # smooth the histograms
        self.hist_w = cv2.filter2D(self.hist_w, -1, k, borderType=cv2.BORDER_REPLICATE)
        self.hist_l = cv2.filter2D(self.hist_l, -1, k, borderType=cv2.BORDER_REPLICATE)

    def water_prob(self, frame):
        h, s = self._hs(frame)
        if self.hist_w is None:                                 # before calibration: "blue = water"
            return np.where((h >= 15) & (h <= 22) & (s >= 6), 0.85, 0.2)
        nw, nl = self.hist_w.sum() + 1e-9, self.hist_l.sum() + 1e-9
        pw, pl = self.hist_w[h, s] / nw + 1e-4, self.hist_l[h, s] / nl + 1e-4
        prior_w = nw / (nw + nl)
        return pw * prior_w / (pw * prior_w + pl * (1 - prior_w))

    def scores(self, pw, valid, thetas):
        idx = np.round(self.lon_f[valid][:, None] - np.asarray(thetas)[None, :] / RES).astype(int) % self.nc
        land = self.land[self.lat_i[valid][:, None], idx]
        p = pw[valid][:, None]
        agree = np.where(land, 1 - p, p)
        w = self.w[valid][:, None]
        return (agree * w).sum(0) / w.sum()

    def update(self, frame, hand_mask=None):
        valid = np.ones(len(self.px), bool)
        if hand_mask is not None:
            valid &= hand_mask[self.py, self.px] == 0
        if valid.mean() < 0.35:                                 # the hand covers most of the globe
            self.status = "held" if self.locked else "searching"
            return self.model.theta
        pw = self.water_prob(frame)
        sc = self.scores(pw, valid, self.COARSE)
        b = int(np.argmax(sc))
        fine = self.COARSE[b] + np.arange(-1.0, 1.01, 0.25)
        scf = self.scores(pw, valid, fine)
        theta, best = float(fine[np.argmax(scf)] % 360), float(scf.max())
        far = np.abs(angdiff(self.COARSE, self.COARSE[b])) > 20
        self.margin = best - float(sc[far].max())               # how much better than other angles
        prev = self.model.theta
        if self.locked:
            at_prev = float(self.scores(pw, valid, [prev])[0])
            if abs(angdiff(theta, prev)) > 30 and best - at_prev < 0.03:
                self.status = "held"                            # a jump without confidence - do not trust it
                return prev
            d = angdiff(theta, prev)
            theta = (prev + 0.4 * d) % 360 if abs(d) < 1.0 else theta   # smooth small jitter
        self.model.theta, self.score, self.locked, self.status = theta, best, True, "tracked"
        return theta


# ----------------------------------------------------------------------------
# "Remember the globe": surface feature points with coordinates on the sphere -> full pose (PnP)
# ----------------------------------------------------------------------------
class GlobeMemory:
    """Feature points (ORB) of the globe surface: for each, a unit vector on the sphere (globe coordinates)
    and a descriptor. Grows when the globe is rotated (new sides), saved to a file."""
    MAX = 9000                                                  # more = slower matching

    def __init__(self, g=None, d=None):
        self.g = np.zeros((0, 3), np.float32) if g is None else np.asarray(g, np.float32)
        self.d = np.zeros((0, 32), np.uint8) if d is None else np.asarray(d, np.uint8)

    def __len__(self):
        return len(self.g)

    def add_from_frame(self, frame, model, orb, mask=None, min_facing=0.45, min_sep_deg=0.6, roi=None):
        """Add points from the frame with a known model pose. -> how many new points were added."""
        geom = model.geom
        m = np.zeros(frame.shape[:2], np.uint8)
        cv2.circle(m, (int(geom.u0), int(geom.v0)), int(geom.r * 0.97), 255, -1)
        if mask is not None:
            m[mask > 0] = 0
        kp, des = orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), m)
        if des is None or not kp:
            return 0
        pts = np.float32([k.pt for k in kp])
        P, ok, facing = geom.hit(pts[:, 0], pts[:, 1])
        keep = ok & (facing > min_facing)
        if not keep.any():
            return 0
        g = (P[keep] @ model.R()).astype(np.float32)           # = R^T P: coordinates on the sphere
        des = des[keep]
        if len(self.g):                                         # do not duplicate places already known
            near = (g @ self.g.T).max(1) > np.cos(np.radians(min_sep_deg))
            g, des = g[~near], des[~near]
        room = self.MAX - len(self.g)
        g, des = g[:max(room, 0)], des[:max(room, 0)]
        if len(g):
            self.g = np.vstack([self.g, g]); self.d = np.vstack([self.d, des])
        return len(g)

    def save(self, path):
        np.savez_compressed(path, g=self.g, d=self.d)

    @classmethod
    def load(cls, path):
        p = Path(path)
        if not p.exists():
            return cls()
        z = np.load(p)
        return cls(z["g"], z["d"])


class GlobeTracker:
    """Finds the remembered globe in the frame: frame points <-> memory points -> PnP -> sphere centre and orientation.
    The globe can be moved, brought closer, tilted. If there are too few points (unlearned side, hand) -
    fallback: search for the angle about the axis using land outlines (globe in the same place).
    While tracking confidently, the memory is extended with new sides of the globe."""
    MIN_INLIERS = 14

    def __init__(self, model, memory=None, hist_w=None, hist_l=None):
        self.model = model
        self.memory = memory if memory is not None else GlobeMemory()
        self.orb = cv2.ORB_create(1500, fastThreshold=12)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.spin = SpinTracker(model, hist_w=hist_w, hist_l=hist_l)
        self._spin_key = self._key()
        self.status, self.mode, self.score, self.inliers, self.locked = "searching", "-", 0.0, 0, False
        self.last_learn = -1e9
        self.frame_i = 0

    # --- compatibility with the rest of the program ---
    @property
    def hist_w(self):
        return self.spin.hist_w

    @property
    def hist_l(self):
        return self.spin.hist_l

    @property
    def px(self):
        return self.spin.px

    @property
    def py(self):
        return self.spin.py

    def learn_colors(self, frame, theta=0.0, valid=None, hand_mask=None):
        self._ensure_spin()                                     # first, points for the current sphere position
        if hand_mask is not None:
            valid = hand_mask[self.spin.py, self.spin.px] == 0
        self.spin.learn_colors(frame, self.model.theta if theta is None else theta, valid)

    def _key(self):
        g = self.model.geom
        return (round(g.u0), round(g.v0), round(g.r))

    def _ensure_spin(self):
        """SpinTracker is tied to the sphere position in the frame; after the globe moves - rebuild the points,
        after the orientation changes - just recompute the base (cheap)."""
        if self._key() != self._spin_key:
            self.spin = SpinTracker(self.model, hist_w=self.spin.hist_w, hist_l=self.spin.hist_l)
            self._spin_key = self._key()
        elif not np.allclose(self.spin.base, self.model.R0, atol=1e-6):
            self.spin.rebase(self.model.R0)

    def _verify(self, frame, hand_mask, span=10.0):
        """Check the pose against an independent source - Natural Earth land outlines.
        -> (agreement at the current pose, by how many degrees about the axis the agreement would be better)."""
        self.model.R0, self.model.theta = self.model.R(), 0.0
        self._ensure_spin()
        valid = np.ones(len(self.spin.px), bool)
        if hand_mask is not None:
            valid &= hand_mask[self.spin.py, self.spin.px] == 0
        if valid.mean() < 0.35:
            return 0.0, 99.0
        offs = np.arange(-span, span + 0.01, 0.5)
        sc = self.spin.scores(self.spin.water_prob(frame), valid, offs)
        return float(sc[len(offs) // 2]), float(offs[int(np.argmax(sc))])

    def coverage(self):
        """How much of the globe is learned: share of 10-degree longitude bands (60°S..60°N) with memory points."""
        if not len(self.memory):
            return 0.0
        lat, lon = vec_to_latlon(self.memory.g)
        lon = lon[np.abs(lat) < 60]
        return float(len(np.unique(((lon + 180) // 10).astype(int))) / 36)

    def tour_step(self, frame, hand_mask=None, now=None, every=0.4):
        """Calibration "tour": the globe stays in place and is rotated slowly. Angle - from land (accurate),
        points are remembered from every position. -> (angle, share of the globe learned)."""
        import time as _t
        now = _t.monotonic() if now is None else now
        self._ensure_spin()
        self.spin.update(frame, hand_mask)
        self.status, self.mode = self.spin.status, "spin"
        if self.spin.status == "tracked" and self.spin.score > 0.85 and now - self.last_learn >= every:
            self.memory.add_from_frame(frame, self.model, self.orb, hand_mask, min_facing=0.6)
            self.last_learn = now
        return self.model.theta, self.coverage()

    def _pnp(self, frame, hand_mask):
        if len(self.memory) < self.MIN_INLIERS:
            return False
        g = self.model.geom
        m = None
        if self.locked:                                         # search near the last sphere position
            region = (g.u0, g.v0, g.r * 1.5)
        else:                                                   # position unknown: first find the sphere by its outline,
            c = detect_globe_circle(frame) if self.frame_i % 3 == 1 or not hasattr(self, "_last_circle") else None
            if c is not None:                                   # so that background points do not use up the ORB limit
                self._last_circle = c
            c = getattr(self, "_last_circle", None)
            region = None if c is None else (c[0], c[1], c[2] * 1.15)
        if region is not None:
            m = np.zeros(frame.shape[:2], np.uint8)
            cv2.circle(m, (int(region[0]), int(region[1])), int(region[2]), 255, -1)
        if hand_mask is not None:
            m = np.full(frame.shape[:2], 255, np.uint8) if m is None else m
            m[hand_mask > 0] = 0
        kp, des = self.orb.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), m)
        if des is None or len(kp) < self.MIN_INLIERS:
            return False
        pairs = self.bf.knnMatch(des, self.memory.d, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.78 * p[1].distance]
        if len(good) < self.MIN_INLIERS:
            return False
        obj = self.memory.g[[q.trainIdx for q in good]].astype(np.float64)
        img = np.float64([kp[q.queryIdx].pt for q in good])
        K = g.K()
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=300, reprojectionError=3.0,
                                                 confidence=0.995, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < self.MIN_INLIERS or len(inl) < 0.25 * len(good):
            return False
        inl = inl.ravel()
        ok, rvec, tvec = cv2.solvePnP(obj[inl], img[inl], K, None, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(rvec)
        C = tvec.ravel()
        if not ok or C[2] <= 1.05 or np.linalg.norm(C) <= 1.05:
            return False
        P = obj[inl] @ R.T                                       # the points must be on the visible side
        if ((P * (C + P)).sum(1) >= 0).mean() > 0.1:
            return False
        proj, _ = cv2.projectPoints(obj[inl], rvec, tvec, K, None)
        if np.linalg.norm(proj.reshape(-1, 2) - img[inl], axis=1).mean() > 3.0:
            return False
        if self.locked:                                          # smooth small position jitter
            dC = np.linalg.norm(C - g.C)
            dA = np.degrees(np.arccos(np.clip((np.trace(R.T @ self.model.R()) - 1) / 2, -1, 1)))
            if dC < 0.01 and dA < 0.6:
                C = g.C + 0.4 * (C - g.C)
        g.set_center(C)
        self.model.R0, self.model.theta = R, 0.0
        self.inliers, self.score = int(len(inl)), float(len(inl)) / len(good)
        return True

    def update(self, frame, hand_mask=None, now=None):
        import time as _t
        now = _t.monotonic() if now is None else now
        self.frame_i += 1
        if self._pnp(frame, hand_mask):
            self.status, self.mode, self.locked = "tracked", "features", True
            if now - self.last_learn > 1.0 and self.inliers >= 25:
                # remember only a pose confirmed by land outlines (otherwise the error accumulates)
                self.land_score, off = self._verify(frame, hand_mask)
                if self.land_score > 0.88 and abs(off) <= 0.5:
                    self.memory.add_from_frame(frame, self.model, self.orb, hand_mask, min_facing=0.6)
                self.last_learn = now
            return self.model.theta
        # fallback: globe in the same place, search only the angle about the axis using land outlines
        self._ensure_spin()
        self.spin.update(frame, hand_mask)
        good = self.spin.status == "tracked" and self.spin.score > 0.85 and getattr(self.spin, "margin", 0) > 0.05
        self.inliers, self.score = 0, self.spin.score
        if good:
            self.status, self.mode, self.locked = "tracked", "spin", True
            if now - self.last_learn > 0.5:                    # unlearned side: remember it
                self.memory.add_from_frame(frame, self.model, self.orb, hand_mask, min_facing=0.6)
                self.last_learn = now
        else:
            self.status = "held" if self.locked else "searching"
        return self.model.theta


# ----------------------------------------------------------------------------
# Virtual globe: drawn over the camera picture, rotated with a left-hand pinch
# ----------------------------------------------------------------------------
def _rodrigues(axis_angle):
    return cv2.Rodrigues(np.asarray(axis_angle, np.float64).reshape(3, 1))[0]


class VirtualGlobe:
    """A globe that does not physically exist: a sphere with the layer data is drawn over the (mirrored) camera frame.
    The right hand points at it, the left hand rotates it: pinch (thumb + index) and move = trackball, with inertia.
    View coordinates: X right, U up, Z towards the viewer; v = R g (g - point in globe coordinates).
    polar=True (default): the poles stay fixed like on a globe stand - the axis never leans sideways. Each pinch
    is locked to one direction, decided by the first LOCK_PX pixels: sideways = only spin around the axis (the
    tilt does not change at all), clearly up/down = only tilt (up to looking straight at a pole).
    polar=False: free trackball (any rotation)."""
    status, mode, inliers, score = "tracked", "virtual", 0, 1.0
    PINCH_ON, PINCH_OFF = 0.33, 0.5          # thumb-index distance / palm size
    LOCK_PX, TILT_RATIO = 25.0, 2.0          # fixed poles: movement before the direction is decided; how vertical = tilt

    def __init__(self, frame_wh, radius_frac=0.42, lat0=20.0, lon0=15.0, step=2, polar=True):
        self.W, self.H = int(frame_wh[0]), int(frame_wh[1])
        self.polar = polar
        self.u0, self.v0 = self.W * 0.5, self.H * 0.52
        self.r = self.H * radius_frac
        self.step = step
        self.reset(lat0, lon0)
        self.coast = [latlon_to_vec(r[::4, 1], r[::4, 0]) for r in land_rings() if len(r) > 20] \
            if LAND_FILE.exists() else []
        grat = []
        for lo in range(-180, 180, 30):
            la = np.linspace(-85, 85, 60)
            grat.append(latlon_to_vec(la, np.full_like(la, lo)))
        for la in range(-60, 61, 30):
            lo = np.linspace(-180, 180, 120)
            grat.append(latlon_to_vec(np.full_like(lo, la), lo))
        self.grat = grat
        self.pinching, self.last_pt, self.pinch_pt, self.last_t = False, None, None, None
        self.vel = np.zeros(2)                                   # drag speed (px/s) for inertia
        self.lock, self.acc = None, np.zeros(2)                  # fixed poles: "spin" or "tilt" for this pinch
        self._grid = None

    # --- orientation ---
    def reset(self, lat0=20.0, lon0=15.0):
        """Point (lat0, lon0) in the centre, north up."""
        self.lat_c = float(np.clip(lat0, -90.0, 90.0))
        self.lon_c = float((lon0 + 180.0) % 360.0 - 180.0)
        self._set_view()
        self.vel = np.zeros(2)

    def _set_view(self):
        """R from the centre point (lat_c, lon_c) with north up. "Up" = the direction of growing latitude,
        which stays defined at the poles too (there it points across the pole)."""
        la, lo = np.radians(self.lat_c), np.radians(self.lon_c)
        fwd = latlon_to_vec(self.lat_c, self.lon_c)
        up = np.array([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)])
        self.R = np.stack([np.cross(up, fwd), up, fwd])

    def north_up(self):
        """Turn north up without changing the place in the centre."""
        lat, lon, _, _ = self.to_latlon(self.u0, self.v0)
        self.reset(lat, lon)

    def set_polar(self, on):
        """Switch between the fixed-poles mode and the free trackball; the place in the centre stays."""
        self.polar = bool(on)
        if self.polar:
            self.north_up()

    def rotate_drag(self, dx, dy):
        """Drag by (dx, dy) pixels: the surface follows the hand."""
        if self.polar:                    # sideways = spin around the axis, up/down = tilt; poles stay put
            self.lon_c = float((self.lon_c - np.degrees(dx / self.r) + 180.0) % 360.0 - 180.0)
            self.lat_c = float(np.clip(self.lat_c + np.degrees(dy / self.r), -90.0, 90.0))
            self._set_view()
            return
        d = np.array([dx, -dy], float) / self.r
        ang = float(np.linalg.norm(d))
        if ang < 1e-6:
            return
        axis = np.array([-d[1], d[0], 0.0]) / ang
        self.R = _rodrigues(axis * ang) @ self.R

    def resize(self, factor):
        self.r = float(np.clip(self.r * factor, self.H * 0.2, self.H * 0.49))
        self._grid = None

    # --- finger -> coordinates ---
    def to_latlon(self, px, py):
        X, U = (px - self.u0) / self.r, (self.v0 - py) / self.r
        d2 = X * X + U * U
        if d2 > 1:
            return 0.0, 0.0, False, 0.0
        Z = np.sqrt(1 - d2)
        lat, lon = vec_to_latlon(np.array([X, U, Z]) @ self.R)
        return float(lat), float(lon), True, float(Z)

    def project(self, G):
        V = np.asarray(G) @ self.R.T
        return self.u0 + V[:, 0] * self.r, self.v0 - V[:, 1] * self.r, V[:, 2] > 0

    # --- left hand: pinch and move ---
    def handle_left(self, lms, w, h, now):
        dt = 0.0 if self.last_t is None else min(now - self.last_t, 0.2)
        self.last_t = now
        if lms is None:
            self.pinching, self.last_pt, self.pinch_pt = False, None, None
        else:
            p = lambda i: np.array([lms[i].x * w, lms[i].y * h])
            size = max(np.linalg.norm(p(0) - p(9)), 1.0)        # wrist - base of the middle finger
            pd = np.linalg.norm(p(4) - p(8)) / size
            self.pinching = pd < (self.PINCH_OFF if self.pinching else self.PINCH_ON)
            self.pinch_pt = (p(4) + p(8)) / 2
            if self.pinching:
                if self.last_pt is None:                         # a new pinch: direction not decided yet
                    self.lock, self.acc = None, np.zeros(2)
                else:
                    delta = self.pinch_pt - self.last_pt
                    if self.polar:
                        delta = self._locked(delta)
                    self.rotate_drag(*delta)
                    if dt > 0:
                        self.vel = 0.6 * self.vel + 0.4 * delta / dt
                self.last_pt = self.pinch_pt
                return
            self.last_pt = None
        if dt > 0 and np.linalg.norm(self.vel) > 20:             # inertia after letting go
            self.rotate_drag(*(self.vel * dt))
            self.vel *= np.exp(-dt / 0.6)
        elif dt > 0:
            self.vel[:] = 0

    def _locked(self, delta):
        """Fixed poles: keep only the spin (x) or only the tilt (y) part of the hand movement."""
        if self.lock is None:
            self.acc = self.acc + delta
            if np.linalg.norm(self.acc) < self.LOCK_PX:
                return np.zeros(2)                               # wait until the direction is clear
            self.lock = "tilt" if abs(self.acc[1]) > self.TILT_RATIO * abs(self.acc[0]) else "spin"
            delta = self.acc                                     # apply the movement made so far
        return np.array([delta[0], 0.0]) if self.lock == "spin" else np.array([0.0, delta[1]])

    # --- drawing ---
    def _disc(self):
        if self._grid is None:
            s = self.step
            xs = np.arange(-self.r, self.r, s)
            X, Y = np.meshgrid(xs, xs)
            X, U = X / self.r, -Y / self.r
            inside = X ** 2 + U ** 2 < 1
            Z = np.sqrt(np.clip(1 - X ** 2 - U ** 2, 0, 1))
            self._grid = (np.stack([X, U, Z], -1), inside, Z, len(xs))
        return self._grid

    def draw(self, frame, layer=None, t=None, alpha=0.6):
        V, inside, Z, n = self._disc()
        G = V[inside] @ self.R                                   # disc points -> coordinates on the globe
        lat, lon = vec_to_latlon(G)
        img = np.zeros((n, n, 3), np.uint8)
        if layer is not None:
            grid = layer.grid(t)
            dlat = abs(layer.lat[1] - layer.lat[0]); dlon = abs(layer.lon[1] - layer.lon[0])
            i = np.clip(np.round((lat - layer.lat[0]) / dlat).astype(int), 0, len(layer.lat) - 1)
            j = np.round((lon - layer.lon[0]) / dlon).astype(int) % len(layer.lon)
            val = grid[i, j]
            x0, x1, y0, y1 = layer.extent()
            if x1 - x0 < 359:                                  # regional layer: nothing outside its area
                lw = (lon + 180) % 360 - 180
                val = np.where((lat >= y0) & (lat <= y1) & (lw >= x0) & (lw <= x1), val, np.nan)
            nrm = layer.norm_array(val)
            rgb = layer.lut[np.nan_to_num(nrm * 255).astype(int)].astype(float)
            rgb[np.isnan(val)] = (170, 170, 165)
        else:
            land = land_raster()
            li = np.clip(((90 - lat) / RES).astype(int), 0, land.shape[0] - 1)
            lj = np.clip(((lon + 180) / RES).astype(int), 0, land.shape[1] - 1)
            rgb = np.where(land[li, lj][:, None], [[150, 170, 110]], [[60, 120, 185]]).astype(float)
        rgb *= (0.55 + 0.45 * Z[inside])[:, None]                # volume: the edge of the sphere is darker
        img[inside] = np.clip(rgb[:, ::-1], 0, 255).astype(np.uint8)   # RGB -> BGR
        big = cv2.resize(img, (2 * int(self.r), 2 * int(self.r)), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros(big.shape[:2], np.uint8)
        cv2.circle(mask, (int(self.r), int(self.r)), int(self.r) - 1, 255, -1, cv2.LINE_AA)
        x0, y0 = int(self.u0 - self.r), int(self.v0 - self.r)
        fx0, fy0 = max(x0, 0), max(y0, 0)
        fx1, fy1 = min(x0 + big.shape[1], frame.shape[1]), min(y0 + big.shape[0], frame.shape[0])
        if fx1 > fx0 and fy1 > fy0:
            sub = frame[fy0:fy1, fx0:fx1].astype(float)
            b = big[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0].astype(float)
            a = (mask[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0] / 255.0 * alpha)[..., None]
            frame[fy0:fy1, fx0:fx1] = (sub * (1 - a) + b * a).astype(np.uint8)
        for vecs, col, th in ((self.grat, (210, 210, 210), 1), (self.coast, (40, 40, 40), 1)):
            for Vv in vecs:
                x, y, vis = self.project(Vv)
                pts = np.stack([x, y], 1)
                start = None
                for k in range(len(vis) + 1):
                    if k < len(vis) and vis[k]:
                        start = k if start is None else start
                    elif start is not None:
                        if k - start > 1:
                            cv2.polylines(frame, [np.int32(pts[start:k])], False, col, th, cv2.LINE_AA)
                        start = None
        cv2.circle(frame, (int(self.u0), int(self.v0)), int(self.r), (255, 255, 255), 2, cv2.LINE_AA)
        if self.pinch_pt is not None:                            # where the left hand is: pinch = rotate
            c = tuple(int(v) for v in self.pinch_pt)
            col = (191, 106, 37) if self.pinching else (160, 160, 160)
            cv2.circle(frame, c, 14 if self.pinching else 9, col, -1 if self.pinching else 2, cv2.LINE_AA)
            word = ("SPIN" if self.lock == "spin" else "TILT" if self.lock == "tilt" else "ROTATE") \
                if self.polar else "ROTATE"
            cv2.putText(frame, word if self.pinching else "pinch to rotate", (c[0] + 16, c[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)


# ----------------------------------------------------------------------------
# Finding the sphere outline in the frame
# ----------------------------------------------------------------------------
def detect_globe_circle(frame):
    """-> (u0, v0, r) of the largest clear circle, or None."""
    h, w = frame.shape[:2]
    s = min(1.0, 640.0 / max(h, w))
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
    gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 5)
    hs = small.shape[0]
    c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=hs / 2, param1=110, param2=45,
                         minRadius=int(0.12 * hs), maxRadius=int(0.6 * hs))
    if c is None:
        return None
    u, v, r = max(c[0], key=lambda x: x[2])
    return float(u / s), float(v / s), float(r / s)


def circle_from_3pts(p1, p2, p3):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-6:
        return None
    ux = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (y1 - y2)) / d
    uy = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (x2 - x1)) / d
    return ux, uy, float(np.hypot(x1 - ux, y1 - uy))


# ----------------------------------------------------------------------------
# Overlay on the frame: outline, grid, coastlines (shows whether the tracking matches)
# ----------------------------------------------------------------------------
class GlobeOverlay:
    def __init__(self, model, coast_step=4):
        self.model = model
        self.coast = [latlon_to_vec(r[::coast_step, 1], r[::coast_step, 0]) for r in land_rings() if len(r) > 20]
        grat = []
        for lo in range(-180, 180, 30):
            la = np.linspace(-85, 85, 60)
            grat.append(latlon_to_vec(la, np.full_like(la, lo)))
        for la in range(-60, 61, 30):
            lo = np.linspace(-180, 180, 120)
            grat.append(latlon_to_vec(np.full_like(lo, la), lo))
        self.grat = grat

    def _poly(self, frame, vecs, color, thick):
        R = self.model.R()
        for V in vecs:
            x, y, vis = self.model.geom.project(V @ R.T)
            pts = np.stack([x, y], 1)
            start = None
            for i in range(len(vis) + 1):                      # draw only the visible pieces
                if i < len(vis) and vis[i]:
                    start = i if start is None else start
                elif start is not None:
                    if i - start > 1:
                        cv2.polylines(frame, [np.int32(pts[start:i])], False, color, thick, cv2.LINE_AA)
                    start = None

    def draw(self, frame, status="tracked"):
        g = self.model.geom
        col = {"tracked": (0, 200, 0), "held": (0, 200, 255), "searching": (0, 0, 255)}.get(status, (200, 200, 200))
        cv2.circle(frame, (int(g.u0), int(g.v0)), int(g.r), col, 2, cv2.LINE_AA)
        self._poly(frame, self.grat, (200, 200, 200), 1)
        self._poly(frame, self.coast, col, 1)


# ----------------------------------------------------------------------------
# Saving the calibration
# ----------------------------------------------------------------------------
GLOBE_MEMORY = HERE / "globe_memory.npz"


def save_globe(path, model, tracker, extra=None):
    d = {"version": GLOBE_VERSION, **model.geom.to_dict(), "C": model.geom.C.tolist(),
         "R0": model.R().tolist(),
         "hist_w": None if tracker.hist_w is None else np.asarray(tracker.hist_w).tolist(),
         "hist_l": None if tracker.hist_l is None else np.asarray(tracker.hist_l).tolist()}
    d.update(extra or {})
    Path(path).write_text(json.dumps(d))
    if isinstance(tracker, GlobeTracker):
        tracker.memory.save(GLOBE_MEMORY)


def load_globe(path=GLOBE_CALIB):
    """-> (model, GlobeTracker) with the last known pose and the point memory, or None."""
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if d.get("version") != GLOBE_VERSION:
        return None
    geom = GlobeGeometry(d["center"], d["radius"], d["f"], d["frame_wh"])
    if d.get("C") is not None:
        geom.set_center(d["C"])
    model = GlobeModel(geom, d["R0"])
    return model, GlobeTracker(model, GlobeMemory.load(GLOBE_MEMORY), d["hist_w"], d["hist_l"])


# ----------------------------------------------------------------------------
# Synthetic globe for the self-test (renders a "camera frame")
# ----------------------------------------------------------------------------
def synthetic_texture(width=2048, seed=3):
    """Texture that looks like a political globe (equirectangular): coloured countries, borders, grid, labels."""
    rng = np.random.default_rng(seed)
    h = width // 2
    sx, sy = width / 360.0, h / 180.0
    tex = np.zeros((h, width, 3), np.uint8)
    tex[:] = (185, 125, 40)                                      # ocean (BGR)
    for y in range(0, h, 4):                                     # slight depth gradient
        tex[y:y + 4] = np.clip(np.array((185, 125, 40)) - abs(y - h / 2) / h * 40, 0, 255)
    cfile = HERE / "data" / "ne_50m_admin_0_countries.geojson"
    polys = []
    if cfile.exists():
        for f in json.loads(cfile.read_text())["features"]:
            g = f["geometry"]
            col = tuple(int(v) for v in rng.integers(90, 230, 3))
            for p in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
                r = np.asarray(p[0], float)
                pts = np.int32(np.stack([(r[:, 0] + 180) * sx, (90 - r[:, 1]) * sy], 1))
                cv2.fillPoly(tex, [pts], col)
                polys.append(pts)
        cv2.polylines(tex, polys, True, (40, 40, 40), 1, cv2.LINE_AA)
    for lo in range(-180, 181, 15):
        cv2.line(tex, (int((lo + 180) * sx), 0), (int((lo + 180) * sx), h), (215, 190, 150), 1)
    for la in range(-75, 76, 15):
        cv2.line(tex, (0, int((90 - la) * sy)), (width, int((90 - la) * sy)), (215, 190, 150), 1)
    words = ["OCEAN", "SEA", "GULF", "BAY", "CAPE", "ISLANDS", "BASIN", "RIDGE", "STRAIT", "TRENCH", "PLATEAU"]
    for _ in range(260):
        x, y = int(rng.integers(0, width - 80)), int(rng.integers(30, h - 30))
        cv2.putText(tex, words[int(rng.integers(len(words)))] + str(int(rng.integers(10, 99))), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    return tex


def render_synthetic(geom, R, frame_shape, seed=0, water=(185, 120, 35), texture=None):
    rng = np.random.default_rng(seed)
    h, w = frame_shape[:2]
    img = (np.full((h, w, 3), 95, np.int16) + rng.integers(-12, 12, (h, w, 3))).astype(np.int16)
    x0, x1 = int(max(0, geom.u0 - geom.r - 3)), int(min(w, geom.u0 + geom.r + 3))
    y0, y1 = int(max(0, geom.v0 - geom.r - 3)), int(min(h, geom.v0 + geom.r + 3))
    X, Y = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    P, ok, facing = geom.hit(X.ravel(), Y.ravel())
    lat, lon = vec_to_latlon(P @ R)
    land = land_raster()
    li = np.clip(((90 - lat) / RES).astype(int), 0, land.shape[0] - 1)
    lj = np.clip(((lon + 180) / RES).astype(int), 0, land.shape[1] - 1)
    is_land = land[li, lj]
    if texture is not None:                                    # globe texture (equirectangular)
        th_, tw_ = texture.shape[:2]
        ti = np.clip(((90 - lat) / 180 * th_).astype(int), 0, th_ - 1)
        tj = np.clip(((lon + 180) / 360 * tw_).astype(int), 0, tw_ - 1)
        col = texture[ti, tj].astype(float)
    else:
        band = np.abs(lat)                                    # land: green near the equator, yellow-brown further away
        land_col = np.stack([60 + band, 150 - band * 0.6, 120 + band * 1.2], 1)
        col = np.where(is_land[:, None], land_col, np.array(water, float)[None, :])
    shade = 0.55 + 0.45 * np.clip(facing, 0, 1)
    col = col * shade[:, None] + rng.normal(0, 6, col.shape)
    sub = img[y0:y1, x0:x1].reshape(-1, 3)
    sub[ok] = col[ok]
    img[y0:y1, x0:x1] = sub.reshape(y1 - y0, x1 - x0, 3)
    return np.clip(img, 0, 255).astype(np.uint8)
