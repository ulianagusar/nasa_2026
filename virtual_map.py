"""Virtual map (--surface vmap): a flat world map drawn over the (mirrored) camera picture.

* The map stays in a fixed rectangle - it cannot be dragged around.
* Zoom, like on a phone screen: hold the LEFT thumb and index finger out like an "L" over the place you want
  (~0.25 s to start), then SPREAD them apart to zoom in or BRING THEM TOGETHER to zoom out - the zoom follows
  the distance between the fingertips. Zooming happens around the point between the fingers where the gesture
  started, so that place stays put. Fold the thumb back (point with the index only) to keep the zoom.
  To get from America to Africa: zoom out, then zoom in over Africa.
* Time: the LEFT index finger alone (thumb folded), slid along the map, exactly as with the paper map.
* The RIGHT index finger points at a place.

Projection: equirectangular (the same as the data), the visible window always keeps a 2:1 aspect.
"""
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
LAND_FILE = HERE / "data" / "ne_50m_land.geojson"
UNIT = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])


class VirtualMap:
    """Flat map in a fixed rectangle of the frame; view window (lon/lat bounds) changes only by zooming."""
    status, mode, inliers, score = "tracked", "virtual", 0, 1.0
    # "Thumb out": distance from the thumb tip to the middle-finger PIP joint / palm size. When pointing with
    # the index only, the thumb lies against the middle finger (small value); in the "L" / pinch pose it is far.
    THUMB_ON, THUMB_OFF = 0.55, 0.40         # with hysteresis
    HOLD_S = 0.25                            # the pose must be held this long before zooming starts
    ZOOM_POWER = 1.5                         # zoom = start_zoom * (spread / start_spread) ** ZOOM_POWER
    MIN_SPREAD = 0.12
    MAX_ZOOM = 24.0

    def __init__(self, frame_wh, step=2, alpha=0.78):
        self.W, self.H = int(frame_wh[0]), int(frame_wh[1])
        mw = 0.9 * self.W
        mh = mw / 2
        if mh > 0.8 * self.H:
            mh = 0.8 * self.H
            mw = 2 * mh
        self.mw, self.mh = int(mw), int(mh)
        self.x0 = int((self.W - self.mw) / 2)
        self.y0 = int((self.H - self.mh) / 2 - 0.04 * self.H)    # room for the time scale underneath
        self.step, self.alpha = step, alpha
        self.reset()
        self.rings = []
        if LAND_FILE.exists():
            land = json.loads(LAND_FILE.read_text())
            for f in land["features"]:
                g = f["geometry"]
                for p in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
                    self.rings.append(np.asarray(p[0], float))
        self.pinching, self.pinch_pt, self.tips = False, None, None
        self._anchor = None                  # (screen point, geo point, zoom and spread at the start of the gesture)
        self._pose_since = None
        self._cache_key, self._cache_img = None, None

    # --- view window ---
    def reset(self):
        """Whole world."""
        self.lon_min, self.lat_max, self.zoom = -180.0, 90.0, 1.0

    @property
    def lon_span(self):
        return 360.0 / self.zoom

    @property
    def lat_span(self):
        return 180.0 / self.zoom

    def _clamp(self):
        self.lon_min = float(np.clip(self.lon_min, -180.0, 180.0 - self.lon_span))
        self.lat_max = float(np.clip(self.lat_max, -90.0 + self.lat_span, 90.0))

    def zoom_about(self, px, py, new_zoom, geo=None):
        """Zoom so that the geographic point under (px, py) stays under (px, py)."""
        fx = (px - self.x0) / self.mw
        fy = (py - self.y0) / self.mh
        if geo is None:
            geo = (self.lat_max - fy * self.lat_span, self.lon_min + fx * self.lon_span)
        self.zoom = float(np.clip(new_zoom, 1.0, self.MAX_ZOOM))
        self.lon_min = geo[1] - fx * self.lon_span
        self.lat_max = geo[0] + fy * self.lat_span
        self._clamp()

    def zoom_center(self, factor):
        self.zoom_about(self.x0 + self.mw / 2, self.y0 + self.mh / 2, self.zoom * factor)

    # --- finger -> coordinates ---
    def to_latlon(self, px, py):
        """-> (lat, lon, ok, 1.0); ok=False outside the map rectangle (same signature as the globes)."""
        fx = (px - self.x0) / self.mw
        fy = (py - self.y0) / self.mh
        ok = 0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0
        return (float(self.lat_max - fy * self.lat_span), float(self.lon_min + fx * self.lon_span), ok, 1.0)

    def to_px(self, lat, lon):
        return (self.x0 + (np.asarray(lon) - self.lon_min) / self.lon_span * self.mw,
                self.y0 + (self.lat_max - np.asarray(lat)) / self.lat_span * self.mh)

    def sheet_H(self):
        """Camera px -> unit square of the map rectangle (for the left-hand time scale, as with the paper map)."""
        corners = np.float32([[self.x0, self.y0], [self.x0 + self.mw, self.y0],
                              [self.x0 + self.mw, self.y0 + self.mh], [self.x0, self.y0 + self.mh]])
        return cv2.getPerspectiveTransform(corners, UNIT)

    # --- left hand: thumb + index spread = zoom gesture ---
    def handle_left(self, lms, w, h, now=None):
        """-> True while the zoom gesture is active (then the left hand does not control time).
        The zoom follows the thumb-index distance, like a two-finger pinch on a phone screen."""
        import time as _t
        now = _t.monotonic() if now is None else now
        if lms is None:
            self.pinching, self.pinch_pt, self.tips, self._anchor, self._pose_since = False, None, None, None, None
            return False
        p = lambda i: np.array([lms[i].x * w, lms[i].y * h])
        palm = max(np.linalg.norm(p(0) - p(9)), 1.0)
        thumb_out = np.linalg.norm(p(4) - p(10)) / palm
        spread = max(np.linalg.norm(p(4) - p(8)) / palm, self.MIN_SPREAD)
        self.tips = (p(4), p(8))
        self.pinch_pt = (p(4) + p(8)) / 2
        pose = thumb_out > (self.THUMB_OFF if self.pinching else self.THUMB_ON)
        if not pose:
            self.pinching, self._anchor, self._pose_since = False, None, None
            return False
        if not self.pinching:                                    # the pose must be held briefly to start
            self._pose_since = now if self._pose_since is None else self._pose_since
            if now - self._pose_since < self.HOLD_S:
                return True        # thumb is out: not zooming yet, but the left hand must not move the time
            self.pinching = True
        if self._anchor is None:                                 # gesture starts: remember place, zoom, spread
            lat, lon, ok, _ = self.to_latlon(*self.pinch_pt)
            anchor_px = tuple(self.pinch_pt)
            if not ok:                                           # fingers outside the map: zoom about the centre
                anchor_px = (self.x0 + self.mw / 2, self.y0 + self.mh / 2)
                lat, lon, _, _ = self.to_latlon(*anchor_px)
            self._anchor = (anchor_px, (lat, lon), self.zoom, spread)
        (ax, ay), geo, z0, s0 = self._anchor
        self.zoom_about(ax, ay, z0 * (spread / s0) ** self.ZOOM_POWER, geo)
        return True

    # --- drawing ---
    def _grid_step(self):
        for s in (1, 2, 5, 10, 15, 30):
            if self.lon_span / s <= 14:
                return s
        return 30

    def _base(self, layer, t):
        key = (getattr(layer, "name", None), None if layer is None else layer.clamp_frame(t),
               round(self.lon_min, 4), round(self.lat_max, 4), round(self.zoom, 4))
        if key == self._cache_key:
            return self._cache_img
        s = self.step
        nx, ny = self.mw // s, self.mh // s
        lon = self.lon_min + (np.arange(nx) + 0.5) / nx * self.lon_span
        lat = self.lat_max - (np.arange(ny) + 0.5) / ny * self.lat_span
        LON, LAT = np.meshgrid(lon, lat)
        if layer is not None:
            grid = layer.grid(t)
            dlat = abs(layer.lat[1] - layer.lat[0]); dlon = abs(layer.lon[1] - layer.lon[0])
            i = np.clip(np.round((LAT - layer.lat[0]) / dlat).astype(int), 0, len(layer.lat) - 1)
            j = np.round((LON - layer.lon[0]) / dlon).astype(int) % len(layer.lon)
            val = grid[i, j]
            x0, x1, y0, y1 = layer.extent()
            if x1 - x0 < 359:                                  # regional layer: nothing outside its area
                lw = (LON + 180) % 360 - 180
                val = np.where((LAT >= y0) & (LAT <= y1) & (lw >= x0) & (lw <= x1), val, np.nan)
            nrm = layer.norm_array(val)
            rgb = layer.lut[np.nan_to_num(nrm * 255).astype(int)].copy()
            rgb[np.isnan(val)] = (170, 170, 165)
        else:
            rgb = np.full((ny, nx, 3), (60, 120, 185), np.uint8)
        img = cv2.resize(np.ascontiguousarray(rgb[:, :, ::-1]), (self.mw, self.mh), interpolation=cv2.INTER_NEAREST)
        # coastlines (more detail when zoomed in)
        sub = 1 if self.zoom >= 3 else 3
        for r in self.rings:
            if r[:, 0].max() < self.lon_min or r[:, 0].min() > self.lon_min + self.lon_span or \
                    r[:, 1].min() > self.lat_max or r[:, 1].max() < self.lat_max - self.lat_span:
                continue
            x = (r[::sub, 0] - self.lon_min) / self.lon_span * self.mw
            y = (self.lat_max - r[::sub, 1]) / self.lat_span * self.mh
            cv2.polylines(img, [np.int32(np.stack([x, y], 1))], True, (40, 40, 40), 1, cv2.LINE_AA)
        # grid with labels at the edges
        st = self._grid_step()
        for lo in np.arange(np.ceil(self.lon_min / st) * st, self.lon_min + self.lon_span, st):
            x = int((lo - self.lon_min) / self.lon_span * self.mw)
            cv2.line(img, (x, 0), (x, self.mh), (215, 215, 215), 1)
            cv2.putText(img, f"{abs(lo):g}{'E' if lo > 0 else 'W' if lo < 0 else ''}", (x + 3, self.mh - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
        for la in np.arange(np.ceil((self.lat_max - self.lat_span) / st) * st, self.lat_max, st):
            y = int((self.lat_max - la) / self.lat_span * self.mh)
            cv2.line(img, (0, y), (self.mw, y), (215, 215, 215), 1)
            cv2.putText(img, f"{abs(la):g}{'N' if la > 0 else 'S' if la < 0 else ''}", (4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
        self._cache_key, self._cache_img = key, img
        return img

    def draw(self, frame, layer=None, t=None):
        img = self._base(layer, t)
        x0, y0 = self.x0, self.y0
        roi = frame[y0:y0 + self.mh, x0:x0 + self.mw]
        frame[y0:y0 + self.mh, x0:x0 + self.mw] = cv2.addWeighted(img, self.alpha, roi, 1 - self.alpha, 0)
        cv2.rectangle(frame, (x0, y0), (x0 + self.mw, y0 + self.mh), (255, 255, 255), 2)
        label = "whole world" if self.zoom <= 1.001 else f"zoom x{self.zoom:.1f}  (0 = whole world)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x0 + 2, y0 + 2), (x0 + 16 + tw, y0 + 14 + th), (40, 40, 40), -1)
        cv2.putText(frame, label, (x0 + 9, y0 + 8 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        if self.tips is not None and self.pinching:            # zoom gesture: line between thumb and index tips
            col = (191, 106, 37)
            t4, t8 = (tuple(int(v) for v in q) for q in self.tips)
            cv2.line(frame, t4, t8, col, 3, cv2.LINE_AA)
            for q in (t4, t8):
                cv2.circle(frame, q, 9, col, -1, cv2.LINE_AA)
            cv2.putText(frame, "ZOOM: spread = in, together = out", (t8[0] + 14, t8[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
            if self._anchor is not None:                         # where the zoom is anchored
                a = tuple(int(v) for v in self._anchor[0])
                cv2.drawMarker(frame, a, col, cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
