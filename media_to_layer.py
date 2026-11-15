"""Video or picture of a map -> Earth Jukebox layer in data/<name>/.

Works for any quantity drawn with a colour scale (temperature, humidity, aerosols, currents ...);
nothing here is tuned to one particular map. The steps:

  1. Palette - the colour scale as a table "colour -> position 0..1 along the scale". It comes from a
     colour bar found automatically (in the frame itself or in a separate picture), from colours typed
     by hand ("#0000ff:0, #ff0000:32") or from a named colour map (viridis, turbo, jet ...).
     Position -> value: linear, logarithmic (concentrations, aerosols) or steps (classes).
  2. Map area in the frame: found automatically, can be corrected by hand. Region = whole world by default
     (the map must be in the plain lat/lon projection: equal steps in degrees).
  3. Every pixel -> the nearest palette colour in Lab (a perceptual colour space). Pixels far from all
     palette colours are not data: background, land, labels, borders, blurred line edges. Thin things that
     never change between frames (coastlines, a date stamp, a logo) are ignored too.
  4. Pixels -> grid cells (median of the positions on the scale). For sparse pictures (current lines, dots)
     several frames inside each time step are combined; small holes between the data are filled from the
     neighbours, and their share is counted. All of this is done on the scale positions, so changing the
     min/max or linear/log afterwards needs no new decoding.
  5. Saved with layers.write_layer, together with import.json (how it was made) and a preset for the next
     video of the same series.

There is no reference data, so the checks are internal: the share of map pixels that matched the scale,
the share of filled cells, and the decoded map re-coloured with the same palette next to the original
(import_page.py shows them before saving).

Command line (the "Add a map" page of finger_map.py does the same interactively):
    .venv/bin/python media_to_layer.py video.mp4 --name humidity --min 0 --max 100 \\
        --start 2020-01-01 --end 2020-12-31 --check check.png          # only the check picture
    .venv/bin/python media_to_layer.py ... --save                        # and save the layer
"""
import argparse
import json
import re
import shutil
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np

import layers as L

HERE = Path(__file__).resolve().parent
PRESET_DIR = HERE / "import_presets"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".mpg", ".mpeg", ".gif"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
WORLD = [-180.0, 180.0, -90.0, 90.0]           # lon0, lon1, lat0, lat1
MATCH_DE = 10.0     # largest allowed colour difference (ΔE in Lab) between a pixel and the scale; the actual
                    # tolerance is measured per file (clean pictures: ~4, compressed video: higher), see tolerance()
K = 6               # pixels per cell side that are looked at (K*K per cell and frame)
MIN_COUNT = 5       # matched pixels a cell needs to get a value
SCALES = ("linear", "log", "steps")
KEEP = ("all", "ocean", "land")


def default_settings(path):
    stem = Path(path).stem
    return {
        "file": str(path), "name": slug(stem), "title": re.sub(r"[_\-]+", " ", stem).strip().capitalize(),
        "units": "", "source": "",
        "palette": {"kind": "frame", "box": None, "reverse": False},
        "scale": "linear", "vmin": None, "vmax": None,
        "start": "", "end": "", "steps": 48,
        "map_box": None, "region": list(WORLD), "keep": "all",
    }


def slug(s):
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(s).strip().lower()).strip("_")
    return s or "new_map"


# ----------------------------------------------------------------------------
# Colours
# ----------------------------------------------------------------------------
def rgb_to_lab(rgb):
    """uint8 RGB [..., 3] -> float32 Lab (L 0..100)."""
    a = np.asarray(rgb, np.float32)
    shp = a.shape
    return cv2.cvtColor(a.reshape(-1, 1, 3) / 255.0, cv2.COLOR_RGB2Lab).reshape(shp)


def _rgb_hex(c):
    return "#%02x%02x%02x" % tuple(int(v) for v in c)


# ----------------------------------------------------------------------------
# Source: a video or a single picture
# ----------------------------------------------------------------------------
class Source:
    def __init__(self, path):
        self.path = Path(path)
        ext = self.path.suffix.lower()
        if ext in IMAGE_EXT:
            img = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Cannot read the picture {self.path.name}")
            self.image, self.n, self.is_video = cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1, False
        else:
            cap = cv2.VideoCapture(str(self.path))
            ok, fr = cap.read()
            if not ok:
                raise ValueError(f"Cannot read the video {self.path.name}")
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 0
            if n <= 1:                                          # some files do not tell: count
                n = 1
                while cap.grab():
                    n += 1
            cap.release()
            self.image, self.n, self.is_video = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), n, True
        self.h, self.w = self.image.shape[:2]

    def frames(self, indices):
        """(index, RGB frame) for the indices, in ascending order (one pass through the video)."""
        if not self.is_video:
            yield 0, self.image
            return
        cap = cv2.VideoCapture(str(self.path))
        i = 0
        try:
            for target in sorted(set(int(v) for v in indices)):
                while i < target:
                    if not cap.grab():
                        return
                    i += 1
                ok, fr = cap.read()
                if not ok:
                    return
                i += 1
                yield target, cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()


# ----------------------------------------------------------------------------
# Palette: colours along the scale
# ----------------------------------------------------------------------------
class Palette:
    """rgb[N] at positions t[N] (0 = the min end, 1 = the max end).
    anchors: [(t, value)] when the colours carry their own values (typed by hand); otherwise the values
    come from min/max."""

    def __init__(self, rgb, t, anchors=None, origin=""):
        self.rgb = np.asarray(rgb, np.uint8).reshape(-1, 3)
        self.t = np.asarray(t, np.float32)
        self.lab = rgb_to_lab(self.rgb)
        self.anchors = anchors
        self.origin = origin
        self._lut = None

    # --- sources ---
    @classmethod
    def from_bar(cls, img, box, reverse=False, origin="colour bar"):
        x0, y0, x1, y1 = [int(v) for v in box]
        crop = img[y0:y1, x0:x1]
        if crop.size == 0 or min(crop.shape[:2]) < 1:
            raise ValueError("The colour scale box is empty")
        vertical = crop.shape[0] >= crop.shape[1]
        across = crop.shape[1] if vertical else crop.shape[0]
        a = across // 4
        b = max(across - across // 4, a + 1)
        line = np.median(crop[:, a:b], axis=1) if vertical else np.median(crop[a:b, :], axis=0)
        if vertical:
            line = line[::-1]                                   # vertical bars: the minimum is at the bottom
        if reverse:
            line = line[::-1]
        line = line.astype(np.uint8)
        lab = rgb_to_lab(line)
        lo, hi = 0, len(line)                                   # drop an outline / blended pixels at the ends
        d = np.linalg.norm(np.diff(lab, axis=0), axis=1)
        jump = max(3.0, 4 * float(np.median(d)))
        for _ in range(4):
            if hi - lo > 8 and d[lo] > jump:
                lo += 1
            if hi - lo > 8 and d[hi - 2] > jump:
                hi -= 1
        line, lab = line[lo:hi], lab[lo:hi]
        keep = _not_ticks(lab)                                   # tick marks drawn across the bar
        pos_all = (np.arange(lo, hi) - lo) / max(hi - lo - 1, 1)
        line, lab = line[keep], lab[keep]
        if len(line) < 4:
            raise ValueError("The colour scale is too small")
        pos = pos_all[keep]
        if len(line) > 256:
            k = np.round(np.linspace(0, len(line) - 1, 256)).astype(int)
            line, pos = line[k], pos[k]
        return cls(line, pos, origin=origin)

    @classmethod
    def from_stops(cls, text):
        """'#0000ff:0, #00ff00:16, red:32' -> palette with its own values (interpolated in Lab)."""
        from matplotlib.colors import to_rgb
        stops = []
        for part in re.split(r"[,;\n]+", text):
            part = part.strip()
            if not part:
                continue
            m = re.match(r"^(.+?)\s*[:=]\s*([-+0-9.eE]+)$", part)
            if not m:
                raise ValueError(f"Cannot read '{part}': write colour:value, e.g. #ff0000:32")
            stops.append((float(m.group(2)), np.array(to_rgb(m.group(1).strip())) * 255))
        if len(stops) < 2:
            raise ValueError("At least two colours with values are needed")
        stops.sort(key=lambda s: s[0])
        n = len(stops)
        lab = rgb_to_lab(np.array([c for _, c in stops]))
        per = max(2, 256 // (n - 1))
        cols_lab, ts = [], []
        for i in range(n - 1):
            for f in np.linspace(0, 1, per, endpoint=(i == n - 2)):
                cols_lab.append(lab[i] * (1 - f) + lab[i + 1] * f)
                ts.append((i + f) / (n - 1))
        lab_arr = np.array(cols_lab, np.float32).reshape(-1, 1, 3)
        rgb = np.clip(cv2.cvtColor(lab_arr, cv2.COLOR_Lab2RGB).reshape(-1, 3) * 255, 0, 255).round()
        anchors = [(i / (n - 1), v) for i, (v, _) in enumerate(stops)]
        return cls(rgb, ts, anchors=anchors, origin="typed colours")

    @classmethod
    def from_cmap(cls, name):
        import matplotlib
        try:
            cm = matplotlib.colormaps[name]
        except KeyError:
            low = {k.lower(): k for k in matplotlib.colormaps}
            if name.lower() not in low:
                raise ValueError(f"Unknown colour map '{name}' (e.g. viridis, turbo, jet, RdBu_r, YlOrRd)")
            cm = matplotlib.colormaps[low[name.lower()]]
        t = np.linspace(0, 1, 256)
        return cls(np.round(cm(t)[:, :3] * 255), t, origin=f"colour map {cm.name}")

    # --- matching ---
    def lut(self):
        """For every colour of a 64x64x64 RGB cube: position of the nearest palette colour and ΔE to it."""
        if self._lut is None:
            q = np.arange(64, dtype=np.uint8) * 4 + 2
            cube = np.stack(np.meshgrid(q, q, q, indexing="ij"), -1).reshape(-1, 3)
            lab = rgb_to_lab(cube)
            pos = np.empty(len(lab), np.float32)
            de = np.empty(len(lab), np.float32)
            for s in range(0, len(lab), 8192):
                d2 = ((lab[s:s + 8192, None, :] - self.lab[None]) ** 2).sum(-1)
                k = d2.argmin(1)
                pos[s:s + 8192] = self.t[k]
                de[s:s + 8192] = np.sqrt(d2[np.arange(len(k)), k])
            self._lut = (pos, de)
        return self._lut

    def match(self, rgb):
        """RGB image -> (position on the scale 0..1, ΔE to the nearest scale colour)."""
        pos, de = self.lut()
        r = np.asarray(rgb, np.uint8)
        idx = (r[..., 0].astype(np.int32) >> 2 << 12) | (r[..., 1].astype(np.int32) >> 2 << 6) | (r[..., 2] >> 2)
        return pos[idx], de[idx]

    # --- positions -> values / colours ---
    def segments(self, min_de=6.0):
        """Edges (in t) between the colour steps of a stepped scale."""
        d = np.linalg.norm(np.diff(self.lab, axis=0), axis=1)
        cut = np.where(d > min_de)[0]
        return (self.t[cut] + self.t[cut + 1]) / 2

    def snap(self, t):
        """Steps: every position -> the middle of its colour step."""
        edges = np.concatenate([[0.0], self.segments(), [1.0]])
        mids = (edges[:-1] + edges[1:]) / 2
        k = np.clip(np.searchsorted(edges, np.nan_to_num(t), side="right") - 1, 0, len(mids) - 1)
        return np.where(np.isfinite(t), mids[k], np.nan)

    def n_steps(self):
        return len(self.segments()) + 1

    def colours_at(self, t):
        """Positions -> RGB (NaN -> None handled by the caller)."""
        k = np.clip(np.searchsorted(self.t, np.nan_to_num(t)), 0, len(self.t) - 1)
        return self.rgb[k]

    def hex_list(self, n=9):
        k = np.round(np.linspace(0, len(self.rgb) - 1, n)).astype(int)
        return [_rgb_hex(c) for c in self.rgb[k]]


def _not_ticks(lab, k=3):
    """Mask of bar pixels that are not tick marks / labels drawn across the bar: a tick is a short piece that
    differs from the bar on both sides while the bar itself goes on (k pixels before and after are alike)."""
    n = len(lab)
    keep = np.ones(n, bool)
    for i in range(n):
        a, b = max(i - k, 0), min(i + k, n - 1)
        if a == i or b == i:
            continue
        mid = lab[a] + (lab[b] - lab[a]) * (i - a) / (b - a)
        around = np.linalg.norm(lab[a] - lab[b])
        if np.linalg.norm(lab[i] - mid) > max(8.0, 2 * around):
            keep[i] = False
    return keep


def to_values(t, palette, scale, vmin, vmax):
    """Positions on the scale -> values (linear, log or steps)."""
    t = np.asarray(t, np.float64)
    if scale == "steps":
        t = palette.snap(t)
    if palette.anchors:
        ts, vs = zip(*palette.anchors)
    else:
        ts, vs = (0.0, 1.0), (vmin, vmax)
    ts, vs = np.array(ts, float), np.array(vs, float)
    if scale == "log" and np.all(vs > 0):
        return np.exp(np.interp(t, ts, np.log(vs)))
    return np.interp(t, ts, vs)


def value_range(palette, settings):
    if palette.anchors:
        vs = [v for _, v in palette.anchors]
        return float(min(vs)), float(max(vs))
    return settings.get("vmin"), settings.get("vmax")


# ----------------------------------------------------------------------------
# Finding things in a frame
# ----------------------------------------------------------------------------
def find_colour_bar(rgb, min_len_frac=0.12, max_gap=4):
    """A colour bar: a straight strip, the same colour across it and changing along it (tick marks and
    video noise may interrupt it for a few pixels). -> (x0, y0, x1, y1) or None."""
    lab_full = rgb_to_lab(rgb)
    best = None
    for vertical in (True, False):
        lab = lab_full if vertical else lab_full.transpose(1, 0, 2)
        h, w = lab.shape[:2]
        maxw = max(8, int(0.5 * w))
        smooth = cv2.blur(lab, (3, 1))                  # video noise: average 3 pixels across the bar
        same = np.linalg.norm(smooth[:, 1:] - smooth[:, :-1], axis=2) < 6.0
        active, found = {}, []                      # (x0, x1) -> {rows, cols, x0s, x1s, last}
        for y in range(h):
            row = np.concatenate([[0], same[y].astype(np.int8), [0]])
            d = np.diff(row)
            starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]   # runs of "same as the right neighbour"
            taken = set()
            for s0, e0 in zip(starts, ends):
                x0, x1 = int(s0), int(e0)                       # pixels x0..x1 have one colour
                width = x1 - x0 + 1
                if width < 6 or width > maxw:
                    continue
                c = lab[y, x0 + width // 4: x1 - width // 4 + 1].mean(0)
                prev = None
                for dx0 in (0, -1, 1, -2, 2):
                    for dx1 in (0, -1, 1, -2, 2):
                        cand = active.get((x0 + dx0, x1 + dx1))
                        if cand is not None and id(cand) not in taken and np.linalg.norm(c - cand["cols"][-1]) < 30:
                            prev = cand
                            break
                    if prev is not None:
                        break
                if prev is None:
                    prev = {"first": y, "cols": [], "x0s": [], "x1s": []}
                else:
                    active.pop((prev["x0s"][-1], prev["x1s"][-1]), None)
                taken.add(id(prev))
                prev["last"] = y
                prev["cols"].append(c); prev["x0s"].append(x0); prev["x1s"].append(x1)
                active[(x0, x1)] = prev
            for key in [k for k, v in active.items() if y - v["last"] > max_gap]:
                found.append(active.pop(key))
        found.extend(active.values())
        for cand in found:
            y0, y1 = cand["first"], cand["last"]
            x0, x1 = int(np.median(cand["x0s"])), int(np.median(cand["x1s"]))
            length = y1 - y0 + 1
            if length < max(40, min_len_frac * h) or length < 3 * (x1 - x0 + 1) or len(cand["cols"]) < 0.8 * length:
                continue
            cols = np.array(cand["cols"])
            path = float(np.linalg.norm(np.diff(cols, axis=0), axis=1).sum())
            span = float(np.linalg.norm(cols[0] - cols[-1]))
            if path < 40 or span < 25:
                continue
            score = path * np.sqrt(length) * np.sqrt(x1 - x0 + 1)
            if best is None or score > best[0]:
                best = (score, vertical, lab, x0, x1, y0, y1)
    if best is None:
        return None
    _, vertical, lab, x0, x1, y0, y1 = best
    y0, y1 = _extend_bar(lab, x0, x1, y0, y1)
    box = (x0, y0, x1 + 1, y1 + 1) if vertical else (y0, x0, y1 + 1, x1 + 1)
    return tuple(int(v) for v in box)


def _extend_bar(lab, x0, x1, y0, y1):
    """Grow a found bar along its axis while its colour goes on smoothly and is not the background
    beside it (light ends of a scale are easily lost on a light page)."""
    h, w = lab.shape[:2]
    bw = x1 - x0 + 1
    a, b = x0 + bw // 4, x1 - bw // 4 + 1
    lo, hi = max(x0 - 3, 0), min(x1 + 3, w - 1)

    def step(y, prev):
        c = lab[y, a:b]
        m = c.mean(0)
        if np.linalg.norm(c - m, axis=1).max() > 6 or np.linalg.norm(m - prev) > 8:
            return None
        if np.linalg.norm(m - lab[y, lo]) < 4 and np.linalg.norm(m - lab[y, hi]) < 4:
            return None                                   # the same as the page around it
        return m

    for direction in (1, -1):
        y = y1 if direction == 1 else y0
        prev = lab[y, a:b].mean(0)
        while 0 <= y + direction < h:
            m = step(y + direction, prev)
            if m is None:
                break
            prev, y = m, y + direction
        if direction == 1:
            y1 = y
        else:
            y0 = y
    return y0, y1


def tolerance(frames, palette, bg=None):
    """How far from the scale colours a data pixel may be: 3x the distance of the closest quarter of the pixels
    belong to the scale (compression noise), between 4 and MATCH_DE."""
    des = []
    for fr in frames[:8]:
        sub = fr[::2, ::2]
        _, de = palette.match(sub)
        keep = de < 15
        if bg is not None:                                 # the page itself is not data
            keep &= np.abs(sub.astype(np.int32) - bg).max(2) > 16
        des.append(de[keep])
    de = np.concatenate(des) if des else np.array([])
    if len(de) < 100:
        return MATCH_DE
    return float(np.clip(3 * np.percentile(de, 25), 4.0, MATCH_DE))


def background_colour(rgb):
    """Page or letterbox colour: one colour that dominates all four borders. None if the picture
    goes up to the border (a polar strip of one colour on one side is not a page)."""
    sides = [rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]
    b = np.concatenate(sides).astype(np.int32)
    q = b // 8
    keys = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    vals, counts = np.unique(keys, return_counts=True)
    k = vals[counts.argmax()]
    colour = np.median(b[keys == k], axis=0)
    for side in sides:
        if (np.abs(side.astype(np.int32) - colour).max(1) <= 12).mean() < 0.6:
            return None
    return colour


def content_rect(rgb, bg):
    if bg is None:
        return (0, 0, rgb.shape[1], rgb.shape[0])
    diff = np.abs(rgb.astype(np.int32) - bg).max(2) > 16
    rows, cols = np.where(diff.any(1))[0], np.where(diff.any(0))[0]
    if len(rows) == 0:
        return (0, 0, rgb.shape[1], rgb.shape[0])
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def static_overlay(frames):
    """Thin things that are the same in every frame: coastlines, borders, labels, a date stamp."""
    if len(frames) < 6:
        return None
    st = np.stack(frames)
    still = (st.max(0).astype(np.int16) - st.min(0)).max(2) < 10
    big = cv2.morphologyEx(still.astype(np.uint8), cv2.MORPH_OPEN,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    return still & ~big


def data_mask(rgb, palette, overlay=None, bg=None, tol=MATCH_DE):
    _, de = palette.match(rgb)
    m = de <= tol
    if overlay is not None:
        m &= ~overlay
    if bg is not None:
        m &= np.abs(rgb.astype(np.int32) - bg).max(2) > 4
    return m


def find_map_box(frames, palette, overlay, bg, exclude=None, tol=MATCH_DE):
    """The map area: the largest region whose colours belong to the scale, snapped to the picture edges."""
    h, w = frames[0].shape[:2]
    match = np.zeros((h, w), bool)
    for fr in frames:
        match |= data_mask(fr, palette, overlay, bg, tol)
    if exclude is not None:
        x0, y0, x1, y1 = exclude
        p = 6
        match[max(0, y0 - p):y1 + p, max(0, x0 - p):x1 + p] = False
    cx0, cy0, cx1, cy1 = content_rect(frames[0], bg)
    k = max(3, int(0.02 * w))
    closed = cv2.morphologyEx(match.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed)
    if n <= 1:
        return (cx0, cy0, cx1, cy1)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = stats[i, :4]
    x0, y0, x1, y1 = int(x), int(y), int(x + bw), int(y + bh)
    if bg is not None:
        # on a page: the map is the solid non-background block under the data (land and empty cells included)
        nonbg = (np.abs(frames[0].astype(np.int32) - bg).max(2) > 16).astype(np.uint8)
        nonbg = cv2.morphologyEx(nonbg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        nb, lbl, st, _ = cv2.connectedComponentsWithStats(nonbg)
        under = lbl[y0:y1, x0:x1][match[y0:y1, x0:x1]]
        under = under[under > 0]
        if len(under):
            j = int(np.bincount(under).argmax())
            bx, by, bww, bhh = st[j, :4]
            if bx <= x0 + 2 and by <= y0 + 2 and bx + bww >= x1 - 2 and by + bhh >= y1 - 2 \
                    and bww * bhh <= 2.0 * bw * bh:
                return (int(bx), int(by), int(bx + bww), int(by + bhh))
    near_x, near_y = 0.006 * (cx1 - cx0), 0.006 * (cy1 - cy0)
    if x0 - cx0 <= near_x:                                 # data almost up to the picture edge -> the edge
        x0 = cx0
    if cx1 - x1 <= near_x:
        x1 = cx1
    if y0 - cy0 <= near_y:
        y0 = cy0
    if cy1 - y1 <= near_y:
        y1 = cy1
    # a whole-world map fills the picture and is 2:1; data often stops before the poles -> the full height
    if x0 == cx0 and x1 == cx1 and abs((cy1 - cy0) - (cx1 - cx0) / 2) <= 0.06 * (cy1 - cy0):
        y0, y1 = cy0, cy1
    return (x0, y0, x1, y1)


# ----------------------------------------------------------------------------
# Grid and land
# ----------------------------------------------------------------------------
class Grid:
    def __init__(self, region, map_w_px):
        lon0, lon1, lat0, lat1 = [float(v) for v in region]
        px_per_deg = map_w_px / max(lon1 - lon0, 1e-6)
        res = 2.0
        for r in (0.25, 0.5, 1.0, 2.0):
            if r * px_per_deg >= K:
                res = r
                break
        self.region, self.res = [lon0, lon1, lat0, lat1], res
        self.nc = max(1, int(round((lon1 - lon0) / res)))
        self.nr = max(1, int(round((lat1 - lat0) / res)))
        self.lon = lon0 + (np.arange(self.nc) + 0.5) * (lon1 - lon0) / self.nc
        self.lat = lat0 + (np.arange(self.nr) + 0.5) * (lat1 - lat0) / self.nr
        self.world = abs((lon1 - lon0) - 360) < 1e-6

    @property
    def shape(self):
        return (self.nr, self.nc)


_LAND = None


def land_share(grid):
    """Share of land in every cell (Natural Earth outlines, data/ne_50m_land.geojson)."""
    global _LAND
    if _LAND is None:
        img = np.zeros((1800, 3600), np.uint8)            # 0.1 degree, north at the top
        if L.COAST_FILE.exists():
            land = json.loads(L.COAST_FILE.read_text())
            for f in land["features"]:
                g = f["geometry"]
                for p in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
                    ring = np.asarray(p[0], float)
                    pts = np.stack([(ring[:, 0] + 180) * 10, (90 - ring[:, 1]) * 10], 1)
                    cv2.fillPoly(img, [np.round(pts).astype(np.int32)], 255)
        _LAND = img.astype(np.float32) / 255
    lon0, lon1, lat0, lat1 = grid.region
    c0, c1 = int(round((lon0 + 180) * 10)), int(round((lon1 + 180) * 10))
    r0, r1 = int(round((90 - lat1) * 10)), int(round((90 - lat0) * 10))
    crop = _LAND[max(r0, 0):max(r1, r0 + 1), max(c0, 0):max(c1, c0 + 1)]
    share = cv2.resize(crop, (grid.nc, grid.nr), interpolation=cv2.INTER_AREA)
    return share[::-1]                                     # south first, like the grid


# ----------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------
def decode_frame(rgb, palette, box, grid, overlay=None, bg=None, tol=MATCH_DE):
    """One frame -> positions on the scale per cell sample: [lat south->north][lon][K*K] (NaN = no data)."""
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1]
    size = (grid.nc * K, grid.nr * K)
    img = cv2.resize(crop, size, interpolation=cv2.INTER_NEAREST)
    t, de = palette.match(img)
    bad = de > tol
    if overlay is not None:
        bad |= cv2.resize(overlay[y0:y1, x0:x1].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    if bg is not None:
        bad |= np.abs(img.astype(np.int32) - bg).max(2) <= 4
    t = np.where(bad, np.nan, t).astype(np.float32)
    blocks = t.reshape(grid.nr, K, grid.nc, K).transpose(0, 2, 1, 3).reshape(grid.nr, grid.nc, K * K)
    return blocks[::-1]


def _nanmedian(a, axis):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def fill_holes(t, world=False, reach=3):
    """Holes between data (up to `reach` cells from it on both sides) <- weighted mean of the neighbours.
    Big empty areas (land under an ocean map, places without data) stay empty. -> (filled, mask of filled)."""
    from scipy.ndimage import gaussian_filter
    known = np.isfinite(t)
    if not known.any():
        return t, np.zeros_like(known)
    ksz = 2 * reach + 1
    pad = ksz if world else 0
    kp = np.pad(known, ((0, 0), (pad, pad)), mode="wrap") if world else known
    hull = cv2.morphologyEx(kp.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((ksz, ksz), np.uint8)).astype(bool)
    hull = hull[:, pad:hull.shape[1] - pad] if world else hull
    target = hull & ~known
    out = np.where(known, t, 0.0).astype(np.float64)
    done = known.copy()
    modes = ("nearest", "wrap" if world else "nearest")
    for s in (1.0, 2.0, 3.0):
        w = done.astype(np.float64)
        num = gaussian_filter(out * w, s, mode=modes)
        den = gaussian_filter(w, s, mode=modes)
        new = target & ~done & (den > 0.05)
        out = np.where(new, num / np.maximum(den, 1e-9), out)
        done |= new
    filled = done & ~known
    return np.where(done, out, np.nan).astype(np.float32), filled


def parse_date(s, end=False):
    s = str(s).strip()
    if not s:
        return None
    m = re.match(r"^(\d{4})(?:[-./](\d{1,2}))?(?:[-./](\d{1,2}))?$", s)
    if not m:
        m2 = re.match(r"^(\d{1,2})[./](\d{1,2})[./](\d{4})$", s)        # 28.03.2007
        if not m2:
            raise ValueError(f"Cannot read the date '{s}' (write 2007-03-28, 2007-03 or 2007)")
        return date(int(m2.group(3)), int(m2.group(2)), int(m2.group(1)))
    y, mo, d = int(m.group(1)), m.group(2), m.group(3)
    if mo is None:
        return date(y, 12, 31) if end else date(y, 1, 1)
    mo = int(mo)
    if d is None:
        if end:
            nxt = date(y + (mo == 12), mo % 12 + 1, 1)
            return nxt - timedelta(days=1)
        return date(y, mo, 1)
    return date(y, mo, int(d))


class Importer:
    """One file being turned into a layer. analyze() -> decode() -> values() / save()."""

    def __init__(self, path, settings=None):
        self.src = Source(path)
        self.s = default_settings(path)
        if settings:
            self.s.update({k: v for k, v in settings.items() if k != "file"})
            self.s["file"] = str(path)
        self.palette = None
        self.bar_box = None              # where the colour scale is (in the frame or in the palette picture)
        self.bar_image = None            # RGB picture the bar was read from
        self.samples, self.sample_idx = [], []
        self.overlay = self.bg = None
        self.tol = MATCH_DE
        self.grid = None
        self.t = self.filled = self.centers = None
        self.stats = {}
        self.warnings = []

    # --- step 1: look at the file ---
    def analyze(self, progress=None, redetect=False):
        """Sample frames, find the colour scale and the map area, how sparse the picture is."""
        src = self.src
        n_s = min(src.n, 24)
        idx = np.unique(np.round(np.linspace(0, src.n - 1, n_s)).astype(int)) if src.is_video else [0]
        self.samples, self.sample_idx = [], []
        for k, (i, fr) in enumerate(src.frames(idx)):
            self.samples.append(fr)
            self.sample_idx.append(i)
            if progress:
                progress(f"Looking at the {'video' if src.is_video else 'picture'}", (k + 1) / len(idx) * 0.5)
        if not self.samples:
            raise ValueError("No frames could be read")
        first = self.samples[0]
        self.bg = background_colour(first)
        self.overlay = static_overlay(self.samples)
        self.build_palette(redetect=redetect)
        self.tol = tolerance(self.samples, self.palette, self.bg)
        if progress:
            progress("Finding the map", 0.7)
        if redetect or not self._box_ok(self.s.get("map_box"), first):
            exclude = self.bar_box if self.s["palette"]["kind"] == "frame" else None
            self.s["map_box"] = list(find_map_box(self.samples, self.palette, self.overlay, self.bg, exclude, self.tol))
        self._measure()
        if progress:
            progress("Done", 1.0)

    def refresh(self, what, progress=None):
        """Recompute after a change: "first" (start), "all" (find everything again), "palette" (scale box,
        flip), "map" (map box), "decode" (steps, region, keep)."""
        if what in ("first", "all"):
            self.analyze(progress, redetect=(what == "all"))
        elif what == "palette":
            self.build_palette()
            self.tol = tolerance(self.samples, self.palette, self.bg)
            if not self._box_ok(self.s.get("map_box"), self.samples[0]):
                exclude = self.bar_box if self.s["palette"]["kind"] == "frame" else None
                self.s["map_box"] = list(find_map_box(self.samples, self.palette, self.overlay, self.bg,
                                                      exclude, self.tol))
            self._measure()
        elif what == "map":
            self._measure()
        self.decode(progress)

    def _box_ok(self, box, img):
        if not box:
            return False
        x0, y0, x1, y1 = box
        return 0 <= x0 < x1 <= img.shape[1] and 0 <= y0 < y1 <= img.shape[0] and x1 - x0 > 8 and y1 - y0 > 8

    def build_palette(self, redetect=False):
        p = self.s["palette"]
        kind = p.get("kind", "frame")
        self.bar_box = self.bar_image = None
        if kind in ("frame", "image"):
            img = self.samples[0] if kind == "frame" else _read_rgb(p["image"])
            box = p.get("box")
            if redetect or not self._box_ok(box, img):
                box = find_colour_bar(img)
                if box is None:
                    raise ValueError("No colour scale found" + (" in the frame" if kind == "frame" else
                                                                  " in the picture") +
                                     ". Draw a box around it (key C) or choose another way to set the colours.")
                p["box"] = list(box)
            self.bar_box, self.bar_image = tuple(box), img
            self.palette = Palette.from_bar(img, box, p.get("reverse", False),
                                            origin="colour bar in the frame" if kind == "frame" else
                                            f"colour bar from {Path(p['image']).name}")
        elif kind == "stops":
            self.palette = Palette.from_stops(p["text"])
        elif kind == "cmap":
            self.palette = Palette.from_cmap(p["name"])
            if p.get("reverse"):
                self.palette = Palette(self.palette.rgb[::-1], self.palette.t, origin=self.palette.origin + " (reversed)")
        else:
            raise ValueError(f"Unknown colour scale source: {kind}")

    def _measure(self):
        """Share of matched pixels in the map, sparse or dense, frames with data (titles, ends)."""
        x0, y0, x1, y1 = self.s["map_box"]
        shares, interior = [], []
        for fr in self.samples:
            m = data_mask(fr[y0:y1, x0:x1], self.palette,
                          None if self.overlay is None else self.overlay[y0:y1, x0:x1], self.bg, self.tol)
            shares.append(float(m.mean()))
            if m.any():
                inner = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                interior.append(inner.sum() / m.sum())
        shares = np.array(shares)
        self.stats["match_share"] = float(np.median(shares))
        self.stats["tolerance"] = self.tol
        self.stats["interior"] = float(np.median(interior)) if interior else 0.0
        self.sparse = self.stats["interior"] < 0.7
        # frames at the start/end with almost no data (titles, credits) are left out of the time range
        lo, hi = 0, self.src.n - 1
        if self.src.is_video and shares.max() > 0:
            good = shares >= 0.3 * np.median(shares[shares > 0])
            if good.any():
                g = np.where(good)[0]
                if g[0] > 0:
                    lo = (self.sample_idx[g[0] - 1] + self.sample_idx[g[0]]) // 2
                if g[-1] < len(shares) - 1:
                    hi = (self.sample_idx[g[-1]] + self.sample_idx[g[-1] + 1]) // 2
        self.frame_range = (int(lo), int(hi))
        if self.stats["match_share"] < 0.05:
            self.warnings.append("Very few map pixels match the colour scale - check the scale and the map box.")

    # --- step 2: decode ---
    def decode(self, progress=None):
        s, src = self.s, self.src
        box = s["map_box"]
        self.grid = grid = Grid(s["region"], box[2] - box[0])
        if src.is_video:
            lo, hi = self.frame_range
            steps = int(max(1, min(int(s.get("steps") or 48), hi - lo + 1)))
            edges = np.linspace(lo, hi + 1, steps + 1)
            window = 5 if self.sparse else 1
            plan = []
            for k in range(steps):
                a, b = edges[k], edges[k + 1] - 1
                idx = [int(round((a + b) / 2))] if window == 1 else \
                    sorted(set(np.round(np.linspace(a, b, window)).astype(int).tolist()))
                plan.append(idx)
            self.centers = [(edges[k] + edges[k + 1] - 1) / 2 for k in range(steps)]
        else:
            plan, self.centers, steps = [[0]], [0.0], 1
        frame_step = {i: k for k, idx in enumerate(plan) for i in idx}
        acc = [[] for _ in range(steps)]
        out = np.full((steps,) + grid.shape, np.nan, np.float32)
        cover = np.zeros((steps,) + grid.shape, np.float32)
        total = sum(len(p) for p in plan)
        done = 0
        for i, fr in src.frames(sorted(frame_step)):
            k = frame_step[i]
            acc[k].append(decode_frame(fr, self.palette, box, grid, self.overlay, self.bg, self.tol))
            done += 1
            if len(acc[k]) == len(plan[k]):                   # step complete -> reduce
                blocks = np.concatenate(acc[k], axis=2)
                cnt = np.isfinite(blocks).sum(2)
                med = _nanmedian(blocks, axis=2)
                out[k] = np.where(cnt >= MIN_COUNT, med, np.nan)
                cover[k] = cnt / blocks.shape[2]
                acc[k] = []
            if progress:
                progress(f"Reading frame {done} of {total}", done / total * 0.9)
        keep = s.get("keep", "all")
        land = land_share(grid) if keep != "all" else None
        filled = np.zeros(out.shape, bool)
        for k in range(steps):
            if keep == "ocean":
                out[k][land > 0.5] = np.nan
            elif keep == "land":
                out[k][land < 0.5] = np.nan
            out[k], filled[k] = fill_holes(out[k], grid.world)
            if keep == "ocean":
                out[k][land > 0.5] = np.nan; filled[k][land > 0.5] = False
            elif keep == "land":
                out[k][land < 0.5] = np.nan; filled[k][land < 0.5] = False
        self.t, self.filled = out, filled
        has = np.isfinite(out)
        self.stats.update({
            "cells_with_data": float(has.mean()),
            "filled_share": float(filled.sum() / max(has.sum(), 1)),
            "steps": steps, "frames_per_step": len(plan[0]),
            "grid_res": grid.res,
        })
        empty = [k for k in range(steps) if has[k].mean() < 0.2 * max(has.mean(), 1e-9)]
        self.warnings = [w for w in self.warnings if not w.startswith("Time steps")]
        if empty:
            self.warnings.append(f"Time steps with almost no data: {', '.join(str(k + 1) for k in empty[:8])}"
                                 + (" ..." if len(empty) > 8 else ""))
        if progress:
            progress("Done", 1.0)

    # --- results ---
    def range(self):
        return value_range(self.palette, self.s)

    def ready(self):
        """-> None if the layer can be saved, else what is missing (a sentence)."""
        vmin, vmax = self.range()
        if vmin is None or vmax is None:
            return "Set the values at the two ends of the colour scale (field Range)."
        if vmin == vmax:
            return "The two ends of the colour scale must have different values."
        if self.s["scale"] == "log" and min(vmin, vmax) <= 0:
            return "A logarithmic scale needs values above zero."
        if self.src.is_video and (not self.s.get("start") or not self.s.get("end")):
            return "Set the dates of the first and the last frame (field Dates)."
        if self.t is None:
            return "The map is not decoded yet."
        if not np.isfinite(self.t).any():
            return "No data was decoded - check the colour scale and the map box."
        return None

    def values(self):
        vmin, vmax = self.range()
        if vmin is None or vmax is None:
            vmin, vmax = 0.0, 1.0
        return to_values(self.t, self.palette, self.s["scale"], vmin, vmax).astype(np.float32)

    def dates(self):
        s = self.s
        if not self.src.is_video:
            d = parse_date(s.get("start", ""))
            return None if d is None else [d.isoformat()]
        d0, d1 = parse_date(s["start"]), parse_date(s["end"], end=True)
        lo, hi = self.frame_range
        out = []
        for c in self.centers:
            f = (c - lo) / max(hi - lo, 1)
            out.append((d0 + timedelta(days=round((d1 - d0).days * f))).isoformat())
        return out

    def save(self, out_dir=L.DATA_DIR):
        msg = self.ready()
        if msg:
            raise ValueError(msg)
        s = self.s
        name = slug(s["name"])
        vmin, vmax = self.range()
        lo_v, hi_v = min(vmin, vmax), max(vmin, vmax)
        vals = self.values()
        dates = self.dates()
        years = None if dates is None else [int(d[:4]) for d in dates]
        scale = s["scale"]
        kind = "video" if self.src.is_video else "picture"
        extra = {
            "cmap": self.palette.hex_list(9) if vmin <= vmax else self.palette.hex_list(9)[::-1],
            "fmt": _fmt_for(lo_v, hi_v, scale),
            "missing_label": "no data in the picture",
            "note": f"decoded from the colours of a {kind}" + ("; frame dates are approximate" if self.src.is_video else ""),
            "speech_note": f"The value is read from the colours of a {kind}.",
        }
        if scale == "log":
            extra["scale"] = "log"
        if not self.src.is_video:                              # one moment: say when (or that it is a picture)
            extra["static_label"] = dates[0] if dates else "from a picture"
        if dates is not None and len(dates) > 1:
            extra["default_frame"] = 0
        folder = L.write_layer(out_dir, name, years, self.grid.lat, self.grid.lon, vals,
                               units=s["units"], title=s["title"] or name, source=s["source"],
                               vmin=lo_v, vmax=hi_v, extra_cfg=extra,
                               dates=dates if dates is not None and len(dates) > 1 else None,
                               scale="log" if scale == "log" else "linear")
        info = {"file": str(self.src.path), "settings": _jsonable(s), "palette": self.palette.origin,
                "grid_res_deg": self.grid.res, "frame_range": list(self.frame_range),
                "stats": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.stats.items()},
                "warnings": self.warnings, "made": date.today().isoformat()}
        (Path(folder) / "import.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
        return Path(folder)


def _fmt_for(lo, hi, scale):
    if scale == "log":
        return "{:.3g}"
    span = abs(hi - lo) or 1.0
    dec = int(np.clip(2 - np.floor(np.log10(span)), 0, 3))
    return "{:.%df}" % dec


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def _read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read the picture {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------
# Presets: settings of a series (same colour scale, same layout)
# ----------------------------------------------------------------------------
PRESET_KEYS = ("title", "units", "source", "palette", "scale", "vmin", "vmax", "steps", "map_box", "region", "keep")


def save_preset(imp, name=None):
    PRESET_DIR.mkdir(exist_ok=True)
    name = slug(name or imp.s["name"])
    p = {k: _jsonable(imp.s.get(k)) for k in PRESET_KEYS}
    p["frame_size"] = [imp.src.w, imp.src.h]
    pal = p.get("palette") or {}
    if pal.get("kind") == "image" and pal.get("image"):              # keep the scale picture with the preset
        src = Path(pal["image"])
        dst = PRESET_DIR / f"{name}_scale{src.suffix.lower()}"
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        pal["image"] = str(dst)
    (PRESET_DIR / f"{name}.json").write_text(json.dumps(p, ensure_ascii=False, indent=2))
    return name


def list_presets():
    return sorted(p.stem for p in PRESET_DIR.glob("*.json")) if PRESET_DIR.exists() else []


def load_preset(name, frame_size=None):
    p = json.loads((PRESET_DIR / f"{name}.json").read_text())
    fs = p.pop("frame_size", None)
    if frame_size is not None and fs is not None and list(fs) != list(frame_size):
        p["map_box"] = None                                   # another frame size: find the map again
        if p.get("palette", {}).get("kind") == "frame":
            p["palette"]["box"] = None
    return p


# ----------------------------------------------------------------------------
# Pictures for checking (used by the page and by --check)
# ----------------------------------------------------------------------------
_RINGS = None


def coast_rings():
    global _RINGS
    if _RINGS is None:
        _RINGS = []
        if L.COAST_FILE.exists():
            for f in json.loads(L.COAST_FILE.read_text())["features"]:
                g = f["geometry"]
                for p in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
                    _RINGS.append(np.asarray(p[0], float))
    return _RINGS


def render_decoded(imp, k, size):
    """Decoded step k re-coloured with the same palette (north up) + coastlines. RGB uint8, size=(w, h)."""
    w, h = size
    t = imp.t[k][::-1]
    rgb = imp.palette.colours_at(t).copy()
    rgb[~np.isfinite(t)] = L.hex_rgb(L.LANDC)
    img = cv2.resize(rgb.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    lon0, lon1, lat0, lat1 = imp.grid.region
    segs = []
    for r in coast_rings():
        if r[:, 0].max() < lon0 or r[:, 0].min() > lon1 or r[:, 1].max() < lat0 or r[:, 1].min() > lat1:
            continue
        x = (r[:, 0] - lon0) / (lon1 - lon0) * w
        y = (lat1 - r[:, 1]) / (lat1 - lat0) * h
        segs.append(np.round(np.stack([x, y], 1)).astype(np.int32))
    cv2.polylines(img, segs, True, (40, 40, 40), 1, cv2.LINE_AA)
    return img


def check_picture(imp, k=0, path=None):
    """Original frame with the boxes | decoded and re-coloured map. -> BGR image (and written to path)."""
    i = int(round(imp.centers[k])) if imp.src.is_video else 0
    fr = next(imp.src.frames([i]))[1].copy()
    if imp.s["map_box"]:
        x0, y0, x1, y1 = imp.s["map_box"]
        cv2.rectangle(fr, (x0, y0), (x1 - 1, y1 - 1), (37, 106, 191), 3)
    if imp.bar_box is not None and imp.s["palette"]["kind"] == "frame":
        x0, y0, x1, y1 = imp.bar_box
        cv2.rectangle(fr, (x0, y0), (x1 - 1, y1 - 1), (208, 59, 59), 3)
    h = 420
    left = cv2.resize(fr, (int(fr.shape[1] * h / fr.shape[0]), h), interpolation=cv2.INTER_AREA)
    lon0, lon1, lat0, lat1 = imp.grid.region
    right = render_decoded(imp, k, (int(h * (lon1 - lon0) / (lat1 - lat0)), h))
    gap = np.full((h, 12, 3), 255, np.uint8)
    out = cv2.cvtColor(np.concatenate([left, gap, right], 1), cv2.COLOR_RGB2BGR)
    if path:
        cv2.imwrite(str(path), out)
    return out


def summary(imp):
    st = imp.stats
    parts = [f"{st.get('match_share', 0) * 100:.0f}% of the map pixels match the colour scale"]
    if "cells_with_data" in st:
        parts.append(f"{st['cells_with_data'] * 100:.0f}% of the cells have data")
        parts.append(f"{st['filled_share'] * 100:.0f}% of them filled from neighbours")
    if imp.src.is_video and "steps" in st:
        parts.append(f"{st['steps']} time steps" + (f", {st['frames_per_step']} frames each"
                                                     if st.get("frames_per_step", 1) > 1 else ""))
    return "; ".join(parts) + "."


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="video or picture of a map")
    ap.add_argument("--preset", help="settings saved after an earlier import (import_presets/<name>.json)")
    ap.add_argument("--name"); ap.add_argument("--title"); ap.add_argument("--units"); ap.add_argument("--source")
    ap.add_argument("--palette", default=None,
                    help="where the colours come from: frame (a colour bar in the frame, default), a picture "
                         "file with the colour bar, cmap:NAME (e.g. cmap:turbo) or stops:'#0000ff:0,#ff0000:32'")
    ap.add_argument("--palette-box", help="x0,y0,x1,y1 of the colour bar in pixels (default: found automatically)")
    ap.add_argument("--reverse", action="store_true", help="the colour bar goes from max to min")
    ap.add_argument("--scale", choices=SCALES)
    ap.add_argument("--min", type=float, help="value at the low end of the colour bar")
    ap.add_argument("--max", type=float, help="value at the high end of the colour bar")
    ap.add_argument("--start", help="date of the first frame (video) or of the picture")
    ap.add_argument("--end", help="date of the last frame (video)")
    ap.add_argument("--steps", type=int, help="time steps to take from a video (default 48)")
    ap.add_argument("--map-box", help="x0,y0,x1,y1 of the map in pixels (default: found automatically)")
    ap.add_argument("--region", help="lon0,lon1,lat0,lat1 shown by the map (default: whole world)")
    ap.add_argument("--keep", choices=KEEP, help="keep all cells, only ocean or only land")
    ap.add_argument("--check", help="write a check picture (original | decoded) to this file")
    ap.add_argument("--save", action="store_true", help="save the layer to data/<name>/ and a preset")
    ap.add_argument("--out", default=str(L.DATA_DIR))
    a = ap.parse_args()

    src = Source(a.file)
    settings = load_preset(a.preset, (src.w, src.h)) if a.preset else {}
    imp = Importer(a.file, settings)
    s = imp.s
    for k in ("name", "title", "units", "source", "scale", "start", "end", "steps", "keep"):
        if getattr(a, k) is not None:
            s[k] = getattr(a, k)
    if a.min is not None:
        s["vmin"] = a.min
    if a.max is not None:
        s["vmax"] = a.max
    if a.palette:
        if a.palette == "frame":
            s["palette"] = {"kind": "frame", "box": None}
        elif a.palette.startswith("cmap:"):
            s["palette"] = {"kind": "cmap", "name": a.palette[5:]}
        elif a.palette.startswith("stops:"):
            s["palette"] = {"kind": "stops", "text": a.palette[6:]}
        else:
            s["palette"] = {"kind": "image", "image": a.palette, "box": None}
    if a.palette_box:
        s["palette"]["box"] = [int(v) for v in a.palette_box.split(",")]
    if a.reverse:
        s["palette"]["reverse"] = True
    if a.map_box:
        s["map_box"] = [int(v) for v in a.map_box.split(",")]
    if a.region:
        s["region"] = [float(v) for v in a.region.split(",")]

    say = lambda text, f: print(f"\r  {text} ({f * 100:.0f}%)   ", end="", flush=True)
    imp.analyze(say)
    print(f"\nColour scale: {imp.palette.origin}" + (f", box {imp.bar_box}" if imp.bar_box else "")
          + f"; map box {s['map_box']}; {'sparse' if imp.sparse else 'filled'} picture")
    imp.decode(say)
    print("\n" + summary(imp))
    for w in imp.warnings:
        print("Warning:", w)
    if a.check:
        check_picture(imp, len(imp.centers) // 2, a.check)
        print(f"Check picture: {a.check}")
    if a.save:
        folder = imp.save(a.out)
        print(f"Saved: {folder}; preset: {save_preset(imp)}")
    else:
        msg = imp.ready()
        print(msg or "Ready to save (add --save).")


if __name__ == "__main__":
    _cli()
