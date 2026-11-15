"""Start menu page "Add a map": a video or a picture of a map -> a new data layer (see media_to_layer.py).

1. Choose the file (the usual Finder window).
2. Where the colours come from: the colour scale in the frame / a separate picture of the scale /
   colours typed by hand / a named colour map / a preset saved after an earlier import.
3. Automatic decoding, then the check page:
     left  - the original, blue box = map, red box = colour scale (drag the corners to fix them,
             M or C and drag to draw them again);
     right - what was read, painted with the same colours, with coastlines on top;
     below - the colour scale, the numbers (how much matched, how much was filled), the fields.
   Fields are edited in a small macOS text window (any language, works with VoiceOver).
   Enter saves the layer to data/<name>/ and a preset; the map is in the list right away.
"""
import subprocess
import sys
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np

import layers as L
import media_to_layer as M

WIN = "Add a map"
W, H = 1280, 800
LEFT, RIGHT = (24, 56, 624, 356), (656, 56, 1256, 356)        # x0, y0, x1, y1 of the two pictures
BLUE, RED = (37, 106, 191), (208, 59, 59)
_LEFT_KEYS = {63234, 2424832, 65361}
_RIGHT_KEYS = {63235, 2555904, 65363}


# ----------------------------------------------------------------------------
# macOS dialogs (a terminal prompt elsewhere)
# ----------------------------------------------------------------------------
def _q(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _osa(script):
    r = subprocess.run(["osascript", "-e", "activate", "-e", script], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def choose_file(prompt):
    if sys.platform != "darwin":
        return input(prompt + " (path, empty = cancel): ").strip() or None
    return _osa(f"POSIX path of (choose file with prompt {_q(prompt)})") or None


def ask_text(prompt, default=""):
    """-> the text, or None if cancelled."""
    if sys.platform != "darwin":
        v = input(f"{prompt} [{default}]: ")
        return v if v else default
    return _osa(f'text returned of (display dialog {_q(prompt)} default answer {_q(default)} '
                f'with title "Earth Jukebox")')


def ask_yes(prompt, yes="Yes"):
    if sys.platform != "darwin":
        return input(prompt + " [y/N] ").strip().lower().startswith("y")
    out = _osa(f'button returned of (display dialog {_q(prompt)} buttons {{"Cancel", {_q(yes)}}} '
               f'default button {_q(yes)} with title "Earth Jukebox")')
    return out == yes


# ----------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------
_F = {}


def fonts():
    if not _F:
        from matplotlib import font_manager
        from PIL import ImageFont
        reg = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
        bold = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
        for k, (p, size) in {"xs": (reg, 13), "s": (reg, 15), "sb": (bold, 15), "m": (bold, 19),
                             "l": (bold, 26)}.items():
            _F[k] = ImageFont.truetype(p, size)
    return _F


def _fit(img_w, img_h, rect):
    """Scale and offset to fit an image into rect, keeping its shape."""
    x0, y0, x1, y1 = rect
    sc = min((x1 - x0) / img_w, (y1 - y0) / img_h)
    w, h = int(img_w * sc), int(img_h * sc)
    return sc, x0 + ((x1 - x0) - w) // 2, y0 + ((y1 - y0) - h) // 2, w, h


class Job:
    """Runs fn(progress) in the background; the page keeps drawing (progress bar) meanwhile."""

    def __init__(self, fn, text="Working"):
        self.text, self.frac, self.error, self.done = text, 0.0, None, False
        threading.Thread(target=self._run, args=(fn,), daemon=True).start()

    def progress(self, text, frac):
        self.text, self.frac = text, frac

    def _run(self, fn):
        try:
            fn(self.progress)
        except Exception as e:                                  # shown on the page and spoken
            self.error = str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}"
            if not isinstance(e, ValueError):
                traceback.print_exc()
        self.done = True


# ----------------------------------------------------------------------------
# Step 2: where the colours come from
# ----------------------------------------------------------------------------
def _source_options(imp):
    what = "video" if imp.src.is_video else "picture"
    opts = [("frame", f"The colour scale is in the {what}"),
            ("image", "The colour scale is in a separate picture"),
            ("stops", "Type the colours and their values"),
            ("cmap", "Name of a colour map (viridis, turbo ...)")]
    opts += [("preset:" + p, f"Preset: {p}") for p in M.list_presets()[:5]]
    return opts


def _render_sources(imp, opts):
    from PIL import Image, ImageDraw
    f = fonts()
    img = Image.new("RGB", (W, H), L.hex_rgb(L.SURFACE))
    d = ImageDraw.Draw(img)
    d.text((40, 30), "Add a map: where do the colours come from?", font=f["l"], fill=L.hex_rgb(L.INK))
    d.text((40, 70), Path(imp.src.path).name, font=f["s"], fill=L.hex_rgb(L.MUTED))
    rows = []
    for k, (_, title) in enumerate(opts):
        y = 120 + k * 64
        rows.append((40, y, 660, y + 54))
        d.rounded_rectangle([40, y, 660, y + 54], radius=10, outline=L.hex_rgb(L.GRIDC), width=2)
        d.text((60, y + 13), str(k + 1), font=f["l"], fill=L.hex_rgb(L.INK2))
        d.text((110, y + 15), title, font=f["m"], fill=L.hex_rgb(L.INK))
    fr = imp.src.image
    sc, ox, oy, w, h = _fit(fr.shape[1], fr.shape[0], (700, 120, 1240, 520))
    img.paste(Image.fromarray(cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)), (ox, oy))
    d.text((40, H - 50), "Number or click - choose     Esc - cancel", font=f["s"], fill=L.hex_rgb(L.MUTED))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), rows


def choose_source(imp, voice):
    """-> True when imp.s["palette"] (or a whole preset) is set, False if cancelled."""
    opts = _source_options(imp)
    spoken = " ".join(f"{k + 1}: {t}." for k, (_, t) in enumerate(opts))
    voice.speak(f"Where do the colours come from? {spoken} Escape to cancel.")
    clicks = []
    cv2.setMouseCallback(WIN, lambda ev, x, y, *_: clicks.append((x, y)) if ev == cv2.EVENT_LBUTTONUP else None)
    while True:
        frame, rows = _render_sources(imp, opts)
        cv2.imshow(WIN, frame)
        key = cv2.waitKeyEx(30)
        choice = None
        if key in (27, ord("q")):
            return False
        if 0 <= key < 256 and ord("1") <= key < ord("1") + len(opts):
            choice = key - ord("1")
        while clicks:
            x, y = clicks.pop()
            for k, (x0, y0, x1, y1) in enumerate(rows):
                if x0 <= x <= x1 and y0 <= y <= y1:
                    choice = k
        if choice is None:
            continue
        kind = opts[choice][0]
        voice.stop()
        try:
            if kind == "frame":
                imp.s["palette"] = {"kind": "frame", "box": None, "reverse": False}
            elif kind == "image":
                p = choose_file("Choose the picture with the colour scale")
                if not p:
                    continue
                imp.s["palette"] = {"kind": "image", "image": p, "box": None, "reverse": False}
            elif kind == "stops":
                t = ask_text("Colours and their values, from low to high. Example:\n"
                             "#313695:0, #ffffbf:15, #a50026:30   (names like blue:0 work too)", "")
                if not t:
                    continue
                M.Palette.from_stops(t)                          # check now: a clear message if it is wrong
                imp.s["palette"] = {"kind": "stops", "text": t}
            elif kind == "cmap":
                t = ask_text("Name of the colour map (matplotlib), e.g. viridis, turbo, jet, YlOrRd, RdBu_r", "viridis")
                if not t:
                    continue
                M.Palette.from_cmap(t.strip())
                imp.s["palette"] = {"kind": "cmap", "name": t.strip(), "reverse": False}
            else:
                imp.s.update(M.load_preset(kind[len("preset:"):], (imp.src.w, imp.src.h)))
            return True
        except ValueError as e:
            print(e)
            voice.speak(str(e))


# ----------------------------------------------------------------------------
# Step 3: the check page
# ----------------------------------------------------------------------------
class CheckPage:
    def __init__(self, imp, voice):
        self.imp, self.voice = imp, voice
        self.job = None
        self.k = 0                                  # time step shown
        self.view = "frame"                         # left picture: "frame" or "scale" (separate scale picture)
        self.draw = None                            # "map" / "bar": the next drag draws this box anew
        self.drag = None                            # (which box, mode, start point, box at start)
        self.message, self.msg_warn = "", False
        self.fields = []
        self.pending_field = None
        self._cache = {}

    # --- background work ---
    def start(self, what, text):
        self.job = Job(lambda progress: self.imp.refresh(what, progress), text)

    def job_finished(self):
        job, self.job = self.job, None
        imp = self.imp
        if job.error:
            self.say(job.error, warn=True)
            return
        self.k = min(self.k, len(imp.centers) - 1)
        missing = imp.ready()
        text = "Check the map. On the left the original, on the right what was read from it. " + M.summary(imp)
        if imp.warnings:
            text += " " + " ".join(imp.warnings)
        text += " " + (missing or "Press Enter to save.") + " Question mark for help."
        self.say(text, warn=bool(missing))

    def say(self, text, warn=False, show=None):
        self.message, self.msg_warn = (show or text), warn
        print(text)
        self.voice.speak(text)

    # --- geometry of the left picture ---
    def left_image(self):
        imp = self.imp
        if self.view == "scale" and imp.s["palette"].get("kind") == "image" and imp.bar_image is not None:
            return imp.bar_image
        if imp.samples:
            if imp.src.is_video and imp.centers is not None and len(imp.sample_idx) > 1:
                c = imp.centers[min(self.k, len(imp.centers) - 1)]
                return imp.samples[int(np.argmin(np.abs(np.array(imp.sample_idx) - c)))]
            return imp.samples[0]
        return imp.src.image

    def boxes(self):
        """Boxes on the left picture: {"map": box, "bar": box} (only those that belong to what is shown)."""
        imp, s = self.imp, self.imp.s
        kind = s["palette"].get("kind")
        out = {}
        if self.view == "frame":
            if s.get("map_box"):
                out["map"] = list(s["map_box"])
            if kind == "frame" and s["palette"].get("box"):
                out["bar"] = list(s["palette"]["box"])
        elif kind == "image" and s["palette"].get("box"):
            out["bar"] = list(s["palette"]["box"])
        return out

    def to_img(self, x, y):
        img = self.left_image()
        sc, ox, oy, _, _ = _fit(img.shape[1], img.shape[0], LEFT)
        return (x - ox) / sc, (y - oy) / sc, sc

    def to_win(self, x, y):
        img = self.left_image()
        sc, ox, oy, _, _ = _fit(img.shape[1], img.shape[0], LEFT)
        return ox + x * sc, oy + y * sc

    # --- mouse ---
    def on_mouse(self, ev, x, y, flags, _):
        if self.job is not None:
            return
        if ev == cv2.EVENT_LBUTTONDOWN:
            if LEFT[0] <= x <= LEFT[2] and LEFT[1] <= y <= LEFT[3]:
                ix, iy, sc = self.to_img(x, y)
                if self.draw:
                    self.drag = (self.draw, "new", (ix, iy), None)
                    return
                for name, b in self.boxes().items():
                    corners = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
                    for ci, (cx, cy) in enumerate(corners):
                        if abs(cx - ix) * sc < 12 and abs(cy - iy) * sc < 12:
                            self.drag = (name, ci, (ix, iy), list(b))
                            return
                for name, b in self.boxes().items():
                    if b[0] <= ix <= b[2] and b[1] <= iy <= b[3]:
                        self.drag = (name, "move", (ix, iy), list(b))
                        return
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag and flags & cv2.EVENT_FLAG_LBUTTON:
            self._drag_to(x, y)
        elif ev == cv2.EVENT_LBUTTONUP:
            if self.drag:
                self._drag_to(x, y)
                self._drop()
            else:
                for k, (x0, y0, x1, y1) in enumerate(self.fields_rects()):
                    if x0 <= x <= x1 and y0 <= y <= y1:
                        self.pending_field = k

    def _set_box(self, name, b):
        img = self.left_image()
        x0, x1 = sorted((int(round(b[0])), int(round(b[2]))))
        y0, y1 = sorted((int(round(b[1])), int(round(b[3]))))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
        if name == "map":
            self.imp.s["map_box"] = [x0, y0, x1, y1]
        else:
            self.imp.s["palette"]["box"] = [x0, y0, x1, y1]

    def _drag_to(self, x, y):
        name, mode, (sx, sy), b0 = self.drag
        ix, iy, _ = self.to_img(x, y)
        if mode == "new":
            b = [sx, sy, ix, iy]
        elif mode == "move":
            dx, dy = ix - sx, iy - sy
            b = [b0[0] + dx, b0[1] + dy, b0[2] + dx, b0[3] + dy]
        else:
            b = list(b0)
            b[0 if mode in (0, 3) else 2] = ix
            b[1 if mode in (0, 1) else 3] = iy
        self._set_box(name, b)

    def _drop(self):
        name = self.drag[0]
        self.drag, self.draw = None, None
        b = self.imp.s["map_box"] if name == "map" else self.imp.s["palette"]["box"]
        if b[2] - b[0] < 6 or b[3] - b[1] < 6:
            self.say("The box is too small, draw it again.", warn=True)
            return
        if name == "map":
            self.start("map", "Reading the map again")
        else:
            self.start("palette", "Reading the colour scale again")

    # --- fields ---
    def field_list(self):
        imp, s = self.imp, self.imp.s
        vmin, vmax = imp.range() if imp.palette is not None else (s.get("vmin"), s.get("vmax"))
        fmt = lambda v: "?" if v is None else f"{v:g}"
        if s["palette"].get("kind") == "stops":
            rng = f"{fmt(vmin)} … {fmt(vmax)} (from the typed colours)"
        else:
            rng = f"{fmt(vmin)} … {fmt(vmax)}"
        if imp.src.is_video:
            dates = f"{s.get('start') or '?'} … {s.get('end') or '?'}"
            steps = str(s.get("steps"))
        else:
            dates, steps = s.get("start") or "(none)", "—"
        r = s["region"]
        region = "whole world" if list(map(float, r)) == M.WORLD else f"lon {r[0]:g}…{r[1]:g}, lat {r[2]:g}…{r[3]:g}"
        return [("Name", s["name"]), ("Title", s["title"]), ("Units", s["units"] or "(none)"),
                ("Source", s["source"] or "(none)"), ("Scale", s["scale"]), ("Range", rng),
                ("Dates", dates), ("Steps", steps), ("Region", region), ("Keep", s["keep"])]

    def fields_rects(self):
        out = []
        for i in range(10):
            col, row = divmod(i, 5)
            x0 = 24 if col == 0 else 656
            y0 = 470 + row * 38
            out.append((x0, y0, x0 + 600, y0 + 34))
        return out

    def edit(self, i):
        imp, s = self.imp, self.imp.s
        name = self.field_list()[i][0]
        self.voice.stop()
        try:
            if name == "Name":
                v = ask_text("Name of the layer (its folder in data/; letters, digits, _)", s["name"])
                if v is not None:
                    s["name"] = M.slug(v)
            elif name == "Title":
                v = ask_text("Title (shown on the map and spoken)", s["title"])
                if v is not None:
                    s["title"] = v.strip()
            elif name == "Units":
                v = ask_text("Units, e.g. °C, %, µg/m³, m/s", s["units"])
                if v is not None:
                    s["units"] = v.strip()
            elif name == "Source":
                v = ask_text("Source (who made the picture / the data)", s["source"])
                if v is not None:
                    s["source"] = v.strip()
            elif name == "Scale":
                s["scale"] = M.SCALES[(M.SCALES.index(s["scale"]) + 1) % len(M.SCALES)]
                extra = ""
                if s["scale"] == "steps" and imp.palette is not None:
                    extra = f" {imp.palette.n_steps()} colour steps found."
                self.say(f"Scale: {s['scale']}.{extra}")
                return
            elif name == "Range":
                if s["palette"].get("kind") == "stops":
                    self.say("The values come from the typed colours.")
                    return
                cur = f"{s['vmin']:g}  {s['vmax']:g}" if s.get("vmin") is not None and s.get("vmax") is not None else ""
                where = "bottom and top" if self._bar_vertical() else "left and right end"
                v = ask_text(f"Values at the two ends of the colour scale ({where}), e.g.  0  100", cur)
                if v is not None:
                    nums = [float(x) for x in v.replace("…", " ").replace("..", " ").replace(",", " ").replace(";", " ").split()]
                    if len(nums) != 2:
                        raise ValueError("Write two numbers: the low end and the high end.")
                    s["vmin"], s["vmax"] = nums
            elif name == "Dates":
                if imp.src.is_video:
                    v = ask_text("Dates of the first and the last frame, e.g.  2020-01-01  2020-12-31  (or 2020-01  2020-12)",
                                 f"{s.get('start', '')}  {s.get('end', '')}".strip())
                    if v is not None:
                        parts = v.replace("…", " ").replace(",", " ").split()
                        if len(parts) != 2:
                            raise ValueError("Write two dates: the first frame and the last frame.")
                        M.parse_date(parts[0]); M.parse_date(parts[1], end=True)
                        s["start"], s["end"] = parts
                else:
                    v = ask_text("Date of the picture (optional), e.g. 2024-08", s.get("start", ""))
                    if v is not None:
                        if v.strip():
                            M.parse_date(v)
                        s["start"] = v.strip()
            elif name == "Steps":
                if not imp.src.is_video:
                    return
                v = ask_text("How many time steps to take from the video", str(s.get("steps", 48)))
                if v is not None and int(v) != s.get("steps"):
                    s["steps"] = max(1, int(v))
                    self.start("decode", "Reading the frames again")
            elif name == "Region":
                r = s["region"]
                v = ask_text("What the map shows: west east south north, e.g.  -25 45 34 72  (Europe), or  world",
                             "world" if list(map(float, r)) == M.WORLD else " ".join(f"{x:g}" for x in r))
                if v is not None:
                    new = list(M.WORLD) if v.strip().lower() in ("world", "whole world", "") else \
                        [float(x) for x in v.replace(",", " ").split()]
                    if len(new) != 4 or not (new[0] < new[1] and new[2] < new[3]):
                        raise ValueError("Write four numbers: west east south north.")
                    if new != s["region"]:
                        s["region"] = new
                        self.start("decode", "Reading the map again")
            elif name == "Keep":
                s["keep"] = M.KEEP[(M.KEEP.index(s["keep"]) + 1) % len(M.KEEP)]
                self.say({"all": "Keep everything.", "ocean": "Keep only the ocean.", "land": "Keep only land."}[s["keep"]])
                self.start("decode", "Reading the map again")
                return
            v = self.field_list()[i][1]
            self.say(f"{name}: {v}.")
        except ValueError as e:
            self.say(str(e), warn=True)

    def _bar_vertical(self):
        b = self.imp.bar_box
        return b is None or (b[3] - b[1]) >= (b[2] - b[0])

    # --- drawing ---
    def render(self):
        from PIL import Image, ImageDraw
        imp, s, f = self.imp, self.imp.s, fonts()
        img = Image.new("RGB", (W, H), L.hex_rgb(L.SURFACE))
        d = ImageDraw.Draw(img)
        d.text((24, 16), f"Add a map · {Path(imp.src.path).name}", font=f["m"], fill=L.hex_rgb(L.INK))
        busy = self.job is not None
        # left: the original (or the separate scale picture) with the boxes
        left = self.left_image()
        sc, ox, oy, w, h = _fit(left.shape[1], left.shape[0], LEFT)
        key = ("left", id(left), w, h)
        if key not in self._cache:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != "left"}
            self._cache[key] = Image.fromarray(cv2.resize(left, (w, h), interpolation=cv2.INTER_AREA))
        img.paste(self._cache[key], (ox, oy))
        for name, b in self.boxes().items():
            col = BLUE if name == "map" else RED
            x0, y0 = self.to_win(b[0], b[1]); x1, y1 = self.to_win(b[2], b[3])
            d.rectangle([x0, y0, x1, y1], outline=col, width=3)
            for cx, cy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                d.rectangle([cx - 5, cy - 5, cx + 5, cy + 5], fill=col)
        cap = "Picture of the colour scale: red = the scale. Drag the corners or the box." if self.view == "scale" else \
            f"Original {'frame' if imp.src.is_video else 'picture'}: blue = map, red = colour scale. Drag the corners or the box."
        d.text((LEFT[0], LEFT[3] + 6), cap,
               font=f["xs"], fill=L.hex_rgb(L.INK2))
        # right: decoded and re-coloured
        if not busy and imp.t is not None and imp.grid is not None:
            lon0, lon1, lat0, lat1 = imp.grid.region
            sc2, ox2, oy2, w2, h2 = _fit(lon1 - lon0, lat1 - lat0, RIGHT)
            key = ("dec", id(imp.t), self.k, w2, h2)
            if key not in self._cache:
                self._cache = {k: v for k, v in self._cache.items() if k[0] != "dec"}
                self._cache[key] = Image.fromarray(M.render_decoded(imp, self.k, (w2, h2)))
            img.paste(self._cache[key], (ox2, oy2))
            when = ""
            if imp.src.is_video:
                try:
                    when = " · " + imp.dates()[self.k] if s.get("start") and s.get("end") else ""
                except ValueError:
                    when = ""
                when = f" · step {self.k + 1} of {len(imp.centers)}{when}"
            d.text((RIGHT[0], RIGHT[3] + 6), f"Read from it, painted with the same colours; lines = coastlines{when}",
                   font=f["xs"], fill=L.hex_rgb(L.INK2))
        else:
            d.rectangle(RIGHT, outline=L.hex_rgb(L.GRIDC), width=1)
        # colour scale and numbers
        if imp.palette is not None:
            strip = imp.palette.colours_at(np.linspace(0, 1, 480))
            img.paste(Image.fromarray(np.repeat(strip[None].astype(np.uint8), 16, 0)), (84, 392))
            vmin, vmax = imp.range()
            lab = lambda v: "?" if v is None else f"{v:g}"
            d.text((24, 392), lab(vmin), font=f["sb"], fill=L.hex_rgb(L.INK))
            d.text((574, 392), lab(vmax), font=f["sb"], fill=L.hex_rgb(L.INK))
            d.text((84, 412), f"{imp.palette.origin} · {s['scale']}" + (f" · {s['units']}" if s["units"] else ""),
                   font=f["xs"], fill=L.hex_rgb(L.MUTED))
        if not busy and imp.stats:
            y = 388
            for line in M.summary(imp).rstrip(".").split("; "):
                d.text((RIGHT[0], y), line, font=f["s"], fill=L.hex_rgb(L.INK2)); y += 20
            for wtxt in imp.warnings[:2]:
                d.text((RIGHT[0], y), wtxt, font=f["s"], fill=L.hex_rgb(L.WARM)); y += 20
        # fields
        for k, ((name, val), (x0, y0, x1, y1)) in enumerate(zip(self.field_list(), self.fields_rects())):
            d.rounded_rectangle([x0, y0, x1, y1], radius=7, outline=L.hex_rgb(L.GRIDC), width=1)
            d.text((x0 + 10, y0 + 7), str((k + 1) % 10), font=f["sb"], fill=L.hex_rgb(L.COOL))
            d.text((x0 + 36, y0 + 7), name, font=f["sb"], fill=L.hex_rgb(L.INK2))
            val = str(val)
            while d.textlength(val, font=f["s"]) > (x1 - x0) - 140 and len(val) > 4:
                val = val[:-2] + "…" if not val.endswith("…") else val[:-2] + "…"
            d.text((x0 + 130, y0 + 7), val, font=f["s"], fill=L.hex_rgb(L.INK))
        # status and keys
        if self.message:                                     # up to two lines
            lines, cur = [], ""
            for word in self.message.split():
                if d.textlength(cur + " " + word, font=f["s"]) > W - 48 and cur:
                    lines.append(cur); cur = word
                else:
                    cur = (cur + " " + word).strip()
            lines.append(cur)
            if len(lines) > 2:
                lines = [lines[0], lines[1][:-1] + "…"]
            for j, line in enumerate(lines):
                d.text((24, 666 + j * 22), line, font=f["s"], fill=L.hex_rgb(L.WARM if self.msg_warn else L.INK))
        mode = {"map": "  ·  now drag a box around the MAP", "bar": "  ·  now drag a box around the COLOUR SCALE"}.get(self.draw, "")
        keys1 = "1–0 or click - edit a field     ← → time step     F - flip the scale     M / C - draw the map / scale box" + mode
        keys2 = "R - find everything again     Tab - show the scale picture     S - say the numbers     ? - help     Enter - save     Esc - cancel"
        d.text((24, 726), keys1, font=f["xs"], fill=L.hex_rgb(L.MUTED))
        d.text((24, 748), keys2, font=f["xs"], fill=L.hex_rgb(L.MUTED))
        if busy:                                              # progress over the pictures
            job = self.job
            d.rectangle([RIGHT[0], 160, RIGHT[2], 250], fill=L.hex_rgb("#ffffff"), outline=L.hex_rgb(L.GRIDC))
            d.text((RIGHT[0] + 20, 175), job.text, font=f["m"], fill=L.hex_rgb(L.INK))
            d.rectangle([RIGHT[0] + 20, 215, RIGHT[2] - 20, 229], outline=L.hex_rgb(L.GRIDC))
            d.rectangle([RIGHT[0] + 20, 215, RIGHT[0] + 20 + int((RIGHT[2] - RIGHT[0] - 40) * job.frac), 229],
                        fill=L.hex_rgb(L.COOL))
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    # --- the loop ---
    def help_text(self):
        return ("Check page. Keys: 1 to 0 edit the fields: name, title, units, source, scale type, range, dates, "
                "time steps, region and what to keep. Left and right arrows go through the time steps. "
                "F flips the colour scale. M then drag draws the map box, C then drag draws the colour scale box. "
                "R finds everything again. S says the numbers. Enter saves the map. Escape cancels.")

    def loop(self):
        """-> the name of the saved layer, or None."""
        imp, s = self.imp, self.imp.s
        cv2.setMouseCallback(WIN, self.on_mouse)
        self.start("first", "Reading the colour scale and the map")
        while True:
            if self.job is not None and self.job.done:
                self.job_finished()
            cv2.imshow(WIN, self.render())
            key = cv2.waitKeyEx(30)
            if self.pending_field is not None:
                i, self.pending_field = self.pending_field, None
                if self.job is None:
                    self.edit(i)
                continue
            if key == -1:
                continue
            if key in (27,) or (key == ord("q") and self.job is None):
                self.voice.speak("Cancelled. Nothing was saved.")
                return None
            if key in (ord("?"), ord("/"), ord("h")):
                self.voice.speak(self.help_text())
                continue
            if self.job is not None:
                continue                                       # busy: only help / cancel
            if 0 <= key < 256 and chr(key).isdigit():
                self.edit((int(chr(key)) - 1) % 10)
            elif key in _LEFT_KEYS or key in _RIGHT_KEYS or key in (ord(","), ord(".")):
                if imp.centers is not None and len(imp.centers) > 1:
                    step = -1 if key in _LEFT_KEYS or key == ord(",") else 1
                    self.k = int(np.clip(self.k + step, 0, len(imp.centers) - 1))
            elif key == ord("f"):
                p = s["palette"]
                if p.get("kind") == "stops":
                    self.say("Typed colours have their own values; change the order there.")
                else:
                    p["reverse"] = not p.get("reverse", False)
                    self.say("Colour scale flipped.")
                    self.start("palette", "Reading the colour scale again")
            elif key == ord("m"):
                self.view, self.draw = "frame", "map"
                self.say("Drag a box around the map.")
            elif key == ord("c"):
                kind = s["palette"].get("kind")
                if kind not in ("frame", "image"):
                    self.say("The colours are typed or named, there is no scale box.")
                else:
                    self.view = "scale" if kind == "image" else "frame"
                    self.draw = "bar"
                    self.say("Drag a box around the colour scale.")
            elif key == 9:                                     # Tab
                if s["palette"].get("kind") == "image":
                    self.view = "scale" if self.view == "frame" else "frame"
            elif key == ord("r"):
                self.start("all", "Finding the colour scale and the map again")
            elif key == ord("s"):
                missing = imp.ready()
                self.say(M.summary(imp) + " " + (missing or "Ready to save."), warn=bool(missing))
            elif key in (13, 10):
                name = self.save()
                if name:
                    return name

    def save(self):
        imp, s = self.imp, self.imp.s
        missing = imp.ready()
        if missing:
            self.say(missing, warn=True)
            return None
        name = M.slug(s["name"])
        if (L.DATA_DIR / name).exists() and not ask_yes(f"A map named '{name}' already exists. Replace it?", "Replace"):
            self.say("Not saved. Change the name (field 1).")
            return None
        try:
            folder = imp.save()
            M.save_preset(imp)
        except Exception as e:
            self.say(f"Could not save: {e}", warn=True)
            return None
        self.say(f"Saved. The map {s['title'] or name} is now in the list.")
        print(f"Layer saved: {folder}")
        while self.voice.speaking():
            cv2.waitKey(30)
        return name


# ----------------------------------------------------------------------------
def run(voice):
    """The whole page. -> name of the new layer or None. Uses (and closes) its own window."""
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    try:
        voice.speak("Add a map. Choose a video or a picture of a map.")
        cv2.imshow(WIN, np.full((H, W, 3), 252, np.uint8))
        cv2.waitKey(1)
        path = choose_file("Choose a video or a picture of a map")
        if not path:
            voice.speak("Cancelled.")
            return None
        if Path(path).suffix.lower() not in M.VIDEO_EXT | M.IMAGE_EXT:
            voice.speak("This is not a video or a picture file.")
            return None
        try:
            imp = M.Importer(path)
        except ValueError as e:
            voice.speak(str(e))
            return None
        if not choose_source(imp, voice):
            voice.speak("Cancelled.")
            return None
        return CheckPage(imp, voice).loop()
    finally:
        cv2.destroyWindow(WIN)
        cv2.waitKey(1)


if __name__ == "__main__":
    import speech as SP
    print(run(SP.Speaker()))
