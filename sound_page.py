"""Start menu page "Sounds for a map": your own recordings as the voice of a layer (see sound_pack.py).

1. Choose the map (layer).
2. Choose what it should sound like: new recordings (one file after another, from the lowest values to the
   highest), a sound set that already exists (e.g. the water), or back to notes.
3. Listen: move the mouse along the colour scale of the map (or the arrow keys) - the mix changes exactly
   as it will under the finger. L - loudness rising/equal, 1 - name of the set, Enter - save and use.
"""
import json
from pathlib import Path

import cv2
import numpy as np

import layers as L
import sound_pack as SPK
from import_page import Job, ask_text, ask_yes, choose_file, fonts

WIN = "Sounds for a map"
W, H = 1100, 700
BAR = (80, 250, 1020, 290)                 # the colour scale the mouse moves along
_LEFT_KEYS = {63234, 2424832, 65361, ord(",")}
_RIGHT_KEYS = {63235, 2555904, 65363, ord(".")}
_UP = {63232, 2490368, 65362}
_DOWN = {63233, 2621440, 65364}


current_sound = SPK.sound_of


def delete_page(specs, voice):
    """Choose a sound set and delete it (after a yes). -> True if something was deleted."""
    titles = {s.name: s.title for s in specs}
    names = [n for n in SPK.list_sets() if n != "water"]
    if not names:
        voice.speak("There are no sound sets to delete. The water set is built in.")
        return False
    rows = []
    for n in names:
        used = [titles.get(u, u) for u in SPK.users(n)]
        rows.append(f"{SPK.describe_set(n)}  —  " + (f"used by: {', '.join(used)}" if used else "not used by any map"))
    k = choose_row("Delete a sound set", "The maps that use it will sound with notes again. "
                   "The built-in water set cannot be deleted.", rows, voice,
                   "Delete a sound set. Which one? " + " ".join(f"{i + 1}: {r}." for i, r in enumerate(rows[:9])))
    if k is None:
        return False
    name = names[k]
    used = SPK.users(name)
    question = f"Delete the sound set '{name}' and its recordings?" + \
        (f"\n{len(used)} map(s) will sound with notes again." if used else "")
    voice.speak(f"Delete the sound set {name}? Press Delete to confirm.")
    if not ask_yes(question, "Delete"):
        voice.speak("Not deleted.")
        return False
    try:
        switched = SPK.delete_set(name)
    except ValueError as e:
        voice.speak(str(e))
        return False
    voice.speak(f"Deleted {name}." + (" " + ", ".join(titles.get(u, u) for u in switched) +
                                       " sound with notes again." if switched else ""))
    print(f"Deleted the sound set '{name}'" + (f"; back to notes: {', '.join(switched)}" if switched else ""))
    return True


def _page(title, subtitle=""):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), L.hex_rgb(L.SURFACE))
    d = ImageDraw.Draw(img)
    d.text((40, 28), title, font=fonts()["l"], fill=L.hex_rgb(L.INK))
    if subtitle:
        d.text((40, 66), subtitle, font=fonts()["s"], fill=L.hex_rgb(L.MUTED))
    return img, d


def _to_bgr(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def choose_row(title, subtitle, rows, voice, spoken, footer="Number, arrows + Enter, or click - choose     Esc - back"):
    """A list page. rows: [text, ...]. -> index or None."""
    sel, clicks = None, []
    cv2.setMouseCallback(WIN, lambda ev, x, y, *_: clicks.append((x, y)) if ev == cv2.EVENT_LBUTTONUP else None)
    voice.speak(spoken)
    top, rh = 110, 52
    while True:
        img, d = _page(title, subtitle)
        rects = []
        for k, text in enumerate(rows):
            y = top + k * rh
            rects.append((40, y, W - 40, y + rh - 8))
            if k == sel:
                d.rounded_rectangle(rects[-1], radius=8, fill=L.hex_rgb("#dbe8f8"))
            d.rounded_rectangle(rects[-1], radius=8, outline=L.hex_rgb(L.GRIDC), width=1)
            num = str(k + 1) if k < 9 else ""
            d.text((58, y + 10), num, font=fonts()["m"], fill=L.hex_rgb(L.COOL))
            d.text((96, y + 11), text, font=fonts()["s"], fill=L.hex_rgb(L.INK))
        d.text((40, H - 44), footer, font=fonts()["s"], fill=L.hex_rgb(L.MUTED))
        cv2.imshow(WIN, _to_bgr(img))
        key = cv2.waitKeyEx(30)
        while clicks:
            x, y = clicks.pop()
            for k, (x0, y0, x1, y1) in enumerate(rects):
                if x0 <= x <= x1 and y0 <= y <= y1:
                    voice.stop()
                    return k
        if key in (27, ord("q"), 8, 127):
            voice.stop()
            return None
        if 0 <= key < 256 and ord("1") <= key <= ord("9") and key - ord("1") < len(rows):
            voice.stop()
            return key - ord("1")
        if key in _UP or key in _DOWN:
            sel = (0 if key in _DOWN else len(rows) - 1) if sel is None else (sel + (1 if key in _DOWN else -1)) % len(rows)
            voice.speak(f"{sel + 1}, {rows[sel]}")
        if key in (13, 10, 32) and sel is not None:
            voice.stop()
            return sel


def wait_job(job, title, voice):
    while not job.done:
        img, d = _page(title)
        d.text((40, 300), job.text, font=fonts()["m"], fill=L.hex_rgb(L.INK))
        d.rectangle([40, 340, W - 40, 356], outline=L.hex_rgb(L.GRIDC))
        d.rectangle([40, 340, 40 + int((W - 80) * job.frac), 356], fill=L.hex_rgb(L.COOL))
        cv2.imshow(WIN, _to_bgr(img))
        cv2.waitKey(30)
    if job.error:
        print(job.error)
        voice.speak(job.error)
    return job.error is None


class Preview:
    """Listen to a set along the colour scale of the layer; save (a new set) and use it."""

    def __init__(self, layer, voice, son, draft=None, set_name=None):
        import finger_map as F
        self.F, self.layer, self.voice, self.son = F, layer, voice, son
        self.draft, self.set_name = draft, set_name
        self.name = SPK.slug(f"{layer.name}_sounds") if draft else set_name
        self.s, self.playing, self.mouse = 0.0, True, []
        self.bank = None
        self.rebuild()

    def rebuild(self):
        F = self.F
        if self.draft is not None:
            self.bank = F.SoundBank(loops=self.draft.loops, centers=self.draft.centers, names=self.draft.names)
        else:
            self.bank = F.SoundBank(SPK.list_sets()[self.set_name])
        self.son.use_bank(self.bank)

    def value_at(self, s):
        lay = self.layer
        if getattr(lay, "log", False):
            return float(np.exp(np.log(lay.vmin) + s * (np.log(lay.vmax) - np.log(lay.vmin))))
        return lay.vmin + s * (lay.vmax - lay.vmin)

    def render(self):
        lay, bank = self.layer, self.bank
        f = fonts()
        what = "new recordings" if self.draft is not None else f"the set '{self.set_name}'"
        img, d = _page(f"Sounds for: {lay.title}", f"Listening to {what}. Move the mouse along the scale "
                                                   "(or ← →): this is how the map will sound under the finger.")
        from PIL import Image
        x0, y0, x1, y1 = BAR
        strip = lay.lut[np.linspace(0, 255, x1 - x0).astype(int)]
        img.paste(Image.fromarray(np.repeat(strip[None], y1 - y0, 0).astype(np.uint8)), (x0, y0))
        d.rectangle(BAR, outline=L.hex_rgb(L.INK2), width=1)
        for s in (0.0, 0.5, 1.0):                                  # values under the scale
            txt = f"{lay.fmt.format(self.value_at(s))} {lay.unit}".strip()
            x = x0 + s * (x1 - x0)
            d.text((x - d.textlength(txt, font=f["s"]) / 2, y1 + 8), txt, font=f["s"], fill=L.hex_rgb(L.MUTED))
        for c, nm in zip(bank.centers, bank.names):                # where each recording is loudest
            x = x0 + float(c) * (x1 - x0)
            d.polygon([(x, y0 - 4), (x - 7, y0 - 16), (x + 7, y0 - 16)], fill=L.hex_rgb(L.INK2))
            d.text((x - d.textlength(nm, font=f["xs"]) / 2, y0 - 36), nm, font=f["xs"], fill=L.hex_rgb(L.INK2))
        cx = x0 + self.s * (x1 - x0)                               # the "finger"
        d.rectangle([cx - 2, y0 - 6, cx + 2, y1 + 6], fill=L.hex_rgb(L.INK))
        val = f"{lay.fmt.format(self.value_at(self.s))} {lay.unit}".strip()
        d.text((cx - d.textlength(val, font=f["m"]) / 2, y1 + 34), val, font=f["m"], fill=L.hex_rgb(L.INK))
        # the mix: how loud every recording is now
        g = bank.mix_gains(self.s)
        d.text((80, 400), "Mix now:", font=f["sb"], fill=L.hex_rgb(L.INK2))
        for k, (nm, gk) in enumerate(zip(bank.names, g)):
            y = 430 + k * 24
            d.text((80, y), nm, font=f["s"], fill=L.hex_rgb(L.INK))
            d.rectangle([260, y + 3, 260 + 300, y + 17], outline=L.hex_rgb(L.GRIDC))
            d.rectangle([260, y + 3, 260 + int(300 * gk ** 2), y + 17], fill=L.hex_rgb(L.COOL))
        # settings
        x = 640
        if self.draft is not None:
            d.text((x, 400), f"1  Name:      {self.name}", font=f["s"], fill=L.hex_rgb(L.INK))
            d.text((x, 428), f"L  Loudness: {self.draft.loudness}"
                   + ("  (higher values a little louder)" if self.draft.loudness == "rising" else "  (all the same)"),
                   font=f["s"], fill=L.hex_rgb(L.INK))
            lens = ", ".join(f"{len(x_) / SPK.SR:.0f}" for x_ in self.draft.loops)
            d.text((x, 456), f"Loops (s): {lens}", font=f["xs"], fill=L.hex_rgb(L.MUTED))
        d.text((x, 490), "Sound: " + ("on" if self.playing else "paused (Space)"), font=f["s"], fill=L.hex_rgb(L.INK2))
        if self.son.stream is None:
            d.text((x, 518), "No sound output - the preview is silent.", font=f["s"], fill=L.hex_rgb(L.WARM))
        keys = "← → or mouse - move along the scale    Space - pause    " + \
               ("1 - name    L - loudness    " if self.draft is not None else "") + "Enter - use it    Esc - back"
        d.text((40, H - 44), keys, font=f["s"], fill=L.hex_rgb(L.MUTED))
        return _to_bgr(img)

    def on_mouse(self, ev, x, y, flags, _):
        x0, y0, x1, y1 = BAR
        if y0 - 60 <= y <= y1 + 60 and x0 - 20 <= x <= x1 + 20 and ev in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self.s = float(np.clip((x - x0) / (x1 - x0), 0, 1))

    def loop(self):
        """-> True if the layer now uses the set."""
        cv2.setMouseCallback(WIN, self.on_mouse)
        n = len(self.bank)
        self.voice.speak(f"{n} sound{'s' if n != 1 else ''}. Move the mouse along the colour scale, or use the arrow keys, "
                         "to hear how the sound changes from low to high values. Enter to use it. Escape to go back.")
        while True:
            if self.playing:
                self.son.play_water(self.s, 0.0)
            else:
                self.son.silence()
            self.son.set_duck(self.voice.speaking())
            cv2.imshow(WIN, self.render())
            key = cv2.waitKeyEx(20)
            if key in (27, ord("q"), 8, 127):
                return False
            if key in _LEFT_KEYS or key in _RIGHT_KEYS:
                self.s = float(np.clip(self.s + (0.05 if key in _RIGHT_KEYS else -0.05), 0, 1))
            elif key == 32:
                self.playing = not self.playing
            elif key == ord("l") and self.draft is not None:
                self.draft.set_loudness("equal" if self.draft.loudness == "rising" else "rising")
                self.rebuild()
                self.voice.speak(f"Loudness: {self.draft.loudness}.")
            elif key == ord("1") and self.draft is not None:
                self.son.silence()
                v = ask_text("Name of the sound set (letters, digits, _)", self.name)
                if v:
                    self.name = SPK.slug(v)
                    self.voice.speak(f"Name: {self.name}.")
            elif key in (13, 10):
                if self.save():
                    return True

    def save(self):
        self.son.silence()
        if self.draft is not None:
            if self.name in SPK.list_sets() and not ask_yes(f"A sound set '{self.name}' already exists. Replace it?", "Replace"):
                self.voice.speak("Not saved. Press 1 to change the name.")
                return False
            if self.name == "water":
                self.voice.speak("The name water is taken by the built-in set. Press 1 to choose another name.")
                return False
            self.draft.save(self.name)
        SPK.attach(self.layer.name, self.name)
        self.voice.speak(f"Done. {self.layer.title} now sounds with {self.name}.")
        print(f"Layer {self.layer.name}: sound set '{self.name}'")
        return True


def run(voice, sound=True, volume=0.8):
    """The whole page. -> True if a layer got a new sound (the caller reloads the layers)."""
    import finger_map as F
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    son = None
    changed = False
    try:
        specs = L.discover_layers()
        while True:
            rows = [f"{s.title}  —  {current_sound(s.name) or 'notes'}" for s in specs]
            can_delete = any(n != "water" for n in SPK.list_sets())
            if can_delete:
                rows.append("Delete a sound set …")
            k = choose_row("Sounds for a map: which map?", "Now: what every map sounds like", rows, voice,
                           "Sounds for a map. Which map should get new sounds? " +
                           " ".join(f"{i + 1}: {s.title}, now {current_sound(s.name) or 'notes'}."
                                    for i, s in enumerate(specs[:9])) +
                           (f" {len(specs) + 1}: delete a sound set." if can_delete and len(specs) < 9 else ""))
            if k is None:
                return changed
            if k == len(specs):                                # the last row: delete a set
                changed |= delete_page(specs, voice)
                continue
            spec = specs[k]
            layer = spec.load()
            sets = SPK.list_sets()
            opts = [("new", "Choose sound files (the sound for the lowest values first)")]
            opts += [("set:" + n, f"Use the set: {SPK.describe_set(n)}") for n in list(sets)[:6]]
            if current_sound(spec.name):
                opts.append(("note", "Back to notes (no recordings)"))
            j = choose_row(f"Sounds for: {layer.title}", f"Now: {current_sound(spec.name) or 'notes'}",
                           [t for _, t in opts], voice,
                           f"{layer.title}. " + " ".join(f"{i + 1}: {t}." for i, (_, t) in enumerate(opts)))
            if j is None:
                continue
            what = opts[j][0]
            if what == "note":
                SPK.attach(spec.name, "note")
                voice.speak(f"{layer.title} sounds with notes again.")
                changed = True
                continue
            draft, set_name = None, None
            if what == "new":
                files = []
                while len(files) < SPK.MAX_SOUNDS:
                    voice.speak("Choose the sound for the lowest values." if not files else
                                f"Sound {len(files) + 1}, for higher values. Cancel when that is all.")
                    p = choose_file("Sound for the LOWEST values of the map" if not files else
                                    f"Sound {len(files) + 1}: for HIGHER values (Cancel = that's all)")
                    voice.stop()
                    if not p:
                        break
                    if Path(p).suffix.lower() not in SPK.AUDIO_EXT:
                        voice.speak("This is not an audio file.")
                        continue
                    files.append(p)
                if not files:
                    continue
                box = {}
                job = Job(lambda progress: box.setdefault("d", SPK.Draft(files, "rising", progress)), "Preparing the sounds")
                if not wait_job(job, f"Sounds for: {layer.title}", voice):
                    continue
                draft = box["d"]
            else:
                set_name = what[len("set:"):]
            if son is None:
                son = F.Sonifier(enabled=sound, volume=volume)
            try:
                ok = Preview(layer, voice, son, draft, set_name).loop()
            finally:
                son.silence()
            changed |= ok
    finally:
        if son is not None:
            son.close()
        cv2.destroyWindow(WIN)
        cv2.waitKey(1)


if __name__ == "__main__":
    import speech as SP
    print(run(SP.Speaker()))
