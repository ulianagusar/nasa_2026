"""Sound sets: your own recordings as the voice of a map layer (like the water of currents_speed).

A sound set = 1..8 recordings ordered from the LOWEST values of a layer to the HIGHEST. They all play in a loop
at the same time; the value under the finger only changes their volumes (neighbouring recordings crossfade
with constant power). One recording only: its volume follows the value.

Making a set (the "Sounds for a map" page of finger_map.py does the same interactively):
  * every file is read (wav, aiff, mp3, m4a, flac, caf ... - macOS afconvert, or ffmpeg if installed);
  * silence at the start and the end is cut, at most MAX_SECONDS are kept;
  * the end is crossfaded into the start, so the loop has no seam;
  * loudness is evened out: "rising" (higher values a little louder, like the water set) or "equal";
  * written to data/sounds/<set>/ as WAV + sounds.json (the same format as data/currents_sounds).
A layer uses a set through data/<layer>/jukebox.json: "sound": "<set>" ("note" = back to notes).

    .venv/bin/python sound_pack.py make fire low.wav mid.mp3 high.m4a --layer temperature
    .venv/bin/python sound_pack.py attach currents_speed water        # a set for a layer
    .venv/bin/python sound_pack.py attach temperature note            # back to notes
    .venv/bin/python sound_pack.py delete fire                        # maps that used it go back to notes
    .venv/bin/python sound_pack.py list
"""
import argparse
import json
import re
import shutil
import subprocess
import tempfile
import wave
from datetime import date
from pathlib import Path

import numpy as np

import layers as L

HERE = Path(__file__).resolve().parent
SOUNDS_DIR = L.DATA_DIR / "sounds"
WATER_DIR = L.DATA_DIR / "currents_sounds"          # the first set; its name is "water"
SR = 44100
MAX_SECONDS = 30.0
CROSSFADE = 1.5
MAX_SOUNDS = 8
AUDIO_EXT = {".wav", ".wave", ".aif", ".aiff", ".aifc", ".mp3", ".m4a", ".aac", ".caf", ".flac", ".mp4", ".ogg", ".opus"}
LOUDNESS = ("rising", "equal")


def slug(s):
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", str(s).strip().lower()).strip("_")
    return s or "sounds"


# ----------------------------------------------------------------------------
# Reading and preparing one recording
# ----------------------------------------------------------------------------
def read_wav(path):
    with wave.open(str(path)) as w:
        ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError(f"{Path(path).name}: only 16-bit WAV can be read directly")
    x = np.frombuffer(raw, "<i2").reshape(-1, ch).astype(np.float32) / 32768.0
    return x, sr


def decode(path):
    """Any audio file -> float32 [frames][2] at 44.1 kHz."""
    path = Path(path)
    if not path.exists():
        raise ValueError(f"No file {path}")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "x.wav"
        ok = False
        if shutil.which("afconvert"):
            r = subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", str(path), str(out)],
                               capture_output=True)
            ok = r.returncode == 0 and out.exists()
        if not ok and shutil.which("ffmpeg"):
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ac", "2", "-ar", str(SR),
                                "-c:a", "pcm_s16le", str(out)], capture_output=True)
            ok = r.returncode == 0 and out.exists()
        if not ok:
            raise ValueError(f"Cannot read the sound {path.name} (is it an audio file?)")
        x, sr = read_wav(out)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, 1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    if sr != SR:                                              # afconvert/ffmpeg already resample; just in case
        t = np.arange(int(len(x) * SR / sr)) * sr / SR
        x = np.stack([np.interp(t, np.arange(len(x)), x[:, c]) for c in range(2)], 1).astype(np.float32)
    return x


def trim_silence(x, db=-50.0):
    env = np.abs(x).max(1)
    loud = np.where(env > 10 ** (db / 20))[0]
    if len(loud) == 0:
        raise ValueError("The recording is silent")
    return x[loud[0]:loud[-1] + 1]


def make_loop(x, crossfade=CROSSFADE):
    """The end crossfades into the start (constant power) -> plays in a loop without a seam."""
    c = int(crossfade * SR)
    if len(x) < 4 * c:
        c = len(x) // 4
    if c < 16:
        return x
    t = np.linspace(0, np.pi / 2, c, dtype=np.float32)[:, None]
    out = x[:len(x) - c].copy()
    out[:c] = x[:c] * np.sin(t) + x[len(x) - c:] * np.cos(t)
    return out


def rms_db(x):
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x - x.mean(0)))) + 1e-12))


def prepare(path, max_seconds=MAX_SECONDS):
    x = trim_silence(decode(path))
    if len(x) < SR * 1.0:
        raise ValueError(f"{Path(path).name} is shorter than 1 second")
    x = x[:int(max_seconds * SR) + int(CROSSFADE * SR)]
    return make_loop(x - x.mean(0))


def targets(n, loudness):
    """Target loudness (RMS, dBFS) of each recording: rising like the water set (-28 -> -18) or equal."""
    if loudness == "equal" or n == 1:
        return np.full(n, -23.0)
    return np.linspace(-28.0, -18.0, n)


def level(loops, loudness):
    """Even out the loudness. -> (loops, applied gains in dB)."""
    out, gains = [], []
    for x, tgt in zip(loops, targets(len(loops), loudness)):
        g = 10 ** ((tgt - rms_db(x)) / 20)
        peak = float(np.abs(x).max()) * g
        if peak > 0.98:                                       # never clip: a very peaky sound stays quieter
            g *= 0.98 / peak
        out.append((x * g).astype(np.float32))
        gains.append(20 * np.log10(g))
    return out, gains


# ----------------------------------------------------------------------------
# A set
# ----------------------------------------------------------------------------
class Draft:
    """Recordings chosen but not saved yet (the page plays them for a preview)."""

    def __init__(self, files, loudness="rising", progress=None):
        if not 1 <= len(files) <= MAX_SOUNDS:
            raise ValueError(f"Choose 1 to {MAX_SOUNDS} sounds")
        self.files = [Path(f) for f in files]
        self.raw = []
        for k, f in enumerate(self.files):
            if progress:
                progress(f"Preparing {f.name}", k / len(self.files))
            self.raw.append(prepare(f))
        self.names = [slug(f.stem) for f in self.files]
        self.set_loudness(loudness)
        if progress:
            progress("Done", 1.0)

    def set_loudness(self, loudness):
        self.loudness = loudness
        self.loops, self.gains_db = level(self.raw, loudness)

    @property
    def centers(self):
        n = len(self.loops)
        return list(np.linspace(0, 1, n)) if n > 1 else [0.5]

    def save(self, name, out_dir=None, note=""):
        name = slug(name)
        folder = Path(out_dir or SOUNDS_DIR) / name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        levels = []
        for k, (x, nm, f, c, gdb) in enumerate(zip(self.loops, self.names, self.files, self.centers, self.gains_db)):
            fn = f"{k + 1}_{nm}.wav"
            with wave.open(str(folder / fn), "wb") as w:
                w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
                w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
            levels.append({"name": nm, "file": fn, "center": round(float(c), 4), "duration_s": round(len(x) / SR, 1),
                           "source_file": f.name, "gain_db": round(float(gdb), 1), "rms_db": round(rms_db(x), 1)})
        meta = {"description": note or f"Sound set '{name}': from the lowest to the highest values",
                "mixing": "all loops play at once; between neighbouring centers gain = sqrt(share) "
                          "(constant power); one sound only: volume 0.2 + 0.8 * s",
                "loudness": self.loudness, "made": date.today().isoformat(), "levels": levels}
        (folder / "sounds.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        return folder


def list_sets():
    """{name: folder} of all sound sets (the water set as "water")."""
    out = {}
    if (WATER_DIR / "sounds.json").exists():
        out["water"] = WATER_DIR
    if SOUNDS_DIR.exists():
        for d in sorted(SOUNDS_DIR.iterdir()):
            if (d / "sounds.json").exists():
                out[d.name] = d
    return out


def describe_set(name):
    sets = list_sets()
    if name not in sets:
        return name
    levels = json.loads((sets[name] / "sounds.json").read_text()).get("levels", [])
    return f"{name} ({len(levels)} sound{'s' if len(levels) != 1 else ''})"


# ----------------------------------------------------------------------------
# Which layer uses which set (data/<layer>/jukebox.json; the sst layer: data/sst/jukebox.json)
# ----------------------------------------------------------------------------
def layer_sound(spec_or_layer):
    lay = spec_or_layer.load() if hasattr(spec_or_layer, "load") else spec_or_layer
    snd = lay.cfg.get("sound")
    return None if not snd or snd == "note" else snd


def sound_of(layer_name):
    """Sound set of a layer without loading its data: its jukebox.json, else the built-in setting. None = notes."""
    ov = L._folder_overrides(L.DATA_DIR / layer_name)
    snd = ov.get("sound", L.BUILTIN.get(layer_name, {}).get("sound"))
    return None if not snd or snd == "note" else snd


def users(set_name):
    """Names of the layers that sound with this set."""
    return [s.name for s in L.discover_layers() if sound_of(s.name) == set_name]


def delete_set(name):
    """Delete a sound set; the maps that used it sound with notes again. The built-in water set is kept.
    -> names of the layers switched back to notes."""
    if name == "water":
        raise ValueError("The water set is built in (ocean currents) and cannot be deleted.")
    sets = list_sets()
    if name not in sets:
        raise ValueError(f"No sound set '{name}'")
    switched = []
    for lname in users(name):
        attach(lname, "note")
        switched.append(lname)
    shutil.rmtree(sets[name])
    return switched


def attach(layer_name, set_name):
    """set_name = a set name, "water", or None/"note" for notes."""
    folder = L.DATA_DIR / layer_name
    folder.mkdir(exist_ok=True)
    p = folder / "jukebox.json"
    cfg = json.loads(p.read_text()) if p.exists() else {}
    cfg.pop("sound_dir", None)
    if not set_name or set_name == "note":
        cfg["sound"] = "note"                                 # also overrides a built-in set
    else:
        if set_name not in list_sets():
            raise ValueError(f"No sound set '{set_name}'")
        cfg["sound"] = set_name
        if set_name != "water":
            cfg["sound_dir"] = str(Path("data") / "sounds" / set_name)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    return p


# ----------------------------------------------------------------------------
def _cli():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="make a sound set from files (lowest values first)")
    m.add_argument("name"); m.add_argument("files", nargs="+")
    m.add_argument("--loudness", choices=LOUDNESS, default="rising")
    m.add_argument("--layer", help="also use it for this layer")
    a_ = sub.add_parser("attach", help="use a set for a layer (note = back to notes)")
    a_.add_argument("layer"); a_.add_argument("set")
    d_ = sub.add_parser("delete", help="delete a sound set (maps that use it go back to notes)")
    d_.add_argument("set")
    sub.add_parser("list", help="sound sets and which layers use them")
    a = ap.parse_args()
    if a.cmd == "make":
        d = Draft(a.files, a.loudness, lambda t, f: print(f"  {t}"))
        folder = d.save(a.name)
        print(f"Saved: {folder}")
        for nm, gdb, x in zip(d.names, d.gains_db, d.loops):
            print(f"  {nm}: {len(x) / SR:.1f} s loop, loudness {rms_db(x):.1f} dB (gain {gdb:+.1f} dB)")
        if a.layer:
            attach(a.layer, slug(a.name))
            print(f"Layer {a.layer} now sounds with '{slug(a.name)}'.")
    elif a.cmd == "attach":
        attach(a.layer, a.set)
        print(f"Layer {a.layer}: {'notes' if a.set == 'note' else a.set}")
    elif a.cmd == "delete":
        switched = delete_set(a.set)
        print(f"Deleted the sound set '{a.set}'." + (f" Back to notes: {', '.join(switched)}" if switched else ""))
    else:
        print("Sound sets: " + (", ".join(describe_set(n) for n in list_sets()) or "none"))
        for s in L.discover_layers():
            print(f"  {s.name:22s} {layer_sound(s) or 'notes'}")


if __name__ == "__main__":
    _cli()
