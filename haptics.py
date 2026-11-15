"""Vibration on the MacBook trackpad (Force Touch): a layer can be FELT, e.g. earthquakes shake under the palm.

A MacBook has no vibration motor, but its trackpad has a haptic actuator (the "click" you feel is made by it).
It can make short ticks of a few strengths; a train of ticks is felt as shaking. You feel it with a hand (or a
finger) resting on the trackpad - e.g. the left palm, while the right index finger points at the map.

    value under the finger (and nearby, for sparse events) -> intensity 0..1
    intensity -> ticks per second (4 -> 22) + tick strength (weak / medium / strong) + irregular rhythm (shaking)

A layer gets it in data/<layer>/jukebox.json (or BUILTIN in layers.py):
    "haptic": true                              intensity = the value under the finger on the layer scale
    "haptic": {"radius_km": 500, "min": 0.15}   events near the finger count too, weaker with distance
                                                (earthquakes: 99.5% of the cells are empty, a finger rarely hits one)
The actuator is a private macOS interface (MultitouchSupport.framework, the one HapticKey and others use);
if it is not there, the program runs without vibration.

    .venv/bin/python haptics.py          # feel it: 6 s from nothing to the strongest shaking (hand on the trackpad)
"""
import ctypes
import math
import random
import re
import subprocess
import threading
import time

import numpy as np

LEVELS = (3, 4, 6)          # actuation waveforms: weak, medium, strong
_MT_PATH = "/System/Library/PrivateFrameworks/MultitouchSupport.framework/MultitouchSupport"


class TrackpadActuator:
    """The haptic actuator of the built-in (or Magic) trackpad. ok=False if there is none."""

    def __init__(self):
        self.mt = self.act = None
        self.ok = False
        try:
            mt = ctypes.CDLL(_MT_PATH)
        except OSError:
            return
        mt.MTActuatorCreateFromDeviceID.argtypes = [ctypes.c_uint64]
        mt.MTActuatorCreateFromDeviceID.restype = ctypes.c_void_p
        mt.MTActuatorOpen.argtypes = [ctypes.c_void_p]
        mt.MTActuatorOpen.restype = ctypes.c_int32
        mt.MTActuatorActuate.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_uint32, ctypes.c_float, ctypes.c_float]
        mt.MTActuatorActuate.restype = ctypes.c_int32
        mt.MTActuatorClose.argtypes = [ctypes.c_void_p]
        mt.MTActuatorClose.restype = ctypes.c_int32
        self.mt = mt
        for dev_id in self._device_ids():                    # several multitouch devices; one has the actuator
            act = mt.MTActuatorCreateFromDeviceID(dev_id)
            if act and mt.MTActuatorOpen(act) == 0:
                self.act, self.ok = act, True
                return

    @staticmethod
    def _device_ids():
        try:
            out = subprocess.run(["ioreg", "-l", "-w0", "-c", "AppleMultitouchDevice"], capture_output=True,
                                 text=True, errors="replace", timeout=10).stdout   # has some binary bytes
        except Exception:
            return []
        ids = [int(v) for v in re.findall(r'"Multitouch ID" = (\d+)', out)]
        return sorted(set(ids), key=ids.index)

    def tick(self, level=2):
        """One tick: level 0 = weak, 1 = medium, 2 = strong."""
        if self.ok:
            self.mt.MTActuatorActuate(self.act, LEVELS[int(np.clip(level, 0, len(LEVELS) - 1))], 0, 0.0, 0.0)

    def close(self):
        if self.ok:
            self.mt.MTActuatorClose(self.act)
            self.ok = False


class Haptics:
    """Shaking with a given intensity, made in a background thread. set() it every frame; 0 = still."""
    MIN = 0.03

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.target = 0.0
        self.act = None
        self._run = False
        self._thread = None

    @property
    def available(self):
        self._open()
        return self.act is not None and self.act.ok

    def _open(self):
        if self.enabled and self.act is None:                # opened on the first use only
            self.act = TrackpadActuator()
            if not self.act.ok:
                print("  Trackpad vibration not available on this Mac.")
            else:
                self._run = True
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()

    def set(self, intensity):
        intensity = 0.0 if intensity is None or not np.isfinite(intensity) else float(np.clip(intensity, 0, 1))
        if intensity > self.MIN:
            self._open()
        self.target = intensity if self.enabled else 0.0

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            self.target = 0.0
        return self.enabled

    def _loop(self):
        rng = random.Random()
        while self._run:
            x = self.target
            if x <= self.MIN:
                time.sleep(0.03)
                continue
            level = 0 if x < 0.35 else 1 if x < 0.7 else 2
            self.act.tick(level)
            if x > 0.85 and rng.random() < 0.35:              # a strong quake: double jolts
                time.sleep(0.012)
                self.act.tick(2)
            rate = 4.0 + 18.0 * x                             # ticks per second
            time.sleep(max(0.02, rng.uniform(0.6, 1.4) / rate))   # irregular, like shaking

    def close(self):
        self._run = False
        self.target = 0.0
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self.act is not None:
            self.act.close()


# ----------------------------------------------------------------------------
# Layer value -> intensity
# ----------------------------------------------------------------------------
def haptic_cfg(layer):
    c = layer.cfg.get("haptic")
    if not c:
        return None
    return {} if c is True else dict(c)


def level_at(layer, lat, lon, t, sample=None):
    """Intensity 0..1 at a point. Without "radius_km": the value under the finger on the layer scale.
    With it: the strongest event within the radius, weaker with distance ((1 - d/r)^1.5), so the shaking
    grows as the finger comes closer to it. "min" - how strongly the weakest value on the scale is felt."""
    cfg = haptic_cfg(layer)
    if cfg is None or lat is None:
        return 0.0
    floor, gamma = float(cfg.get("min", 0.15)), float(cfg.get("gamma", 1.0))
    feel = lambda n: floor + (1 - floor) * np.power(np.clip(n, 0, 1), gamma)
    radius = float(cfg.get("radius_km", 0))
    if radius <= 0:
        if sample is None:
            sample = layer.sample(lat, lon, t)
        if sample is None or sample.get("missing") or sample.get("norm") is None:
            return 0.0
        return float(feel(sample["norm"]))
    g = layer.grid(t)
    dlat = abs(layer.lat[1] - layer.lat[0])
    dlon = abs(layer.lon[1] - layer.lon[0])
    reach_lat = radius / 111.2
    i = np.where(np.abs(layer.lat - lat) <= reach_lat + dlat)[0]
    if len(i) == 0:
        return 0.0
    reach_lon = min(180.0, radius / (111.2 * max(math.cos(math.radians(min(abs(lat), 89.0))), 0.02)))
    dl = (layer.lon - lon + 180.0) % 360.0 - 180.0
    j = np.where(np.abs(dl) <= reach_lon + dlon)[0]
    if len(j) == 0:
        return 0.0
    sub = g[np.ix_(i, j)]
    if not np.isfinite(sub).any():
        return 0.0
    la, lo = np.meshgrid(layer.lat[i], layer.lon[j], indexing="ij")
    p1, p2 = np.radians(lat), np.radians(la)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo - lon) / 2) ** 2
    d = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    w = np.clip(1 - d / radius, 0, 1) ** 1.5
    n = layer.norm_array(sub)
    val = np.where(np.isfinite(sub), w * feel(np.nan_to_num(n)), 0.0)
    return float(val.max())


if __name__ == "__main__":
    h = Haptics()
    if not h.available:
        raise SystemExit("No trackpad actuator found.")
    print("Put your hand on the trackpad: 6 s from nothing to the strongest shaking ...")
    t0 = time.time()
    while time.time() - t0 < 6:
        x = (time.time() - t0) / 6
        h.set(x)
        print(f"\r  intensity {x:.2f}", end="", flush=True)
        time.sleep(0.05)
    h.close()
    print("\nDone.")
