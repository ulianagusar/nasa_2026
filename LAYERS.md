# Earth Jukebox data layers

`finger_map.py` knows nothing about specific data. Everything it shows and speaks comes from **layers**
([layers.py](layers.py)). A layer is always a **map**: the value depends on the place the right finger
points at. Layers are discovered automatically at start-up and switched with the `l` or `1`–`9` keys
in the program window. For layers with several time frames, the left finger sets the time (its
position along the map); on the globe surfaces the time is set with the keys.

```bash
.venv/bin/python layers.py list          # which layers the program sees
.venv/bin/python finger_map.py --layer temperature --year 1990
```

## Layers available now

| Name | Type | What | Source |
|---|---|---|---|
| `sst` | map, 1 moment | Sea surface temperature, the latest month + the 1971–2000 normal | `sst.mnmean.nc` (NOAA ERSST v5) |
| `temperature` | 147 frames, years 1880–2026 | Land and ocean temperature anomaly vs 1951–1980 | `data/temperature/` (NASA GISTEMP v4) |
| `ocean_sst_video` | 48 frames, weeks 03.2007–02.2008 | Ocean temperature decoded from the colours of a video (ECCO2 model). The "normal" is the yearly mean of each cell | `data/ocean_sst_video/` (NASA SVS 3912) |
| `ocean_flow_activity` | 48 frames, weeks 03.2007–02.2008 | Share of drawn current lines in a cell (0–1) - a visual index, not a speed | `data/ocean_flow_activity/` (NASA SVS 3912) |
| `currents_speed` | 8 frames, months 01–08.2014 | Ocean surface current speed, m/s, monthly mean. **Heard as water** (calm river → roaring sea) instead of a note | `data/currents_speed/` (OSCAR, NASA PO.DAAC via NOAA ERDDAP) |

Time in a layer is a sequence of **frames**: years (`temperature`) or dates (`ocean_*`). The left finger
goes through all frames from the left to the right edge of the map. When the layer is switched, the
program picks the frame closest to the same moment in time.

Folders in `data/` without `*.meta.json` (for example the curve `data/co2_curve/` - one number per year for
the whole planet) are skipped: there is nothing on the map to point at.

## How to add a new layer

There are four ways. The first three need the numbers; the fourth reads them from the colours of a picture.

### 1. I have a NetCDF file (`.nc`) with a `(time, lat, lon)` variable

```bash
.venv/bin/python layers.py from-netcdf path/to/file.nc --var variable_name --name my_layer \
    --title "Precipitation" --units "mm/day" --source "NASA GPM"
```

The converter averages time by year, flips latitudes to run south to north, converts longitudes
to −180…180 and writes the folder `data/my_layer/`. The layer then appears in the program right away.
For anomalies (values around zero) add `--diverging` to get a symmetric scale
(blue–grey–red). If the years should not be averaged: `--monthly`.

### 2. I already have an array in Python

```python
import layers as L
L.write_layer("data", "ndvi", years, lat, lon, data,      # data: [year][lat][lon], NaN = no data
              units="", title="Vegetation greenness (NDVI)", source="NASA MODIS")
```

### 3. Writing the format by hand (this is how `temperature` is made)

A folder `data/<name>/` with two files:

- `<name>.meta.json` - `{"years": [...], "lat": [...], "lon": [...], "vmin": .., "vmax": .., "units": "..", "description": "..", "source": ".."}`.
  For a map without years - `"years": null`. If the frames are more frequent than yearly (weeks, months), add
  `"dates": ["2007-03-28", ...]` - one date per frame; `years` is then just the year of each frame.
- `<name>.u8.bin` - a `uint8` array of shape `[years][lat][lon]`: `0..254` = values from `vmin` to `vmax`,
  `255` = no data. (Instead you can provide `<name>.json` with a `data` field and `null` for gaps.)
- If both files exist, the program uses **the `.json` when it is smaller than 20 MB** (more accurate: no
  quantization and no clipping at `vmin`/`vmax`), otherwise the `.u8.bin`. Override with `"prefer": "u8"` or `"json"` in `jukebox.json`.

The order of latitudes and longitudes does not matter - the program sorts them.

### 4. I only have a video or a picture of a map (with a colour scale)

In the program: start menu → **7. Add a map from a video**. Or from the terminal:

```bash
.venv/bin/python media_to_layer.py video.mp4 --name humidity --title "Relative humidity" --units % \
    --min 0 --max 100 --start 2020-01-01 --end 2020-12-31 --check check.png      # look first
.venv/bin/python media_to_layer.py ... --save                                      # then save
```

Nothing is tuned to one kind of map; any quantity drawn with a colour scale works ([media_to_layer.py](media_to_layer.py)):

1. **Colours.** The colour bar is found automatically, in the frame itself or in a separate picture
   (`--palette scale.png`). The colours can also be typed (`--palette "stops:#313695:0,#ffffbf:15,#a50026:30"`)
   or named (`--palette cmap:turbo`). You give the values at the two ends of the bar; the scale is
   `linear`, `log` (concentrations, aerosols: e.g. 0.4 … 409.6) or `steps` (classes).
2. **Map area.** Found automatically (the solid block under the data); drag the blue box on the page to fix it.
   The map must be in plain lat/lon (equal degree steps). For a part of the world set the region:
   `--region=-25,45,34,72` (west, east, south, north).
3. **Reading.** Every pixel → the nearest colour of the scale in Lab. Pixels too far from all scale colours
   (background, land, labels, blurred edges) are not data; the allowed distance is measured per file
   (clean pictures are strict, compressed video gets more room). Thin things that never change between
   frames (coastlines, a date stamp, a logo) are ignored.
4. **Cells.** Median per cell. For sparse pictures (current lines, dots) 5 frames inside each time step are
   combined; small holes between data are filled from the neighbours (their share is shown).
   `--keep ocean` / `land` drops the other cells.
5. **Time.** A video gives `--steps` time steps (48 by default) evenly between `--start` and `--end`; title or
   credits frames with almost no data at the start or end are left out.

There is no reference data, so the page shows internal checks: the share of map pixels that matched the scale,
the share of filled cells, and the decoded map **painted with the same colours next to the original**, with
coastlines on top (a shifted map box shows up at once). Nothing is saved before Enter.

What is saved: the usual `data/<name>/` (the layer keeps the original colours), plus `import.json` (file,
settings, numbers - how the layer was made) and a preset in `import_presets/`, so the next video of the
same series needs only a check.

Tested on: a map rendered from `wind_speed` (dense video: mean error 1.5% of the scale; sparse lines:
7%), `sst_overview.png` against `sst.mnmean.nc` (mean error 0.23 °C), a log scale from `data/PM2.5_colorbar.png`
(median error 2.7%).

## Layer settings: `data/<name>/jukebox.json` (optional)

Overrides any value from the metadata. All keys are optional:

```json
{
  "title": "Precipitation",
  "units": "mm/day",
  "fmt": "{:.1f}",
  "cmap": "blues",
  "vmin": 0, "vmax": 12,
  "baseline": 3.0,
  "base_label": "the 1981–2010 average",
  "tension_scale": 2.0,
  "default_year": 2025,
  "missing_label": "no measurements",
  "note": "yearly mean",
  "speech_note": "This is the yearly mean."
}
```

| Key | What it does |
|---|---|
| `title`, `units`, `source`, `note` | Texts in the panel and the map title |
| `fmt` | Number format (`"{:+.2f}"` - with a sign, for anomalies) |
| `cmap` | `diverging`, `sequential`, `blues`, `greens` or a list of hex colours `["#fff", "#000"]` |
| `vmin`, `vmax` | Limits of the colour scale **and of the note pitch** (vmin → C3, vmax → E5) |
| `baseline` | The baseline ("normal"): a number or `"mean"` - the mean over all frames in each cell separately. The "tension" of the sound is measured from it |
| `base_label` | How the baseline is named in the panel and by the voice, e.g. `"the 1971–2000 normal"` |
| `tension_scale` | How many units above the baseline = maximum alarm in the sound |
| `default_year` | Which year to start from (e.g. the last complete one) |
| `default_frame` | Which frame to start from (index from 0) - for dated layers |
| `partial_years` | Years with incomplete data, e.g. `[2026]` - the panel shows "(partial year)" |
| `prefer` | `"json"` or `"u8"` - which file to read the data from |
| `speech_note` | A sentence the voice adds after the value (e.g. "This is the difference from the 1951–1980 average.") |
| `max_near_km` | If a point has no data, use the nearest cell only within this distance (default 300 km) |
| `sound` | A sound set instead of a note: `"water"` or the name of a set in `data/sounds/` (see below); `"note"` - notes (also overrides a built-in set) |
| `sound_dir` | Folder with the water recordings and their `sounds.json` (default `data/currents_sounds`) |
| `scale` | `"log"` - colours and note pitch follow the logarithm of the value (for values over several orders of magnitude). Set automatically by the importer |

## What the voice says

After the right finger has been still for 3 s, the voice announces: **the place** (country and part of it,
world region, natural region; or sea/ocean and the nearest country), **the time** (year or frame date),
**the layer name and value**, the comparison with the baseline and the `speech_note`. The layer name is
the `title` without the explanation in brackets. The place is found offline from the Natural Earth
outlines (`data/ne_50m_*.geojson`, downloaded automatically) - see [speech.py](speech.py).
The voice is **Kokoro**, a small neural text-to-speech model (82M parameters, Apache 2.0) that runs offline on
the CPU, about 3× faster than real time on an M2. Its files are not in git; get them once with
`.venv/bin/python speech.py --download` (~350 MB into `models/`). Long messages are spoken sentence by sentence
(the first one starts while the next ones are made, ~1 s); every sentence is kept in `.tts_cache/`, so the menu,
the help and repeated phrases start at once. Other voices: `--voice bf_emma` (British), `af_bella`, `am_michael`,
`bm_george` ... Without the model the best installed macOS voice is used (`--voice Daniel` picks one).
Listen to a voice: `.venv/bin/python speech.py bf_emma`.
The `v` key (or menu item **8. Voice**) turns the spoken messages off and on; the sounds and the chime stay.
The choice is kept for the menu and the next sessions; `--no-voice` starts with the voice off.

## How a layer becomes sound

The same for every layer ([finger_map.py](finger_map.py), `Sonifier.play`):

| Layer data | Sound |
|---|---|
| value on the `vmin…vmax` scale | note pitch (13 pentatonic notes C3…E5) and timbre brightness |
| `(value − baseline) / tension_scale` | beating and a pulsing "heartbeat" when above the baseline |
| finger longitude | stereo (west on the left, east on the right) |
| no data | soft noise |

If no `baseline` is set and the scale is not symmetric around zero, there is no "tension" - only the pitch is heard.

### Water instead of a note (`"sound": "water"`)

Used by `currents_speed`. `data/currents_sounds/` holds four looped recordings, from the calmest to the roughest
(`calm`, `stream`, `rapids`, `roar`), listed in `sounds.json`. All four play at the same time; the value under
the finger only changes their volumes:

```
s = (value - vmin) / (vmax - vmin)          # 0..1
p = s * (N - 1)                             # N = 4 recordings
gain_i = sqrt(max(0, 1 - |p - i|))          # constant total power - no volume dips between sounds
```

Longitude still sets the left/right balance. The mp3 files are decoded once with macOS `afconvert` into
`data/currents_sounds/.decoded/`. The sound sources and licences are listed in `data/currents_sounds/CREDITS.txt`
- only `stream` is described as CC0; check the other three before publishing.

### Your own sounds for any map (sound sets)

Water is just the first **sound set**. Any layer can get its own: start menu → **9. Sounds for a map** →
choose the map → choose 1–8 audio files one after another, **from the lowest values to the highest**
(wav, aiff, mp3, m4a, flac, caf ...) → listen: move the mouse along the colour scale of the map (or ← →) and
hear the mix change exactly as it will under the finger → Enter. The same page can give a map an existing set
(e.g. the water) or turn it back to notes. The last row of the map list, **Delete a sound set**, removes a set with its recordings (after a yes); the maps that used it sound with notes again. The built-in water set cannot be deleted. From the terminal:

```bash
.venv/bin/python sound_pack.py make fire calm_fire.wav crackle.m4a blaze.mp3 --layer temperature
.venv/bin/python sound_pack.py attach temperature note        # back to notes
.venv/bin/python sound_pack.py delete fire                    # delete a set; its maps go back to notes
.venv/bin/python sound_pack.py list                           # sets and which layer uses which
```

What happens to the files ([sound_pack.py](sound_pack.py)): silence at the ends is cut, at most 30 s are kept,
the end is crossfaded into the start (a loop without a seam), the loudness is evened out - `rising` (higher
values a little louder, like the water; default) or `equal` (L on the page). The set is written to
`data/sounds/<set>/` (WAV + `sounds.json` in the same format as the water) and the layer gets
`"sound": "<set>"` in its `jukebox.json` (for `sst`: `data/sst/jukebox.json`).

Mixing: every recording has a `center` on the 0..1 scale (evenly spread by default; can be changed in
`sounds.json`); between two neighbouring centers the two recordings crossfade with constant power
(`gain = sqrt(share)`). With one recording only, its volume follows the value (0.2 → 1).

## Useful

- `layers.py list` - check that a new layer is found and its years are right.
- `finger_map.py --selftest` - checks all layers, the converter, the globe modes and the sound without a camera;
  puts a picture of every layer and demo WAV files into `selftest_out/`.
- The `data/` folder is in `.gitignore` (large files). A small layer can be added to git separately:
  `git add -f data/<name>`.
