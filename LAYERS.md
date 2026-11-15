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

Time in a layer is a sequence of **frames**: years (`temperature`) or dates (`ocean_*`). The left finger
goes through all frames from the left to the right edge of the map. When the layer is switched, the
program picks the frame closest to the same moment in time.

Folders in `data/` without `*.meta.json` (for example the curve `data/co2_curve/` - one number per year for
the whole planet) are skipped: there is nothing on the map to point at.

## How to add a new layer

There are three ways, from the simplest.

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

## What the voice says

After the right finger has been still for 3 s, the voice announces: **the place** (country and part of it,
world region, natural region; or sea/ocean and the nearest country), **the time** (year or frame date),
**the layer name and value**, the comparison with the baseline and the `speech_note`. The layer name is
the `title` without the explanation in brackets. The place is found offline from the Natural Earth
outlines (`data/ne_50m_*.geojson`, downloaded automatically) - see [speech.py](speech.py).
The voice is the best installed English macOS voice (Premium/Enhanced voices are picked first);
choose another one with `--voice Daniel`.

## How a layer becomes sound

The same for every layer ([finger_map.py](finger_map.py), `Sonifier.play`):

| Layer data | Sound |
|---|---|
| value on the `vmin…vmax` scale | note pitch (13 pentatonic notes C3…E5) and timbre brightness |
| `(value − baseline) / tension_scale` | beating and a pulsing "heartbeat" when above the baseline |
| finger longitude | stereo (west on the left, east on the right) |
| no data | soft noise |

If no `baseline` is set and the scale is not symmetric around zero, there is no "tension" - only the pitch is heard.

## Useful

- `layers.py list` - check that a new layer is found and its years are right.
- `finger_map.py --selftest` - checks all layers, the converter, the globe modes and the sound without a camera;
  puts a picture of every layer and demo WAV files into `selftest_out/`.
- The `data/` folder is in `.gitignore` (large files). A small layer can be added to git separately:
  `git add -f data/<name>`.
