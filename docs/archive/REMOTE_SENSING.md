# Satellite / Remote-Sensing Features (Phase 4)

## Status: blocked by missing station coordinates, infrastructure implemented

Run `scripts/phase4_remote_sensing_feasibility.py` against the real Ayase and
Naka canonical datasets:

```
PHASE 4 REMOTE SENSING: BLOCKED BY MISSING STATION COORDINATES
no coordinates on record for 9/9 station(s): ['46八条橋', '48豊橋', '49松富橋',
'50行幸橋', '51道橋', '52内匠橋', '54槐戸橋', '55畷橋', '57綾瀬川合流点前'].
```

All 9 real monitoring stations across both rivers have no recorded
latitude/longitude anywhere in this repository - only Japanese names. This is
not a temporal problem: the observation dates (2022-2025) sit comfortably
inside both Sentinel-2 (operating since 2015) and Landsat coverage. It is a
missing-prerequisite-data problem, and it is the *only* thing blocking this
phase - `pyproj` (coordinate reprojection) is already an installed
dependency, and the feature-extraction machinery below needs no further
libraries to run once real coordinates and real imagery exist.

**Why coordinates were not estimated:** guessing a station's position from
its name or its approximate position along the river would introduce
unverifiable spatial error into every NDVI/NDWI/MNDWI value computed from it,
silently. The only acceptable source is a citable public record - e.g. the
Saitama prefecture monitoring program's own station registry - which has not
been located and integrated into this repository. Until it is, no feature in
this document should be reported for a real station.

## What is implemented (`src/aquanexus/remote_sensing.py`)

All of the following are implemented and tested against synthetic data
(`tests/test_remote_sensing.py`, 14 tests) - none has been run against real
imagery, because none can be pointed at a real station coordinate yet:

- **Spectral indices**: `ndvi`, `ndwi` (McFeeters), `mndwi` (Xu 2006's
  SWIR-substituted variant, chosen to suppress the built-up-area false
  positives plain NDWI is prone to). All are safe at a zero denominator
  (return `NaN`, never raise or emit `inf`).
- **Cloud filtering**: `cloud_free_mask` against Sentinel-2's L2A Scene
  Classification Layer, keeping only vegetation/bare-soil/water/unclassified
  pixels (SCL codes 4/5/6/7) and excluding shadow, cloud, cirrus, snow,
  saturated, no-data, and (conservatively) dark-area pixels.
- **Spatial buffers**: `RasterGrid` + `circular_buffer_mask` +
  `buffer_statistics`, built on plain NumPy rather than `rasterio` (an
  optional, currently-uninstalled dependency this project's `geo` extra
  already lists) - the circular-buffer geometry needs nothing `rasterio`
  provides beyond an affine transform, which `RasterGrid` represents
  directly. `buffer_statistics` reports `valid_fraction` alongside the mean,
  so a buffer that is mostly cloud is visibly weaker evidence than one that
  is mostly clear, not silently averaged as if the two were equivalent.
- **Temporal matching**: `nearest_acquisition`, using the same relative-
  tolerance honesty as `ml.forecasting`'s lag features - an acquisition 40
  days from a field sample is not matched to it as if same-day.
- **The entry guard**: `require_station_coordinates`, which every future
  feature-extraction call must pass through, and which is exactly what
  raises today.

## Candidate buffer radii (not yet used with real data)

`CANDIDATE_BUFFERS_M = (250, 500, 1000, 2000)` metres - chosen for
consideration, not yet justified against a real study area:

- 250 m: a tight near-bank riparian zone; at Sentinel-2's finest (10 m) band
  resolution, roughly 78 pixels of buffer area - resolvable, not pixel-starved.
- 500 m / 1 km: intermediate riparian-to-catchment context.
- 2 km: catchment-scale land-cover context.

Which of these is scientifically appropriate depends on the actual channel
width and adjacent land use at each station - unknowable without the station
coordinates this phase is blocked on. Do not treat this list as a finding;
it is a candidate set for when the blocker clears.

## What would unblock this phase

1. A citable coordinate for each of the 9 stations listed above, from a
   public source (not estimated).
2. A chosen satellite source and access path (Sentinel-2 via a STAC catalogue
   such as Microsoft's Planetary Computer, or Landsat via USGS) - not yet
   selected, since there is nothing to fetch for yet.
3. Only then: an ablation experiment, baseline vs. baseline + remote-sensing
   features, on the identical validation split already used for every other
   phase (`docs/VALIDATION.md`). No performance claim should be made before
   that ablation runs on real, not synthetic, pixels.

## What must not happen

- No NDVI/NDWI/MNDWI value should be reported for any real AquaNexus station
  until `require_station_coordinates` stops raising for it.
- No ablation result ("remote sensing improved R² by X") should be reported
  without that ablation having actually been run on real imagery under the
  project's canonical validation protocol.
