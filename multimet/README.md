# Open-MultiMet Gridded Archive Builders

This package contains the ETL pipelines that assemble the **unified, analysis
ready gridded meteorological archives** behind Open-MultiMet.

Each upstream provider publishes its data in a different format, on a different
grid, with a different time convention, and split across a different number of
files. These builders normalize all of that into one Zarr store per product,
with a single contiguous daily time axis and a stable schema, so downstream
consumers can open one store and slice it by date.

| Module | Product | Resolution | Coverage | Output store |
| --- | --- | --- | --- | --- |
| [`build_cpc_archive.py`](build_cpc_archive.py) | NOAA CPC Global Unified gauge-based daily precipitation | 0.5° | 1979 → present | `gs://open-multimet/gridded-data-archives/CPC/daily_surface.zarr` |
| [`build_hres_archive.py`](build_hres_archive.py) | ECMWF IFS HRES daily surface forecast, lead days 1–10 | 0.25° | 2016 → present | `gs://open-multimet/gridded-data-archives/HRES/daily_surface.zarr` |
| [`build_imerg_archive.py`](build_imerg_archive.py) | NASA GPM IMERG Early V07 daily precipitation | 0.1° | 2000 → present | `gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr` |

> **Scope.** These builders produce *gridded* archives. Reducing those grids to
> basin-averaged time series is a separate concern and is not part of this
> package.

---

## Installation

The builders ship with the package and are installed as console scripts:

```bash
pip install -e .

build-cpc-archive --help
build-hres-archive --help
build-imerg-archive --help
```

Equivalently, run them as modules:

```bash
python -m multimet.build_cpc_archive --help
python -m multimet.build_hres_archive --help
python -m multimet.build_imerg_archive --help
```

### Dependencies

Everything needed for the CPC builder and for the WeatherBench 2 portion of the
HRES builder is in `environments/environment_cpu.yml`.

Decoding the **ECMWF Open Data** GRIB2 archive (HRES from 2023-07-13 onward)
additionally requires ecCodes, which is imported lazily so that the rest of the
package works without it:

```bash
conda install -c conda-forge python-eccodes eccodes
```

---

## CPC: `build_cpc_archive`

### Source

NOAA PSL publishes one NetCDF file per year:

```
https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.{year}.nc
```

Files are downloaded into a local cache (`--cache_dir`) before processing, and
the download is retried with exponential backoff.

### Transformations

NOAA PSL's native axis layout differs from the Caravan MultiMet convention in
three ways, all of which the builder corrects:

1. **Latitude is flipped.** PSL orders latitude north → south
   (`+89.75 … −89.75`); the archive stores it ascending (`−89.75 … +89.75`) so
   that xarray label-based selection and interpolation behave correctly.
2. **Longitude is rolled.** PSL uses a `[0, 360)` axis (`0.25 … 359.75`); the
   archive uses a signed `[−180, 180)` axis (`−179.75 … 179.75`).
3. **Missing values become NaN.** NOAA flags absent gauge analysis with a large
   negative fill value (`-9.96921e36`); any negative value is masked to
   `np.nan`.

### Output schema

```
Dimensions:            (time, latitude, longitude)
Coordinates:
  * time               datetime64[ns]     daily, midnight UTC
  * latitude           float32   360      -89.75 .. 89.75  (ascending)
  * longitude          float32   720      -179.75 .. 179.75
Data variables:
    cpc_precipitation  float32   (time, latitude, longitude)   mm/day
Chunking:              (30, 360, 720)
```

### Usage

```bash
# Full archive from scratch.
build-cpc-archive --start_year 1979 --overwrite

# Incremental update: resumes from the last date already in the store.
build-cpc-archive

# Bounded backfill into a local store, single process.
build-cpc-archive \
  --start_year 2020 --end_year 2021 \
  --start_date 2020-06-01 --end_date 2021-05-31 \
  --target_zarr /tmp/cpc_test.zarr \
  --num_workers 1
```

---

## HRES: `build_hres_archive`

### Sources

ECMWF IFS HRES is not available from any single archive over the full period, so
the builder stitches three sources together and presents them as one store:

| Date range | Source | Format |
| --- | --- | --- |
| 2016-01-01 → 2023-01-10 | WeatherBench 2 | public Zarr |
| 2023-01-11 → 2023-07-12 | Google Flood Forecasting archive | NetCDF |
| 2023-07-13 → present | ECMWF Open Data | GRIB2 |

Each date is routed to the source that owns it. If a date cannot be retrieved
from any source, the builder writes an **all-NaN slice** rather than skipping
it, which keeps the time axis contiguous — downstream consumers can rely on
`time` being a gap-free daily index and detect missing data via NaN.

### Aggregation

Every 00z initialization is reduced to 10 daily lead steps:

- **`temperature_2m`, `surface_pressure`** — 24-hour mean, computed from the
  four 6-hourly steps that fall inside each lead day.
- **`total_precipitation`** — 24-hour total. WeatherBench 2 already publishes a
  24h accumulation; ECMWF Open Data publishes a run-cumulative total, so it is
  differenced into per-day increments (and floored at zero, since a negative
  precipitation increment can only be numerical noise).
- **`surface_net_solar_radiation`, `surface_net_thermal_radiation`** — also
  differenced from run-cumulative totals, but **not** floored, because net
  thermal radiation is legitimately negative.

> **Known gap.** WeatherBench 2 does not archive `ssr`/`str`. Dates served by
> that source therefore carry NaN for both radiation variables.

ECMWF Open Data served a 0.4° beta grid before switching to 0.25°. The builder
detects which grid a date used and bilinearly upsamples the 0.4° grid so the
store keeps a single consistent resolution.

### Output schema

```
Dimensions:                       (time, lead_time, latitude, longitude)
Coordinates:
  * time                          datetime64[ns]      forecast init date (00z)
  * lead_time                     int32     10        1 .. 10 (days ahead)
  * latitude                      float32   721       -90 .. 90
  * longitude                     float32   1440      0 .. 359.75
Data variables:  (all float32, dims (time, lead_time, latitude, longitude))
    temperature_2m                        K
    surface_pressure                      Pa
    total_precipitation                   m
    surface_net_solar_radiation           J/m^2
    surface_net_thermal_radiation         J/m^2
Chunking:                         (1, 10, 721, 1440)
```

One chunk per forecast date keeps parallel writers lock-free.

### Usage

```bash
# Full archive from scratch.
build-hres-archive --start_date 2016-01-01 --overwrite

# Incremental update: resumes from the last date already in the store.
build-hres-archive

# Repair: recompute a date range and overwrite it in place, leaving the
# surrounding time axis untouched.
build-hres-archive \
  --start_date 2023-02-01 --end_date 2023-02-28 \
  --in_place
```

---

## Operational behaviour

Both builders share the same operational model.

### Resume by default

Running a builder against an existing store **resumes** it: the builder reads
the store's time coordinate, determines what is already present, and extracts
only the missing dates. Re-running a completed build is a no-op. This makes the
builders safe to schedule as a recurring job.

### Write modes

| Mode | Flag | Effect |
| --- | --- | --- |
| Resume | *(default)* | Append only dates not already present. |
| Overwrite | `--overwrite` | Delete the store and rebuild from scratch. |
| In place | `--in_place` (HRES) | Rewrite dates that already exist, without changing the length of the time axis. |

`--in_place` is the repair path: use it when upstream reissues data for dates
you have already ingested. It fails loudly if a requested date is not already in
the store, so it can never silently corrupt the time index.

### Target locations

`--target_zarr` accepts either a cloud URL or a local path. The two are told
apart by [`storage.resolve_zarr_target`](storage.py):

| You pass | Interpreted as |
| --- | --- |
| `gs://bucket/key.zarr`, `s3://...` | Remote, used verbatim |
| `bucket/key.zarr` | Remote — a **bare path is assumed to be GCS** and is rewritten to `gs://bucket/key.zarr` |
| `/data/out.zarr`, `./out.zarr`, `../out.zarr`, `~/out.zarr` | Local |
| `C:\data\out.zarr`, `C:/data/out.zarr`, `\\server\share\out.zarr` | Local |
| `file:///data/out.zarr` | Local, scheme stripped |

> **Note**
> The bare-path rule means a relative path such as `output/cpc.zarr` is treated
> as the GCS location `gs://output/cpc.zarr`, not as a directory beside you.
> Prefix it with `./` to write locally.

Writing to a remote target requires `gcsfs`; without it the write fails
immediately rather than retrying, since a missing driver is not a transient
error.

### Parallelism

`--num_workers` controls a process pool. CPC parallelizes across years; HRES
parallelizes across forecast dates. Workers only extract and transform — all
Zarr writes happen serially in the parent process, so no locking is required.
Set `--num_workers 1` for deterministic, easily debuggable runs.

### Batching and crash safety

Extracted dates are accumulated into batches (`--batch_size` for HRES, one year
at a time for CPC) and written together. Writes are retried with exponential
backoff. If a run dies mid-build, the next run resumes from the last
successfully committed batch.

> **Note.** A batch's data chunks and its time coordinate are written by the
> same `to_zarr` call, but a crash between those two operations can leave data
> chunks whose time labels were never committed. On the next run the builder
> resumes from the last *labelled* date and rewrites those dates, which appends
> a duplicated block. Validate the time axis is strictly increasing after a
> crashed run.

---

## Testing

The unit and integration suites are fully hermetic — they fabricate synthetic
upstream data on the local filesystem and never contact NOAA PSL,
WeatherBench 2, ECMWF, or GCS.

```bash
# Everything hermetic. Canaries are skipped automatically.
pytest multimet/test

# Just the fast unit tests.
pytest multimet/test -m unit

# End-to-end builds against local Zarr stores.
pytest multimet/test -m integration

# Skip the slower multi-process equivalence check.
pytest multimet/test -m "not slow"
```

| File | Scope |
| --- | --- |
| [`test_storage.py`](test/test_storage.py) | Zarr target classification: cloud URLs, POSIX paths, Windows paths, UNC shares |
| [`test_build_cpc_archive.py`](test/test_build_cpc_archive.py) | Grid standardization, NaN masking, date filtering, Zarr write primitive, CLI |
| [`test_build_hres_archive.py`](test/test_build_hres_archive.py) | Accumulation differencing, WB2 aggregation, batch schema, in-place writes, missing-data fallback, CLI |
| [`test_gridded_archive_integration.py`](test/test_gridded_archive_integration.py) | Full builds: resume, overwrite, in-place, gap handling, serial vs. parallel equivalence |

These run automatically on every pull request via
[`.github/workflows/pytest-ci.yml`](../.github/workflows/pytest-ci.yml) (Linux,
macOS, Windows) and
[`.github/workflows/multimet-ci.yml`](../.github/workflows/multimet-ci.yml)
(dedicated fast lane with unit and integration reported separately).

### Canaries

[`test_canary.py`](test/test_canary.py) is **not** part of the hermetic suite.
It checks whether the three upstream feeds are still working, which is a
question about third parties rather than about this code. Canaries are skipped
unless you opt in:

```bash
pytest multimet/test -m canary --run-canary
```

| Canary | Checks |
| --- | --- |
| NOAA PSL | The yearly NetCDF is published, exceeds the 1 MiB cache threshold, and still parses onto the MultiMet grid with live values |
| WeatherBench 2 | The remote Zarr opens anonymously and a date inside its window yields real, physically plausible fields |
| ECMWF open data | A forecast from the last 5 days is published under the expected prefix, and `OPEN_DATA_START_DATE` is still exactly the first available date |
| ECMWF GRIB decoding | Real GRIB2 messages decode and regrid onto 721 × 1440, on **both** the `0p25` and `0p4-beta` grids (requires `eccodes`; skipped otherwise) |

Every canary asserts on the *content* of what came back, never just that a call
succeeded. That is deliberate: `_extract_single_date` degrades to an all-NaN
slice when a source fails, so a broken upstream otherwise produces a
successful-looking build full of holes.

> **Note**
> Canaries run nightly via
> [`.github/workflows/multimet-canary.yml`](../.github/workflows/multimet-canary.yml),
> which has no `pull_request` trigger by design. An ECMWF outage is not a
> reason to block a merge.

---

## Adding a new product

The two builders deliberately share a shape. To add a third:

1. Write a source class (or function) exposing
   `extract_date(date, ...) -> dict[str, np.ndarray] | None`, returning `None`
   when upstream has no data for that date.
2. Define the archive's schema once, as a `build_batch_dataset`-style helper, and
   route every write path through it.
3. Reuse the `write_batch_to_zarr` / resume / `--overwrite` structure so the new
   builder is operationally identical to the existing two.
4. Add a hermetic test module with synthetic upstream data, marked `unit`, plus
   an end-to-end case marked `integration`.
