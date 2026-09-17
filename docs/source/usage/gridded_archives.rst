=========================
Gridded Archive Builders
=========================

The :mod:`multimet` package contains the ETL pipelines that assemble the
**unified, analysis ready gridded meteorological archives** behind
Open-MultiMet.

Each upstream provider publishes its data in a different format, on a different
grid, with a different time convention, and split across a different number of
files. These builders normalize all of that into one Zarr store per product,
with a single contiguous daily time axis and a stable schema, so that
downstream consumers can open one store and slice it by date.

.. list-table::
   :header-rows: 1
   :widths: 20 30 12 18 20

   * - Module
     - Product
     - Resolution
     - Coverage
     - Output store
   * - :mod:`multimet.build_cpc_archive`
     - NOAA CPC Global Unified gauge-based daily precipitation
     - 0.5°
     - 1979 → present
     - ``gs://open-multimet/data/cpc/daily_surface.zarr``
   * - :mod:`multimet.build_hres_archive`
     - ECMWF IFS HRES daily surface forecast, lead days 1–10
     - 0.25°
     - 2016 → present
     - ``gs://open-multimet/data/hres/daily_surface.zarr``

.. note::

   These builders produce *gridded* archives. Reducing those grids to
   basin-averaged time series is a separate concern and is not part of this
   package.

------------
Installation
------------

The builders are installed with the package and exposed as console scripts:

.. code-block:: bash

   pip install -e .

   build-cpc-archive --help
   build-hres-archive --help

Equivalently, run them as modules:

.. code-block:: bash

   python -m multimet.build_cpc_archive --help
   python -m multimet.build_hres_archive --help

Everything needed for the CPC builder and for the WeatherBench 2 portion of the
HRES builder is in ``environments/environment_cpu.yml``. Decoding the **ECMWF
Open Data** GRIB2 archive (HRES from 2023-07-13 onward) additionally requires
ecCodes, which is imported lazily so the rest of the package works without it:

.. code-block:: bash

   conda install -c conda-forge python-eccodes eccodes

------------------------------
CPC precipitation archive
------------------------------

Source
^^^^^^

NOAA PSL publishes one NetCDF file per year at
``https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.{year}.nc``.
Files are downloaded into a local cache (``--cache_dir``) before processing, and
the download is retried with exponential backoff.

Transformations
^^^^^^^^^^^^^^^

NOAA PSL's native axis layout differs from the Caravan MultiMet convention in
three ways, all of which the builder corrects:

#. **Latitude is flipped.** PSL orders latitude north → south
   (``+89.75 … −89.75``); the archive stores it ascending
   (``−89.75 … +89.75``) so that xarray label-based selection and interpolation
   behave correctly.
#. **Longitude is rolled.** PSL uses a ``[0, 360)`` axis (``0.25 … 359.75``);
   the archive uses a signed ``[−180, 180)`` axis (``−179.75 … 179.75``).
#. **Missing values become NaN.** NOAA flags absent gauge analysis with a large
   negative fill value (``-9.96921e36``); any negative value is masked to
   ``np.nan``.

Output schema
^^^^^^^^^^^^^

.. code-block:: text

   Dimensions:            (time, latitude, longitude)
   Coordinates:
     * time               datetime64[ns]     daily, midnight UTC
     * latitude           float32   360      -89.75 .. 89.75  (ascending)
     * longitude          float32   720      -179.75 .. 179.75
   Data variables:
       cpc_precipitation  float32   (time, latitude, longitude)   mm/day
   Chunking:              (30, 360, 720)

Usage
^^^^^

.. code-block:: bash

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

------------------------
HRES forecast archive
------------------------

Sources
^^^^^^^

ECMWF IFS HRES is not available from any single archive over the full period,
so the builder stitches three sources together and presents them as one store:

.. list-table::
   :header-rows: 1

   * - Date range
     - Source
     - Format
   * - 2016-01-01 → 2023-01-10
     - WeatherBench 2
     - public Zarr
   * - 2023-01-11 → 2023-07-12
     - Google Flood Forecasting archive
     - NetCDF
   * - 2023-07-13 → present
     - ECMWF Open Data
     - GRIB2

Each date is routed to the source that owns it. If a date cannot be retrieved
from any source, the builder writes an **all-NaN slice** rather than skipping
it, which keeps the time axis contiguous — downstream consumers can rely on
``time`` being a gap-free daily index and detect missing data via NaN.

Aggregation
^^^^^^^^^^^

Every 00z initialization is reduced to 10 daily lead steps:

- ``temperature_2m``, ``surface_pressure`` — 24-hour mean, computed from the
  four 6-hourly steps that fall inside each lead day.
- ``total_precipitation`` — 24-hour total. WeatherBench 2 already publishes a
  24h accumulation; ECMWF Open Data publishes a run-cumulative total, so it is
  differenced into per-day increments (and floored at zero, since a negative
  precipitation increment can only be numerical noise).
- ``surface_net_solar_radiation``, ``surface_net_thermal_radiation`` — also
  differenced from run-cumulative totals, but **not** floored, because net
  thermal radiation is legitimately negative.

.. warning::

   WeatherBench 2 does not archive ``ssr``/``str``. Dates served by that source
   therefore carry NaN for both radiation variables.

ECMWF Open Data served a 0.4° beta grid before switching to 0.25°. The builder
detects which grid a date used and bilinearly upsamples the 0.4° grid so the
store keeps a single consistent resolution.

Output schema
^^^^^^^^^^^^^

.. code-block:: text

   Dimensions:                       (time, lead_time, latitude, longitude)
   Coordinates:
     * time                          datetime64[ns]   forecast init date (00z)
     * lead_time                     int32     10     1 .. 10 (days ahead)
     * latitude                      float32   721    -90 .. 90
     * longitude                     float32   1440   0 .. 359.75
   Data variables:  (all float32, dims (time, lead_time, latitude, longitude))
       temperature_2m                        K
       surface_pressure                      Pa
       total_precipitation                   m
       surface_net_solar_radiation           J/m^2
       surface_net_thermal_radiation         J/m^2
   Chunking:                         (1, 10, 721, 1440)

One chunk per forecast date keeps parallel writers lock-free.

Usage
^^^^^

.. code-block:: bash

   # Full archive from scratch.
   build-hres-archive --start_date 2016-01-01 --overwrite

   # Incremental update: resumes from the last date already in the store.
   build-hres-archive

   # Repair: recompute a date range and overwrite it in place, leaving the
   # surrounding time axis untouched.
   build-hres-archive \
     --start_date 2023-02-01 --end_date 2023-02-28 \
     --in_place

----------------------
Operational behaviour
----------------------

Both builders share the same operational model.

Resume by default
^^^^^^^^^^^^^^^^^

Running a builder against an existing store **resumes** it: the builder reads
the store's time coordinate, determines what is already present, and extracts
only the missing dates. Re-running a completed build is a no-op. This makes the
builders safe to schedule as a recurring job.

Write modes
^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Mode
     - Flag
     - Effect
   * - Resume
     - *(default)*
     - Append only dates not already present.
   * - Overwrite
     - ``--overwrite``
     - Delete the store and rebuild from scratch.
   * - In place
     - ``--in_place`` (HRES)
     - Rewrite dates that already exist, without changing the length of the
       time axis.

``--in_place`` is the repair path: use it when upstream reissues data for dates
you have already ingested. It fails loudly if a requested date is not already in
the store, so it can never silently corrupt the time index.

Parallelism
^^^^^^^^^^^

``--num_workers`` controls a process pool. CPC parallelizes across years; HRES
parallelizes across forecast dates. Workers only extract and transform — all
Zarr writes happen serially in the parent process, so no locking is required.
Set ``--num_workers 1`` for deterministic, easily debuggable runs.

Batching and crash safety
^^^^^^^^^^^^^^^^^^^^^^^^^

Extracted dates are accumulated into batches (``--batch_size`` for HRES, one
year at a time for CPC) and written together. Writes are retried with
exponential backoff. If a run dies mid-build, the next run resumes from the last
successfully committed batch.

.. note::

   A batch's data chunks and its time coordinate are written by the same
   ``to_zarr`` call, but a crash between those two operations can leave data
   chunks whose time labels were never committed. On the next run the builder
   resumes from the last *labelled* date and rewrites those dates, which appends
   a duplicated block. Validate that the time axis is strictly increasing after
   a crashed run.

-------
Testing
-------

The test suite is fully hermetic — it fabricates synthetic upstream data on the
local filesystem and never contacts NOAA PSL, WeatherBench 2, ECMWF, or GCS.

.. code-block:: bash

   pytest multimet/test                  # everything
   pytest multimet/test -m unit          # fast unit tests
   pytest multimet/test -m integration   # end-to-end builds
   pytest multimet/test -m "not slow"    # skip the multi-process check

Both suites run automatically on every pull request.
