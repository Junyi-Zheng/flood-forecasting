# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ETL pipeline to build the unified Open-MultiMet daily HRES surface archive.

Ingests ECMWF IFS HRES forecasts from public cloud archives:

#. WeatherBench 2 (2016-01-01 to 2023-01-10): public Zarr archive.
#. ECMWF Open Data (2023-07-13 to present): operational GRIB2 archive.

The intermediate window (2023-01-11 to 2023-07-12) between the end of the
WeatherBench 2 archive and the start of the ECMWF Open Data archive is
initialized with NaN slices so that the daily time coordinate stays contiguous
and can be backfilled in-place.

Aggregates each daily 00z forecast initialization into 10 daily lead steps:

- ``temperature_2m``: 24h mean (K)
- ``surface_pressure``: 24h mean (Pa)
- ``total_precipitation``: 24h accumulated interval (m)
- ``surface_net_solar_radiation``: 24h accumulation (J/m^2)
- ``surface_net_thermal_radiation``: 24h accumulation (J/m^2)

Dates that cannot be retrieved from any source are written as all-NaN slices so
that the time axis stays contiguous.

Outputs directly to ``gs://open-multimet/gridded-data-archives/HRES/daily_surface.zarr``.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from typing import Dict, Optional, Sequence, Tuple

import fsspec
try:
  import gcsfs
except ImportError:
  gcsfs = None

import numpy as np
import pandas as pd
import tqdm
import xarray as xr
import zarr

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

from multimet.storage import NON_RETRYABLE_ERRORS, resolve_zarr_target

DEFAULT_PROJECT = "global-ungauged-experiments"
DEFAULT_TARGET_ZARR = "open-multimet/gridded-data-archives/HRES/daily_surface.zarr"
WB2_HRES_ZARR = "weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
ECMWF_OPEN_DATA_PREFIX = "ecmwf-open-data"

# First forecast initialization date available in WeatherBench 2.
DEFAULT_START_DATE = "2016-01-01"

WB2_CUTOFF_DATE = pd.Timestamp("2023-01-10")
OPEN_DATA_START_DATE = pd.Timestamp("2023-07-13")

LEAD_STEPS_WB2 = [24, 48, 72, 96, 120, 144, 168, 192, 216, 240]

# Number of daily lead steps (lead day 1 .. 10) stored for every init date.
NUM_LEAD_DAYS = 10

# Canonical 0.25 degree HRES grid used by the unified archive.
HRES_LATS = np.linspace(-90.0, 90.0, 721, dtype=np.float32)
HRES_LONS = np.linspace(0.0, 359.75, 1440, dtype=np.float32)

# Surface variables written to the archive, in canonical order.
HRES_VARIABLES = (
    "temperature_2m",
    "surface_pressure",
    "total_precipitation",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
)

HRES_ATTRS = {
    "title": "Open-MultiMet ECMWF HRES Daily Surface Forecast Archive",
    "spatial_resolution": "0.25 degree",
    "description": (
        "Daily-aggregated surface forecast variables (lead days 1..10) from"
        " ECMWF IFS HRES"
    ),
    "license": "CC-BY-4.0",
    "institution": "ECMWF / Open-MultiMet",
}


def deaccumulate(
    accumulated: np.ndarray, clip_negative: bool = False
) -> np.ndarray:
  """Converts a run-cumulative forecast stack into per-lead-day increments.

  ECMWF reports ``tp``, ``ssr`` and ``str`` as totals accumulated since the
  forecast initialization time, so lead day ``d`` must be differenced against
  lead day ``d - 1`` to recover the value for that day alone. Lead day 1 is
  already a single-day total and is passed through unchanged.

  Args:
    accumulated: Array whose leading axis is the lead-day axis.
    clip_negative: If ``True``, negative increments are floored at zero. Use
      this for precipitation, where a negative increment can only be numerical
      noise. Radiation fluxes are genuinely signed and must not be clipped.

  Returns:
    An array of the same shape holding per-lead-day increments.
  """
  daily = np.empty_like(accumulated)
  daily[0] = accumulated[0]
  difference = accumulated[1:] - accumulated[:-1]
  daily[1:] = np.maximum(0.0, difference) if clip_negative else difference
  return daily


class WeatherBench2Source:
  """Extracts daily aggregates from WeatherBench 2 HRES Zarr archive."""

  def __init__(self, zarr_path: str = WB2_HRES_ZARR):
    full_path, is_remote = resolve_zarr_target(zarr_path)
    if is_remote and gcsfs is not None:
      fs = gcsfs.GCSFileSystem(token="anon")
      self.mapper = fs.get_mapper(full_path.replace("gs://", ""))
      self.ds = xr.open_zarr(self.mapper, decode_timedelta=False)
    else:
      self.ds = xr.open_zarr(full_path, decode_timedelta=False)
    self.latitudes = self.ds["latitude"].values.astype(np.float32)
    self.longitudes = self.ds["longitude"].values.astype(np.float32)

  def extract_date(self, date: pd.Timestamp) -> Optional[Dict[str, np.ndarray]]:
    """Extracts 10 lead days for a single forecast date."""
    target_time = np.datetime64(date.strftime("%Y-%m-%dT00:00:00"))
    if target_time not in self.ds["time"].values:
      return None

    day_sel = self.ds.sel(time=target_time)

    # 1. Precipitation: total_precipitation_24hr is precomputed at 24h multiples
    tp_daily = day_sel["total_precipitation_24hr"].sel(
        prediction_timedelta=LEAD_STEPS_WB2
    ).values.astype(np.float32)

    # 2. Temperature and Pressure: 24h mean over each lead day
    temp_slices = []
    pres_slices = []
    for d in range(1, 11):
      steps = [d * 24 - 18, d * 24 - 12, d * 24 - 6, d * 24]
      t_mean = (
          day_sel["2m_temperature"]
          .sel(prediction_timedelta=steps)
          .mean(dim="prediction_timedelta")
          .values
      )
      p_mean = (
          day_sel["surface_pressure"]
          .sel(prediction_timedelta=steps)
          .mean(dim="prediction_timedelta")
          .values
      )
      temp_slices.append(t_mean)
      pres_slices.append(p_mean)

    temp_daily = np.stack(temp_slices, axis=0).astype(np.float32)
    pres_daily = np.stack(pres_slices, axis=0).astype(np.float32)

    # Solar and thermal radiation are unavailable in WB2 HRES
    nan_grid = np.full((10, len(self.latitudes), len(self.longitudes)), np.nan, dtype=np.float32)

    return {
        "temperature_2m": temp_daily,
        "surface_pressure": pres_daily,
        "total_precipitation": tp_daily,
        "surface_net_solar_radiation": nan_grid,
        "surface_net_thermal_radiation": nan_grid.copy(),
    }



class ECMWFOpenDataSource:
  """Extracts daily surface aggregates from ECMWF Open Data GRIB2 using index offsets."""

  def __init__(self):
    self.fs = gcsfs.GCSFileSystem(token="anon") if gcsfs is not None else None
    self.lead_steps = [24, 48, 72, 96, 120, 144, 168, 192, 216, 240]

  def extract_date(
      self, date: pd.Timestamp, target_lat: np.ndarray, target_lon: np.ndarray
  ) -> Optional[Dict[str, np.ndarray]]:
    if self.fs is None:
      return None

    import json
    import eccodes
    from scipy.ndimage import zoom

    date_str = date.strftime("%Y%m%d")
    candidates = [
        f"ecmwf-open-data/{date_str}/00z/ifs/0p25/oper/{date_str}000000",
        f"ecmwf-open-data/{date_str}/00z/0p4-beta/oper/{date_str}000000",
    ]

    prefix = None
    is_0p4 = False
    for cand in candidates:
      if self.fs.exists(f"{cand}-24h-oper-fc.index"):
        prefix = cand
        if "0p4" in cand:
          is_0p4 = True
        break

    if prefix is None:
      return None

    raw_steps = {var: [] for var in ["2t", "sp", "tp", "ssr", "str"]}

    try:
      for step in self.lead_steps:
        idx_path = f"{prefix}-{step}h-oper-fc.index"
        grib_path = f"{prefix}-{step}h-oper-fc.grib2"

        offsets = {}
        with self.fs.open(idx_path, "r") as f:
          for line in f:
            msg = json.loads(line)
            param = msg.get("param")
            if msg.get("levtype") == "sfc" and param in raw_steps:
              offsets[param] = (msg["_offset"], msg["_length"])

        with self.fs.open(grib_path, "rb") as f:
          for param in raw_steps:
            if param in offsets:
              off, length = offsets[param]
              f.seek(off)
              raw_bytes = f.read(length)
              gid = eccodes.codes_new_from_message(raw_bytes)
              vals = eccodes.codes_get_values(gid)
              eccodes.codes_release(gid)

              if is_0p4:
                grid_0p4 = vals.reshape(451, 900)
                grid_0p25 = zoom(grid_0p4, (721 / 451, 1440 / 900), order=1).astype(np.float32)
                raw_steps[param].append(grid_0p25)
              else:
                raw_steps[param].append(vals.reshape(721, 1440).astype(np.float32))
            else:
              raw_steps[param].append(np.full((len(target_lat), len(target_lon)), np.nan, dtype=np.float32))

      return {
          "temperature_2m": np.stack(raw_steps["2t"], axis=0),
          "surface_pressure": np.stack(raw_steps["sp"], axis=0),
          # ECMWF publishes tp/ssr/str as totals accumulated since the forecast
          # initialization, so they must be differenced into daily increments.
          "total_precipitation": deaccumulate(
              np.stack(raw_steps["tp"], axis=0), clip_negative=True
          ),
          "surface_net_solar_radiation": deaccumulate(
              np.stack(raw_steps["ssr"], axis=0)
          ),
          "surface_net_thermal_radiation": deaccumulate(
              np.stack(raw_steps["str"], axis=0)
          ),
      }
    except Exception as e:
      logging.warning("Error decoding Open Data for %s: %s", date_str, e)
      return None


_worker_wb2: Optional[WeatherBench2Source] = None
_worker_open_data: Optional[ECMWFOpenDataSource] = None
_target_lat: Optional[np.ndarray] = None
_target_lon: Optional[np.ndarray] = None


def _init_worker(project: str) -> None:
  del project  # Unused by public anonymous sources.
  global _worker_wb2, _worker_open_data, _target_lat, _target_lon
  _worker_wb2 = WeatherBench2Source()
  _worker_open_data = ECMWFOpenDataSource()
  _target_lat = _worker_wb2.latitudes
  _target_lon = _worker_wb2.longitudes


def _extract_single_date(
    dt: pd.Timestamp,
) -> Tuple[pd.Timestamp, Dict[str, np.ndarray]]:
  global _worker_wb2, _worker_open_data, _target_lat, _target_lon
  date_data = None
  for extract_attempt in range(3):
    try:
      if dt <= WB2_CUTOFF_DATE:
        date_data = _worker_wb2.extract_date(dt)
      elif dt < OPEN_DATA_START_DATE:
        # 6-month gap between WeatherBench 2 (2023-01-10) and ECMWF Open Data (2023-07-13).
        # Initialized with NaN slice to preserve a contiguous daily time axis.
        date_data = None
        break
      else:
        date_data = _worker_open_data.extract_date(dt, _target_lat, _target_lon)
      if date_data is not None:
        break
    except Exception as e:
      if "No module named 'eccodes'" in str(e) or (isinstance(e, ModuleNotFoundError) and "eccodes" in str(e)):
        logging.error("FATAL: 'eccodes' is not installed in this python environment! Please install python-eccodes/eccodes.")
        raise
      logging.warning(
          "Error extracting date %s (attempt %d/3): %s",
          dt.strftime("%Y-%m-%d"),
          extract_attempt + 1,
          e,
      )
      import time

      time.sleep(2)

  if date_data is None:
    logging.warning(
        "No data found for date %s, inserting NaN slice",
        dt.strftime("%Y-%m-%d"),
    )
    shape = (NUM_LEAD_DAYS, len(_target_lat), len(_target_lon))
    date_data = {
        var: np.full(shape, np.nan, dtype=np.float32)
        for var in HRES_VARIABLES
    }
  return dt, date_data


def build_batch_dataset(
    batch_dates: Sequence[pd.Timestamp],
    batch_data: Dict[str, Sequence[np.ndarray]],
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> xr.Dataset:
  """Assembles one write batch into the canonical HRES archive schema.

  This is the single definition of the archive's on-disk layout. Every write
  path (parallel, sequential, and the trailing partial batch) goes through it,
  which guarantees the dimension order, coordinates, dtypes, and global
  attributes stay identical across the whole store.

  Args:
    batch_dates: Forecast initialization dates in the batch, in write order.
    batch_data: Mapping of variable name to a list of ``(lead_time, lat, lon)``
      arrays, one per entry in ``batch_dates``.
    latitudes: Latitude coordinate values of the target grid.
    longitudes: Longitude coordinate values of the target grid.

  Returns:
    An ``xarray.Dataset`` with dims ``(time, lead_time, latitude, longitude)``.
  """
  return xr.Dataset(
      data_vars={
          var: (
              ["time", "lead_time", "latitude", "longitude"],
              np.stack(values, axis=0),
          )
          for var, values in batch_data.items()
      },
      coords={
          "time": list(batch_dates),
          "lead_time": np.arange(1, NUM_LEAD_DAYS + 1, dtype=np.int32),
          "latitude": latitudes,
          "longitude": longitudes,
      },
      attrs=dict(HRES_ATTRS),
  )


def write_batch_to_zarr(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    project: str = DEFAULT_PROJECT,
    is_initial_write: bool = False,
    consolidated: bool = False,
    max_retries: int = 5,
) -> None:
  """Writes or appends a batch of dates to the target Zarr store with retries."""
  full_url, is_remote = resolve_zarr_target(target_zarr_url)
  is_local = not is_remote
  clean_path = full_url.replace("gs://", "") if is_remote else full_url
  mapper = full_url

  for attempt in range(max_retries):
    try:
      if not is_local:
        if gcsfs is not None:
          fs = gcsfs.GCSFileSystem(project=project, requester_pays=project)
          mapper = fs.get_mapper(clean_path)
        else:
          mapper = fsspec.get_mapper(full_url)

      if is_initial_write:
        logging.info("Writing initial Zarr schema to %s...", full_url)
        encoding = {
            var: {"chunks": (1, 10, len(ds_batch["latitude"]), len(ds_batch["longitude"]))}
            for var in ds_batch.data_vars
        }
        ds_batch.to_zarr(mapper, mode="w", consolidated=consolidated, encoding=encoding)
      else:
        logging.info("Appending %d dates along time dimension...", len(ds_batch["time"]))
        ds_batch.to_zarr(mapper, mode="a", append_dim="time", consolidated=consolidated)
      return
    except NON_RETRYABLE_ERRORS:
      # A missing storage driver or a malformed location will fail the same
      # way on every attempt, so retrying only delays the real error.
      raise
    except Exception as e:
      wait_secs = 5 * (2 ** attempt)
      logging.warning(
          "Error writing batch to Zarr (attempt %d/%d): %s. Retrying in %ds...",
          attempt + 1,
          max_retries,
          e,
          wait_secs,
      )
      if attempt == max_retries - 1:
        raise
      import time
      time.sleep(wait_secs)


def write_batch_in_place(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    project: str = DEFAULT_PROJECT,
    date_to_idx: Optional[Dict[str, int]] = None,
    max_retries: int = 5,
) -> None:
  """Writes a batch of dates directly in-place into existing Zarr array slices."""
  full_url, is_remote = resolve_zarr_target(target_zarr_url)
  if not is_remote:
    mapper = full_url
  elif gcsfs is not None:
    fs = gcsfs.GCSFileSystem(project=project, requester_pays=project)
    mapper = fs.get_mapper(full_url.replace("gs://", ""))
  else:
    mapper = fsspec.get_mapper(full_url)

  root = zarr.open_group(mapper, mode="r+")

  if date_to_idx is None:
    existing_ds = xr.open_zarr(mapper, consolidated=False)
    time_pd = pd.to_datetime(existing_ds["time"].values)
    date_to_idx = {t.strftime("%Y-%m-%d"): i for i, t in enumerate(time_pd)}

  batch_dates = pd.to_datetime(ds_batch["time"].values)
  indices = [date_to_idx.get(t.strftime("%Y-%m-%d")) for t in batch_dates]

  if any(idx is None for idx in indices):
    missing = [
        t.strftime("%Y-%m-%d")
        for t, idx in zip(batch_dates, indices, strict=True)
        if idx is None
    ]
    raise ValueError(f"Cannot write in-place: dates {missing} not in target Zarr time coordinate.")

  is_contiguous = indices[-1] - indices[0] == len(indices) - 1 and indices == list(range(indices[0], indices[-1] + 1))

  for attempt in range(max_retries):
    try:
      logging.info(
          "Writing %d dates in-place to Zarr (time indices %d to %d)...",
          len(indices),
          indices[0],
          indices[-1],
      )
      for var in ds_batch.data_vars:
        vals = ds_batch[var].values
        if is_contiguous:
          root[var][indices[0] : indices[-1] + 1] = vals
        else:
          for i, date_idx in enumerate(indices):
            root[var][date_idx] = vals[i]
      return
    except Exception as e:
      wait_secs = 5 * (2 ** attempt)
      logging.warning(
          "Error writing in-place batch to Zarr (attempt %d/%d): %s. Retrying in %ds...",
          attempt + 1,
          max_retries,
          e,
          wait_secs,
      )
      if attempt == max_retries - 1:
        raise
      import time
      time.sleep(wait_secs)


def build_hres_archive(
    start_date: str,
    end_date: str,
    target_zarr: str = DEFAULT_TARGET_ZARR,
    project: str = DEFAULT_PROJECT,
    batch_size: int = 10,
    overwrite: bool = False,
    in_place: bool = False,
    num_workers: Optional[int] = None,
) -> None:
  """Main entry point to execute the HRES archive build."""
  logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
  full_target_url, is_remote = resolve_zarr_target(target_zarr)
  is_local = not is_remote
  clean_target = (
      full_target_url.replace("gs://", "") if is_remote else full_target_url
  )

  if num_workers is None or num_workers <= 0:
    num_workers = min(32, os.cpu_count() or 4)

  dates = pd.date_range(start_date, end_date, freq="1D")
  logging.info("Building HRES archive for %d dates: %s to %s", len(dates), start_date, end_date)
  logging.info(
      "Target: %s (Project: %s, Workers: %d, Batch Size: %d, In-Place: %s)",
      full_target_url,
      project,
      num_workers,
      batch_size,
      in_place,
  )

  fs = None
  store_exists = False
  has_consolidated = False
  if is_local:
    store_exists = os.path.exists(os.path.join(clean_target, ".zmetadata")) or os.path.exists(os.path.join(clean_target, "zarr.json"))
    has_consolidated = os.path.exists(os.path.join(clean_target, ".zmetadata"))
  elif gcsfs is not None:
    fs = gcsfs.GCSFileSystem(project=project, requester_pays=project)
    has_consolidated = fs.exists(f"{clean_target}/.zmetadata")
    store_exists = has_consolidated or fs.exists(f"{clean_target}/zarr.json")
  else:
    try:
      m = fsspec.get_mapper(full_target_url)
      has_consolidated = ".zmetadata" in m
      store_exists = has_consolidated or "zarr.json" in m
    except Exception:
      store_exists = False

  date_to_idx = None
  if in_place:
    if not store_exists:
      raise ValueError(f"Cannot run --in_place: target store {full_target_url} does not exist.")
    if is_local:
      mapper = clean_target
    elif fs is not None:
      mapper = fs.get_mapper(clean_target)
    else:
      mapper = fsspec.get_mapper(full_target_url)
    existing_ds = xr.open_zarr(mapper, consolidated=False)
    time_pd = pd.to_datetime(existing_ds["time"].values)
    date_to_idx = {t.strftime("%Y-%m-%d"): i for i, t in enumerate(time_pd)}
    logging.info(
        "In-place update mode enabled across %d dates (%s to %s).",
        len(dates),
        start_date,
        end_date,
    )
  elif store_exists and overwrite:
    logging.info("Overwriting existing store at %s...", full_target_url)
    if is_local:
      import shutil
      if os.path.exists(clean_target):
        shutil.rmtree(clean_target)
    elif fs is not None and fs.exists(clean_target):
      fs.rm(clean_target, recursive=True)
    store_exists = False

  elif store_exists and not overwrite:
    try:
      if is_local:
        mapper = clean_target
      elif fs is not None:
        mapper = fs.get_mapper(clean_target)
      else:
        mapper = fsspec.get_mapper(full_target_url)
      existing_ds = xr.open_zarr(mapper, consolidated=has_consolidated)
      max_existing_time = pd.Timestamp(existing_ds["time"].values.max())
      logging.info(
          "Existing store has %d dates up to %s.",
          len(existing_ds["time"]),
          max_existing_time.strftime("%Y-%m-%d"),
      )
      resume_start = max_existing_time + pd.Timedelta(days=1)
      if resume_start > pd.Timestamp(end_date):
        logging.info("Store already contains all dates up to %s. Nothing to do!", end_date)
        return
      dates = pd.date_range(resume_start, end_date, freq="1D")
      logging.info(
          "Resuming build from %s (%d dates to process)",
          dates[0].strftime("%Y-%m-%d"),
          len(dates),
      )
    except Exception as e:
      logging.warning("Could not read existing store for resume: %s. Rebuilding...", e)
      store_exists = False

  is_first_write = not store_exists or overwrite

  batch_dates = []
  batch_data = {var: [] for var in HRES_VARIABLES}

  target_lat = HRES_LATS
  target_lon = HRES_LONS

  def flush_batch() -> None:
    """Writes the currently accumulated batch and resets the accumulators."""
    nonlocal batch_dates, batch_data, is_first_write
    if not batch_dates:
      return
    ds_batch = build_batch_dataset(
        batch_dates, batch_data, target_lat, target_lon
    )
    if in_place:
      write_batch_in_place(
          ds_batch, full_target_url, project=project, date_to_idx=date_to_idx
      )
    else:
      write_batch_to_zarr(
          ds_batch,
          full_target_url,
          project=project,
          is_initial_write=is_first_write,
          consolidated=has_consolidated,
      )
      is_first_write = False
    batch_dates = []
    batch_data = {var: [] for var in HRES_VARIABLES}

  if num_workers > 1:
    import multiprocessing as mp
    mp_ctx = mp.get_context("spawn")

    logging.info("Spawning worker pool with %d processes...", num_workers)
    with mp_ctx.Pool(processes=num_workers, initializer=_init_worker, initargs=(project,)) as pool:
      iterator = pool.imap(_extract_single_date, dates, chunksize=1)
      for dt, date_data in tqdm.tqdm(
          iterator, total=len(dates), desc=f"Processing HRES ({num_workers} workers)"
      ):
        batch_dates.append(dt)
        for k in batch_data:
          batch_data[k].append(date_data[k])
        if len(batch_dates) >= batch_size:
          flush_batch()
  else:
    logging.info("Running extraction sequentially (1 worker)...")
    _init_worker(project)
    target_lat = _target_lat
    target_lon = _target_lon
    for dt in tqdm.tqdm(dates, desc="Processing HRES (sequential)"):
      dt, date_data = _extract_single_date(dt)
      batch_dates.append(dt)
      for k in batch_data:
        batch_data[k].append(date_data[k])
      if len(batch_dates) >= batch_size:
        flush_batch()

  flush_batch()

  logging.info("HRES archive build complete for %s to %s!", start_date, end_date)


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the command-line parser for the HRES archive builder."""
  parser = argparse.ArgumentParser(
      prog="build-hres-archive",
      description="Build the unified HRES daily surface forecast archive.",
  )
  parser.add_argument(
      "--start_date",
      type=str,
      default=DEFAULT_START_DATE,
      help="First forecast initialization date to build (YYYY-MM-DD).",
  )
  parser.add_argument(
      "--end_date",
      type=str,
      default=datetime.date.today().isoformat(),
      help="Last forecast initialization date to build (YYYY-MM-DD).",
  )
  parser.add_argument(
      "--target_zarr",
      type=str,
      default=DEFAULT_TARGET_ZARR,
      help="Destination Zarr store (GCS path, or a local path for testing).",
  )
  parser.add_argument(
      "--project",
      type=str,
      default=DEFAULT_PROJECT,
      help="GCP project used for billing/authentication of GCS requests.",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=10,
      help="Number of forecast dates accumulated before each Zarr write.",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Delete and rebuild the target store instead of resuming it.",
  )
  parser.add_argument(
      "--in_place",
      action="store_true",
      help="Rewrite dates that already exist in the target store in place.",
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=min(32, os.cpu_count() or 4),
      help="Number of parallel extraction worker processes.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entry point for ``build-hres-archive``."""
  args = build_arg_parser().parse_args(argv)
  build_hres_archive(
      start_date=args.start_date,
      end_date=args.end_date,
      target_zarr=args.target_zarr,
      project=args.project,
      batch_size=args.batch_size,
      overwrite=args.overwrite,
      in_place=args.in_place,
      num_workers=args.num_workers,
  )
  if argv is None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
  main(sys.argv[1:])
  sys.stdout.flush()
  sys.stderr.flush()
  os._exit(0)
