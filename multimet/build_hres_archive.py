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

Ingests ECMWF IFS HRES forecasts from three sources:
1. WeatherBench 2 (2016-01-01 to 2023-01-10): Zarr archive
2. Google Flood Forecasting (2023-01-11 to 2023-07-11): 00z NetCDF archive
3. ECMWF Open Data (2023-07-12 to Present): Operational GRIB2 archive

Aggregates each daily 00z forecast initialization into 10 daily lead steps:
- temperature_2m: 24h mean (K)
- surface_pressure: 24h mean (Pa)
- total_precipitation: 24h accumulated interval (m)
- surface_net_solar_radiation: 24h flux / accumulation (J/m^2 or W/m^2)
- surface_net_thermal_radiation: 24h flux / accumulation (J/m^2 or W/m^2)

Outputs directly to gs://open-multimet/data/hres/daily_surface.zarr
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import fsspec
try:
  import gcsfs
except ImportError:
  gcsfs = None

try:
  from cloud.bigstore.util import bigstore_file_register
except ImportError:
  pass

try:
  from pyglib import gfile
except ImportError:
  gfile = None

import numpy as np
import pandas as pd
import tqdm
import xarray as xr
import zarr

DEFAULT_PROJECT = "global-ungauged-experiments"
DEFAULT_TARGET_ZARR = "open-multimet/data/hres/daily_surface.zarr"
WB2_HRES_ZARR = "weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
FLOOD_FORECASTING_NC_PATTERN = (
    "ecmwf-downloads/flood-forecasting/single-levels/daily-surface-regridded/{date}-tp-2t-sp-ssr-str-sf.nc"
)
ECMWF_OPEN_DATA_PREFIX = "ecmwf-open-data"

WB2_CUTOFF_DATE = pd.Timestamp("2023-01-10")
OPEN_DATA_START_DATE = pd.Timestamp("2023-07-13")

LEAD_STEPS_WB2 = [24, 48, 72, 96, 120, 144, 168, 192, 216, 240]


class WeatherBench2Source:
  """Extracts daily aggregates from WeatherBench 2 HRES Zarr archive."""

  def __init__(self, zarr_path: str = WB2_HRES_ZARR):
    full_path = zarr_path if zarr_path.startswith("gs://") else f"gs://{zarr_path}"
    if gcsfs is not None:
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


class FloodForecastingGapSource:
  """Extracts daily aggregates from Google Flood Forecasting NetCDF archive."""

  def __init__(
      self,
      pattern: str = FLOOD_FORECASTING_NC_PATTERN,
      project: str = DEFAULT_PROJECT,
  ):
    self.pattern = pattern
    self.project = project
    if gcsfs is not None:
      self.fs = gcsfs.GCSFileSystem(project=self.project, requester_pays=self.project)
    else:
      try:
        self.fs = fsspec.filesystem("gs", project=self.project)
      except Exception:
        self.fs = None

  def extract_date(
      self, date: pd.Timestamp, target_lat: np.ndarray, target_lon: np.ndarray
  ) -> Optional[Dict[str, np.ndarray]]:
    """Extracts 10 lead days for a date within the gap period."""
    path = self.pattern.format(date=date.strftime("%Y-%m-%d"))
    clean_path = path.replace("gs://", "")
    full_path = f"gs://{clean_path}"

    exists = False
    if self.fs is not None:
      exists = self.fs.exists(clean_path) or self.fs.exists(full_path)
    elif gfile is not None:
      exists = gfile.Exists(full_path)

    if not exists:
      return None

    try:
      if self.fs is not None:
        opener = self.fs.open(clean_path)
      elif gfile is not None:
        opener = gfile.Open(full_path, "rb")
      else:
        opener = fsspec.open(full_path, "rb")

      with opener as f:
        ds = xr.open_dataset(f, engine="h5netcdf")
        vars_map = {
            "2t": "temperature_2m",
            "sp": "surface_pressure",
            "tp": "total_precipitation",
            "ssr": "surface_net_solar_radiation",
            "str": "surface_net_thermal_radiation",
        }
        res = {}
        for src_var, dst_var in vars_map.items():
          if src_var in ds:
            res[dst_var] = ds[src_var].values[:10, :, :].astype(np.float32)
          else:
            res[dst_var] = np.full((10, len(target_lat), len(target_lon)), np.nan, dtype=np.float32)
        return res
    except Exception as e:
      logging.warning("Error reading Flood Forecasting file %s: %s", path, e)
      return None


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

      tp_stack = np.stack(raw_steps["tp"], axis=0)
      ssr_stack = np.stack(raw_steps["ssr"], axis=0)
      str_stack = np.stack(raw_steps["str"], axis=0)

      daily_tp = np.empty_like(tp_stack)
      daily_tp[0] = tp_stack[0]
      daily_tp[1:] = np.maximum(0.0, tp_stack[1:] - tp_stack[:-1])

      daily_ssr = np.empty_like(ssr_stack)
      daily_ssr[0] = ssr_stack[0]
      daily_ssr[1:] = ssr_stack[1:] - ssr_stack[:-1]

      daily_str = np.empty_like(str_stack)
      daily_str[0] = str_stack[0]
      daily_str[1:] = str_stack[1:] - str_stack[:-1]

      return {
          "temperature_2m": np.stack(raw_steps["2t"], axis=0),
          "surface_pressure": np.stack(raw_steps["sp"], axis=0),
          "total_precipitation": daily_tp,
          "surface_net_solar_radiation": daily_ssr,
          "surface_net_thermal_radiation": daily_str,
      }
    except Exception as e:
      logging.warning("Error decoding Open Data for %s: %s", date_str, e)
      return None


_worker_wb2: Optional[WeatherBench2Source] = None
_worker_gap: Optional[FloodForecastingGapSource] = None
_worker_open_data: Optional[ECMWFOpenDataSource] = None
_target_lat: Optional[np.ndarray] = None
_target_lon: Optional[np.ndarray] = None


def _init_worker(project: str) -> None:
  global _worker_wb2, _worker_gap, _worker_open_data, _target_lat, _target_lon
  _worker_wb2 = WeatherBench2Source()
  _worker_gap = FloodForecastingGapSource(project=project)
  _worker_open_data = ECMWFOpenDataSource()
  _target_lat = _worker_wb2.latitudes
  _target_lon = _worker_wb2.longitudes


def _extract_single_date(
    dt: pd.Timestamp,
) -> Tuple[pd.Timestamp, Dict[str, np.ndarray]]:
  global _worker_wb2, _worker_gap, _worker_open_data, _target_lat, _target_lon
  date_data = None
  for extract_attempt in range(3):
    try:
      if dt <= WB2_CUTOFF_DATE:
        date_data = _worker_wb2.extract_date(dt)
      elif dt < OPEN_DATA_START_DATE:
        # 6-month gap between WeatherBench 2 (2023-01-10) and ECMWF Open Data (2023-07-13).
        # Fill with NaN slice (to be backfilled from local CNS / internal storage).
        date_data = None
        break
      else:
        date_data = _worker_open_data.extract_date(dt, _target_lat, _target_lon)
      if date_data is not None:
        break
    except Exception as e:
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
    nan_grid = np.full(
        (10, len(_target_lat), len(_target_lon)), np.nan, dtype=np.float32
    )
    date_data = {
        "temperature_2m": nan_grid,
        "surface_pressure": nan_grid.copy(),
        "total_precipitation": nan_grid.copy(),
        "surface_net_solar_radiation": nan_grid.copy(),
        "surface_net_thermal_radiation": nan_grid.copy(),
    }
  return dt, date_data


def write_batch_to_zarr(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    project: str = DEFAULT_PROJECT,
    is_initial_write: bool = False,
    consolidated: bool = False,
    max_retries: int = 5,
) -> None:
  """Writes or appends a batch of dates to the target Zarr store with retries."""
  is_local = target_zarr_url.startswith("/") or target_zarr_url.startswith("./")
  if is_local:
    full_url = target_zarr_url
    clean_path = target_zarr_url
    mapper = target_zarr_url
  else:
    full_url = target_zarr_url if target_zarr_url.startswith("gs://") else f"gs://{target_zarr_url}"
    clean_path = full_url.replace("gs://", "")

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


def build_hres_archive(
    start_date: str,
    end_date: str,
    target_zarr: str = DEFAULT_TARGET_ZARR,
    project: str = DEFAULT_PROJECT,
    batch_size: int = 10,
    overwrite: bool = False,
    num_workers: Optional[int] = None,
) -> None:
  """Main entry point to execute the HRES archive build."""
  logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
  is_local = target_zarr.startswith("/") or target_zarr.startswith("./")
  if is_local:
    full_target_url = target_zarr
    clean_target = target_zarr
  else:
    full_target_url = target_zarr if target_zarr.startswith("gs://") else f"gs://{target_zarr}"
    clean_target = full_target_url.replace("gs://", "")

  if num_workers is None or num_workers <= 0:
    num_workers = min(32, os.cpu_count() or 4)

  dates = pd.date_range(start_date, end_date, freq="1D")
  logging.info("Building HRES archive for %d dates: %s to %s", len(dates), start_date, end_date)
  logging.info(
      "Target: %s (Project: %s, Workers: %d, Batch Size: %d)",
      full_target_url,
      project,
      num_workers,
      batch_size,
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

  if store_exists and overwrite:
    logging.info("Overwriting existing store at %s...", full_target_url)
    if is_local:
      import shutil
      if os.path.exists(clean_target):
        shutil.rmtree(clean_target)
    elif fs is not None and fs.exists(clean_target):
      fs.rm(clean_target, recursive=True)
    store_exists = False

  if store_exists and not overwrite:
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
  batch_data = {
      "temperature_2m": [],
      "surface_pressure": [],
      "total_precipitation": [],
      "surface_net_solar_radiation": [],
      "surface_net_thermal_radiation": [],
  }

  target_lat = np.linspace(-90.0, 90.0, 721, dtype=np.float32)
  target_lon = np.linspace(0.0, 359.75, 1440, dtype=np.float32)

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
        for k in batch_data.keys():
          batch_data[k].append(date_data[k])

        if len(batch_dates) >= batch_size:
          ds_batch = xr.Dataset(
              data_vars={
                  k: (["time", "lead_time", "latitude", "longitude"], np.stack(v, axis=0))
                  for k, v in batch_data.items()
              },
              coords={
                  "time": batch_dates,
                  "lead_time": np.arange(1, 11, dtype=np.int32),
                  "latitude": target_lat,
                  "longitude": target_lon,
              },
              attrs={
                  "title": "Open-MultiMet ECMWF HRES Daily Surface Forecast Archive",
                  "spatial_resolution": "0.25 degree",
                  "description": "Daily-aggregated surface forecast variables (lead days 1..10) from ECMWF IFS HRES",
                  "license": "CC-BY-4.0",
                  "institution": "ECMWF / Open-MultiMet",
              },
          )
          write_batch_to_zarr(
              ds_batch, full_target_url, project=project, is_initial_write=is_first_write, consolidated=has_consolidated
          )
          is_first_write = False
          batch_dates = []
          batch_data = {k: [] for k in batch_data.keys()}
  else:
    logging.info("Running extraction sequentially (1 worker)...")
    _init_worker(project)
    target_lat = _target_lat
    target_lon = _target_lon
    for dt in tqdm.tqdm(dates, desc="Processing HRES (sequential)"):
      dt, date_data = _extract_single_date(dt)
      batch_dates.append(dt)
      for k in batch_data.keys():
        batch_data[k].append(date_data[k])

      if len(batch_dates) >= batch_size:
        ds_batch = xr.Dataset(
            data_vars={
                k: (["time", "lead_time", "latitude", "longitude"], np.stack(v, axis=0))
                for k, v in batch_data.items()
            },
            coords={
                "time": batch_dates,
                "lead_time": np.arange(1, 11, dtype=np.int32),
                "latitude": target_lat,
                "longitude": target_lon,
            },
            attrs={
                "title": "Open-MultiMet ECMWF HRES Daily Surface Forecast Archive",
                "spatial_resolution": "0.25 degree",
                "description": "Daily-aggregated surface forecast variables (lead days 1..10) from ECMWF IFS HRES",
                "license": "CC-BY-4.0",
                "institution": "ECMWF / Open-MultiMet",
            },
        )
        write_batch_to_zarr(
            ds_batch, full_target_url, project=project, is_initial_write=is_first_write, consolidated=has_consolidated
        )
        is_first_write = False
        batch_dates = []
        batch_data = {k: [] for k in batch_data.keys()}

  if batch_dates:
    ds_batch = xr.Dataset(
        data_vars={
            k: (["time", "lead_time", "latitude", "longitude"], np.stack(v, axis=0))
            for k, v in batch_data.items()
        },
        coords={
            "time": batch_dates,
            "lead_time": np.arange(1, 11, dtype=np.int32),
            "latitude": target_lat,
            "longitude": target_lon,
        },
        attrs={
            "title": "Open-MultiMet ECMWF HRES Daily Surface Forecast Archive",
            "spatial_resolution": "0.25 degree",
            "description": "Daily-aggregated surface forecast variables (lead days 1..10) from ECMWF IFS HRES",
            "license": "CC-BY-4.0",
            "institution": "ECMWF / Open-MultiMet",
        },
    )
    write_batch_to_zarr(
        ds_batch, full_target_url, project=project, is_initial_write=is_first_write, consolidated=has_consolidated
    )

  logging.info("HRES archive build complete for %s to %s!", start_date, end_date)


try:
  from absl import flags
  FLAGS = flags.FLAGS
  flags.DEFINE_string("start_date", "2016-01-01", "Start date (YYYY-MM-DD)")
  flags.DEFINE_string("end_date", "2026-09-14", "End date (YYYY-MM-DD)")
  flags.DEFINE_string("target_zarr", DEFAULT_TARGET_ZARR, "GCS target path")
  flags.DEFINE_string("project", DEFAULT_PROJECT, "GCP project ID")
  flags.DEFINE_integer("batch_size", 10, "Days per write batch")
  flags.DEFINE_boolean("overwrite", False, "Overwrite existing store")
  flags.DEFINE_integer(
      "num_workers",
      min(32, os.cpu_count() or 4),
      "Number of parallel extraction workers",
  )
except Exception:
  FLAGS = None


def main(argv: Sequence[str] | None = None) -> None:
  if FLAGS is not None and hasattr(FLAGS, "start_date"):
    start_date = FLAGS.start_date
    end_date = FLAGS.end_date
    target_zarr = FLAGS.target_zarr
    project = FLAGS.project
    batch_size = FLAGS.batch_size
    overwrite = FLAGS.overwrite
    num_workers = FLAGS.num_workers
  else:
    parser = argparse.ArgumentParser(description="Build unified HRES daily surface archive on GCS.")
    parser.add_argument("--start_date", type=str, default="2016-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default="2026-09-14", help="End date (YYYY-MM-DD)")
    parser.add_argument("--target_zarr", type=str, default=DEFAULT_TARGET_ZARR, help="GCS target path")
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT, help="GCP project ID")
    parser.add_argument("--batch_size", type=int, default=10, help="Days per write batch")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing store")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=min(32, os.cpu_count() or 4),
        help="Number of parallel extraction worker processes (default: up to 32 cores)",
    )
    parsed_args, _ = parser.parse_known_args(argv[1:] if argv and len(argv) > 1 else None)
    start_date = parsed_args.start_date
    end_date = parsed_args.end_date
    target_zarr = parsed_args.target_zarr
    project = parsed_args.project
    batch_size = parsed_args.batch_size
    overwrite = parsed_args.overwrite
    num_workers = parsed_args.num_workers

  build_hres_archive(
      start_date=start_date,
      end_date=end_date,
      target_zarr=target_zarr,
      project=project,
      batch_size=batch_size,
      overwrite=overwrite,
      num_workers=num_workers,
  )


if __name__ == "__main__":
  try:
    from absl import app
    app.run(main)
  except ImportError:
    main(sys.argv)

