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

"""ETL pipeline to build the unified Open-MultiMet daily CPC surface archive.

Ingests NOAA CPC Global Unified Daily Precipitation from the NOAA PSL (Physical
Sciences Laboratory) yearly NetCDF archives published at
``https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.{year}.nc``.

Standardizes spatial dimensions to the Caravan MultiMet specification:

- Dimensions: ``(time, latitude, longitude)``
- Latitude: 360 points from -89.75 to 89.75 (0.5 deg resolution, ascending)
- Longitude: 720 points from -179.75 to 179.75 (0.5 deg, shifted from
  ``[0, 360)``)
- Variable: ``cpc_precipitation`` (float32, mm/day, NaN over missing values)

Outputs directly to ``gs://open-multimet/data/cpc/daily_surface.zarr``.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import shutil
import sys
import tempfile
import time
from typing import Optional, Sequence, Tuple
import urllib.request

import fsspec
try:
  import gcsfs
except ImportError:
  gcsfs = None

import numpy as np
import pandas as pd
import tqdm
import xarray as xr

from multimet.storage import NON_RETRYABLE_ERRORS, resolve_zarr_target

DEFAULT_PROJECT = "global-ungauged-experiments"
DEFAULT_TARGET_ZARR = "open-multimet/gridded-data-archives/CPC/daily_surface.zarr"
DEFAULT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "cpc_cache")

# NOAA PSL publishes CPC Global Unified Precipitation from 1979 onwards.
DEFAULT_START_YEAR = 1979
DEFAULT_END_YEAR = datetime.date.today().year

# Standard CPC 0.5 deg coordinates
CPC_LATS = np.linspace(-89.75, 89.75, 360, dtype=np.float32)
CPC_LONS = np.linspace(-179.75, 179.75, 720, dtype=np.float32)

# Name of the single data variable written to the archive.
CPC_VARIABLE = "cpc_precipitation"

NOAA_PSL_URL_TEMPLATE = (
    "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.{year}.nc"
)


def ensure_psl_cpc_netcdf(
    year: int,
    cache_dir: str = DEFAULT_CACHE_DIR,
    max_retries: int = 4,
) -> str:
  """Downloads and caches yearly NOAA PSL CPC NetCDF file if not already present."""
  os.makedirs(cache_dir, exist_ok=True)
  local_path = os.path.join(cache_dir, f"precip.{year}.nc")
  if os.path.exists(local_path) and os.path.getsize(local_path) > 1024 * 1024:
    return local_path

  url = NOAA_PSL_URL_TEMPLATE.format(year=year)
  temp_path = f"{local_path}.tmp.{os.getpid()}.{time.time_ns()}"
  logging.info("Downloading NOAA PSL CPC NetCDF for %d from %s...", year, url)

  last_error = None
  for attempt in range(max_retries):
    try:
      if os.path.exists(local_path) and os.path.getsize(local_path) > 1024 * 1024:
        return local_path
      req = urllib.request.Request(
          url,
          headers={"User-Agent": "OpenMultiMet/1.1 (Google Research)"},
      )
      with urllib.request.urlopen(req, timeout=180) as response, open(
          temp_path, "wb"
      ) as out_f:
        shutil.copyfileobj(response, out_f)
      if not (os.path.exists(local_path) and os.path.getsize(local_path) > 1024 * 1024):
        os.replace(temp_path, local_path)
      logging.info(
          "✓ Cached %s (%.1f MB)", local_path, os.path.getsize(local_path) / 1e6
      )
      return local_path
    except Exception as e:
      last_error = e
      wait_s = 3 * (2**attempt)
      logging.warning(
          "Failed download attempt %d/%d for year %d: %s. Retrying in %ds...",
          attempt + 1,
          max_retries,
          year,
          e,
          wait_s,
      )
      if os.path.exists(temp_path):
        try:
          os.remove(temp_path)
        except OSError:
          pass
      time.sleep(wait_s)

  raise RuntimeError(
      f"Failed to download NOAA PSL NetCDF for {year} after {max_retries} attempts: {last_error}"
  )


def process_cpc_netcdf_to_dataset(
    nc_path: str,
    target_start_date: Optional[pd.Timestamp] = None,
    target_end_date: Optional[pd.Timestamp] = None,
) -> xr.Dataset:
  """Reads a yearly NOAA PSL NetCDF file and standardizes it to Caravan MultiMet schema.

  Performs spatial transformation:
  1. Inverts latitude axis (PSL: [89.75 .. -89.75] -> [-89.75 .. 89.75]).
  2. Shifts longitude axis (PSL: [0.25 .. 359.75] -> [-179.75 .. 179.75]).
  3. Masks missing values (< 0 or fill values) to np.nan.

  Args:
    nc_path: Path to local precip.{year}.nc file.
    target_start_date: Optional lower bound filter on dates.
    target_end_date: Optional upper bound filter on dates.

  Returns:
    Standardized xarray Dataset with dimensions (time, latitude, longitude).
  """
  with xr.open_dataset(nc_path, decode_timedelta=False) as ds:
    precip_raw = ds["precip"].values
    time_raw = pd.to_datetime(ds["time"].values)

  # Invert latitude: from north->south [89.75 .. -89.75] to south->north [-89.75 .. 89.75]
  precip_lat_inv = precip_raw[:, ::-1, :]

  # Shift longitude: split at 180 deg (index 360) and concatenate [360:] + [:360]
  # Original: [0.25 .. 179.75, 180.25 .. 359.75]
  # Shifted : [-179.75 .. -0.25, 0.25 .. 179.75]
  precip_shifted = np.concatenate(
      [precip_lat_inv[:, :, 360:], precip_lat_inv[:, :, :360]], axis=2
  )

  # Mask missing values (< 0) with NaN
  precip_clean = np.where(precip_shifted < 0, np.nan, precip_shifted).astype(
      np.float32
  )

  dates = pd.DatetimeIndex(time_raw.strftime("%Y-%m-%d"))

  # Date filtering if requested
  if target_start_date is not None or target_end_date is not None:
    mask = np.ones(len(dates), dtype=bool)
    if target_start_date is not None:
      mask &= dates >= target_start_date
    if target_end_date is not None:
      mask &= dates <= target_end_date

    dates = dates[mask]
    precip_clean = precip_clean[mask]

  if len(dates) == 0:
    return None

  return xr.Dataset(
      data_vars={
          "cpc_precipitation": (
              ["time", "latitude", "longitude"],
              precip_clean,
              {
                  "units": "mm/day",
                  "long_name": (
                      "CPC Global Unified Gauge-Based Daily Precipitation"
                  ),
                  "standard_name": "precipitation_amount",
              },
          ),
      },
      coords={
          "time": dates.values,
          "latitude": CPC_LATS,
          "longitude": CPC_LONS,
      },
      attrs={
          "title": (
              "Open-MultiMet NOAA CPC Global Unified Daily Precipitation"
              " Archive"
          ),
          "spatial_resolution": "0.50 degree",
          "description": (
              "Daily gauge-based global precipitation analysis from NOAA PSL"
              " standardized to Caravan MultiMet v1.1"
          ),
          "license": (
              "Usage Restrictions: None (Public Domain / NOAA PSL: "
              "https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html)"
          ),
          "institution": "NOAA PSL / CPC / Open-MultiMet",
          "citation": (
              "Chen et al. (2008) J. Geophys. Res. 113, D04110; Xie et al."
              " (2007) J. Hydrometeorol. 8, 607-626."
          ),
          "product": "CPC",
          "version": "1.1",
      },
  )


def write_batch_to_zarr(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    project: str = DEFAULT_PROJECT,
    is_initial_write: bool = False,
    consolidated: bool = False,
    max_retries: int = 5,
) -> None:
  """Writes or appends a batch of dates to the target Zarr store with exponential retries."""
  full_url, is_gcs = resolve_zarr_target(target_zarr_url)

  clean_path = full_url.replace("gs://", "") if is_gcs else full_url

  for attempt in range(max_retries):
    try:
      if is_gcs:
        if gcsfs is not None:
          fs = gcsfs.GCSFileSystem(project=project)
          mapper = fs.get_mapper(clean_path)
        else:
          mapper = fsspec.get_mapper(full_url)
      else:
        mapper = full_url

      if is_initial_write:
        logging.info("Writing initial Zarr schema to %s...", full_url)
        # Optimal chunking: 30 days x 360 lat x 720 lon (~31MB uncompressed, ~5MB compressed)
        chunk_days = min(30, len(ds_batch["time"]))
        encoding = {
            "cpc_precipitation": {
                "chunks": (
                    chunk_days,
                    len(ds_batch["latitude"]),
                    len(ds_batch["longitude"]),
                ),
            }
        }
        ds_batch.to_zarr(
            mapper, mode="w", consolidated=consolidated, encoding=encoding
        )
        logging.info("✓ Successfully initialized Zarr store.")
      else:
        logging.info(
            "Appending %d dates along time dimension...", len(ds_batch["time"])
        )
        ds_batch.to_zarr(
            mapper, mode="a", append_dim="time", consolidated=consolidated
        )
        logging.info("✓ Successfully appended dates.")
      return
    except NON_RETRYABLE_ERRORS:
      # A missing storage driver or a malformed location will fail the same
      # way on every attempt, so retrying only delays the real error.
      raise
    except Exception as e:
      wait_secs = 5 * (2**attempt)
      logging.warning(
          "Error writing batch to Zarr (attempt %d/%d): %s. Retrying in %ds...",
          attempt + 1,
          max_retries,
          e,
          wait_secs,
      )
      if attempt == max_retries - 1:
        raise
      time.sleep(wait_secs)


_worker_cache_dir: str = DEFAULT_CACHE_DIR
_worker_start_date: Optional[pd.Timestamp] = None
_worker_end_date: Optional[pd.Timestamp] = None
_worker_cleanup_cache: bool = False
_worker_year_starts: dict[int, pd.Timestamp] = {}


def _init_cpc_worker(
    cache_dir: str,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    cleanup_cache: bool,
    year_starts: Optional[dict[int, pd.Timestamp]] = None,
) -> None:
  global _worker_cache_dir, _worker_start_date, _worker_end_date, _worker_cleanup_cache, _worker_year_starts
  _worker_cache_dir = cache_dir
  _worker_start_date = start_date
  _worker_end_date = end_date
  _worker_cleanup_cache = cleanup_cache
  _worker_year_starts = year_starts or {}


def _extract_single_year(year: int) -> Tuple[int, Optional[xr.Dataset]]:
  """Worker task to download and transform a single year of CPC precipitation."""
  global _worker_cache_dir, _worker_start_date, _worker_end_date, _worker_cleanup_cache, _worker_year_starts
  t0 = time.time()
  logging.info(
      "Worker [%d] downloading CPC PSL NetCDF for year %d...",
      os.getpid(),
      year,
  )
  nc_path = ensure_psl_cpc_netcdf(year, cache_dir=_worker_cache_dir)

  y_start = _worker_year_starts.get(year, _worker_start_date)
  logging.info(
      "Worker [%d] standardizing CPC dataset for year %d (start_date: %s)...",
      os.getpid(),
      year,
      y_start.strftime("%Y-%m-%d") if y_start else "None",
  )
  ds_year = process_cpc_netcdf_to_dataset(
      nc_path,
      target_start_date=y_start,
      target_end_date=_worker_end_date,
  )

  if _worker_cleanup_cache and os.path.exists(nc_path):
    try:
      os.remove(nc_path)
      logging.info("Cleaned up cached file: %s", nc_path)
    except OSError:
      pass

  logging.info(
      "Worker [%d] finished year %d in %.2fs",
      os.getpid(),
      year,
      time.time() - t0,
  )
  return year, ds_year


def build_cpc_archive(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    target_zarr: str = DEFAULT_TARGET_ZARR,
    project: str = DEFAULT_PROJECT,
    cache_dir: str = DEFAULT_CACHE_DIR,
    cleanup_cache: bool = False,
    overwrite: bool = False,
    num_workers: Optional[int] = None,
) -> None:
  """Main entry point to execute the NOAA CPC daily gridded archive build."""
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  full_target_url, is_gcs = resolve_zarr_target(target_zarr)

  clean_target = (
      full_target_url.replace("gs://", "") if is_gcs else full_target_url
  )

  if num_workers is None or num_workers <= 0:
    num_workers = min(32, os.cpu_count() or 4)

  t_start_filter = pd.Timestamp(start_date) if start_date else None
  t_end_filter = pd.Timestamp(end_date) if end_date else None

  if t_start_filter:
    start_year = max(start_year, t_start_filter.year)
  if t_end_filter:
    end_year = min(end_year, t_end_filter.year)

  years = list(range(start_year, end_year + 1))
  logging.info(
      "Building CPC archive for %d years: %d to %d (Workers: %d)",
      len(years),
      start_year,
      end_year,
      num_workers,
  )
  logging.info("Target: %s (Project: %s)", full_target_url, project)

  fs = None
  store_exists = False
  if is_gcs:
    if gcsfs is not None:
      fs = gcsfs.GCSFileSystem(project=project)
      store_exists = fs.exists(f"{clean_target}/.zmetadata") or fs.exists(
          f"{clean_target}/zarr.json"
      )
    else:
      try:
        m = fsspec.get_mapper(full_target_url)
        store_exists = ".zmetadata" in m or "zarr.json" in m
      except Exception:
        store_exists = False
  else:
    store_exists = os.path.exists(
        os.path.join(full_target_url, ".zmetadata")
    ) or os.path.exists(os.path.join(full_target_url, "zarr.json"))

  if store_exists and overwrite:
    logging.info("Overwriting existing store at %s...", full_target_url)
    if is_gcs and fs is not None and fs.exists(clean_target):
      fs.rm(clean_target, recursive=True)
    elif not is_gcs and os.path.exists(full_target_url):
      shutil.rmtree(full_target_url)
    store_exists = False

  is_first_write = not store_exists or overwrite
  processed_years = set()
  year_starts: dict[int, pd.Timestamp] = {}
  has_consolidated = False

  if store_exists and not overwrite:
    try:
      mapper = (
          fs.get_mapper(clean_target)
          if (is_gcs and fs is not None)
          else (
              fsspec.get_mapper(full_target_url) if is_gcs else full_target_url
          )
      )
      if is_gcs and fs is not None:
        has_consolidated = fs.exists(f"{clean_target}/.zmetadata")
      elif not is_gcs:
        has_consolidated = os.path.exists(os.path.join(full_target_url, ".zmetadata"))

      existing_ds = xr.open_zarr(mapper, consolidated=has_consolidated)
      existing_times = pd.to_datetime(existing_ds["time"].values)
      max_existing_time = pd.Timestamp(existing_times.max())
      min_existing_time = pd.Timestamp(existing_times.min())
      logging.info(
          "Existing store has %d dates from %s to %s (Consolidated: %s).",
          len(existing_times),
          min_existing_time.strftime("%Y-%m-%d"),
          max_existing_time.strftime("%Y-%m-%d"),
          has_consolidated,
      )

      year_starts = {}
      # Determine which years are already fully written
      for y in years:
        y_dates = existing_times[existing_times.year == y]
        expected_days = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
        if len(y_dates) >= expected_days:
          processed_years.add(y)
        elif len(y_dates) > 0:
          next_date = pd.Timestamp(y_dates.max()) + pd.Timedelta(days=1)
          if t_start_filter is not None:
            year_starts[y] = max(t_start_filter, next_date)
          else:
            year_starts[y] = next_date
          logging.info(
              "Year %d partially present (%d dates up to %s). Resuming from %s.",
              y,
              len(y_dates),
              y_dates.max().strftime("%Y-%m-%d"),
              year_starts[y].strftime("%Y-%m-%d"),
          )

      # Filter remaining years
      remaining_years = [y for y in years if y not in processed_years]
      if not remaining_years:
        logging.info("Store already contains all requested years. Done!")
        return
      logging.info(
          "Resuming build. %d years already present, %d years remaining: %s",
          len(processed_years),
          len(remaining_years),
          remaining_years,
      )
      years = remaining_years
      is_first_write = False
    except Exception as e:
      logging.warning(
          "Could not read existing store for resume: %s. Proceeding with caution...",
          e,
      )

  # Process each year
  total_days_processed = 0
  t0_total = time.time()

  try:
    if num_workers > 1 and len(years) > 1:
      import multiprocessing as mp

      if sys.executable and os.path.exists(sys.executable):
        try:
          mp_ctx = mp.get_context("spawn")
        except Exception:
          mp_ctx = mp.get_context("fork" if hasattr(os, "fork") else None)
      else:
        mp_ctx = mp.get_context("fork" if hasattr(os, "fork") else None)

      logging.info("Spawning worker pool with %d processes...", num_workers)
      with mp_ctx.Pool(
          processes=num_workers,
          initializer=_init_cpc_worker,
          initargs=(cache_dir, t_start_filter, t_end_filter, cleanup_cache, year_starts),
      ) as pool:
        iterator = pool.imap(_extract_single_year, years, chunksize=1)
        for y, ds_year in tqdm.tqdm(
            iterator, total=len(years), desc=f"Processing CPC ({num_workers} workers)"
        ):
          t0_write = time.time()
          if ds_year is None:
            logging.warning("No data returned for year %d within date filters.", y)
            continue

          write_batch_to_zarr(
              ds_year,
              full_target_url,
              project=project,
              is_initial_write=is_first_write,
              consolidated=has_consolidated,
          )
          is_first_write = False

          num_days = len(ds_year["time"])
          total_days_processed += num_days
          logging.info(
              "✓ Year %d completed (%d days) [write took %.2fs]",
              y,
              num_days,
              time.time() - t0_write,
          )
    else:
      logging.info(
          "Running extraction sequentially (%d worker)...",
          1 if num_workers <= 1 else num_workers,
      )
      _init_cpc_worker(cache_dir, t_start_filter, t_end_filter, cleanup_cache, year_starts)
      iterator = (_extract_single_year(y) for y in years)
      for y, ds_year in tqdm.tqdm(
          iterator, total=len(years), desc="Processing CPC (sequential)"
      ):
        t0_write = time.time()
        if ds_year is None:
          logging.warning("No data returned for year %d within date filters.", y)
          continue

        write_batch_to_zarr(
            ds_year,
            full_target_url,
            project=project,
            is_initial_write=is_first_write,
            consolidated=has_consolidated,
        )
        is_first_write = False

        num_days = len(ds_year["time"])
        total_days_processed += num_days
        logging.info(
            "✓ Year %d completed (%d days) [write took %.2fs]",
            y,
            num_days,
            time.time() - t0_write,
        )

    logging.info(
        "🎉 CPC Archive build complete! Processed %d total days across %d years in %.1fs.",
        total_days_processed,
        len(years),
        time.time() - t0_total,
    )
    logging.info("Destination store: %s", full_target_url)
  finally:
    if cleanup_cache and os.path.exists(cache_dir):
      try:
        shutil.rmtree(cache_dir)
        logging.info("Cleaned up cache directory: %s", cache_dir)
      except OSError:
        pass


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the command-line parser for the CPC archive builder."""
  parser = argparse.ArgumentParser(
      prog="build-cpc-archive",
      description="Build the unified CPC daily precipitation archive.",
  )
  parser.add_argument(
      "--start_year",
      type=int,
      default=DEFAULT_START_YEAR,
      help="First NOAA PSL yearly file to ingest (e.g. 1979).",
  )
  parser.add_argument(
      "--end_year",
      type=int,
      default=DEFAULT_END_YEAR,
      help="Last NOAA PSL yearly file to ingest (inclusive).",
  )
  parser.add_argument(
      "--start_date",
      type=str,
      default=None,
      help="Optional lower bound date filter within the year range.",
  )
  parser.add_argument(
      "--end_date",
      type=str,
      default=None,
      help="Optional upper bound date filter within the year range.",
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
      "--cache_dir",
      type=str,
      default=DEFAULT_CACHE_DIR,
      help="Local directory used to cache downloaded NOAA PSL NetCDF files.",
  )
  parser.add_argument(
      "--cleanup_cache",
      action="store_true",
      help="Delete downloaded NetCDF files once they have been written.",
  )
  parser.add_argument(
      "--overwrite",
      action="store_true",
      help="Delete and rebuild the target store instead of resuming it.",
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=min(32, os.cpu_count() or 4),
      help="Number of parallel extraction worker processes.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entry point for ``build-cpc-archive``."""
  args = build_arg_parser().parse_args(argv)
  build_cpc_archive(
      start_year=args.start_year,
      end_year=args.end_year,
      start_date=args.start_date,
      end_date=args.end_date,
      target_zarr=args.target_zarr,
      project=args.project,
      cache_dir=args.cache_dir,
      cleanup_cache=args.cleanup_cache,
      overwrite=args.overwrite,
      num_workers=args.num_workers,
  )


if __name__ == "__main__":
  main(sys.argv[1:])
  sys.stdout.flush()
  sys.stderr.flush()
  # Hard-exit: worker processes hold open HTTP connections to NOAA PSL that can
  # otherwise keep the interpreter alive for minutes after the build finishes.
  os._exit(0)
