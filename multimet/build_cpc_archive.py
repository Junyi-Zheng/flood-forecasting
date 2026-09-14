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

Ingests NOAA CPC Global Unified Daily Precipitation from NOAA PSL (Physical
Sciences Laboratory) yearly NetCDF archives:
  https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.{year}.nc

Standardizes spatial dimensions to the Caravan MultiMet specification:
- Dimensions: (time, latitude, longitude)
- Latitude: 360 points from -89.75 to 89.75 (0.5 deg resolution, ascending)
- Longitude: 720 points from -179.75 to 179.75 (0.5 deg resolution, shifted from [0, 360])
- Variable: cpc_precipitation (float32, mm/day, NaN over missing values)

Outputs directly to gs://open-multimet/data/cpc/daily_surface.zarr
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import shutil
import sys
import time
from typing import Optional, Sequence
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
import zarr

DEFAULT_PROJECT = "global-ungauged-experiments"
DEFAULT_TARGET_ZARR = "open-multimet/data/cpc/daily_surface.zarr"
DEFAULT_CACHE_DIR = "/tmp/cpc_cache"
DEFAULT_START_YEAR = 1979
DEFAULT_END_YEAR = 2026

# Standard CPC 0.5 deg coordinates
CPC_LATS = np.linspace(-89.75, 89.75, 360, dtype=np.float32)
CPC_LONS = np.linspace(-179.75, 179.75, 720, dtype=np.float32)

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
  temp_path = f"{local_path}.tmp"
  logging.info("Downloading NOAA PSL CPC NetCDF for %d from %s...", year, url)

  last_error = None
  for attempt in range(max_retries):
    try:
      req = urllib.request.Request(
          url,
          headers={"User-Agent": "OpenMultiMet/1.1 (Google Research)"},
      )
      with urllib.request.urlopen(req, timeout=180) as response, open(
          temp_path, "wb"
      ) as out_f:
        shutil.copyfileobj(response, out_f)
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
    max_retries: int = 5,
) -> None:
  """Writes or appends a batch of dates to the target Zarr store with exponential retries."""
  if target_zarr_url.startswith("/") or target_zarr_url.startswith("./"):
    full_url = target_zarr_url
    is_gcs = False
  elif target_zarr_url.startswith("gs://"):
    full_url = target_zarr_url
    is_gcs = True
  else:
    full_url = f"gs://{target_zarr_url}"
    is_gcs = True

  clean_path = full_url.replace("gs://", "") if is_gcs else full_url

  for attempt in range(max_retries):
    try:
      if is_gcs:
        if gcsfs is not None:
          fs = gcsfs.GCSFileSystem(project=project, requester_pays=project)
          mapper = fs.get_mapper(clean_path)
        else:
          mapper = fsspec.get_mapper(full_url)
      else:
        mapper = target_zarr_url


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
            mapper, mode="w", consolidated=True, encoding=encoding
        )
        logging.info("✓ Successfully initialized Zarr store.")
      else:
        logging.info(
            "Appending %d dates along time dimension...", len(ds_batch["time"])
        )
        ds_batch.to_zarr(mapper, mode="a", append_dim="time", consolidated=True)
        logging.info("✓ Successfully appended dates.")
      return
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
) -> None:
  """Main entry point to execute the NOAA CPC daily gridded archive build."""
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  if target_zarr.startswith("/") or target_zarr.startswith("./"):
    full_target_url = target_zarr
    is_gcs = False
  elif target_zarr.startswith("gs://"):
    full_target_url = target_zarr
    is_gcs = True
  else:
    full_target_url = f"gs://{target_zarr}"
    is_gcs = True

  clean_target = (
      full_target_url.replace("gs://", "") if is_gcs else full_target_url
  )


  t_start_filter = pd.Timestamp(start_date) if start_date else None
  t_end_filter = pd.Timestamp(end_date) if end_date else None

  if t_start_filter:
    start_year = max(start_year, t_start_filter.year)
  if t_end_filter:
    end_year = min(end_year, t_end_filter.year)

  years = list(range(start_year, end_year + 1))
  logging.info(
      "Building CPC archive for %d years: %d to %d",
      len(years),
      start_year,
      end_year,
  )
  logging.info("Target: %s (Project: %s)", full_target_url, project)

  fs = None
  store_exists = False
  if is_gcs:
    if gcsfs is not None:
      fs = gcsfs.GCSFileSystem(project=project, requester_pays=project)
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

  if store_exists and not overwrite:
    try:
      mapper = (
          fs.get_mapper(clean_target)
          if (is_gcs and fs is not None)
          else (
              fsspec.get_mapper(full_target_url) if is_gcs else full_target_url
          )
      )
      existing_ds = xr.open_zarr(mapper, consolidated=True)
      existing_times = pd.to_datetime(existing_ds["time"].values)
      max_existing_time = pd.Timestamp(existing_times.max())
      min_existing_time = pd.Timestamp(existing_times.min())
      logging.info(
          "Existing store has %d dates from %s to %s.",
          len(existing_times),
          min_existing_time.strftime("%Y-%m-%d"),
          max_existing_time.strftime("%Y-%m-%d"),
      )

      # Determine which years are already fully written
      for y in years:
        y_dates = existing_times[existing_times.year == y]
        # Full year check (365 or 366 days for past years)
        expected_days = 366 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 365
        if y < datetime.date.today().year and len(y_dates) >= expected_days:
          processed_years.add(y)

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

  for y in tqdm.tqdm(years, desc="Processing CPC Years"):
    t0_year = time.time()
    logging.info("=== Processing Year %d ===", y)

    # 1. Download NetCDF from NOAA PSL
    nc_path = ensure_psl_cpc_netcdf(y, cache_dir=cache_dir)

    # 2. Standardize to Caravan MultiMet Dataset
    ds_year = process_cpc_netcdf_to_dataset(
        nc_path,
        target_start_date=t_start_filter,
        target_end_date=t_end_filter,
    )

    if ds_year is None:
      logging.warning("No data returned for year %d within date filters.", y)
      continue

    # 3. Write / Append to Zarr
    write_batch_to_zarr(
        ds_year,
        full_target_url,
        project=project,
        is_initial_write=is_first_write,
    )
    is_first_write = False

    num_days = len(ds_year["time"])
    total_days_processed += num_days
    logging.info(
        "✓ Year %d completed (%d days) in %.2fs",
        y,
        num_days,
        time.time() - t0_year,
    )

    # Optional cache cleanup
    if cleanup_cache and os.path.exists(nc_path):
      try:
        os.remove(nc_path)
        logging.info("Cleaned up cached file: %s", nc_path)
      except OSError:
        pass

  logging.info(
      "🎉 CPC Archive build complete! Processed %d total days across %d years in %.1fs.",
      total_days_processed,
      len(years),
      time.time() - t0_total,
  )
  logging.info("Destination store: %s", full_target_url)


try:
  from absl import flags
  FLAGS = flags.FLAGS
  flags.DEFINE_integer(
      "start_year", DEFAULT_START_YEAR, "Start year (e.g. 1979)"
  )
  flags.DEFINE_integer("end_year", DEFAULT_END_YEAR, "End year (e.g. 2026)")
  flags.DEFINE_string(
      "start_date", None, "Optional start date filter (YYYY-MM-DD)"
  )
  flags.DEFINE_string("end_date", None, "Optional end date filter (YYYY-MM-DD)")
  flags.DEFINE_string("target_zarr", DEFAULT_TARGET_ZARR, "GCS target path")
  flags.DEFINE_string("project", DEFAULT_PROJECT, "GCP project ID")
  flags.DEFINE_string("cache_dir", DEFAULT_CACHE_DIR, "Local NetCDF cache dir")
  flags.DEFINE_boolean(
      "cleanup_cache",
      False,
      "Remove downloaded NetCDF files after writing to Zarr",
  )
  flags.DEFINE_boolean("overwrite", False, "Overwrite existing store")
except Exception:
  FLAGS = None


def main(argv: Sequence[str] | None = None) -> None:
  if FLAGS is not None and hasattr(FLAGS, "start_year"):
    start_year = FLAGS.start_year
    end_year = FLAGS.end_year
    start_date = FLAGS.start_date
    end_date = FLAGS.end_date
    target_zarr = FLAGS.target_zarr
    project = FLAGS.project
    cache_dir = FLAGS.cache_dir
    cleanup_cache = FLAGS.cleanup_cache
    overwrite = FLAGS.overwrite
  else:
    parser = argparse.ArgumentParser(
        description="Build unified CPC daily precipitation archive on GCS."
    )
    parser.add_argument(
        "--start_year",
        type=int,
        default=DEFAULT_START_YEAR,
        help="Start year (e.g. 1979)",
    )
    parser.add_argument(
        "--end_year",
        type=int,
        default=DEFAULT_END_YEAR,
        help="End year (e.g. 2026)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Optional start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="Optional end date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--target_zarr",
        type=str,
        default=DEFAULT_TARGET_ZARR,
        help="GCS target path",
    )
    parser.add_argument(
        "--project", type=str, default=DEFAULT_PROJECT, help="GCP project ID"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Local NetCDF cache dir",
    )
    parser.add_argument(
        "--cleanup_cache",
        action="store_true",
        help="Remove downloaded NetCDF files after writing",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing store"
    )

    parsed_args, _ = parser.parse_known_args(
        argv[1:] if argv and len(argv) > 1 else None
    )
    start_year = parsed_args.start_year
    end_year = parsed_args.end_year
    start_date = parsed_args.start_date
    end_date = parsed_args.end_date
    target_zarr = parsed_args.target_zarr
    project = parsed_args.project
    cache_dir = parsed_args.cache_dir
    cleanup_cache = parsed_args.cleanup_cache
    overwrite = parsed_args.overwrite

  build_cpc_archive(
      start_year=start_year,
      end_year=end_year,
      start_date=start_date,
      end_date=end_date,
      target_zarr=target_zarr,
      project=project,
      cache_dir=cache_dir,
      cleanup_cache=cleanup_cache,
      overwrite=overwrite,
  )
  sys.stdout.flush()
  sys.stderr.flush()
  os._exit(0)



if __name__ == "__main__":
  try:
    from absl import app
    app.run(main)
  except ImportError:
    main(sys.argv)
