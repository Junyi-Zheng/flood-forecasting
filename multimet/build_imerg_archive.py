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

"""ETL pipeline to build the unified Open-MultiMet daily IMERG surface archive.

Ingests NASA GPM IMERG Early V07 daily precipitation at its native 0.1 degree
resolution:

- Spatial resolution: 0.1 degree x 0.1 degree global grid
  - ``latitude``: 1,800 points from -89.95 to 89.95 (ascending)
  - ``longitude``: 3,600 points from -179.95 to 179.95 (ascending)
- Temporal resolution: Daily aggregation (00:00:00 UTC to 24:00:00 UTC)
- Variable: ``imerg_precipitation`` (mm/day, ``float32``)

Supported upstream sources:

#. ``gesdisc`` (default): Downloads the official Level 3 Daily NetCDF-4 product
   (``3B-DAY-E.MS.MRG.3IMERG.*.V07*.nc4``) from NASA GES DISC using Earthdata
   Login credentials (via ``~/.netrc``, environment variables, or CLI flags).
#. ``local``: Ingests pre-downloaded daily NetCDF-4 files (``.nc4`` / ``.nc``)
   or 48 half-hourly HDF5 granules (``.RT-H5`` / ``.HDF5``) from a local
   directory.

Outputs directly to ``gs://open-multimet/gridded-data-archives/IMERG/daily_surface.zarr``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import http.client
import io
import logging
import netrc
import os
import random
import shutil
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple
import urllib.parse

import fsspec

try:
  import gcsfs
except ImportError:
  gcsfs = None

try:
  import h5py
except ImportError:
  h5py = None

import numpy as np
import pandas as pd
import requests
import tqdm
import xarray as xr
import zarr

from multimet.storage import NON_RETRYABLE_ERRORS, resolve_zarr_target

DEFAULT_PROJECT = "global-ungauged-experiments"
DEFAULT_TARGET_ZARR = "open-multimet/gridded-data-archives/IMERG/daily_surface.zarr"
DEFAULT_GESDISC_URL = "https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDE.07"
DEFAULT_START_DATE = "2000-06-01"
DEFAULT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "imerg_cache")

LAT_COUNT = 1800
LON_COUNT = 3600
IMERG_LATS = np.linspace(-89.95, 89.95, LAT_COUNT, dtype=np.float32)
IMERG_LONS = np.linspace(-179.95, 179.95, LON_COUNT, dtype=np.float32)
LATS = IMERG_LATS
LONS = IMERG_LONS
IMERG_VARIABLE = "imerg_precipitation"

IMERG_ATTRS = {
    "title": "Open-MultiMet NASA GPM IMERG Early V07 Daily Surface Archive",
    "spatial_resolution": "0.1 degree x 0.1 degree",
    "temporal_resolution": "Daily (00:00:00 UTC to 24:00:00 UTC, left-labeled)",
    "units": "precipitation [mm]",
    "citation": (
        "Huffman, G.J., E.F. Stocker, D.T. Bolvin, E.J. Nelkin, Jackson"
        " Tan (2024), GPM IMERG Early Precipitation L3 Half Hourly 0.1"
        " degree x 0.1 degree V07, Greenbelt, MD, GES DISC,"
        " 10.5067/GPM/IMERG/3B-HH-E/07"
    ),
    "license": "CC-BY-4.0",
    "institution": "NASA GSFC / Open-MultiMet",
}

_netcdf_lock = threading.Lock()


def get_earthdata_credentials_from_netrc(
    netrc_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
  """Reads NASA Earthdata credentials from ``~/.netrc`` if present."""
  path = netrc_path or os.path.expanduser("~/.netrc")
  if not os.path.exists(path):
    return None, None
  try:
    parsed = netrc.netrc(path)
    for host in ("urs.earthdata.nasa.gov", "gpm1.gesdisc.eosdis.nasa.gov"):
      auth_info = parsed.authenticators(host)
      if auth_info:
        return auth_info[0], auth_info[2]
  except Exception as e:  # noqa: BLE001
    logging.warning("Could not parse netrc at %s: %s", path, e)
  return None, None


class EarthdataSession(requests.Session):
  """Custom ``requests.Session`` that preserves auth across NASA URS redirects."""

  AUTH_HOST = "urs.earthdata.nasa.gov"

  def __init__(
      self,
      username: Optional[str] = None,
      password: Optional[str] = None,
      token: Optional[str] = None,
      netrc_path: Optional[str] = None,
  ):
    super().__init__()
    token = token or os.environ.get("EARTHDATA_TOKEN")
    username = username or os.environ.get("EARTHDATA_USERNAME")
    password = password or os.environ.get("EARTHDATA_PASSWORD")

    if not (username and password) and not token:
      netrc_user, netrc_pass = get_earthdata_credentials_from_netrc(netrc_path)
      if netrc_user and netrc_pass:
        username, password = netrc_user, netrc_pass

    self.token = token
    self.username = username
    self.password = password

    if token:
      self.headers.update({"Authorization": f"Bearer {token}"})
    elif username and password:
      self.auth = (username, password)

  def rebuild_auth(self, prepared_request, response) -> None:
    """Preserves Authorization header across redirects to/from NASA URS."""
    headers = prepared_request.headers
    url = prepared_request.url

    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.hostname == self.AUTH_HOST:
      if self.token:
        headers["Authorization"] = f"Bearer {self.token}"
      elif self.username and self.password:
        prepared_request.prepare_auth((self.username, self.password))
      return

    if "Authorization" in headers:
      original_parsed = urllib.parse.urlparse(response.request.url)
      redirect_parsed = urllib.parse.urlparse(url)
      if (
          original_parsed.hostname != redirect_parsed.hostname
          and redirect_parsed.hostname != self.AUTH_HOST
          and original_parsed.hostname != self.AUTH_HOST
      ):
        del headers["Authorization"]

    super().rebuild_auth(prepared_request, response)


def download_daily_imerg(
    url: str,
    dest_path: str,
    session: Optional[requests.Session] = None,
    max_retries: int = 5,
) -> str:
  """Downloads a daily IMERG NetCDF4 file from NASA GES DISC with retries."""
  if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024:
    return dest_path

  if session is None:
    session = EarthdataSession()

  os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
  temp_path = f"{dest_path}.tmp.{os.getpid()}.{time.time_ns()}"

  last_err: Optional[Exception] = None
  for attempt in range(max_retries):
    try:
      if attempt == 0:
        time.sleep(random.uniform(0.05, 0.5))
      else:
        sleep_sec = (2**attempt) + random.uniform(0.5, 2.0)
        logging.warning(
            "Retrying NASA GES DISC download (%d/%d) in %.1fs for: %s",
            attempt + 1,
            max_retries,
            sleep_sec,
            url,
        )
        time.sleep(sleep_sec)

      with session.get(url, stream=True, timeout=120) as resp:
        if resp.status_code in (401, 403):
          raise PermissionError(
              f"NASA GES DISC returned HTTP {resp.status_code} Unauthorized for"
              f" URL:\n  {url}\nAccess to NASA IMERG data requires NASA"
              " Earthdata Login authentication."
          )
        if resp.status_code == 404:
          raise FileNotFoundError(f"404 Not Found: {url}")
        if resp.status_code in (429, 500, 502, 503, 504):
          last_err = requests.HTTPError(
              f"{resp.status_code} Server Error: {resp.reason} for url: {url}",
              response=resp,
          )
          continue

        resp.raise_for_status()
        with open(temp_path, "wb") as f:
          for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
              f.write(chunk)

        if not (os.path.exists(dest_path) and os.path.getsize(dest_path) > 1024):
          os.replace(temp_path, dest_path)
        return dest_path
    except (FileNotFoundError, PermissionError):
      raise
    except (
        requests.RequestException,
        http.client.RemoteDisconnected,
        TimeoutError,
    ) as e:
      last_err = e
      continue
    finally:
      if os.path.exists(temp_path):
        try:
          os.remove(temp_path)
        except OSError:
          pass

  if last_err is not None:
    raise last_err
  raise RuntimeError(f"Failed to download {url} after {max_retries} attempts.")


def parse_imerg_netcdf_to_grid(nc_path: str) -> np.ndarray:
  """Reads a NASA IMERG daily NetCDF4 file into a (1800, 3600) float32 grid."""
  with _netcdf_lock:
    with xr.open_dataset(nc_path) as ds:
      da = ds["precipitation"] if "precipitation" in ds else ds["precipitationCal"]
      if "time" in da.dims:
        da = da.squeeze("time")
      if da.dims == ("lon", "lat"):
        da = da.transpose("lat", "lon")

      vals = da.values.astype(np.float32)
      vals = np.where(vals < 0.0, np.nan, vals)
      return vals


def _read_and_parse_h5_granule(fpath: str) -> Optional[np.ndarray]:
  """Reads a single half-hourly IMERG HDF5 granule into a (1800, 3600) array."""
  try:
    with open(fpath, "rb") as f:
      content = f.read()

    if h5py is None:
      raise ImportError("h5py is required to parse raw HDF5 granules.")

    with h5py.File(io.BytesIO(content), "r") as h5:
      grid = h5["Grid"]
      if "precipitation" in grid:
        ds = grid["precipitation"]
      elif "precipitationCal" in grid:
        ds = grid["precipitationCal"]
      else:
        return None

      raw = np.squeeze(ds[()])
      grid_rate = np.transpose(raw).astype(np.float32)
      return grid_rate
  except Exception as e:  # noqa: BLE001
    logging.warning("Failed reading granule %s: %s", fpath, e)
    return None


class GESDISCImergSource:
  """Downloads and extracts daily gridded precipitation from NASA GES DISC NetCDF-4 files."""

  def __init__(
      self,
      base_url: str = DEFAULT_GESDISC_URL,
      username: Optional[str] = None,
      password: Optional[str] = None,
      token: Optional[str] = None,
      netrc_path: Optional[str] = None,
      cache_dir: Optional[str] = None,
      cleanup_cache: bool = False,
  ):
    self.base_url = base_url.rstrip("/")
    self.username = username
    self.password = password
    self.token = token
    self.netrc_path = netrc_path
    self.cleanup_cache = cleanup_cache
    self._thread_local = threading.local()
    self.cache_dir = cache_dir or os.environ.get(
        "MULTIMET_IMERG_CACHE", DEFAULT_CACHE_DIR
    )
    os.makedirs(self.cache_dir, exist_ok=True)

  @property
  def session(self) -> EarthdataSession:
    if not hasattr(self._thread_local, "session"):
      self._thread_local.session = EarthdataSession(
          username=self.username,
          password=self.password,
          token=self.token,
          netrc_path=self.netrc_path,
      )
    return self._thread_local.session

  def extract_date(self, date: pd.Timestamp) -> Optional[np.ndarray]:
    """Downloads daily NetCDF-4 granule and returns (1800, 3600) array in mm."""
    date = pd.to_datetime(date)
    date_str = date.strftime("%Y%m%d")
    year = date.year
    month = date.month

    candidate_suffixes = (
        ["V07C", "V07B", "V07", "V07A"]
        if year >= 2026
        else ["V07B", "V07C", "V07", "V07A"]
    )

    cached_path = None
    for suffix in candidate_suffixes:
      cand_fn = f"3B-DAY-E.MS.MRG.3IMERG.{date_str}-S000000-E235959.{suffix}.nc4"
      cand_path = os.path.join(self.cache_dir, cand_fn)
      if os.path.exists(cand_path) and os.path.getsize(cand_path) > 1000:
        cached_path = cand_path
        break

    if cached_path is None:
      for suffix in candidate_suffixes:
        cand_fn = f"3B-DAY-E.MS.MRG.3IMERG.{date_str}-S000000-E235959.{suffix}.nc4"
        cand_url = f"{self.base_url}/{year}/{month:02d}/{cand_fn}"
        cand_path = os.path.join(self.cache_dir, cand_fn)
        try:
          download_daily_imerg(cand_url, cand_path, session=self.session)
          cached_path = cand_path
          break
        except FileNotFoundError:
          continue
        except PermissionError:
          raise
        except Exception as e:  # noqa: BLE001
          if "404" in str(e):
            continue
          logging.warning(
              "Failed downloading IMERG for %s (%s): %s", date_str, cand_fn, e
          )

    if cached_path is None or not os.path.exists(cached_path):
      logging.warning(
          "No valid IMERG granule found for %s across candidate versions",
          date_str,
      )
      return None

    try:
      return parse_imerg_netcdf_to_grid(cached_path)
    except Exception as e:  # noqa: BLE001
      logging.warning("Error reading NetCDF file %s: %s", cached_path, e)
      return None
    finally:
      if self.cleanup_cache and cached_path and os.path.exists(cached_path):
        try:
          os.remove(cached_path)
        except OSError:
          pass


class LocalImergSource:
  """Extracts daily gridded precipitation from a local directory of NetCDF-4 or HDF5 files."""

  def __init__(self, local_dir: str, granule_workers: int = 8):
    self.local_dir = local_dir
    self.granule_workers = max(1, granule_workers)

  def extract_date(self, date: pd.Timestamp) -> Optional[np.ndarray]:
    date = pd.to_datetime(date)
    date_str = date.strftime("%Y%m%d")
    month_str = date.strftime("%Y%m")

    if not os.path.isdir(self.local_dir):
      return None

    # 1. Check for daily NetCDF-4 files in local_dir
    for fname in sorted(os.listdir(self.local_dir)):
      if date_str in fname and fname.endswith((".nc4", ".nc")):
        nc_path = os.path.join(self.local_dir, fname)
        try:
          return parse_imerg_netcdf_to_grid(nc_path)
        except Exception as e:  # noqa: BLE001
          logging.warning("Error reading local NetCDF file %s: %s", nc_path, e)
          return None

    # 2. Check for 48 half-hourly HDF5 granules in local_dir or local_dir/{YYYYMM}
    search_dirs = [self.local_dir, os.path.join(self.local_dir, month_str)]
    h5_files: List[str] = []
    for dpath in search_dirs:
      if os.path.isdir(dpath):
        for fname in os.listdir(dpath):
          if date_str in fname and fname.endswith((".RT-H5", ".HDF5", ".h5")):
            h5_files.append(os.path.join(dpath, fname))
        if h5_files:
          break

    if not h5_files:
      return None

    h5_files = sorted(h5_files)
    if len(h5_files) != 48:
      logging.warning(
          "Date %s has %d/48 HDF5 granules in %s. Skipping date.",
          date.strftime("%Y-%m-%d"),
          len(h5_files),
          self.local_dir,
      )
      return None

    daily_sum = np.zeros((LAT_COUNT, LON_COUNT), dtype=np.float64)
    valid_counts = np.zeros((LAT_COUNT, LON_COUNT), dtype=np.int32)

    for fpath in h5_files:
      grid_rate = _read_and_parse_h5_granule(fpath)
      if grid_rate is None:
        continue
      valid_mask = (grid_rate >= 0.0) & (~np.isnan(grid_rate))
      daily_sum[valid_mask] += grid_rate[valid_mask] * 0.5
      valid_counts[valid_mask] += 1

    has_data = valid_counts > 0
    result = np.full((LAT_COUNT, LON_COUNT), np.nan, dtype=np.float32)
    result[has_data] = daily_sum[has_data].astype(np.float32)
    return result


def build_batch_dataset(
    batch_dates: Sequence[pd.Timestamp],
    batch_grids: Sequence[np.ndarray],
    latitudes: np.ndarray = IMERG_LATS,
    longitudes: np.ndarray = IMERG_LONS,
) -> xr.Dataset:
  """Assembles a batch of daily IMERG grids into the canonical Zarr schema."""
  return xr.Dataset(
      data_vars={
          IMERG_VARIABLE: (
              ["time", "latitude", "longitude"],
              np.stack(batch_grids, axis=0).astype(np.float32),
          ),
      },
      coords={
          "time": list(batch_dates),
          "latitude": latitudes,
          "longitude": longitudes,
      },
      attrs=dict(IMERG_ATTRS),
  )


def write_batch_to_zarr(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    project: str = DEFAULT_PROJECT,
    is_initial_write: bool = False,
    consolidated: bool = True,
    max_retries: int = 5,
) -> None:
  """Writes or appends a batch of dates to the target Zarr store with retries."""
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
        encoding = {
            var: {
                "chunks": (
                    1,
                    len(ds_batch["latitude"]),
                    len(ds_batch["longitude"]),
                )
            }
            for var in ds_batch.data_vars
        }
        ds_batch.to_zarr(
            mapper, mode="w", consolidated=consolidated, encoding=encoding
        )
      else:
        logging.info(
            "Appending %d dates along time dimension...", len(ds_batch["time"])
        )
        ds_batch.to_zarr(
            mapper, mode="a", append_dim="time", consolidated=consolidated
        )
      return
    except NON_RETRYABLE_ERRORS:
      raise
    except Exception as e:  # noqa: BLE001
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


def write_batch_in_place(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    project: str = DEFAULT_PROJECT,
    date_to_idx: Optional[Dict[str, int]] = None,
    max_retries: int = 5,
) -> None:
  """Writes a batch of dates directly in-place into existing Zarr array slices."""
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

      if date_to_idx is None:
        existing_ds = xr.open_zarr(mapper, consolidated=False)
        time_pd = pd.to_datetime(existing_ds["time"].values)
        date_to_idx = {t.strftime("%Y-%m-%d"): i for i, t in enumerate(time_pd)}

      batch_times = pd.to_datetime(ds_batch["time"].values)
      missing_dates = [
          t.strftime("%Y-%m-%d")
          for t in batch_times
          if t.strftime("%Y-%m-%d") not in date_to_idx
      ]
      if missing_dates:
        raise ValueError(
            f"Dates not found in target store for in-place write: {missing_dates}"
        )

      indices = [date_to_idx[t.strftime("%Y-%m-%d")] for t in batch_times]
      root = zarr.open_group(mapper, mode="r+")
      is_contiguous = indices == list(
          range(indices[0], indices[0] + len(indices))
      )

      for var in ds_batch.data_vars:
        vals = ds_batch[var].values
        if is_contiguous:
          root[var][indices[0] : indices[-1] + 1] = vals
        else:
          for i, target_idx in enumerate(indices):
            root[var][target_idx : target_idx + 1] = vals[i : i + 1]
      return
    except (NON_RETRYABLE_ERRORS, ValueError):
      raise
    except Exception as e:  # noqa: BLE001
      wait_secs = 5 * (2**attempt)
      logging.warning(
          "Error writing batch in-place (attempt %d/%d): %s. Retrying in %ds...",
          attempt + 1,
          max_retries,
          e,
          wait_secs,
      )
      if attempt == max_retries - 1:
        raise
      time.sleep(wait_secs)


def build_imerg_archive(
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = None,
    source_type: str = "auto",
    target_zarr: str = DEFAULT_TARGET_ZARR,
    project: str = DEFAULT_PROJECT,
    batch_size: int = 30,
    num_workers: int = 4,
    granule_workers: int = 8,
    cache_dir: str = DEFAULT_CACHE_DIR,
    cleanup_cache: bool = False,
    overwrite: bool = False,
    in_place: bool = False,
    local_dir: Optional[str] = None,
    earthdata_username: Optional[str] = None,
    earthdata_password: Optional[str] = None,
    earthdata_token: Optional[str] = None,
    netrc_path: Optional[str] = None,
) -> None:
  """Builds or updates the unified IMERG daily native resolution Zarr archive."""
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
  )
  full_target_url, is_gcs = resolve_zarr_target(target_zarr)
  clean_target = full_target_url.replace("gs://", "") if is_gcs else full_target_url

  if end_date is None:
    end_date = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()

  dates = pd.date_range(start_date, end_date, freq="1D")
  logging.info(
      "Building IMERG archive for %d dates: %s to %s",
      len(dates),
      start_date,
      end_date,
  )
  logging.info("Target: %s (Project: %s)", full_target_url, project)

  fs = None
  store_exists = False
  has_consolidated = False
  if is_gcs:
    if gcsfs is not None:
      fs = gcsfs.GCSFileSystem(project=project)
      has_consolidated = fs.exists(f"{clean_target}/.zmetadata")
      store_exists = has_consolidated or fs.exists(f"{clean_target}/zarr.json")
    else:
      try:
        m = fsspec.get_mapper(full_target_url)
        has_consolidated = ".zmetadata" in m
        store_exists = has_consolidated or "zarr.json" in m
      except Exception:  # noqa: BLE001
        store_exists = False
  else:
    has_consolidated = os.path.exists(
        os.path.join(full_target_url, ".zmetadata")
    )
    store_exists = has_consolidated or os.path.exists(
        os.path.join(full_target_url, "zarr.json")
    )

  date_to_idx = None
  if in_place:
    if not store_exists:
      raise ValueError(
          f"Cannot run --in_place: target store {full_target_url} does not exist."
      )
    mapper = (
        (fs.get_mapper(clean_target) if fs is not None else fsspec.get_mapper(full_target_url))
        if is_gcs
        else full_target_url
    )
    existing_ds = xr.open_zarr(mapper, consolidated=False)
    time_pd = pd.to_datetime(existing_ds["time"].values)
    date_to_idx = {t.strftime("%Y-%m-%d"): i for i, t in enumerate(time_pd)}
  elif store_exists and overwrite:
    logging.info("Overwriting existing store at %s...", full_target_url)
    if is_gcs:
      if fs is not None and fs.exists(clean_target):
        fs.rm(clean_target, recursive=True)
    else:
      if os.path.exists(full_target_url):
        shutil.rmtree(full_target_url)
    store_exists = False
  elif store_exists and not overwrite:
    try:
      mapper = (
          (fs.get_mapper(clean_target) if fs is not None else fsspec.get_mapper(full_target_url))
          if is_gcs
          else full_target_url
      )
      existing_ds = xr.open_zarr(mapper, consolidated=has_consolidated)
      max_existing_time = pd.Timestamp(existing_ds["time"].values.max())
      logging.info(
          "Existing store has %d dates up to %s.",
          len(existing_ds["time"]),
          max_existing_time.strftime("%Y-%m-%d"),
      )
      resume_start = max_existing_time + pd.Timedelta(days=1)
      if resume_start > pd.Timestamp(end_date):
        logging.info(
            "Store already contains all dates up to %s. Nothing to do!",
            end_date,
        )
        return
      dates = pd.date_range(resume_start, end_date, freq="1D")
      logging.info(
          "Resuming build from %s (%d dates to process)",
          dates[0].strftime("%Y-%m-%d"),
          len(dates),
      )
    except Exception as e:  # noqa: BLE001
      logging.warning(
          "Could not read existing store for resume: %s. Rebuilding...", e
      )
      store_exists = False

  is_first_write = not store_exists or overwrite

  if source_type == "local" or (local_dir and os.path.isdir(local_dir)):
    if not local_dir:
      raise ValueError("--local_dir must be provided when --source=local.")
    logging.info("Using local directory archive from %s", local_dir)
    source = LocalImergSource(local_dir=local_dir, granule_workers=granule_workers)
  else:
    logging.info("Using NASA GES DISC public daily NetCDF archive")
    source = GESDISCImergSource(
        username=earthdata_username,
        password=earthdata_password,
        token=earthdata_token,
        netrc_path=netrc_path,
        cache_dir=cache_dir,
        cleanup_cache=cleanup_cache,
    )

  try:
    total_dates = len(dates)
    for i in tqdm.trange(
        0, total_dates, batch_size, desc="Processing IMERG Batches"
    ):
      chunk_dates = dates[i : i + batch_size]

      if num_workers > 1 and len(chunk_dates) > 1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(chunk_dates), num_workers)
        ) as executor:
          future_to_date = {
              executor.submit(source.extract_date, dt): dt for dt in chunk_dates
          }
          results_map = {}
          for future in concurrent.futures.as_completed(future_to_date):
            dt_item = future_to_date[future]
            try:
              results_map[dt_item] = future.result()
            except PermissionError:
              raise
            except Exception as e:  # noqa: BLE001
              logging.warning(
                  "Error extracting date %s: %s",
                  dt_item.strftime("%Y-%m-%d"),
                  e,
              )
              results_map[dt_item] = None

        batch_dates = list(chunk_dates)
        batch_grids = []
        for dt in chunk_dates:
          grid = results_map.get(dt)
          if grid is None:
            logging.warning(
                "No data found for date %s, inserting NaN grid",
                dt.strftime("%Y-%m-%d"),
            )
            grid = np.full((LAT_COUNT, LON_COUNT), np.nan, dtype=np.float32)
          batch_grids.append(grid)
      else:
        batch_dates = []
        batch_grids = []
        for dt in chunk_dates:
          grid = source.extract_date(dt)
          if grid is None:
            logging.warning(
                "No data found for date %s, inserting NaN grid",
                dt.strftime("%Y-%m-%d"),
            )
            grid = np.full((LAT_COUNT, LON_COUNT), np.nan, dtype=np.float32)
          batch_dates.append(dt)
          batch_grids.append(grid)

      ds_batch = build_batch_dataset(batch_dates, batch_grids)
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
            consolidated=has_consolidated or is_first_write,
        )
        is_first_write = False

    logging.info(
        "IMERG archive build complete for %s to %s!", start_date, end_date
    )
  finally:
    if cleanup_cache and cache_dir and os.path.exists(cache_dir):
      try:
        shutil.rmtree(cache_dir)
        logging.info("Cleaned up cache directory: %s", cache_dir)
      except OSError:
        pass


def build_arg_parser() -> argparse.ArgumentParser:
  """Builds the command-line parser for the IMERG archive builder."""
  parser = argparse.ArgumentParser(
      prog="build-imerg-archive",
      description="Build the unified IMERG daily surface precipitation archive.",
  )
  parser.add_argument(
      "--start_date",
      type=str,
      default=DEFAULT_START_DATE,
      help="First date to ingest (YYYY-MM-DD, default 2000-06-01).",
  )
  parser.add_argument(
      "--end_date",
      type=str,
      default=None,
      help="Last date to ingest (YYYY-MM-DD, default yesterday UTC).",
  )
  parser.add_argument(
      "--source",
      type=str,
      default="auto",
      choices=["auto", "gesdisc", "local"],
      help="Upstream source type: 'auto' / 'gesdisc' (NASA GES DISC) or 'local'.",
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
      default=30,
      help="Number of daily slices accumulated before each Zarr write.",
  )
  parser.add_argument(
      "--num_workers",
      type=int,
      default=4,
      help="Number of concurrent date download/extraction workers.",
  )
  parser.add_argument(
      "--granule_workers",
      type=int,
      default=8,
      help="Number of concurrent HDF5 granule workers per date (local HDF5 mode).",
  )
  parser.add_argument(
      "--cache_dir",
      type=str,
      default=DEFAULT_CACHE_DIR,
      help="Local directory used to stage downloaded NASA GES DISC NetCDF files.",
  )
  parser.add_argument(
      "--cleanup_cache",
      action="store_true",
      help="Delete downloaded NetCDF files immediately after processing and clean cache_dir on exit.",
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
      "--local_dir",
      type=str,
      default=None,
      help="Local directory containing pre-downloaded NetCDF-4 or HDF5 granules.",
  )
  parser.add_argument(
      "--earthdata_username",
      type=str,
      default=None,
      help="NASA Earthdata Login username.",
  )
  parser.add_argument(
      "--earthdata_password",
      type=str,
      default=None,
      help="NASA Earthdata Login password.",
  )
  parser.add_argument(
      "--earthdata_token",
      type=str,
      default=None,
      help="NASA Earthdata Bearer token.",
  )
  parser.add_argument(
      "--netrc_path",
      type=str,
      default=None,
      help="Custom path to a .netrc file containing Earthdata credentials.",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entry point for ``build-imerg-archive``."""
  args = build_arg_parser().parse_args(argv)
  build_imerg_archive(
      start_date=args.start_date,
      end_date=args.end_date,
      source_type=args.source,
      target_zarr=args.target_zarr,
      project=args.project,
      batch_size=args.batch_size,
      num_workers=args.num_workers,
      granule_workers=args.granule_workers,
      cache_dir=args.cache_dir,
      cleanup_cache=args.cleanup_cache,
      overwrite=args.overwrite,
      in_place=args.in_place,
      local_dir=args.local_dir,
      earthdata_username=args.earthdata_username,
      earthdata_password=args.earthdata_password,
      earthdata_token=args.earthdata_token,
      netrc_path=args.netrc_path,
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
