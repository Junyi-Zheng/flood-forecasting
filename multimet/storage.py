# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Storage location handling shared by the gridded archive builders.

The builders accept a Zarr target that may be either a cloud URL or a path on
the local filesystem, and they have to decide which it is before choosing
between an ``fsspec`` mapper and a plain path. Getting that decision wrong is
expensive and confusing: a local path misread as a bucket name produces a URL
like ``gs://C:\\Users\\...`` that fails deep inside the storage stack rather
than at the call site.

This module keeps that classification in one place so both builders agree.
"""

from __future__ import annotations

import atexit
import logging
import os
import re
import sys
import time
from typing import TYPE_CHECKING, Any

import fsspec
import pandas as pd
import zarr

try:
  import gcsfs  # type: ignore[import-untyped]
  import gcsfs.core  # type: ignore[import-untyped]
except ImportError:
  gcsfs = None

if TYPE_CHECKING:
  import xarray as xr

_logger = logging.getLogger(__name__)

# Matches a URI scheme prefix such as "gs://", "s3://" or "file://".
#
# A Windows drive letter cannot match this: "C:\\data" and "C:/data" have no
# "//" after the colon.
_URI_SCHEME_RE = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*)://")

# Matches a Windows drive-qualified path such as "C:\\data" or "C:/data".
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")

_FILE_SCHEME_PREFIX = "file://"

# Errors that will fail identically on every attempt, so a retry loop should
# re-raise them immediately rather than sleeping between attempts. The usual
# case is a location whose protocol has no installed driver (for example a
# ``gs://`` URL without ``gcsfs``), which fsspec reports as an ImportError.
NON_RETRYABLE_ERRORS = (ImportError, TypeError)

_SHUTTING_DOWN = False


def _mark_shutting_down() -> None:
  """Flags interpreter exit and silences benign asyncio teardown logs."""
  global _SHUTTING_DOWN  # noqa: PLW0603
  _SHUTTING_DOWN = True
  logging.getLogger("asyncio").setLevel(logging.CRITICAL)


def resolve_zarr_target(target: str) -> tuple[str, bool]:
  """Classifies a Zarr target as either remote or local.

  The rules, in order:

  1. An explicit URI scheme wins. ``file://`` is unwrapped to a bare local
     path; anything else (``gs://``, ``s3://``, ...) is remote.
  2. Anything that looks like a local filesystem path is local. This covers
     POSIX absolute paths, ``./`` and ``../`` prefixes, ``~``, Windows
     drive-qualified paths, UNC paths, and any path containing a backslash.
  3. Anything else is a bare ``bucket/key`` and is assumed to live in Google
     Cloud Storage, which is what the default targets look like.

  Args:
    target: A Zarr store location.

  Returns:
    A ``(location, is_remote)`` pair. When ``is_remote`` is true, ``location``
    is a fully qualified URL suitable for ``fsspec``. When it is false,
    ``location`` is a plain filesystem path.

  Raises:
    ValueError: If ``target`` is empty.
  """
  if not target:
    raise ValueError("Zarr target must be a non-empty string.")

  scheme_match = _URI_SCHEME_RE.match(target)
  if scheme_match:
    if scheme_match.group("scheme").lower() == "file":
      return target[len(_FILE_SCHEME_PREFIX) :], False
    return target, True

  looks_local = (
      _WINDOWS_DRIVE_RE.match(target) is not None
      # UNC share, e.g. \\server\share. Also caught by the backslash test
      # below, but spelled out because it is a genuinely distinct case.
      or target.startswith("\\\\")
      # Any backslash means a Windows-style path; bucket keys never use them.
      or "\\" in target
      or os.path.isabs(target)
      or target.startswith(("./", "../", "~"))
  )
  if looks_local:
    return target, False

  return f"gs://{target}", True


def is_remote_target(target: str) -> bool:
  """Returns whether ``target`` refers to a remote (cloud) store."""
  return resolve_zarr_target(target)[1]


def get_zarr_mapper(target_zarr_url: str, project: str) -> tuple[str, bool, Any]:
  """Resolves a Zarr target into ``(full_url, is_gcs, mapper)``."""
  full_url, is_gcs = resolve_zarr_target(target_zarr_url)
  clean_path = full_url.replace("gs://", "") if is_gcs else full_url
  if is_gcs:
    if gcsfs is not None:
      fs = gcsfs.GCSFileSystem(project=project)
      mapper = fs.get_mapper(clean_path)
    else:
      mapper = fsspec.get_mapper(full_url)
  else:
    mapper = full_url
  return full_url, is_gcs, mapper


def decode_zarr_time_index(mapper: Any) -> pd.DatetimeIndex:  # noqa: ANN401
  """Reads and decodes the ``time`` coordinate directly from a Zarr group."""
  root = zarr.open_group(mapper, mode="r")
  raw_time = root["time"][:]
  attrs = dict(root["time"].attrs)
  units = attrs.get("units", "")
  if isinstance(units, str) and "days since" in units:
    origin = units.split("days since")[-1].strip().split()[0]
    return pd.to_datetime(raw_time, unit="D", origin=origin)
  return pd.to_datetime(raw_time)


def write_dataset_batch_to_zarr(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    *,
    project: str,
    is_initial_write: bool = False,
    consolidated: bool = True,
    time_chunk_size: int = 1,
    max_retries: int = 5,
) -> None:
  """Writes or appends a batch ``xr.Dataset`` to a Zarr store with backoff."""
  for attempt in range(max_retries):
    try:
      full_url, _, mapper = get_zarr_mapper(target_zarr_url, project)
      if is_initial_write:
        _logger.info("Writing initial Zarr schema to %s...", full_url)
        encoding = {
            var: {
                "chunks": (time_chunk_size,)
                + tuple(
                    len(ds_batch[dim])
                    for dim in ds_batch[var].dims
                    if dim != "time"
                )
            }
            for var in ds_batch.data_vars
        }
        ds_batch.to_zarr(
            mapper, mode="w", consolidated=consolidated, encoding=encoding
        )
      else:
        _logger.info(
            "Appending %d dates along time dimension...", len(ds_batch["time"])
        )
        ds_batch.to_zarr(
            mapper, mode="a", append_dim="time", consolidated=consolidated
        )
      return
    except NON_RETRYABLE_ERRORS:
      raise
    except Exception as err:  # noqa: BLE001
      wait_secs = 5 * (2**attempt)
      _logger.warning(
          "Error writing batch to Zarr (attempt %d/%d): %s. Retrying in %ds...",
          attempt + 1,
          max_retries,
          err,
          wait_secs,
      )
      if attempt == max_retries - 1:
        raise
      time.sleep(wait_secs)


def write_dataset_batch_in_place(
    ds_batch: xr.Dataset,
    target_zarr_url: str,
    *,
    project: str,
    date_to_idx: dict[str, int] | None = None,
    max_retries: int = 5,
) -> None:
  """Writes a batch of dates directly in-place into existing Zarr slices."""
  for attempt in range(max_retries):
    try:
      _, _, mapper = get_zarr_mapper(target_zarr_url, project)
      if date_to_idx is None:
        time_pd = decode_zarr_time_index(mapper)
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
    except (*NON_RETRYABLE_ERRORS, ValueError):
      raise
    except Exception as err:  # noqa: BLE001
      wait_secs = 5 * (2**attempt)
      _logger.warning(
          "Error writing batch in-place (attempt %d/%d): %s. Retrying in %ds...",
          attempt + 1,
          max_retries,
          err,
          wait_secs,
      )
      if attempt == max_retries - 1:
        raise
      time.sleep(wait_secs)


def patch_gcsfs_session_shutdown() -> None:
  """Silences the benign cross-loop aiohttp RuntimeError at interpreter exit.

  During interpreter finalization (``sys.is_finalizing()``), ``gcsfs``'s
  ``weakref.finalize`` callback ``GCSFileSystem.close_session`` attempts to
  schedule ``session.close()`` on ``fsspec.asyn.loop[0]``. When ``zarr`` v3 is
  also active, the connector futures may belong to ``zarr``'s loop, raising
  ``RuntimeError: Task ... got Future ... attached to a different loop``.
  During normal runtime ``orig_close`` is called unchanged; only during
  interpreter finalization (or if ``orig_close`` raises ``RuntimeError``) do we
  fall back to ``connector._close()``.
  """
  if gcsfs is None:
    return

  orig_close = getattr(gcsfs.core.GCSFileSystem, "close_session", None)
  if orig_close is None or getattr(orig_close, "_multimet_patched", False):
    return

  atexit.register(_mark_shutting_down)

  def _safe_close_session(
      loop: object, session: object, asynchronous: bool = False
  ) -> None:
    if _SHUTTING_DOWN or sys.is_finalizing():
      try:
        if not getattr(session, "closed", True):
          connector = getattr(session, "_connector", None)
          if connector is not None:
            connector._close()
      except Exception:  # noqa: BLE001
        pass
      return
    try:
      orig_close(loop, session, asynchronous=asynchronous)
    except Exception:  # noqa: BLE001
      try:
        connector = getattr(session, "_connector", None)
        if connector is not None:
          connector._close()
      except Exception:  # noqa: BLE001
        pass

  _safe_close_session._multimet_patched = True  # type: ignore[attr-defined]
  gcsfs.core.GCSFileSystem.close_session = staticmethod(_safe_close_session)


patch_gcsfs_session_shutdown()

