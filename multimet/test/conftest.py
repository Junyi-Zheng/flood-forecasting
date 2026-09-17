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

"""Shared fixtures for the gridded archive builder tests.

Every fixture here produces *synthetic* data on the local filesystem. The unit
and integration suites never touch NOAA PSL, WeatherBench 2, ECMWF Open Data,
or GCS, so they are hermetic and safe to run in CI.

The one exception is ``test_canary.py``, which deliberately talks to those live
services. Those tests are skipped unless ``--run-canary`` is passed; see
:func:`pytest_collection_modifyitems` below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from multimet import build_hres_archive as hres_module

_CANARY_FLAG = "--run-canary"


def pytest_addoption(parser: pytest.Parser) -> None:
  """Registers the opt-in flag for live upstream canaries."""
  parser.addoption(
      _CANARY_FLAG,
      action="store_true",
      default=False,
      help=(
          "Run canaries against live third-party feeds (NOAA PSL, "
          "WeatherBench 2, ECMWF Open Data). Requires network access."
      ),
  )


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
  """Skips canary tests unless they were explicitly requested.

  This is a flag rather than a marker expression in ``addopts`` on purpose. A
  ``-m`` filter in ``addopts`` is silently replaced by any ``-m`` passed on the
  command line, and the repository-wide CI already runs ``-m "not gpu"`` --
  which would quietly re-enable every canary in the blocking test job.
  """
  if config.getoption(_CANARY_FLAG):
    return

  skip_canary = pytest.mark.skip(
      reason=f"live upstream canary; pass {_CANARY_FLAG} to run"
  )
  for item in items:
    if "canary" in item.keywords:
      item.add_marker(skip_canary)

# NOAA PSL publishes CPC on a fixed 0.5 degree global grid. The builder relies
# on the exact axis layout, so the synthetic files must reproduce it verbatim:
# latitude descends from +89.75, longitude runs 0.25 .. 359.75.
PSL_LATS = np.linspace(89.75, -89.75, 360, dtype=np.float32)
PSL_LONS = np.linspace(0.25, 359.75, 720, dtype=np.float32)

# NOAA's fill value for missing gauge analysis cells.
PSL_MISSING_VALUE = -9.96921e36

# Tiny stand-in grid for the HRES tests. The real archive is 721 x 1440, which
# is far too large to materialise in a unit test; the builder is grid agnostic
# so a 4 x 8 grid exercises exactly the same code paths.
FAKE_HRES_LATS = np.linspace(-90.0, 90.0, 4, dtype=np.float32)
FAKE_HRES_LONS = np.linspace(0.0, 315.0, 8, dtype=np.float32)


def make_psl_precip_array(
    dates: pd.DatetimeIndex,
    *,
    fill_value: float = 1.0,
    missing_cells: Iterable[tuple[int, int, int]] = (),
) -> np.ndarray:
  """Builds a synthetic NOAA PSL ``precip`` array.

  Args:
    dates: Dates to generate, one slice each.
    fill_value: Base value written to every cell. Each day ``i`` is offset by
      ``i`` so that tests can distinguish slices from one another.
    missing_cells: ``(time, lat, lon)`` indices to set to NOAA's fill value.

  Returns:
    Array of shape ``(len(dates), 360, 720)`` in native PSL axis order.
  """
  data = np.empty((len(dates), len(PSL_LATS), len(PSL_LONS)), dtype=np.float32)
  for i in range(len(dates)):
    data[i] = fill_value + i
  for index in missing_cells:
    data[index] = PSL_MISSING_VALUE
  return data


def write_psl_netcdf(
    directory: Path,
    year: int,
    dates: pd.DatetimeIndex,
    *,
    fill_value: float = 1.0,
    missing_cells: Iterable[tuple[int, int, int]] = (),
) -> Path:
  """Writes a synthetic ``precip.{year}.nc`` file in NOAA PSL layout."""
  data = make_psl_precip_array(
      dates, fill_value=fill_value, missing_cells=missing_cells
  )
  dataset = xr.Dataset(
      data_vars={"precip": (["time", "lat", "lon"], data)},
      coords={"time": dates, "lat": PSL_LATS, "lon": PSL_LONS},
  )
  path = directory / f"precip.{year}.nc"
  dataset.to_netcdf(path)
  dataset.close()
  return path


@pytest.fixture
def psl_cache(tmp_path: Path) -> Path:
  """Directory used as the builder's NetCDF download cache."""
  cache = tmp_path / "psl_cache"
  cache.mkdir()
  return cache


@pytest.fixture
def write_psl_year(psl_cache: Path) -> Callable[..., Path]:
  """Factory that drops a synthetic NOAA PSL year into the download cache.

  Pre-seeding the cache means ``ensure_psl_cpc_netcdf`` short-circuits before
  it would otherwise reach out to downloads.psl.noaa.gov.
  """

  def _write(
      year: int,
      start: str,
      end: str,
      *,
      fill_value: float = 1.0,
      missing_cells: Iterable[tuple[int, int, int]] = (),
  ) -> Path:
    dates = pd.date_range(start, end, freq="D")
    return write_psl_netcdf(
        psl_cache,
        year,
        dates,
        fill_value=fill_value,
        missing_cells=missing_cells,
    )

  return _write


class FakeHRESSource:
  """In-memory stand-in for the three real HRES upstream sources.

  Returns a deterministic, per-date, per-variable constant so that tests can
  assert exactly which slice ended up at which time index. Dates that are not
  in ``available`` return ``None``, which is how the real sources signal
  "upstream data is missing" and which makes the builder write a NaN slice.
  """

  def __init__(
      self,
      available: Optional[Iterable[str]] = None,
      offset: float = 0.0,
      latitudes: np.ndarray = FAKE_HRES_LATS,
      longitudes: np.ndarray = FAKE_HRES_LONS,
  ):
    self.available = None if available is None else set(available)
    self.offset = offset
    self.latitudes = latitudes
    self.longitudes = longitudes
    self.requested: List[str] = []

  def value_for(self, date: pd.Timestamp, variable: str) -> float:
    """Deterministic value written for a given date/variable pair."""
    day_of_year = float(pd.Timestamp(date).dayofyear)
    variable_index = float(hres_module.HRES_VARIABLES.index(variable))
    return day_of_year + 100.0 * variable_index + self.offset

  def extract_date(
      self,
      date: pd.Timestamp,
      target_lat: Optional[np.ndarray] = None,
      target_lon: Optional[np.ndarray] = None,
  ) -> Optional[Dict[str, np.ndarray]]:
    """Mirrors the ``extract_date`` contract of the real source classes."""
    del target_lat, target_lon  # Fakes always emit the canonical fake grid.
    key = pd.Timestamp(date).strftime("%Y-%m-%d")
    self.requested.append(key)
    if self.available is not None and key not in self.available:
      return None
    shape = (
        hres_module.NUM_LEAD_DAYS,
        len(self.latitudes),
        len(self.longitudes),
    )
    return {
        variable: np.full(shape, self.value_for(date, variable), np.float32)
        for variable in hres_module.HRES_VARIABLES
    }


@pytest.fixture
def fake_hres_source(monkeypatch: pytest.MonkeyPatch) -> Callable[..., FakeHRESSource]:
  """Factory that swaps every HRES upstream source for an in-memory fake.

  ``build_hres_archive`` reaches the network exclusively through the module
  level worker globals that ``_init_worker`` populates. Replacing
  ``_init_worker`` therefore isolates the builder completely, while still
  exercising the real batching, resume, write, and in-place code paths.
  """

  def _install(
      available: Optional[Iterable[str]] = None,
      offset: float = 0.0,
  ) -> FakeHRESSource:
    source = FakeHRESSource(available=available, offset=offset)

    def fake_init_worker(project: str) -> None:
      del project  # No credentials are needed for the fake.
      hres_module._worker_wb2 = source
      hres_module._worker_gap = source
      hres_module._worker_open_data = source
      hres_module._target_lat = source.latitudes
      hres_module._target_lon = source.longitudes

    monkeypatch.setattr(hres_module, "_init_worker", fake_init_worker)
    # Route every requested date through the fake regardless of which real
    # source would normally own that date range.
    monkeypatch.setattr(
        hres_module, "WB2_CUTOFF_DATE", pd.Timestamp("2100-01-01")
    )
    return source

  return _install
