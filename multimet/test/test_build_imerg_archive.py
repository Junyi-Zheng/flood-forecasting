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

"""Unit and integration tests for :mod:`multimet.build_imerg_archive`."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from multimet import build_imerg_archive as imerg_module
from multimet.build_imerg_archive import (
  DEFAULT_START_DATE,
  IMERG_ATTRS,
  IMERG_VARIABLE,
  LAT_COUNT,
  LON_COUNT,
  GESDISCImergSource,
  LocalImergSource,
  build_arg_parser,
  build_batch_dataset,
  build_imerg_archive,
  parse_imerg_netcdf_to_grid,
  write_batch_in_place,
)

pytestmark = pytest.mark.unit


def _write_synthetic_imerg_nc4(
    path: Path,
    fill_value: float = 5.0,
    var_name: str = "precipitation",
    transpose_dims: bool = True,
) -> None:
  """Writes a synthetic daily IMERG NetCDF-4 file."""
  lats = np.linspace(-89.95, 89.95, LAT_COUNT, dtype=np.float32)
  lons = np.linspace(-179.95, 179.95, LON_COUNT, dtype=np.float32)
  if transpose_dims:
    data = np.full((1, LON_COUNT, LAT_COUNT), fill_value, dtype=np.float32)
    data[0, 0, 0] = -9999.9  # Sentinel to verify NaN masking
    ds = xr.Dataset(
        {var_name: (["time", "lon", "lat"], data)},
        coords={"time": [pd.Timestamp("2024-01-01")], "lon": lons, "lat": lats},
    )
  else:
    data = np.full((LAT_COUNT, LON_COUNT), fill_value, dtype=np.float32)
    ds = xr.Dataset(
        {var_name: (["lat", "lon"], data)},
        coords={"lat": lats, "lon": lons},
    )
  ds.to_netcdf(path)


class TestNetCDFParsing:
  """Tests ``parse_imerg_netcdf_to_grid``."""

  def test_transposes_lon_lat_and_masks_sentinels(self, tmp_path: Path) -> None:
    nc_path = tmp_path / "sample.nc4"
    _write_synthetic_imerg_nc4(nc_path, fill_value=12.5, transpose_dims=True)

    grid = parse_imerg_netcdf_to_grid(str(nc_path))

    assert grid.shape == (LAT_COUNT, LON_COUNT)
    assert grid.dtype == np.float32
    assert np.isnan(grid[0, 0])
    assert float(grid[1, 1]) == pytest.approx(12.5)

  def test_supports_precipitation_cal_variable(self, tmp_path: Path) -> None:
    nc_path = tmp_path / "sample_cal.nc4"
    _write_synthetic_imerg_nc4(
        nc_path, fill_value=3.25, var_name="precipitationCal", transpose_dims=False
    )

    grid = parse_imerg_netcdf_to_grid(str(nc_path))

    assert grid.shape == (LAT_COUNT, LON_COUNT)
    assert float(grid[0, 0]) == pytest.approx(3.25)


class TestGESDISCSourceAndCleanup:
  """Tests ``GESDISCImergSource`` suffix resolution and immediate cache cleanup."""

  def test_resolves_cached_v07c_and_cleans_up_file(self, tmp_path: Path) -> None:
    cache_dir = tmp_path / "imerg_cache"
    cache_dir.mkdir()
    fn = "3B-DAY-E.MS.MRG.3IMERG.20260201-S000000-E235959.V07C.nc4"
    staged_file = cache_dir / fn
    _write_synthetic_imerg_nc4(staged_file, fill_value=7.0)

    source = GESDISCImergSource(cache_dir=str(cache_dir), cleanup_cache=True)
    grid = source.extract_date(pd.Timestamp("2026-02-01"))

    assert grid is not None
    assert grid.shape == (LAT_COUNT, LON_COUNT)
    assert float(np.nanmax(grid)) == pytest.approx(7.0)
    assert not staged_file.exists(), "cleanup_cache=True must remove the NetCDF after reading"


class TestLocalSource:
  """Tests ``LocalImergSource`` with NetCDF-4 and 48 half-hourly HDF5 granules."""

  def test_reads_local_nc4(self, tmp_path: Path) -> None:
    nc_file = tmp_path / "3B-DAY-E.MS.MRG.3IMERG.20230510-S000000-E235959.V07B.nc4"
    _write_synthetic_imerg_nc4(nc_file, fill_value=4.5)

    source = LocalImergSource(str(tmp_path))
    grid = source.extract_date(pd.Timestamp("2023-05-10"))

    assert grid is not None
    assert float(np.nanmax(grid)) == pytest.approx(4.5)

  def test_accumulates_48_half_hourly_h5_granules(
      self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    monkeypatch.setattr(imerg_module, "LAT_COUNT", 4)
    monkeypatch.setattr(imerg_module, "LON_COUNT", 6)
    month_dir = tmp_path / "202305"
    month_dir.mkdir()

    # Write 48 half-hourly granules each with rate = 2.0 mm/hr -> daily total = 48 * 2.0 * 0.5 = 48.0 mm
    for i in range(48):
      fpath = month_dir / f"3B-HHR-E.MS.MRG.3IMERG.20230515-S{i:02d}0000.{i:04d}.V07B.RT-H5"
      with h5py.File(fpath, "w") as h5:
        grp = h5.create_group("Grid")
        grp.create_dataset("precipitation", data=np.full((6, 4), 2.0, dtype=np.float32))

    source = LocalImergSource(str(tmp_path), granule_workers=1)
    grid = source.extract_date(pd.Timestamp("2023-05-15"))

    assert grid is not None
    assert grid.shape == (4, 6)
    assert np.allclose(grid, 48.0)


class TestZarrWriteAndBuild:
  """Tests batch dataset schema, resume, in-place writes, and full cache cleanup."""

  def test_build_and_resume_and_in_place(
      self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    small_lats = np.linspace(-89.95, 89.95, 8, dtype=np.float32)
    small_lons = np.linspace(-179.95, 179.95, 12, dtype=np.float32)
    monkeypatch.setattr(imerg_module, "LAT_COUNT", 8)
    monkeypatch.setattr(imerg_module, "LON_COUNT", 12)
    monkeypatch.setattr(imerg_module, "IMERG_LATS", small_lats)
    monkeypatch.setattr(imerg_module, "IMERG_LONS", small_lons)

    local_dir = tmp_path / "local_nc"
    local_dir.mkdir()
    for d_idx, d_str in enumerate(["20240101", "20240102", "20240103"], start=1):
      ds = xr.Dataset(
          {"precipitation": (["lat", "lon"], np.full((8, 12), float(d_idx), dtype=np.float32))},
          coords={"lat": small_lats, "lon": small_lons},
      )
      ds.to_netcdf(local_dir / f"imerg_{d_str}.nc4")

    target_store = str(tmp_path / "imerg_out.zarr")
    cache_dir = tmp_path / "temp_cache"
    cache_dir.mkdir()

    # 1. Initial build for 2 days
    build_imerg_archive(
        start_date="2024-01-01",
        end_date="2024-01-02",
        source_type="local",
        local_dir=str(local_dir),
        target_zarr=target_store,
        cache_dir=str(cache_dir),
        cleanup_cache=True,
        batch_size=2,
        num_workers=1,
    )
    assert not cache_dir.exists(), "cleanup_cache=True must remove cache_dir in finally block"

    with xr.open_zarr(target_store, consolidated=False) as store:
      assert len(store["time"]) == 2
      assert store.attrs["title"] == IMERG_ATTRS["title"]
      assert float(store[IMERG_VARIABLE].isel(time=0).mean()) == pytest.approx(1.0)
      assert float(store[IMERG_VARIABLE].isel(time=1).mean()) == pytest.approx(2.0)

    # 2. Resume to add 2024-01-03
    build_imerg_archive(
        start_date="2024-01-01",
        end_date="2024-01-03",
        source_type="local",
        local_dir=str(local_dir),
        target_zarr=target_store,
        batch_size=2,
        num_workers=1,
    )
    with xr.open_zarr(target_store, consolidated=False) as store:
      assert len(store["time"]) == 3
      assert float(store[IMERG_VARIABLE].isel(time=2).mean()) == pytest.approx(3.0)

    # 3. In-place rewrite of 2024-01-02
    replacement = build_batch_dataset(
        [pd.Timestamp("2024-01-02")],
        [np.full((8, 12), 99.0, dtype=np.float32)],
        latitudes=small_lats,
        longitudes=small_lons,
    )
    write_batch_in_place(replacement, target_store)
    with xr.open_zarr(target_store, consolidated=False) as store:
      assert float(store[IMERG_VARIABLE].isel(time=1).mean()) == pytest.approx(99.0)


class TestCommandLine:
  """Tests ``build-imerg-archive`` argument parsing."""

  def test_defaults(self) -> None:
    args = build_arg_parser().parse_args([])
    assert args.start_date == DEFAULT_START_DATE
    assert args.cleanup_cache is False
    assert args.overwrite is False
    assert args.in_place is False

  def test_query_cmr_granules_parses_data_links(
      self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    class _FakeResp:
      def raise_for_status(self) -> None:
        pass

      def json(self) -> dict:
        return {
            "feed": {
                "entry": [
                    {
                        "links": [
                            {
                                "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                                "href": "https://gpm1.gesdisc.eosdis.nasa.gov/data/3B-DAY-E.V07B.nc4",
                            },
                            {
                                "rel": "http://esipfed.org/ns/fedsearch/1.1/metadata#",
                                "href": "https://gpm1.gesdisc.eosdis.nasa.gov/data/3B-DAY-E.V07B.nc4.xml",
                            },
                        ]
                    }
                ]
            }
        }

    monkeypatch.setattr(imerg_module.requests, "get", lambda *a, **k: _FakeResp())
    urls = imerg_module.query_cmr_granules(
        imerg_module.IMERG_DAILY_SHORT_NAME, pd.Timestamp("2024-06-01")
    )
    assert urls == ["https://gpm1.gesdisc.eosdis.nasa.gov/data/3B-DAY-E.V07B.nc4"]
