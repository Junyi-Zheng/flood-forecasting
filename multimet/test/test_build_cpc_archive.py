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

"""Unit tests for NOAA CPC daily gridded archive builder."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from multimet.build_cpc_archive import (
    CPC_LATS,
    CPC_LONS,
    build_cpc_archive,
    process_cpc_netcdf_to_dataset,
    write_batch_to_zarr,
)


class TestBuildCPCArchive(unittest.TestCase):

  def setUp(self):
    self.test_dir = tempfile.mkdtemp()
    self.sample_nc = os.path.join(self.test_dir, "precip.2020.nc")

    # Create synthetic NOAA PSL format NetCDF:
    # 3 days, lat 89.75 .. -89.75 (descending), lon 0.25 .. 359.75
    times = pd.date_range("2020-01-01", "2020-01-03", freq="D")
    psl_lats = np.linspace(89.75, -89.75, 360, dtype=np.float32)
    psl_lons = np.linspace(0.25, 359.75, 720, dtype=np.float32)

    data = np.ones((len(times), len(psl_lats), len(psl_lons)), dtype=np.float32)
    # Set missing value in one cell
    data[0, 0, 0] = -9.96921e36

    synthetic_ds = xr.Dataset(
        data_vars={
            "precip": (["time", "lat", "lon"], data),
        },
        coords={
            "time": times,
            "lat": psl_lats,
            "lon": psl_lons,
        },
    )
    synthetic_ds.to_netcdf(self.sample_nc)

  def tearDown(self):
    shutil.rmtree(self.test_dir)

  def test_process_cpc_netcdf_transformation(self):
    ds = process_cpc_netcdf_to_dataset(self.sample_nc)

    self.assertIsNotNone(ds)
    self.assertIn("cpc_precipitation", ds.data_vars)
    self.assertEqual(ds["cpc_precipitation"].shape, (3, 360, 720))

    # Latitudes must be ascending (-89.75 to 89.75)
    np.testing.assert_allclose(ds["latitude"].values, CPC_LATS)
    self.assertAlmostEqual(float(ds["latitude"].values[0]), -89.75)
    self.assertAlmostEqual(float(ds["latitude"].values[-1]), 89.75)

    # Longitudes must be shifted (-179.75 to 179.75)
    np.testing.assert_allclose(ds["longitude"].values, CPC_LONS)
    self.assertAlmostEqual(float(ds["longitude"].values[0]), -179.75)
    self.assertAlmostEqual(float(ds["longitude"].values[-1]), 179.75)

    # Missing value must be NaN
    # Original (0, 0, 0) was at lat 89.75, lon 0.25
    # In transformed: lat 89.75 is at index -1 (359)
    # lon 0.25 is at index 360
    self.assertTrue(np.isnan(ds["cpc_precipitation"].values[0, 359, 360]))

    # Valid values should be 1.0
    self.assertEqual(ds["cpc_precipitation"].values[0, 0, 0], 1.0)

  def test_date_filtering(self):
    ds = process_cpc_netcdf_to_dataset(
        self.sample_nc,
        target_start_date=pd.Timestamp("2020-01-02"),
        target_end_date=pd.Timestamp("2020-01-02"),
    )
    self.assertIsNotNone(ds)
    self.assertEqual(len(ds["time"]), 1)
    self.assertEqual(
        pd.Timestamp(ds["time"].values[0]), pd.Timestamp("2020-01-02")
    )

  def test_write_and_append_local_zarr(self):
    target_zarr = os.path.join(self.test_dir, "test_archive.zarr")

    ds1 = process_cpc_netcdf_to_dataset(
        self.sample_nc,
        target_start_date=pd.Timestamp("2020-01-01"),
        target_end_date=pd.Timestamp("2020-01-01"),
    )
    write_batch_to_zarr(
        ds1, target_zarr, is_initial_write=True
    )

    # Check store exists and has 1 day
    store1 = xr.open_zarr(target_zarr, consolidated=True)
    self.assertEqual(len(store1["time"]), 1)
    self.assertEqual(store1["cpc_precipitation"].shape, (1, 360, 720))

    # Append day 2 and day 3
    ds2 = process_cpc_netcdf_to_dataset(
        self.sample_nc,
        target_start_date=pd.Timestamp("2020-01-02"),
        target_end_date=pd.Timestamp("2020-01-03"),
    )
    write_batch_to_zarr(
        ds2, target_zarr, is_initial_write=False
    )

    # Check store has 3 days
    store2 = xr.open_zarr(target_zarr, consolidated=True)
    self.assertEqual(len(store2["time"]), 3)
    self.assertEqual(store2["cpc_precipitation"].shape, (3, 360, 720))


if __name__ == "__main__":
  unittest.main()
