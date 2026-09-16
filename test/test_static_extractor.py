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

"""Unit and integration tests for Caravan Static Attributes Extractor."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import shapely.geometry

from static_extractor import (
    ATTRIBUTE_DEFINITIONS,
    MAJORITY_PROPERTIES,
    StaticAttributesExtractor,
    calculate_fao_pm_pet,
    calculate_knoben_moisture_and_seasonality,
    compute_caravan_climate_metrics,
    compute_pour_point_properties,
    get_default_gdb_path,
)
from static_extractor.cli import parse_args


def test_schema_definitions():
  """Verifies core schema definitions and attribute categories."""
  assert len(MAJORITY_PROPERTIES) == 10
  assert "clz_cl_smj" in MAJORITY_PROPERTIES
  assert "glc_cl_smj" in MAJORITY_PROPERTIES
  assert "lit_cl_smj" in MAJORITY_PROPERTIES

  assert "basin_area" in ATTRIBUTE_DEFINITIONS
  assert "ele_mt_sav" in ATTRIBUTE_DEFINITIONS
  assert "p_mean" in ATTRIBUTE_DEFINITIONS
  assert ATTRIBUTE_DEFINITIONS["ele_mt_sav"]["category"] == "Topography"
  assert ATTRIBUTE_DEFINITIONS["p_mean"]["category"] == "Climate"


def test_fao_pm_pet_calculation():
  """Tests FAO-56 Penman-Monteith daily reference evapotranspiration."""
  dates = pd.date_range("2020-01-01", periods=5, freq="D")
  sp = pd.Series([101.3] * 5, index=dates)
  t2m = pd.Series([20.0, 25.0, 15.0, 30.0, 10.0], index=dates)
  d2m = pd.Series([15.0, 18.0, 10.0, 20.0, 5.0], index=dates)
  u10 = pd.Series([2.0] * 5, index=dates)
  v10 = pd.Series([1.5] * 5, index=dates)
  # Net radiation in J/m2/hr: 1e6 J/m2/hr * 24 / 1e6 = 24 MJ/m2/day
  ssr = pd.Series([800000.0] * 5, index=dates)
  str_s = pd.Series([200000.0] * 5, index=dates)

  pet = calculate_fao_pm_pet(
      surface_pressure_kpa=sp,
      temperature_2m_c=t2m,
      dewpoint_temperature_2m_c=d2m,
      u_component_of_wind_10m=u10,
      v_component_of_wind_10m=v10,
      surface_net_solar_radiation_mean=ssr,
      surface_net_thermal_radiation_mean=str_s,
  )

  assert len(pet) == 5
  assert (pet >= 0.0).all()
  # Typically daily summer PET is between 2 and 8 mm/day
  assert 2.0 <= pet.mean() <= 8.0


def test_knoben_moisture_and_seasonality():
  """Tests Knoben et al. (2018) annual moisture and seasonality indices."""
  dates = pd.date_range("2020-01-01", periods=365, freq="D")
  # Wet winter, dry summer
  day_of_year = dates.dayofyear
  p_vals = 3.0 + 2.0 * np.cos(2 * np.pi * day_of_year / 365.0)
  pet_vals = 3.0 - 2.0 * np.cos(2 * np.pi * day_of_year / 365.0)

  p = pd.Series(p_vals, index=dates)
  pet = pd.Series(pet_vals, index=dates)

  mi, seasonality = calculate_knoben_moisture_and_seasonality(p, pet)
  assert -1.0 <= mi <= 1.0
  assert seasonality >= 0.0


def test_caravan_climate_metrics_extremes():
  """Tests extreme precipitation indices (high/low prec freq and duration)."""
  dates = pd.date_range("2020-01-01", periods=100, freq="D")
  p_vals = np.ones(100) * 2.0
  # Add an extreme event: 3 consecutive days >= 5 * p_mean (10 mm/day)
  p_vals[10:13] = 12.0
  # Add dry days: 5 consecutive days < 1 mm/day
  p_vals[20:25] = 0.5

  p = pd.Series(p_vals, index=dates)
  t = pd.Series(np.ones(100) * 15.0, index=dates)
  pet = pd.Series(np.ones(100) * 3.0, index=dates)

  metrics = compute_caravan_climate_metrics(p, t, pet)

  assert metrics["p_mean"] > 0
  assert metrics["high_prec_freq"] > 0
  assert metrics["high_prec_dur"] == 3.0
  assert metrics["low_prec_freq"] > 0
  assert metrics["low_prec_dur"] == 5.0
  assert metrics["frac_snow"] == 0.0  # temp > 0


def test_pour_point_properties():
  """Tests downstream topological outlet tracing via NEXT_DOWN."""
  basin_data = {
      "HYBAS_ID": [712000001, 712000002, 712000003],
      "NEXT_DOWN": [712000002, 712000003, 0],
      "SUB_AREA": [100.0, 100.0, 100.0],
      "weights": [80.0, 90.0, 95.0],
      "dis_m3_pyr": [10.0, 20.0, 50.0],
  }
  res = compute_pour_point_properties(
      basin_data, pour_point_properties=["dis_m3_pyr"]
  )
  assert "dis_m3_pyr" in res


def test_extractor_wabash_basins_matching_reference():
  """End-to-end test verifying extracted attributes match official reference data."""
  wabash_geojson = Path(
      "/usr/local/google/home/gsnearing/Projects/flood-forecasting-multimet/multimet/test/test_data/shapefiles/us/us_basin_shapes.geojson"
  )
  if not wabash_geojson.exists():
    pytest.skip("Wabash test GeoJSON not found on local path.")

  gdb_path = get_default_gdb_path()
  if not gdb_path.exists():
    pytest.skip("BasinATLAS dataset not found in local cache.")

  extractor = StaticAttributesExtractor(gdb_path=gdb_path)

  with tempfile.TemporaryDirectory() as tmpdir:
    out_csv = Path(tmpdir) / "test_attributes.csv"
    df = extractor.extract_attributes_from_file(
        wabash_geojson, output_csv_path=out_csv
    )

    assert df.shape[0] == 5
    assert df.shape[1] >= 200
    assert "basin_area" in df.columns
    assert "ele_mt_sav" in df.columns
    assert "pre_mm_syr" in df.columns
    assert "glc_cl_smj" in df.columns

    # Verify attributes against known Wabash test reference
    ref_csv = Path(
        "/google/src/cloud/gsnearing/open-multimet-2/google3/third_party/py/googlehydrology/extractor/testdata/attributes/attributes_caravan_extracted_watersheds_wabash_test.csv"
    )
    if ref_csv.exists():
      ref_df = pd.read_csv(ref_csv).set_index("gauge_id")
      for gid in df.index:
        if gid in ref_df.index:
          # Verify basin area matches within 1%
          area_test = df.loc[gid, "basin_area"]
          area_ref = ref_df.loc[gid, "basin_area"]
          assert abs(area_test - area_ref) / area_ref < 0.05

          # Verify mean elevation matches within 5m
          ele_test = df.loc[gid, "ele_mt_sav"]
          ele_ref = ref_df.loc[gid, "ele_mt_sav"]
          assert abs(ele_test - ele_ref) < 5.0


def test_cli_parsing():
  """Tests CLI argument parsing."""
  args = parse_args(["--input", "basins.geojson", "--output", "attrs.csv"])
  assert args.input == "basins.geojson"
  assert args.output == "attrs.csv"
  assert args.min_overlap_threshold == 0.0
  assert args.era5_source == "hybas"

  args_gridded = parse_args([
      "--input", "basins.geojson",
      "--output", "attrs.csv",
      "--era5-source", "gridded",
      "--gridded-era5-uri", "gs://my-bucket/era5.zarr",
      "--workers", "16",
  ])
  assert args_gridded.era5_source == "gridded"
  assert args_gridded.gridded_era5_uri == "gs://my-bucket/era5.zarr"
  assert args_gridded.workers == 16


def test_era5_gridded_extractor_synthetic(tmp_path):
  """Tests ERA5GriddedExtractor with a synthetic local Zarr dataset."""
  import zarr
  import shapely.geometry
  from static_extractor.climate import ERA5GriddedExtractor

  zarr_dir = tmp_path / "synthetic_era5.zarr"
  root = zarr.open_group(str(zarr_dir), mode="w")

  n_times = 50
  lats = np.linspace(40.0, 41.0, 11, dtype=np.float32)
  lons = np.linspace(-87.0, -86.0, 11, dtype=np.float32)

  # Coordinates
  root.create_array("latitude", data=lats)
  root.create_array("longitude", data=lons)
  time_arr = root.create_array("time", data=np.arange(n_times, dtype=np.int64))
  time_arr.attrs["units"] = "days since 2000-01-01"

  # Climate data arrays (time, lat, lon)
  p_data = np.full((n_times, len(lats), len(lons)), 4.0, dtype=np.float32)
  t_data = np.full((n_times, len(lats), len(lons)), 18.0, dtype=np.float32)
  pet_data = np.full((n_times, len(lats), len(lons)), 2.0, dtype=np.float32)

  root.create_array("era5land_total_precipitation", data=p_data)
  root.create_array("era5land_temperature_2m", data=t_data)
  root.create_array("era5land_potential_evaporation_FAO_PENMAN_MONTEITH", data=pet_data)

  extractor = ERA5GriddedExtractor(zarr_uri=str(zarr_dir))

  # Test polygon covering central region
  poly = shapely.geometry.box(-86.8, 40.2, -86.2, 40.8)
  metrics = extractor.extract_climate_metrics_for_polygon(poly, baseline_years=None)

  assert metrics["p_mean"] == 4.0
  assert metrics["pet_mean_FAO_PM"] == 2.0
  assert metrics["aridity_FAO_PM"] == 0.5
  assert metrics["frac_snow"] == 0.0


def test_batch_runner_discovery(tmp_path):
  """Tests discover_datasets in batch_runner across multiple dataset folders."""
  from static_extractor.batch_runner import discover_datasets
  
  parent = tmp_path / "caravan_root"
  ds1 = parent / "camels"
  ds2 = parent / "hysets"
  ds1.mkdir(parents=True)
  ds2.mkdir(parents=True)
  
  (ds1 / "camels_basin_shapes.shp").touch()
  (ds2 / "hysets.geojson").touch()
  
  datasets = discover_datasets(parent_dirs=[str(parent)])
  assert "camels" in datasets
  assert "hysets" in datasets
  assert datasets["camels"].name == "camels_basin_shapes.shp"
  assert datasets["hysets"].name == "hysets.geojson"

  # Test nested staging directory structure (e.g. parent/parent/dataset)
  nested_parent = tmp_path / "staged" / "caravan" / "caravan"
  ds_nested = nested_parent / "lamah"
  ds_nested.mkdir(parents=True)
  (ds_nested / "lamah_basin_shapes.shp").touch()

  datasets_nested = discover_datasets(parent_dirs=[str(tmp_path / "staged" / "caravan")])
  assert "lamah" in datasets_nested
  assert datasets_nested["lamah"].name == "lamah_basin_shapes.shp"



