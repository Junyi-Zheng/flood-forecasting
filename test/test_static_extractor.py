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
  """End-to-end test verifying extracted attributes match reference data."""
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
        wabash_geojson, output_csv_path=out_csv, workers=2
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


def test_benchmark_metrics_continuous():
  """Tests continuous statistical validation metrics calculation."""
  from static_extractor.benchmark import compute_continuous_metrics

  y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
  y_pred = np.array([10.1, 19.9, 30.2, 39.8, 50.1])

  res = compute_continuous_metrics(y_true, y_pred)
  assert res["n"] == 5
  assert res["pearson_r"] > 0.999
  assert res["spearman_rho"] == 1.0
  assert res["r2"] > 0.999
  assert res["mae"] < 0.2
  assert res["rmse"] < 0.2
  assert res["max_abs_error"] == 0.2
  assert res["med_rel_error_pct"] < 1.0
  assert res["max_rel_error_pct"] < 1.5


def test_benchmark_metrics_categorical():
  """Tests categorical classification accuracy calculation."""
  from static_extractor.benchmark import compute_categorical_metrics

  y_true = np.array([1, 2, 3, 4, 5, 2, 1, 3])
  y_pred = np.array([1, 2, 3, 4, 5, 2, 1, 4])  # 7 out of 8 match

  res = compute_categorical_metrics(y_true, y_pred)
  assert res["n"] == 8
  assert res["accuracy_pct"] == 87.5
  assert res["classes_count"] == 5


def test_benchmark_attribute_categorization():
  """Tests categorization of all standard attribute names."""
  from static_extractor.benchmark import get_attribute_category

  assert get_attribute_category("ele_mt_sav") == "Topography"
  assert get_attribute_category("tmp_dc_syr") == "Climate (HydroATLAS)"
  assert get_attribute_category("p_mean") == "Caravan ERA5 Climate"
  assert get_attribute_category("aridity_ERA5_LAND") == "Caravan ERA5 Climate"
  assert get_attribute_category("run_mm_syr") == "Hydrology"
  assert get_attribute_category("cly_pc_sav") == "Soils & Geology"
  assert get_attribute_category("for_pc_sse") == "Land Cover"
  assert get_attribute_category("ppd_pk_sav") == "Anthropogenic"
  assert get_attribute_category("glc_cl_smj") == "Land Cover"
  assert get_attribute_category("wet_cl_smj") == "Hydrology"


def test_batch_runner_clean_cache_flag(tmp_path):
  """Verifies that --clean-cache removes cache_root after batch execution."""
  from static_extractor.batch_runner import parse_args, main
  fake_cache = tmp_path / "cache_dir"
  fake_cache.mkdir(parents=True, exist_ok=True)
  (fake_cache / "staged_shapefiles").mkdir(parents=True, exist_ok=True)
  (fake_cache / "test.txt").write_text("hello")

  # Test parser recognition
  args = parse_args(["-o", str(tmp_path / "out"), "--clean-cache", "--cache-dir", str(fake_cache)])
  assert args.clean_cache is True

  # Verify cleanup behavior in main finally block
  with pytest.raises(SystemExit):
    main(["-o", str(tmp_path / "out"), "--clean-cache", "--cache-dir", str(fake_cache)])
  assert not fake_cache.exists()


def test_batch_runner_gcs_output_and_args(tmp_path, monkeypatch):
  """Verifies GCS output handling and multi parent-dir argument parsing."""
  from unittest.mock import MagicMock
  from static_extractor.batch_runner import parse_args, run_batch_extraction

  args = parse_args([
      "-p", "gs://open-multimet/data/caravan_shapefiles/caravan/",
      "-p", "gs://open-multimet/data/caravan_shapefiles/caravan_extensions/",
      "-o", "gs://open-multimet/data/caravan_static_attributes/",
      "--workers", "14",
      "--combine",
  ])
  assert len(args.parent_dirs) == 2
  assert args.output_dir == "gs://open-multimet/data/caravan_static_attributes/"
  assert args.workers == 14
  assert args.combine is True

  # Also test space-separated multi paths for a single -p flag
  args_multi = parse_args([
      "-p", "dir1", "dir2", "dir3",
      "-o", "/tmp/out",
  ])
  assert len(args_multi.parent_dirs) == 1
  assert len(args_multi.parent_dirs[0]) == 3

  # Test upload_to_gcs is called during run_batch_extraction when GCS output is set
  mock_upload = MagicMock()
  monkeypatch.setattr("static_extractor.batch_runner.upload_to_gcs", mock_upload)
  monkeypatch.setattr("static_extractor.batch_runner.gcs_path_exists", lambda uri: False)

  dummy_df = pd.DataFrame({"basin_id": ["b1"], "ele_mt_sav": [100.0]})
  dummy_shp = tmp_path / "test.shp"
  dummy_shp.write_text("dummy")

  mock_extractor = MagicMock()
  mock_extractor.extract_attributes_from_file.return_value = dummy_df
  monkeypatch.setattr(
      "static_extractor.batch_runner.StaticAttributesExtractor",
      lambda **kwargs: mock_extractor,
  )

  results = run_batch_extraction(
      dataset_map={"test_ds": dummy_shp},
      output_dir="gs://open-multimet/data/caravan_static_attributes/",
      workers=1,
      staging_cache_dir=tmp_path / "staged",
      combine=True,
      resume=False,
  )

  assert "test_ds" in results
  # Should have uploaded dataset CSV and combined CSV
  assert mock_upload.call_count == 2


def test_batch_runner_progress_and_quiet_logging(tmp_path):
  """Verifies that --verbose and --no-progress flags are properly parsed and setup_logging configures root logger."""
  import logging
  from static_extractor.batch_runner import parse_args, setup_logging

  # Test default parser flags
  args = parse_args(["-o", str(tmp_path / "out")])
  assert args.verbose is False
  assert args.show_progress is True

  # Test verbose and no-progress flags
  args_v = parse_args(["-o", str(tmp_path / "out"), "-v", "--no-progress"])
  assert args_v.verbose is True
  assert args_v.show_progress is False

  # Test setup_logging suppresses info logs when verbose=False
  setup_logging(verbose=False)
  assert logging.getLogger("static_extractor").level == logging.WARNING

  setup_logging(verbose=True)
  assert logging.getLogger().level == logging.DEBUG


def test_export_subdataset_partitioned_files(tmp_path):
  """Tests partitioning of extracted attributes into HydroATLAS, Caravan, and Parquet tables."""
  from static_extractor.batch_runner import export_subdataset_partitioned_files

  ds_dir = tmp_path / "camels"
  ds_dir.mkdir()
  coords_file = ds_dir / "coordinates.csv"
  coords_file.write_text(
      "gauge_id,gauge_lat,gauge_lon,original_id\n"
      "camels_01,45.0,-70.0,ORIG_01\n"
      "camels_02,46.0,-71.0,ORIG_02\n"
  )
  dummy_shp = ds_dir / "camels_basin_shapes.shp"
  dummy_shp.write_text("dummy")

  df = pd.DataFrame(
      {
          "basin_area": [100.0, 200.0],
          "ele_mt_sav": [500.0, 600.0],
          "p_mean": [3.5, 4.0],
          "pet_mean": [2.5, 2.8],
          "aridity": [0.7, 0.8],
      },
      index=["camels_01", "camels_02"],
  )
  df.index.name = "gauge_id"

  out_sub = tmp_path / "out" / "camels"
  res = export_subdataset_partitioned_files(
      df=df,
      ds_name="camels",
      output_sub_dir=out_sub,
      vector_path=dummy_shp,
  )

  assert res["hydroatlas"].exists()
  assert res["caravan"].exists()
  assert res["parquet"].exists()

  df_hydro = pd.read_csv(res["hydroatlas"], index_col=0)
  assert "basin_area" in df_hydro.columns
  assert "ele_mt_sav" in df_hydro.columns
  assert "p_mean" not in df_hydro.columns
  assert "gauge_lat" not in df_hydro.columns

  df_caravan = pd.read_csv(res["caravan"], index_col=0)
  assert "p_mean" in df_caravan.columns
  assert "pet_mean" in df_caravan.columns
  assert "aridity" in df_caravan.columns
  assert "gauge_lat" in df_caravan.columns
  assert "gauge_lon" in df_caravan.columns
  assert df_caravan.loc["camels_01", "gauge_lat"] == 45.0

  df_parquet = pd.read_parquet(res["parquet"])
  assert len(df_parquet) == 2
  assert "basin_area" in df_parquet.columns
  assert "p_mean" in df_parquet.columns
  assert "gauge_lat" in df_parquet.columns


def test_batch_runner_partitioned_caravan_new(tmp_path, monkeypatch):
  """Verifies that caravan-new paths automatically trigger subdataset partitioning and uploads."""
  from unittest.mock import MagicMock
  from static_extractor.batch_runner import run_batch_extraction

  uploaded_uris = []
  def mock_upload(local_file, gcs_dest):
    uploaded_uris.append((Path(local_file).name, gcs_dest))
    return True

  monkeypatch.setattr("static_extractor.batch_runner.upload_to_gcs", mock_upload)
  monkeypatch.setattr("static_extractor.batch_runner.gcs_path_exists", lambda uri: False)

  ds_dir = tmp_path / "camels"
  ds_dir.mkdir()
  (ds_dir / "coordinates.csv").write_text("gauge_id,gauge_lat,gauge_lon\ncamels_01,44.0,-68.0\n")
  dummy_shp = ds_dir / "camels_basin_shapes.shp"
  dummy_shp.write_text("dummy")

  dummy_df = pd.DataFrame(
      {"basin_area": [150.0], "ele_mt_sav": [300.0], "p_mean": [3.2]},
      index=["camels_01"],
  )
  dummy_df.index.name = "gauge_id"

  mock_extractor = MagicMock()
  mock_extractor.extract_attributes_from_file.return_value = dummy_df
  monkeypatch.setattr(
      "static_extractor.batch_runner.StaticAttributesExtractor",
      lambda **kwargs: mock_extractor,
  )

  # 1. Run extraction into caravan-new target GCS path
  results = run_batch_extraction(
      dataset_map={"camels": dummy_shp},
      output_dir="gs://open-multimet/caravan-new/caravan-original/attributes/",
      workers=1,
      staging_cache_dir=tmp_path / "staged",
      resume=True,
  )

  assert "camels" in results
  # Verify 3 files were uploaded to gs://open-multimet/caravan-new/caravan-original/attributes/camels/
  uploaded_filenames = [u[0] for u in uploaded_uris]
  assert "attributes_hydroatlas_camels.csv" in uploaded_filenames
  assert "attributes_caravan_camels.csv" in uploaded_filenames
  assert "attributes_camels.parquet" in uploaded_filenames
  for _, dest in uploaded_uris:
    assert dest.endswith("attributes/camels/")

  # 2. Re-running with resume=True should skip camels (parquet already exists locally)
  prev_upload_count = len(uploaded_uris)
  results_resume = run_batch_extraction(
      dataset_map={"camels": dummy_shp},
      output_dir="gs://open-multimet/caravan-new/caravan-original/attributes/",
      workers=1,
      staging_cache_dir=tmp_path / "staged",
      resume=True,
  )
  assert "camels" in results_resume
  assert len(uploaded_uris) == prev_upload_count


def test_batch_runner_preserve_caravan_dirs(tmp_path, monkeypatch):
  """Verifies that --preserve-caravan-dirs partitions output per the contract into <collection>/attributes/<subdataset>/."""
  from unittest.mock import MagicMock
  from static_extractor.batch_runner import parse_args, run_batch_extraction

  args = parse_args(["-o", "gs://open-multimet/caravan-new/", "--preserve-caravan-dirs"])
  assert args.preserve_caravan_dirs is True

  uploaded_uris = []
  def mock_upload(local_file, gcs_dest):
    uploaded_uris.append((Path(local_file).name, gcs_dest))
    return True

  monkeypatch.setattr("static_extractor.batch_runner.upload_to_gcs", mock_upload)
  monkeypatch.setattr("static_extractor.batch_runner.gcs_path_exists", lambda uri: False)

  # Setup 3 dummy datasets spanning all 3 collections
  # 1. camels (caravan-original)
  # 2. camelsde (caravan-extensions)
  # 3. camelsfr (google-internal)
  dataset_map = {}
  for ds_name in ["camels", "camelsde", "camelsfr"]:
    d = tmp_path / ds_name
    d.mkdir()
    (d / "coordinates.csv").write_text(f"gauge_id,gauge_lat,gauge_lon\n{ds_name}_01,45.0,-70.0\n")
    shp = d / f"{ds_name}_basin_shapes.shp"
    shp.write_text("dummy")
    dataset_map[ds_name] = shp

  mock_extractor = MagicMock()
  def mock_extract(input_path, **kwargs):
    ds = Path(input_path).stem.replace("_basin_shapes", "")
    df = pd.DataFrame(
        {"basin_area": [100.0], "ele_mt_sav": [400.0], "p_mean": [3.0]},
        index=[f"{ds}_01"],
    )
    df.index.name = "gauge_id"
    return df

  mock_extractor.extract_attributes_from_file.side_effect = mock_extract
  monkeypatch.setattr(
      "static_extractor.batch_runner.StaticAttributesExtractor",
      lambda **kwargs: mock_extractor,
  )

  results = run_batch_extraction(
      dataset_map=dataset_map,
      output_dir="gs://open-multimet/caravan-new/",
      preserve_caravan_dirs=True,
      workers=1,
      staging_cache_dir=tmp_path / "staged",
      resume=False,
  )

  assert len(results) == 3
  # Check uploaded destinations
  dest_map = {name: dest for name, dest in uploaded_uris}

  # camels -> caravan-original/attributes/camels/
  assert any(
      dest == "gs://open-multimet/caravan-new/caravan-original/attributes/camels/"
      for _, dest in uploaded_uris
  )
  # camelsde -> caravan-extensions/attributes/camelsde/
  assert any(
      dest == "gs://open-multimet/caravan-new/caravan-extensions/attributes/camelsde/"
      for _, dest in uploaded_uris
  )
  # camelsfr -> google-internal/attributes/camelsfr/
  assert any(
      dest == "gs://open-multimet/caravan-new/google-internal/attributes/camelsfr/"
      for _, dest in uploaded_uris
  )
