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

"""Static Catchment Attribute Extraction Engine for Caravan & HydroATLAS.

Interfaces with global BasinATLAS (HydroATLAS v1.0 Level 12) local geodatabase
or shapefile to compute exact area-weighted physiographic, hydro-environmental,
soil, land-cover, climatology, and anthropogenic attributes for arbitrary user-uploaded
or delineated watershed polygons following the official Caravan aggregation methodology:
- Area-weighted majority voting for discrete categorical classes
- Downstream topological outlet tracing via NEXT_DOWN for pour-point metrics
- Area-weighted averaging for continuous physiographic & hydro-climatic properties
- Canonical 40-year ERA5 climate indices (1981-2020)
"""

from __future__ import annotations

from collections import defaultdict
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
shape = shapely.geometry.shape
Point = shapely.geometry.Point
Polygon = shapely.geometry.Polygon
MultiPolygon = shapely.geometry.MultiPolygon
box = shapely.geometry.box

try:
  import pyogrio
except ImportError:
  pyogrio = None

try:
  import xarray as xr
except ImportError:
  xr = None

from static_extractor.climate import (
    ERA5ClimateLoader,
    ERA5GriddedExtractor,
    compute_caravan_climate_metrics,
)
from static_extractor.config import (
    ADDITIONAL_PROPERTIES,
    ATTRIBUTE_DEFINITIONS,
    DEFAULT_ERA5_SOURCE,
    GCS_ERA5_GRIDDED_ZARR_URI,
    GCS_HYDROATLAS_GDB_URI,
    IGNORE_PROPERTIES,
    MAJORITY_PROPERTIES,
    POUR_POINT_PROPERTIES,
    UPSTREAM_PROPERTIES,
    get_default_era5_cache_dir,
    get_default_gdb_path,
)
from static_extractor.gcs import download_hydroatlas_from_gcs

logger = logging.getLogger(__name__)

_WORKER_EXTRACTOR: Optional[StaticAttributesExtractor] = None


def _get_worker_extractor(
    gdb_path: str,
    era5_cache_dir: Optional[str],
    gridded_era5_uri: Optional[str],
) -> StaticAttributesExtractor:
  global _WORKER_EXTRACTOR
  if _WORKER_EXTRACTOR is None:
    _WORKER_EXTRACTOR = StaticAttributesExtractor(
        gdb_path=gdb_path,
        era5_cache_dir=era5_cache_dir,
        gridded_era5_uri=gridded_era5_uri,
        auto_download=False,
    )
  return _WORKER_EXTRACTOR


def _worker_extract_polygon(args: tuple) -> Dict[str, Any]:
  (
      geom,
      gid,
      min_overlap_threshold,
      era5_source,
      gdb_path,
      era5_cache_dir,
      gridded_era5_uri,
  ) = args
  ext = _get_worker_extractor(gdb_path, era5_cache_dir, gridded_era5_uri)
  return ext.extract_attributes_for_polygon(
      geom,
      catchment_id=gid,
      min_overlap_threshold=min_overlap_threshold,
      era5_source=era5_source,
  )



def compute_pour_point_properties(
    basin_data: Dict[str, List[Any]],
    min_overlap_threshold: float = 0.0,
    pour_point_properties: Optional[List[str]] = None,
) -> Dict[str, float]:
  """Computes Caravan pour-point metrics by following NEXT_DOWN to find the outlet sub-basins."""
  props_to_compute = pour_point_properties or POUR_POINT_PROPERTIES
  if not props_to_compute:
    return {}

  weights = np.array(basin_data.get("weights", []))
  sub_areas = np.array(basin_data.get("SUB_AREA", []))
  if len(weights) == 0 or len(sub_areas) == 0:
    return {p: 0.0 for p in props_to_compute}

  percentage_overlap = np.where(sub_areas > 0, weights / sub_areas, 0.0)
  if len(percentage_overlap) == 0:
    return {p: 0.0 for p in props_to_compute}

  current_basin_pos = int(np.argmax(percentage_overlap))
  next_down_id = basin_data["NEXT_DOWN"][current_basin_pos]

  # Traverse downstream until leaving the polygon or hitting ocean (0)
  while True:
    if next_down_id == 0:
      break
    if next_down_id not in basin_data["HYBAS_ID"]:
      break
    next_down_pos = basin_data["HYBAS_ID"].index(next_down_id)
    if percentage_overlap[next_down_pos] < 0.5:
      break
    next_down_id = basin_data["NEXT_DOWN"][next_down_pos]

  # Find all sub-basins draining into the terminal downstream outlet
  direct_upstream_polygons = []
  for i, next_down in enumerate(basin_data["NEXT_DOWN"]):
    if (next_down == next_down_id) and (
        (basin_data["weights"][i] > min_overlap_threshold)
        or (percentage_overlap[i] > 0.5)
    ):
      direct_upstream_polygons.append(i)

  if not direct_upstream_polygons:
    direct_upstream_polygons = [current_basin_pos]

  aggregated = {}
  for prop in props_to_compute:
    if prop in basin_data:
      aggregated[prop] = float(
          sum(basin_data[prop][i] for i in direct_upstream_polygons)
      )
    else:
      aggregated[prop] = 0.0
  return aggregated


class StaticAttributesExtractor:
  """Extracts and computes exact Caravan & HydroATLAS static catchment attributes."""

  def __init__(
      self,
      gdb_path: Optional[Union[str, Path]] = None,
      era5_cache_dir: Optional[Union[str, Path]] = None,
      auto_download: bool = True,
      era5_source: str = DEFAULT_ERA5_SOURCE,
      gridded_era5_uri: Optional[str] = None,
  ):
    """Initializes the StaticAttributesExtractor.

    Authoritative data sources are strictly:
      - HydroATLAS: gs://open-multimet/data/hydroatlas/BasinATLAS_v10.gdb/
      - ERA5 Climate (hybas): gs://open-multimet/data/hydroatlas/era5_climate/
      - ERA5 Gridded (gridded): gs://open-multimet/data/era5_land/daily_surface.zarr

    Args:
      gdb_path: Path to runtime staging BasinATLAS_v10.gdb directory. If None,
        defaults to ~/.cache/googlehydrology/hydroatlas/BasinATLAS_v10.gdb.
      era5_cache_dir: Path to runtime staging directory for ERA5 climate files.
      auto_download: Whether to automatically download BasinATLAS_v10.gdb from
        GCS if not staged locally. Defaults to True.
      era5_source: Sourcing mode for ERA5 climate attributes. Options:
        - "hybas": Fast area-weighted aggregation of precalculated Level 12
          sub-basin climate metrics (default, ~20ms per basin).
        - "gridded": Recalculate directly on the fly from archived gridded ERA5
          daily surface Zarr data on GCS.
      gridded_era5_uri: Optional GCS URI or path to gridded daily ERA5 Zarr store.
    """
    self.era5_source = era5_source.lower() if era5_source else "hybas"
    self.gridded_era5_uri = gridded_era5_uri or GCS_ERA5_GRIDDED_ZARR_URI

    if gdb_path is not None:
      self.gdb_path = Path(gdb_path)
    else:
      self.gdb_path = get_default_gdb_path()

    if (not self.gdb_path.exists() or not any(self.gdb_path.iterdir())) and auto_download:
      logger.info(
          "BasinATLAS GDB not found in runtime cache %s. Automatically downloading from %s...",
          self.gdb_path,
          GCS_HYDROATLAS_GDB_URI,
      )
      self.gdb_path = download_hydroatlas_from_gcs(target_dir=self.gdb_path)

    # Determine if target is a FileGDB directory or shapefile
    self.is_shapefile = str(self.gdb_path).endswith(".shp")
    self.layer_name = None if self.is_shapefile else "BasinATLAS_v10_lev12"

    if self.gdb_path.exists():
      try:
        if pyogrio is not None:
          if self.is_shapefile:
            info = pyogrio.read_info(self.gdb_path)
          else:
            info = pyogrio.read_info(self.gdb_path, layer=self.layer_name)
          self.all_gdb_fields = list(info["fields"])
        else:
          kwargs = {} if self.is_shapefile else {"layer": self.layer_name}
          sample_df = gpd.read_file(self.gdb_path, rows=1, **kwargs)
          self.all_gdb_fields = list(sample_df.columns)
      except Exception as e:
        logger.warning(
            "Could not read fields from %s (%s). Initializing with schema definitions.",
            self.gdb_path,
            e,
        )
        self.all_gdb_fields = list(ATTRIBUTE_DEFINITIONS.keys())
    else:
      logger.warning(
          "BasinATLAS dataset not found at %s. Initialized with schema definitions.",
          self.gdb_path,
      )
      self.all_gdb_fields = list(ATTRIBUTE_DEFINITIONS.keys())

    self.use_properties = [
        p
        for p in self.all_gdb_fields
        if p not in IGNORE_PROPERTIES + UPSTREAM_PROPERTIES
    ]
    self.caravan_feature_names = [
        p for p in self.use_properties if p not in ADDITIONAL_PROPERTIES
    ]

    # Initialize ERA5 climate loaders
    self.era5_loader = ERA5ClimateLoader(cache_dir=era5_cache_dir)
    self.gridded_extractor = ERA5GriddedExtractor(zarr_uri=self.gridded_era5_uri)

  def _read_subbasins_in_bbox(
      self, bbox: Tuple[float, float, float, float]
  ) -> gpd.GeoDataFrame:
    """Reads Level 12 sub-basins within bounding box from GDB or shapefile."""
    if not self.gdb_path.exists():
      raise FileNotFoundError(
          f"BasinATLAS dataset not found at {self.gdb_path}. "
          "Please download it using static_extractor.gcs.download_hydroatlas_from_gcs() "
          "or set HYDROATLAS_GDB_PATH environment variable."
      )

    read_kwargs = {"bbox": bbox}
    if not self.is_shapefile:
      read_kwargs["layer"] = self.layer_name

    if pyogrio is not None:
      return pyogrio.read_dataframe(self.gdb_path, **read_kwargs)
    return gpd.read_file(self.gdb_path, **read_kwargs)

  def extract_attributes_for_polygon(
      self,
      polygon_geojson: Union[Dict, Polygon, MultiPolygon, gpd.GeoSeries, gpd.GeoDataFrame],
      catchment_id: Optional[str] = None,
      min_overlap_threshold: float = 0.0,
      baseline_years: Tuple[int, int] = (1981, 2020),
      timeseries_df: Optional[pd.DataFrame] = None,
      era5_source: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Calculates exact Caravan HydroATLAS static attributes for an arbitrary watershed polygon.

    Args:
      polygon_geojson: GeoJSON Feature, Geometry dict, Shapely Polygon/MultiPolygon,
        or GeoDataFrame row.
      catchment_id: Optional catchment identifier string.
      min_overlap_threshold: Minimum area threshold in km2 for filtering small overlap slivers.
      baseline_years: Tuple of start and end years for climate baseline (default 1981-2020).
      timeseries_df: Optional daily timeseries DataFrame containing columns
        (total_precipitation or prcp, temperature or 2m_temperature,
        potential_evaporation or pet) to compute climate indices directly.

    Returns:
      Dictionary containing:
      - catchment_id: Identifier string
      - caravan_attributes: Full 197+ Caravan feature dictionary
      - summary: User-friendly summary metrics
      - categories: Categorized display dictionary for visualization
      - processed_attributes: Formatted attribute metrics with units
      - intersected_subbasins_count: Total HydroATLAS Level 12 units intersected
      - total_area_km2: Drainage area in km²
    """
    # 1. Parse Input Geometry
    if isinstance(polygon_geojson, (gpd.GeoDataFrame, gpd.GeoSeries)):
      geom = polygon_geojson.geometry.iloc[0] if hasattr(polygon_geojson, "geometry") else polygon_geojson.iloc[0]
      if catchment_id is None and hasattr(polygon_geojson, "columns"):
        for id_col in ["gauge_id", "catchment_id", "id", "basin_id"]:
          if id_col in polygon_geojson.columns:
            catchment_id = str(polygon_geojson[id_col].iloc[0])
            break
    elif isinstance(polygon_geojson, dict):
      if polygon_geojson.get("type") == "Feature":
        geom_dict = polygon_geojson["geometry"]
        props = polygon_geojson.get("properties", {})
        catchment_id = (
            catchment_id
            or props.get("catchment_id")
            or props.get("gauge_id")
            or props.get("id")
            or "custom_catchment"
        )
      else:
        geom_dict = polygon_geojson
        catchment_id = catchment_id or "custom_catchment"
      geom = shape(geom_dict)
    else:
      geom = polygon_geojson
      catchment_id = catchment_id or "custom_catchment"

    if not geom.is_valid:
      geom = geom.buffer(0)

    if geom.area <= 0:
      raise ValueError("Target polygon area must be greater than 0.")

    minx, miny, maxx, maxy = geom.bounds

    # 2. Read BasinATLAS Level 12 Sub-basins within Bounding Box
    bbox = (minx - 0.02, miny - 0.02, maxx + 0.02, maxy + 0.02)
    try:
      gdf_subbasins = self._read_subbasins_in_bbox(bbox)
    except Exception as e:
      logger.warning("Error querying bounding box %s: %s", bbox, e)
      gdf_subbasins = gpd.GeoDataFrame()

    if len(gdf_subbasins) == 0:
      logger.info("No sub-basins found in immediate bbox %s. Trying broader bbox.", bbox)
      bbox_wide = (minx - 0.1, miny - 0.1, maxx + 0.1, maxy + 0.1)
      try:
        gdf_subbasins = self._read_subbasins_in_bbox(bbox_wide)
      except Exception:
        gdf_subbasins = gpd.GeoDataFrame()

    if len(gdf_subbasins) == 0:
      raise ValueError(
          f"No BasinATLAS Level 12 units found intersecting polygon bounds {bbox}. "
          "Ensure the polygon coordinates are in EPSG:4326 (latitude/longitude)."
      )

    # 3. Calculate exact geometric intersections and area weights in km²
    intersections = gdf_subbasins.geometry.intersection(geom)
    valid_mask = ~intersections.is_empty
    gdf_matched = gdf_subbasins[valid_mask].copy()
    gdf_matched["intersect_geom"] = intersections[valid_mask]

    if len(gdf_matched) == 0:
      centroid = geom.centroid
      distances = gdf_subbasins.geometry.distance(centroid)
      nearest_idx = distances.idxmin()
      gdf_matched = gdf_subbasins.loc[[nearest_idx]].copy()
      gdf_matched["intersect_geom"] = [geom]

    # Geodesic area scaling using latitude projection
    mean_lat = geom.centroid.y
    deg_to_km = 111.32
    lat_scale = deg_to_km
    lon_scale = deg_to_km * np.cos(np.radians(mean_lat))

    gdf_matched["intersect_area_km2"] = [
        float(g.area * lat_scale * lon_scale)
        for g in gdf_matched["intersect_geom"]
    ]

    # 4. Collect Sub-basin Data with Caravan Overlap Rules
    basin_data = defaultdict(list)
    for _, row in gdf_matched.iterrows():
      int_area = float(row["intersect_area_km2"])
      sub_area = (
          float(row["SUB_AREA"])
          if ("SUB_AREA" in row and row["SUB_AREA"] > 0)
          else int_area
      )

      # Caravan filtering threshold: either > min_overlap_threshold or >50% of sub-basin
      if (int_area > min_overlap_threshold) or (int_area / sub_area > 0.5):
        for prop in self.use_properties:
          if prop in row:
            basin_data[prop].append(row[prop])
        basin_data["weights"].append(int_area)

      basin_data["area_fragments"].append(int_area)

    # Fallback if all sub-basins were filtered out
    if not basin_data["weights"]:
      for prop in self.use_properties:
        if prop in gdf_matched.columns:
          basin_data[prop].append(gdf_matched.iloc[0][prop])
      basin_data["weights"].append(
          float(gdf_matched.iloc[0]["intersect_area_km2"])
      )

    weights = np.array(basin_data["weights"])
    mask = weights > 0
    masked_weights = weights[mask]

    # 5. Aggregate Caravan Properties
    caravan_attributes: Dict[str, Any] = {}

    for key in self.use_properties:
      if key in [
          "weights",
          "UP_AREA",
          "area_fragments",
          "HYBAS_ID",
          "NEXT_DOWN",
          "SUB_AREA",
          "geometry",
          "geom",
          "Shape",
      ]:
        continue
      if key in POUR_POINT_PROPERTIES:
        continue
      if key not in basin_data:
        continue

      try:
        val = np.array(basin_data[key], dtype=float)
      except (ValueError, TypeError):
        continue
      masked_val = val[mask]

      # Caravan rule for wetland classes: no wetland (-999 / -9999 / <0) is mapped to class 13
      if key == "wet_cl_smj":
        masked_val = np.where(
            (masked_val == -999) | (masked_val == -9999) | (masked_val < 0),
            13,
            masked_val,
        )

      valid_idx = (masked_val > -900) & (~np.isnan(masked_val))

      if not np.any(valid_idx):
        caravan_attributes[key] = np.nan
      else:
        if key in MAJORITY_PROPERTIES:
          # Area-Weighted Majority Vote
          valid_vals = masked_val[valid_idx].astype(int)
          valid_w = masked_weights[valid_idx]
          val_counts = np.bincount(valid_vals, weights=valid_w)
          caravan_attributes[key] = int(val_counts.argmax())
        else:
          # Area-Weighted Average
          caravan_attributes[key] = float(
              np.average(
                  masked_val[valid_idx], weights=masked_weights[valid_idx]
              )
          )

    # 6. Extract / Compute ERA5-Land Climate Attributes (1981-2020)
    era5_indices = {}
    actual_era5_source = (era5_source or self.era5_source).lower()

    if timeseries_df is not None and not timeseries_df.empty:
      # Compute directly from provided daily timeseries DataFrame
      p_col = next((c for c in ["total_precipitation", "prcp", "precip", "tp"] if c in timeseries_df.columns), None)
      t_col = next((c for c in ["temperature", "2m_temperature", "temp", "t2m"] if c in timeseries_df.columns), None)
      pet_era5_col = next((c for c in ["potential_evaporation", "pet_era5", "pev"] if c in timeseries_df.columns), None)
      pet_fao_col = next((c for c in ["pet_fao", "pet_mean_FAO_PM", "fao_pet"] if c in timeseries_df.columns), None)

      if p_col and t_col and pet_era5_col:
        p_series = timeseries_df[p_col]
        t_series = timeseries_df[t_col]
        pet_era5_series = timeseries_df[pet_era5_col]
        pet_fao_series = timeseries_df[pet_fao_col] if pet_fao_col else None
        era5_indices = compute_caravan_climate_metrics(
            precipitation=p_series,
            temperature=t_series,
            pet_era5=pet_era5_series,
            pet_fao=pet_fao_series,
        )
    elif actual_era5_source == "gridded":
      # Recalculate directly on the fly from archived gridded ERA5 data on GCS
      try:
        era5_indices = self.gridded_extractor.extract_climate_metrics_for_polygon(
            geom, baseline_years=baseline_years
        )
      except Exception as e:
        logger.warning(
            "Gridded ERA5 extraction failed for catchment '%s': %s. Falling back to HYBAS statistics.",
            catchment_id,
            e,
        )
        era5_indices = {}

    if not era5_indices or all(pd.isna(v) for v in era5_indices.values()):
      # Load from Level 12 precomputed continental climate indices table
      hybas_ids = [int(hid) for hid in gdf_matched["HYBAS_ID"].values]
      intersect_weights = [
          float(w) for w in gdf_matched["intersect_area_km2"].values
      ]
      era5_indices = self.era5_loader.get_indices_for_subbasins(
          hybas_ids, intersect_weights
      )

    for k, v in era5_indices.items():
      caravan_attributes[k] = v

    # 7. Drainage Area & Aggregation Fraction
    total_frag_area = float(sum(basin_data["area_fragments"]))
    caravan_attributes["area"] = total_frag_area
    caravan_attributes["basin_area"] = total_frag_area  # Canonical Caravan attribute name
    caravan_attributes["area_fraction_used_for_aggregation"] = (
        float(sum(masked_weights) / total_frag_area)
        if total_frag_area > 0
        else 1.0
    )

    # 8. Curated UI Schema Formatting
    processed_attributes: Dict[str, Any] = {}
    categories_dict: Dict[str, List[Dict[str, Any]]] = {
        "Topography": [],
        "Climate": [],
        "Soils": [],
        "Land Cover": [],
        "Hydrology": [],
        "Anthropogenic": [],
    }

    for attr_key, defn in ATTRIBUTE_DEFINITIONS.items():
      if attr_key in caravan_attributes:
        raw_val = caravan_attributes[attr_key]
        if pd.isna(raw_val):
          scaled_val = 0.0
        else:
          scaled_val = round(float(raw_val) * defn["scale"], 3)
          if defn["unit"] in ["m", "mm/yr", "people", "M m³"]:
            scaled_val = (
                round(scaled_val, 1)
                if defn["unit"] != "people"
                else int(round(scaled_val))
            )
          elif defn["unit"].startswith("class"):
            scaled_val = int(scaled_val)

        item = {
            "key": attr_key,
            "name": defn["name"],
            "value": scaled_val,
            "unit": defn["unit"],
            "description": defn["desc"],
            "category": defn["category"],
        }
        processed_attributes[attr_key] = scaled_val
        categories_dict[defn["category"]].append(item)

    # 9. Summary Metrics
    summary = {
        "catchment_id": catchment_id,
        "elevation_mean_m": processed_attributes.get("ele_mt_sav", 0.0),
        "slope_mean_deg": processed_attributes.get("slp_dg_sav", 0.0),
        "annual_precip_mm": processed_attributes.get("pre_mm_syr", 0.0),
        "annual_temp_c": processed_attributes.get("tmp_dc_syr", 0.0),
        "aridity_index": processed_attributes.get("ari_ix_sav", 0.0),
        "era5_p_mean_mm_day": processed_attributes.get("p_mean", 0.0),
        "era5_pet_mean_mm_day": processed_attributes.get(
            "pet_mean_ERA5_LAND", 0.0
        ),
        "era5_fao_pet_mean_mm_day": processed_attributes.get(
            "pet_mean_FAO_PM", 0.0
        ),
        "era5_aridity": processed_attributes.get("aridity_ERA5_LAND", 0.0),
        "era5_fao_aridity": processed_attributes.get("aridity_FAO_PM", 0.0),
        "era5_frac_snow_pc": processed_attributes.get("frac_snow", 0.0),
        "forest_fraction_pc": processed_attributes.get("for_pc_sse", 0.0),
        "cropland_fraction_pc": processed_attributes.get("crp_pc_sse", 0.0),
        "urban_fraction_pc": processed_attributes.get("urb_pc_sse", 0.0),
        "dominant_land_cover_class": caravan_attributes.get("glc_cl_smj", 0),
        "soil_clay_pc": processed_attributes.get("cly_pc_sav", 0.0),
        "soil_sand_pc": processed_attributes.get("snd_pc_sav", 0.0),
        "soil_silt_pc": processed_attributes.get("slt_pc_sav", 0.0),
        "groundwater_table_depth_cm": processed_attributes.get(
            "gwt_cm_sav", 0.0
        ),
        "soil_water_content_pc": processed_attributes.get("swc_pc_syr", 0.0),
        "inundation_max_pc": processed_attributes.get("inu_pc_smx", 0.0),
        "total_area_km2": round(total_frag_area, 2),
        "intersected_subbasins": len(gdf_matched),
        "subbasin_ids": [int(hid) for hid in gdf_matched["HYBAS_ID"].values],
    }

    return {
        "catchment_id": catchment_id,
        "caravan_attributes": caravan_attributes,
        "raw_attributes": caravan_attributes,
        "summary": summary,
        "categories": categories_dict,
        "processed_attributes": processed_attributes,
        "intersected_subbasins_count": len(gdf_matched),
        "total_area_km2": round(total_frag_area, 2),
    }

  def extract_attributes_batch(
      self,
      features: List[Union[Dict[str, Any], Polygon, MultiPolygon]],
      min_overlap_threshold: float = 0.0,
      era5_source: Optional[str] = None,
  ) -> List[Dict[str, Any]]:
    """Extracts exact Caravan attributes for a list of watershed features."""
    results = []
    for i, feat in enumerate(features):
      c_id = None
      if isinstance(feat, dict):
        props = feat.get("properties", {})
        c_id = (
            props.get("catchment_id")
            or props.get("gauge_id")
            or props.get("id")
            or f"basin_{i+1}"
        )
      else:
        c_id = f"basin_{i+1}"
      res = self.extract_attributes_for_polygon(
          feat,
          catchment_id=c_id,
          min_overlap_threshold=min_overlap_threshold,
          era5_source=era5_source,
      )
      results.append(res)
    return results

  def extract_attributes_from_file(
      self,
      input_path: Union[str, Path],
      output_csv_path: Optional[Union[str, Path]] = None,
      id_column: Optional[str] = None,
      min_overlap_threshold: float = 0.0,
      era5_source: Optional[str] = None,
      workers: int = 1,
  ) -> pd.DataFrame:
    """Extracts Caravan attributes for all features in a vector file (Shapefile, GeoJSON, GeoPackage).

    Args:
      input_path: Path to vector polygon file.
      output_csv_path: Optional path to save extracted attributes CSV.
      id_column: Name of column to use for basin / gauge ID.
      min_overlap_threshold: Minimum area threshold in km2.
      era5_source: Optional ERA5 sourcing mode override ('hybas' or 'gridded').
      workers: Number of parallel processes to use (default 1).

    Returns:
      Pandas DataFrame with extracted attributes, indexed by gauge_id.
    """
    gdf = gpd.read_file(input_path)
    if gdf.crs is not None and not gdf.crs.is_geographic:
      gdf = gdf.to_crs(epsg=4326)

    # Determine ID column
    if id_column is None:
      for candidate in ["gauge_id", "catchment_id", "id", "basin_id", "HYBAS_ID", "gauge_id_"]:
        if candidate in gdf.columns:
          id_column = candidate
          break

    tasks = []
    for idx, row in gdf.iterrows():
      gid = str(row[id_column]) if id_column and id_column in row else f"basin_{idx+1}"
      tasks.append((row.geometry, gid))

    if workers > 1 and len(tasks) > 1:
      import concurrent.futures
      worker_args = [
          (
              geom,
              gid,
              min_overlap_threshold,
              era5_source,
              str(self.gdb_path),
              str(self.era5_cache_dir),
              self.gridded_era5_uri,
          )
          for geom, gid in tasks
      ]
      logger.info(
          "Processing %d catchments in parallel with %d workers...",
          len(tasks),
          workers,
      )
      with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_worker_extract_polygon, worker_args))
    else:
      results = []
      for geom, gid in tasks:
        res = self.extract_attributes_for_polygon(
            geom,
            catchment_id=gid,
            min_overlap_threshold=min_overlap_threshold,
            era5_source=era5_source,
        )
        results.append(res)

    df = self.export_caravan_csv(
        results, output_csv_path=output_csv_path if output_csv_path else None
    )
    return df

  def export_caravan_csv(
      self,
      results: Union[List[Dict[str, Any]], pd.DataFrame],
      output_csv_path: Optional[Union[str, Path]] = None,
  ) -> pd.DataFrame:
    """Formats and exports Caravan attributes to standard CSV."""
    if isinstance(results, pd.DataFrame):
      df = results
    else:
      rows = []
      gauge_ids = []
      for r in results:
        gid = r.get("catchment_id", "basin")
        gauge_ids.append(gid)
        rows.append(r["caravan_attributes"])
      df = pd.DataFrame(rows, index=gauge_ids)
      df.index.name = "gauge_id"

    # Sort columns alphabetically, but ensure basin_area is early
    sorted_cols = sorted(df.columns)
    if "basin_area" in sorted_cols:
      sorted_cols.remove("basin_area")
      sorted_cols = ["basin_area"] + sorted_cols
    df = df[sorted_cols].sort_index(axis=0)

    if output_csv_path:
      p = Path(output_csv_path)
      p.parent.mkdir(parents=True, exist_ok=True)
      df.to_csv(p)
      logger.info("Saved Caravan static attributes to %s (shape: %s)", p, df.shape)

    return df

  def append_attributes_to_zarr(
      self,
      master_zarr_path: Union[str, Path],
      basin_id: str,
      attributes_dict: Dict[str, Any],
  ) -> None:
    """Appends static Caravan & HydroATLAS attributes to a master Zarr store along 'basin' dim."""
    if xr is None:
      raise ImportError("xarray is required to append attributes to Zarr.")

    master_path = Path(master_zarr_path)
    if not master_path.exists():
      return

    try:
      ds = xr.open_zarr(str(master_path)).load()
      if "basin" not in ds.dims:
        return

      basin_list = list(ds["basin"].values)
      if basin_id not in basin_list:
        return

      basin_idx = basin_list.index(basin_id)
      caravan_attrs = attributes_dict.get("caravan_attributes", {})

      for key, val in caravan_attrs.items():
        if isinstance(val, (int, float, np.integer, np.floating)):
          var_name = f"caravan_{key}"
          if var_name not in ds:
            arr = np.full((len(basin_list),), np.nan, dtype=np.float32)
            arr[basin_idx] = float(val)
            ds[var_name] = (["basin"], arr)
          else:
            ds[var_name].values[basin_idx] = float(val)

      ds.to_zarr(str(master_path), mode="w", consolidated=True)
      logger.info(
          "Appended Caravan static attributes for %s to %s", basin_id, master_path
      )
    except Exception as e:
      logger.warning("Could not append static attributes to Zarr store: %s", e)
