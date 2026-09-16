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

"""Multi-dataset batch runner for Caravan static attribute extraction."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from static_extractor.extractor import StaticAttributesExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("static_extractor.batch_runner")

SUPPORTED_EXTENSIONS = [".shp", ".geojson", ".gpkg", ".json", ".parquet", ".geoparquet"]


def sync_gcs_directory(gcs_uri: str, local_dest: Path) -> Path:
  """Syncs a GCS directory to a local directory using gcloud storage or gsutil."""
  local_dest.mkdir(parents=True, exist_ok=True)
  logger.info("Syncing %s to local staging directory %s...", gcs_uri, local_dest)

  gcs_uri_clean = gcs_uri if gcs_uri.endswith("/") else gcs_uri + "/"

  # Try gcloud storage rsync first (preferred)
  try:
    res = subprocess.run(
        ["gcloud", "storage", "rsync", "-r", gcs_uri_clean, str(local_dest)],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0:
      return local_dest
    logger.debug("gcloud storage rsync failed: %s. Trying gsutil...", res.stderr)
  except Exception as e:
    logger.debug("gcloud storage rsync error: %s. Trying gsutil...", e)

  # Try gsutil rsync
  try:
    res = subprocess.run(
        ["gsutil", "-m", "rsync", "-r", gcs_uri_clean, str(local_dest)],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0:
      return local_dest
    logger.debug("gsutil rsync failed: %s. Trying gcloud storage cp...", res.stderr)
  except Exception as e:
    logger.debug("gsutil rsync error: %s", e)

  # Fallback to gcloud storage cp
  res = subprocess.run(
      ["gcloud", "storage", "cp", "-r", gcs_uri_clean, str(local_dest)],
      capture_output=True,
      text=True,
      check=False,
  )
  if res.returncode != 0:
    logger.warning("Could not sync GCS URI %s: %s", gcs_uri, res.stderr)
  return local_dest


def find_vector_file_in_dir(dataset_dir: Path) -> Optional[Path]:
  """Finds the primary watershed polygon file in a dataset directory."""
  dataset_name = dataset_dir.name.lower()

  # Check for typical Caravan naming first (e.g. camels_basin_shapes.shp)
  preferred_names = [
      f"{dataset_name}_basin_shapes.shp",
      f"{dataset_name}_basins.shp",
      f"{dataset_name}.shp",
      f"{dataset_name}.geojson",
      f"{dataset_name}.geoparquet",
      f"{dataset_name}.parquet",
  ]
  for pref in preferred_names:
    p = dataset_dir / pref
    if p.exists():
      return p

  # Check any shapefiles, excluding auxiliary/gauge points if basin shapes exist
  shps = list(dataset_dir.glob("*.shp"))
  if shps:
    basin_shps = [
        s
        for s in shps
        if "gauge" not in s.name.lower() and "point" not in s.name.lower()
    ]
    return basin_shps[0] if basin_shps else shps[0]

  # Check GeoJSON, GPKG, or GeoParquet
  for ext in [".geojson", ".gpkg", ".json", ".parquet", ".geoparquet"]:
    matches = list(dataset_dir.glob(f"*{ext}"))
    if matches:
      return matches[0]

  return None


def find_all_dataset_dirs(root_dir: Path) -> Dict[str, Path]:
  """Recursively finds all dataset directories containing vector files under root_dir."""
  datasets: Dict[str, Path] = {}
  # Check if root_dir directly contains multiple vector files (e.g. continental geoparquet files)
  direct_vfs = [
      f for f in root_dir.iterdir()
      if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
      and "gauge" not in f.name.lower() and "point" not in f.name.lower()
  ]
  if len(direct_vfs) > 1:
    for vf in sorted(direct_vfs):
      ds_key = f"{root_dir.name}_{vf.stem}" if root_dir.name not in ["staged_shapefiles", "data", "shapes"] else vf.stem
      datasets[ds_key] = vf
    return datasets

  # First check if root_dir itself is a single dataset directory
  root_vf = find_vector_file_in_dir(root_dir)
  if root_vf:
    datasets[root_dir.name] = root_vf
    return datasets

  for dirpath, dirnames, _ in os.walk(root_dir):
    d = Path(dirpath)
    if d == root_dir:
      continue
    # Check if this subdirectory has multiple vector files directly
    d_vfs = [
        f for f in d.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        and "gauge" not in f.name.lower() and "point" not in f.name.lower()
    ]
    if len(d_vfs) > 1:
      for vf in sorted(d_vfs):
        ds_key = f"{d.name}_{vf.stem}" if d.name not in ["staged_shapefiles", "data", "shapes"] else vf.stem
        datasets[ds_key] = vf
      dirnames.clear()
      continue

    vf = find_vector_file_in_dir(d)
    if vf:
      datasets[d.name] = vf
      # Stop recursing into subdirectories of a discovered dataset
      dirnames.clear()
  return datasets


def discover_datasets(
    parent_dirs: Optional[List[str]] = None,
    input_dirs: Optional[List[str]] = None,
    input_files: Optional[List[str]] = None,
    staging_cache_dir: Optional[Path] = None,
) -> Dict[str, Path]:
  """Discovers dataset names and their corresponding vector files.

  Returns a mapping of dataset_name -> vector_file_path.
  """
  dataset_map: Dict[str, Path] = {}

  if staging_cache_dir is None:
    staging_cache_dir = (
        Path.home() / ".cache" / "googlehydrology" / "staged_shapefiles"
    )

  # 1. Process specific input files
  if input_files:
    for f_str in input_files:
      p = Path(f_str)
      if p.exists():
        d_name = p.stem.replace("_basin_shapes", "").replace("_basins", "")
        dataset_map[d_name] = p

  # 2. Process specific input directories
  if input_dirs:
    for d_str in input_dirs:
      if d_str.startswith("gs://"):
        d_name = d_str.rstrip("/").split("/")[-1]
        local_dir = sync_gcs_directory(d_str, staging_cache_dir / d_name)
      else:
        local_dir = Path(d_str)

      if local_dir.is_dir():
        found = find_all_dataset_dirs(local_dir)
        if found:
          dataset_map.update(found)
        else:
          logger.warning("No vector polygon file found in %s", local_dir)

  # 3. Process parent directories (containing subdirectories for each dataset)
  if parent_dirs:
    for p_str in parent_dirs:
      if p_str.startswith("gs://"):
        parent_name = p_str.rstrip("/").split("/")[-1]
        local_parent = sync_gcs_directory(
            p_str, staging_cache_dir / parent_name
        )
      else:
        local_parent = Path(p_str)

      if not local_parent.is_dir():
        logger.warning(
            "Parent directory %s does not exist or is not a directory.",
            local_parent,
        )
        continue

      found = find_all_dataset_dirs(local_parent)
      if found:
        dataset_map.update(found)
      else:
        logger.warning(
            "No dataset vector files found under parent directory %s",
            local_parent,
        )

  return dataset_map


def run_batch_extraction(
    dataset_map: Dict[str, Path],
    output_dir: Union[str, Path],
    workers: int = 1,
    era5_source: str = "hybas",
    gridded_era5_uri: Optional[str] = None,
    gdb_path: Optional[str] = None,
    era5_cache_dir: Optional[str] = None,
    min_overlap_threshold: float = 0.0,
    combine: bool = False,
    resume: bool = True,
) -> Dict[str, pd.DataFrame]:
  """Runs static attribute extraction across all discovered datasets."""
  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  logger.info("Initializing StaticAttributesExtractor (era5_source=%s)...", era5_source)
  extractor = StaticAttributesExtractor(
      gdb_path=gdb_path,
      era5_cache_dir=era5_cache_dir,
      era5_source=era5_source,
      gridded_era5_uri=gridded_era5_uri,
      auto_download=True,
  )

  extracted_dfs: Dict[str, pd.DataFrame] = {}
  total_datasets = len(dataset_map)
  start_all = time.time()

  logger.info("Found %d datasets to process: %s", total_datasets, list(dataset_map.keys()))

  for i, (ds_name, vector_path) in enumerate(dataset_map.items(), start=1):
    out_file = out_dir / f"attributes_caravan_{ds_name}.csv"
    if resume and out_file.exists() and out_file.stat().st_size > 500:
      logger.info(
          "[%d/%d] Skipping %s (already completed: %s)",
          i,
          total_datasets,
          ds_name,
          out_file,
      )
      df = pd.read_csv(out_file, index_col=0)
      extracted_dfs[ds_name] = df
      continue

    logger.info(
        "==========================================================\n"
        "[%d/%d] Processing dataset '%s' from %s (workers=%d)...\n"
        "==========================================================",
        i,
        total_datasets,
        ds_name,
        vector_path,
        workers,
    )
    t0 = time.time()
    try:
      df = extractor.extract_attributes_from_file(
          input_path=vector_path,
          output_csv_path=out_file,
          min_overlap_threshold=min_overlap_threshold,
          era5_source=era5_source,
          workers=workers,
      )
      elapsed = time.time() - t0
      logger.info(
          "Successfully completed '%s': %d basins extracted in %.1f seconds (saved to %s)",
          ds_name,
          len(df),
          elapsed,
          out_file,
      )
      extracted_dfs[ds_name] = df
    except Exception as e:
      logger.exception("Error extracting attributes for dataset '%s': %s", ds_name, e)

  total_elapsed = time.time() - start_all
  total_basins = sum(len(df) for df in extracted_dfs.values())
  logger.info(
      "All datasets processed: %d total basins across %d datasets in %.1f seconds.",
      total_basins,
      len(extracted_dfs),
      total_elapsed,
  )

  if combine and extracted_dfs:
    combined_path = out_dir / "attributes_caravan_combined.csv"
    logger.info("Combining all %d datasets into %s...", len(extracted_dfs), combined_path)
    combined_df = pd.concat(list(extracted_dfs.values()), axis=0)
    combined_df.to_csv(combined_path)
    logger.info("Combined CSV written: %d total rows.", len(combined_df))

  return extracted_dfs


def parse_args(args=None):
  parser = argparse.ArgumentParser(
      description="Batch runner for Caravan static attribute extraction across multiple datasets.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--parent-dir",
      "-p",
      action="append",
      dest="parent_dirs",
      help="Parent directory containing dataset subdirectories (e.g. /path/to/caravan/ or gs://...). Can specify multiple times.",
  )
  parser.add_argument(
      "--input-dirs",
      "-d",
      nargs="+",
      default=None,
      help="List of dataset directories (e.g., path/to/camels path/to/hysets).",
  )
  parser.add_argument(
      "--input-files",
      "-f",
      nargs="+",
      default=None,
      help="List of specific vector polygon files (.shp, .geojson, .gpkg).",
  )
  parser.add_argument(
      "--output-dir",
      "-o",
      required=True,
      type=str,
      help="Directory to save extracted attributes CSV files.",
  )
  parser.add_argument(
      "--workers",
      "-w",
      type=int,
      default=1,
      help="Number of parallel worker processes to use.",
  )
  parser.add_argument(
      "--era5-source",
      choices=["hybas", "gridded"],
      default="hybas",
      help="Source for ERA5 climate metrics: 'hybas' (fast precalculated) or 'gridded' (on-the-fly Zarr recalculation).",
  )
  parser.add_argument(
      "--gridded-era5-uri",
      default=None,
      type=str,
      help="GCS URI or path to gridded daily ERA5 Zarr store.",
  )
  parser.add_argument(
      "--gdb-path",
      "-g",
      default=None,
      type=str,
      help="Optional path to local BasinATLAS_v10.gdb.",
  )
  parser.add_argument(
      "--era5-cache-dir",
      default=None,
      type=str,
      help="Optional directory for ERA5 climate tables.",
  )
  parser.add_argument(
      "--min-overlap-threshold",
      default=0.0,
      type=float,
      help="Minimum sub-basin intersection area threshold in km².",
  )
  parser.add_argument(
      "--cache-dir",
      default=None,
      type=str,
      help="Base directory for runtime cache (defaults to ~/.cache/googlehydrology).",
  )
  parser.add_argument(
      "--no-download",
      action="store_true",
      help="Disable automatic GCS downloads. Requires local files to be present.",
  )
  parser.add_argument(
      "--clean-staging",
      action="store_true",
      help="Automatically clean up staged shapefiles after batch extraction finishes.",
  )
  parser.add_argument(
      "--clean-cache",
      action="store_true",
      help="Automatically clean up the entire local cache directory (~/.cache/googlehydrology) after extraction finishes.",
  )
  parser.add_argument(
      "--combine",
      action="store_true",
      help="Also save a combined attributes_caravan_combined.csv containing all processed datasets.",
  )
  parser.add_argument(
      "--no-resume",
      action="store_false",
      dest="resume",
      default=True,
      help="Do not skip datasets that have already been extracted.",
  )
  return parser.parse_args(args)


def main(args=None):
  parsed = parse_args(args)
  cache_root = Path(parsed.cache_dir) if parsed.cache_dir else Path.home() / ".cache" / "googlehydrology"
  staging_cache = cache_root / "staged_shapefiles"
  gdb_path = parsed.gdb_path or (cache_root / "hydroatlas" / "BasinATLAS_v10.gdb")
  era5_cache_dir = parsed.era5_cache_dir or (cache_root / "era5_climate")

  try:
    if not parsed.parent_dirs and not parsed.input_dirs and not parsed.input_files:
      logger.error("Must provide at least one of --parent-dir, --input-dirs, or --input-files.")
      sys.exit(1)

    dataset_map = discover_datasets(
        parent_dirs=parsed.parent_dirs,
        input_dirs=parsed.input_dirs,
        input_files=parsed.input_files,
        staging_cache_dir=staging_cache,
    )

    if not dataset_map:
      logger.error("No valid dataset vector files found matching provided paths.")
      sys.exit(1)

    run_batch_extraction(
        dataset_map=dataset_map,
        output_dir=parsed.output_dir,
        workers=parsed.workers,
        era5_source=parsed.era5_source,
        gridded_era5_uri=parsed.gridded_era5_uri,
        gdb_path=str(gdb_path),
        era5_cache_dir=str(era5_cache_dir),
        min_overlap_threshold=parsed.min_overlap_threshold,
        combine=parsed.combine,
        resume=parsed.resume,
    )
  finally:
    if parsed.clean_cache and cache_root.exists():
      import shutil
      logger.info("Cleaning up cache root directory %s...", cache_root)
      shutil.rmtree(cache_root, ignore_errors=True)
    elif parsed.clean_staging and staging_cache.exists():
      import shutil
      logger.info("Cleaning up staged shapefiles directory %s...", staging_cache)
      shutil.rmtree(staging_cache, ignore_errors=True)



if __name__ == "__main__":
  main()
