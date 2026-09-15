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

"""Command-line interface for Caravan static attributes extraction."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from static_extractor.extractor import StaticAttributesExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("static_extractor")


def parse_args(args=None):
  parser = argparse.ArgumentParser(
      description="Extract Caravan HydroATLAS and ERA5 static attributes for watershed polygons.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--input",
      "-i",
      required=True,
      type=str,
      help="Path to input vector watershed polygon file (GeoJSON, Shapefile, GPKG).",
  )
  parser.add_argument(
      "--output",
      "-o",
      required=True,
      type=str,
      help="Path to output CSV file for extracted attributes.",
  )
  parser.add_argument(
      "--gdb-path",
      "-g",
      default=None,
      type=str,
      help="Path to BasinATLAS_v10.gdb directory or BasinATLAS_v10_lev12.shp. Defaults to cache or auto-discovery.",
  )
  parser.add_argument(
      "--era5-cache-dir",
      default=None,
      type=str,
      help="Directory where continental ERA5 climate index files are cached.",
  )
  parser.add_argument(
      "--id-column",
      default=None,
      type=str,
      help="Column name in vector file containing gauge or catchment ID.",
  )
  parser.add_argument(
      "--min-overlap-threshold",
      default=0.0,
      type=float,
      help="Minimum sub-basin intersection area threshold in km².",
  )
  parser.add_argument(
      "--auto-download",
      action="store_true",
      default=True,
      help="Automatically download data from canonical Google Cloud Storage if not staged locally.",
  )
  parser.add_argument(
      "--era5-source",
      choices=["hybas", "gridded"],
      default="hybas",
      help="Source for ERA5 climate metrics: 'hybas' (fast area-weighted aggregation of precalculated Level 12 sub-basin statistics) or 'gridded' (recalculated on the fly from archived gridded ERA5 daily surface data on GCS).",
  )
  parser.add_argument(
      "--gridded-era5-uri",
      default=None,
      type=str,
      help="GCS URI or path to gridded daily ERA5 Zarr store. Defaults to gs://open-multimet/data/era5_land/daily_surface.zarr.",
  )
  return parser.parse_args(args)


def main(args=None):
  parsed = parse_args(args)
  input_path = Path(parsed.input)
  if not input_path.exists():
    logger.error("Input file '%s' does not exist.", input_path)
    sys.exit(1)

  logger.info("Initializing Caravan Static Attributes Extractor (ERA5 source: %s)...", parsed.era5_source)
  extractor = StaticAttributesExtractor(
      gdb_path=parsed.gdb_path,
      era5_cache_dir=parsed.era5_cache_dir,
      auto_download=parsed.auto_download,
      era5_source=parsed.era5_source,
      gridded_era5_uri=parsed.gridded_era5_uri,
  )

  logger.info("Extracting static attributes from '%s'...", input_path)
  df = extractor.extract_attributes_from_file(
      input_path=input_path,
      output_csv_path=parsed.output,
      id_column=parsed.id_column,
      min_overlap_threshold=parsed.min_overlap_threshold,
  )

  logger.info(
      "Successfully extracted %d attributes for %d catchments. Saved to %s",
      df.shape[1],
      df.shape[0],
      parsed.output,
  )


if __name__ == "__main__":
  main()
