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

"""Climate metrics and ERA5 indices computation following Caravan methodology.

Implements FAO-56 Penman-Monteith potential evapotranspiration, Knoben et al. (2018)
climate indices, Addor et al. (2017) extreme precipitation indices, and
area-weighted Level 12 continental ERA5 precomputed climate indices caching.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from static_extractor.config import (
    CNS_ERA5_CLIMATE_BASE,
    CONTINENT_MAP,
    GCS_ERA5_CLIMATE_URI,
    get_default_era5_cache_dir,
)

logger = logging.getLogger(__name__)


def calculate_fao_pm_pet(
    surface_pressure_kpa: pd.Series,
    temperature_2m_c: pd.Series,
    dewpoint_temperature_2m_c: pd.Series,
    u_component_of_wind_10m: pd.Series,
    v_component_of_wind_10m: pd.Series,
    surface_net_solar_radiation_mean: pd.Series,
    surface_net_thermal_radiation_mean: pd.Series,
) -> pd.Series:
  """Calculates daily potential evapotranspiration (PET) following FAO-56 Penman-Monteith guidelines.

  Args:
    surface_pressure_kpa: Daily mean atmospheric surface pressure in kPa.
    temperature_2m_c: Daily mean 2-meter air temperature in °C.
    dewpoint_temperature_2m_c: Daily mean 2-meter dewpoint temperature in °C.
    u_component_of_wind_10m: Daily mean 10-meter eastward wind component in m/s.
    v_component_of_wind_10m: Daily mean 10-meter northward wind component in m/s.
    surface_net_solar_radiation_mean: Mean surface net solar radiation (J/m²/hr or W/m²).
    surface_net_thermal_radiation_mean: Mean surface net thermal radiation (J/m²/hr or W/m²).

  Returns:
    Daily potential evapotranspiration series in mm/day.
  """
  # 1. 2m Wind speed adjustment from 10m components (FAO-56 Eq. 47)
  temp_windspeed10m_m_s = np.sqrt(
      u_component_of_wind_10m**2 + v_component_of_wind_10m**2
  )
  windspeed2m_m_s = temp_windspeed10m_m_s * 4.87 / (np.log(67.8 * 10.0 - 5.42))

  # 2. Net radiation (MJ/m2/day)
  # When inputs are hourly rates in J/m2/hr: (ssr + str) * 24 / 1e6
  net_radiation_mj_m2 = (
      (surface_net_solar_radiation_mean + surface_net_thermal_radiation_mean)
      * 24.0
      / 1e6
  )

  # 3. Thermodynamic Constants
  lmbda = 2.45  # Latent heat of vaporization [MJ kg-1]
  cp = 1.013e-3  # Specific heat at constant pressure [MJ kg-1 °C-1]
  eps = 0.622  # Ratio molecular weight of water vapour/dry air

  soil_heat_flux = np.zeros_like(surface_pressure_kpa)
  p_kpa = surface_pressure_kpa
  psychometric_kpa_c = cp * p_kpa / (eps * lmbda)
  svp_kpa = 0.6108 * np.exp(
      (17.27 * temperature_2m_c) / (temperature_2m_c + 237.3)
  )
  delta_kpa_c = 4098.0 * svp_kpa / (temperature_2m_c + 237.3) ** 2
  avp_kpa = 0.6108 * np.exp(
      (17.27 * dewpoint_temperature_2m_c)
      / (dewpoint_temperature_2m_c + 237.3)
  )
  svpdeficit_kpa = np.maximum(0.0, svp_kpa - avp_kpa)

  # 4. FAO-56 Equation
  numerator = (
      0.408 * delta_kpa_c * (net_radiation_mj_m2 - soil_heat_flux)
      + psychometric_kpa_c
      * (900.0 / (temperature_2m_c + 273.15))
      * windspeed2m_m_s
      * svpdeficit_kpa
  )
  denominator = delta_kpa_c + psychometric_kpa_c * (
      1.0 + 0.34 * windspeed2m_m_s
  )

  et0_mm_day = numerator / denominator
  return pd.Series(
      np.clip(et0_mm_day, 0.0, None), index=temperature_2m_c.index
  )


def calculate_knoben_moisture_and_seasonality(
    precipitation: pd.Series,
    pet: pd.Series,
) -> Tuple[float, float]:
  """Computes Knoben et al. (2018) annual moisture index and seasonality index.

  Reference:
    Knoben, W. J., et al. (2018). Climate indices for catchment comparison:
    A global evaluation of the moisture and seasonality indices.
    Hydrological Processes, 32(23), 3502-3518.

  Args:
    precipitation: Daily precipitation series (mm/day).
    pet: Daily potential evapotranspiration series (mm/day).

  Returns:
    Tuple of (annual_moisture_index, seasonality_index).
  """
  mean_monthly_precip = precipitation.groupby(precipitation.index.month).mean()
  mean_monthly_pet = pet.groupby(pet.index.month).mean()

  # Monthly moisture index: piecewise formulation in Knoben et al. (2018) Eq. 1-2
  p_gt_et = (
      1.0
      - mean_monthly_pet.loc[mean_monthly_precip > mean_monthly_pet]
      / mean_monthly_precip.loc[mean_monthly_precip > mean_monthly_pet]
  )
  srs = pd.Series(
      np.zeros(len(mean_monthly_pet), dtype=np.float32), index=mean_monthly_pet.index, name="dummy"
  )
  p_eq_et = srs.loc[mean_monthly_precip == mean_monthly_pet]
  p_lt_et = (
      mean_monthly_precip.loc[mean_monthly_precip < mean_monthly_pet]
      / mean_monthly_pet.loc[mean_monthly_precip < mean_monthly_pet]
      - 1.0
  )
  monthly_moisture_index = pd.concat([p_gt_et, p_eq_et, p_lt_et])

  annual_moisture_index = float(monthly_moisture_index.mean())
  seasonality = float(
      monthly_moisture_index.max() - monthly_moisture_index.min()
  )
  return annual_moisture_index, seasonality


def _split_list(indices: np.ndarray) -> List[List[int]]:
  """Splits an array of consecutive integer indices into consecutive groups."""
  if len(indices) == 0:
    return []
  groups = []
  current_group = [int(indices[0])]
  for i in range(1, len(indices)):
    if indices[i] == indices[i - 1] + 1:
      current_group.append(int(indices[i]))
    else:
      groups.append(current_group)
      current_group = [int(indices[i])]
  if current_group:
    groups.append(current_group)
  return groups


def compute_caravan_climate_metrics(
    precipitation: pd.Series,
    temperature: pd.Series,
    pet_era5: pd.Series,
    pet_fao: Optional[pd.Series] = None,
) -> Dict[str, float]:
  """Computes all canonical Caravan climate indices.

  According to Addor et al. (2017) and Knoben et al. (2018).

  Args:
    precipitation: Daily basin-mean precipitation series (mm/day).
    temperature: Daily basin-mean 2m air temperature series (°C).
    pet_era5: Daily basin-mean ERA5-Land native potential evaporation (mm/day).
    pet_fao: Optional daily basin-mean FAO-56 Penman-Monteith PET (mm/day).

  Returns:
    Dictionary mapping all Caravan climate index names to their computed values.
  """
  p = np.asarray(precipitation.values, dtype=float)
  p_mean = float(np.nanmean(p)) if len(p) > 0 else np.nan

  # 1. ERA5-Land Native PEV Metrics
  e_era5 = np.asarray(pet_era5.values, dtype=float)
  pet_mean_era5 = float(np.nanmean(e_era5)) if len(e_era5) > 0 else np.nan
  aridity_era5 = (
      float(pet_mean_era5 / p_mean)
      if (p_mean and not np.isnan(p_mean) and p_mean > 0)
      else np.nan
  )
  mi_era5, seas_era5 = calculate_knoben_moisture_and_seasonality(
      precipitation, pet_era5
  )

  # 2. FAO-56 Penman-Monteith Metrics
  if pet_fao is not None:
    e_fao = np.asarray(pet_fao.values, dtype=float)
    pet_mean_fao = float(np.nanmean(e_fao)) if len(e_fao) > 0 else np.nan
    aridity_fao = (
        float(pet_mean_fao / p_mean)
        if (p_mean and not np.isnan(p_mean) and p_mean > 0)
        else np.nan
    )
    mi_fao, seas_fao = calculate_knoben_moisture_and_seasonality(
        precipitation, pet_fao
    )
  else:
    pet_mean_fao = pet_mean_era5
    aridity_fao = aridity_era5
    mi_fao = mi_era5
    seas_fao = seas_era5

  # 3. Fraction of Snow (Knoben et al. 2018, Eq. 4)
  mean_monthly_precip = precipitation.groupby(precipitation.index.month).mean()
  mean_monthly_temp = temperature.groupby(temperature.index.month).mean()
  tot_monthly_p = mean_monthly_precip.sum()
  if tot_monthly_p > 0:
    frac_snow = float(
        mean_monthly_precip.loc[mean_monthly_temp < 0.0].sum() / tot_monthly_p
    )
  else:
    frac_snow = 0.0

  # 4. Extreme Precipitation Frequency & Duration (Addor et al. 2017)
  if p_mean and not np.isnan(p_mean) and p_mean > 0:
    high_p_thresh = 5.0 * p_mean
    high_prec_freq = float(
        len(precipitation.loc[precipitation >= high_p_thresh])
        / len(precipitation)
    )

    high_idx = np.where(p >= high_p_thresh)[0]
    if high_idx.size > 0:
      groups = _split_list(high_idx)
      high_prec_dur = (
          float(np.mean([len(g) for g in groups])) if groups else 0.0
      )
    else:
      high_prec_dur = 0.0

    low_p_thresh = 1.0
    low_prec_freq = float(
        len(precipitation.loc[precipitation < low_p_thresh]) / len(precipitation)
    )

    low_idx = np.where(p < low_p_thresh)[0]
    if low_idx.size > 0:
      groups = _split_list(low_idx)
      low_prec_dur = float(np.mean([len(g) for g in groups])) if groups else 0.0
    else:
      low_prec_dur = 0.0
  else:
    high_prec_freq = 0.0
    high_prec_dur = 0.0
    low_prec_freq = 0.0
    low_prec_dur = 0.0

  return {
      "p_mean": round(p_mean, 4),
      # FAO-56 Penman-Monteith (Standard Caravan)
      "pet_mean": round(pet_mean_fao, 4),
      "pet_mean_FAO_PM": round(pet_mean_fao, 4),
      "aridity": round(aridity_fao, 4),
      "aridity_FAO_PM": round(aridity_fao, 4),
      "moisture_index": round(mi_fao, 4),
      "moisture_index_FAO_PM": round(mi_fao, 4),
      "seasonality": round(seas_fao, 4),
      "seasonality_FAO_PM": round(seas_fao, 4),
      # ERA5-Land Native PEV
      "pet_mean_ERA5_LAND": round(pet_mean_era5, 4),
      "aridity_ERA5_LAND": round(aridity_era5, 4),
      "moisture_index_ERA5_LAND": round(mi_era5, 4),
      "seasonality_ERA5_LAND": round(seas_era5, 4),
      # Climate Extremes & Snow
      "frac_snow": round(frac_snow, 4),
      "high_prec_freq": round(high_prec_freq, 4),
      "high_prec_dur": round(high_prec_dur, 4),
      "low_prec_freq": round(low_prec_freq, 4),
      "low_prec_dur": round(low_prec_dur, 4),
  }


class ERA5ClimateLoader:
  """Loads and aggregates Level 12 precomputed ERA5 climate indices.

  Checks local cache, then downloads from GCS (gs://open-multimet/data/hydroatlas/era5_climate),
  with internal fallback to CNS.
  """

  def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
    self.cache_dir = Path(cache_dir) if cache_dir else get_default_era5_cache_dir()
    self.cache_dir.mkdir(parents=True, exist_ok=True)
    self.loaded_continents: Set[str] = set()
    self.records: Dict[int, Dict[str, Any]] = {}

  def _download_from_gcs(self, continent_code: str, target_file: Path) -> bool:
    """Attempts to download continent file from GCS."""
    gcs_src = f"{GCS_ERA5_CLIMATE_URI}/{continent_code}_climate_indices.txt"
    try:
      import gcsfs

      fs = gcsfs.GCSFileSystem(token="anon")
      remote_path = gcs_src.replace("gs://", "")
      if fs.exists(remote_path):
        fs.get(remote_path, str(target_file))
        return True
    except Exception:
      pass

    if shutil.which("gcloud"):
      try:
        cmd = ["gcloud", "storage", "cp", gcs_src, str(target_file)]
        res = subprocess.run(cmd, capture_output=True, timeout=30)
        if res.returncode == 0 and target_file.exists():
          return True
      except Exception:
        pass
    return False

  def _download_from_cns(self, continent_code: str, target_file: Path) -> bool:
    """Attempts to download continent file from CNS if on Google infrastructure."""
    if not shutil.which("fileutil"):
      return False
    cns_path = f"{CNS_ERA5_CLIMATE_BASE}/{continent_code}_climate_indices.txt"
    try:
      cmd = ["fileutil", "cp", cns_path, str(target_file)]
      res = subprocess.run(cmd, capture_output=True, timeout=30)
      return res.returncode == 0 and target_file.exists()
    except Exception:
      return False

  def _ensure_file_on_disk(self, continent_code: str) -> bool:
    txt_path = self.cache_dir / f"{continent_code}_climate_indices.txt"
    if txt_path.exists() and txt_path.stat().st_size > 0:
      return True

    if self._download_from_gcs(continent_code, txt_path):
      return True

    if self._download_from_cns(continent_code, txt_path):
      return True

    logger.warning(
        "Could not locate %s_climate_indices.txt locally, on GCS, or on CNS.",
        continent_code,
    )
    return False

  def ensure_continent(
      self, continent_code: str, target_ids: Optional[Set[int]] = None
  ) -> None:
    """Ensures continental climate index file is available and cached in memory."""
    if target_ids is None and continent_code in self.loaded_continents:
      return

    if not self._ensure_file_on_disk(continent_code):
      return

    txt_path = self.cache_dir / f"{continent_code}_climate_indices.txt"
    if txt_path.exists():
      target_strs = {str(tid) for tid in target_ids} if target_ids else None
      count = 0
      with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
          line = line.strip()
          if not line:
            continue
          if target_strs is not None and not any(ts in line for ts in target_strs):
            continue
          try:
            item = json.loads(line)
            gid = item.get("gauge_id", "")
            if gid.startswith("hybas_"):
              hid = int(gid.split("_")[1])
              if target_ids is None or hid in target_ids:
                self.records[hid] = item
                count += 1
          except Exception:
            pass
      if target_ids is None:
        self.loaded_continents.add(continent_code)
      logger.info(
          "Loaded %d Level 12 climate records for continent '%s'", count, continent_code
      )

  def get_indices_for_subbasins(
      self, hybas_ids: List[int], weights: List[float]
  ) -> Dict[str, float]:
    """Calculates area-weighted average ERA5 climate indices for a set of Level 12 sub-basins."""
    needed_continents = set()
    for hid in hybas_ids:
      try:
        first_digit = int(str(hid)[0])
        if first_digit in CONTINENT_MAP:
          needed_continents.add(CONTINENT_MAP[first_digit])
      except Exception:
        pass

    target_id_set = set(hybas_ids)
    for c in needed_continents:
      self.ensure_continent(c, target_ids=target_id_set)

    keys = [
        "p_mean",
        "pet_mean",
        "aridity",
        "frac_snow",
        "moisture_index",
        "seasonality",
        "high_prec_freq",
        "high_prec_dur",
        "low_prec_freq",
        "low_prec_dur",
    ]

    valid_weights = []
    valid_records = []
    for hid, w in zip(hybas_ids, weights):
      if hid in self.records:
        valid_records.append(self.records[hid])
        valid_weights.append(float(w))

    if not valid_records or sum(valid_weights) == 0:
      return {
          "p_mean": np.nan,
          "pet_mean": np.nan,
          "pet_mean_FAO_PM": np.nan,
          "pet_mean_ERA5_LAND": np.nan,
          "aridity": np.nan,
          "aridity_FAO_PM": np.nan,
          "aridity_ERA5_LAND": np.nan,
          "frac_snow": np.nan,
          "moisture_index": np.nan,
          "moisture_index_FAO_PM": np.nan,
          "moisture_index_ERA5_LAND": np.nan,
          "seasonality": np.nan,
          "seasonality_FAO_PM": np.nan,
          "seasonality_ERA5_LAND": np.nan,
          "high_prec_freq": np.nan,
          "high_prec_dur": np.nan,
          "low_prec_freq": np.nan,
          "low_prec_dur": np.nan,
      }

    tot_w = sum(valid_weights)
    norm_w = np.array(valid_weights) / tot_w

    raw_res = {}
    for k in keys:
      vals = np.array([r.get(k, np.nan) for r in valid_records])
      raw_res[k] = float(np.sum(vals * norm_w))

    return {
        "p_mean": raw_res["p_mean"],
        "pet_mean": raw_res["pet_mean"],
        "pet_mean_FAO_PM": raw_res["pet_mean"],
        "pet_mean_ERA5_LAND": raw_res["pet_mean"],
        "aridity": raw_res["aridity"],
        "aridity_FAO_PM": raw_res["aridity"],
        "aridity_ERA5_LAND": raw_res["aridity"],
        "frac_snow": raw_res["frac_snow"],
        "moisture_index": raw_res["moisture_index"],
        "moisture_index_FAO_PM": raw_res["moisture_index"],
        "moisture_index_ERA5_LAND": raw_res["moisture_index"],
        "seasonality": raw_res["seasonality"],
        "seasonality_FAO_PM": raw_res["seasonality"],
        "seasonality_ERA5_LAND": raw_res["seasonality"],
        "high_prec_freq": raw_res["high_prec_freq"],
        "high_prec_dur": raw_res["high_prec_dur"],
        "low_prec_freq": raw_res["low_prec_freq"],
        "low_prec_dur": raw_res["low_prec_dur"],
    }
