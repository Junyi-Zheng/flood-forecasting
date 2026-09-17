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

"""Canaries for the upstream feeds the gridded archive builders depend on.

A canary answers a different question from the rest of this suite:

* ``unit`` / ``integration`` tests ask *"is our code correct?"*. They are
  hermetic, deterministic, and gate every pull request.
* A **canary** asks *"is the third-party feed still working?"*. It talks to
  the live network, it is not deterministic, and it must never gate a pull
  request -- an ECMWF outage is not a reason to block a merge.

These are therefore deselected by default. Run them explicitly::

    pytest multimet/test -m canary

The builders are unusually dependent on this kind of check because
:func:`multimet.build_hres_archive._extract_single_date` deliberately degrades
to an all-NaN slice when a source fails, so a broken upstream produces a
successful-looking build full of holes rather than a crash. Every assertion
here checks the *content* of what came back, never just that a call returned.
"""

from __future__ import annotations

import datetime
import urllib.request

import numpy as np
import pandas as pd
import pytest

from multimet import build_cpc_archive as cpc_module
from multimet import build_hres_archive as hres_module

pytestmark = pytest.mark.canary

# A slice is considered live if at least this fraction of cells are finite.
# Real fields are never entirely finite (HRES radiation has gaps, CPC is
# gauge-based and undefined over much of the ocean), so this is a floor rather
# than a target.
MIN_FINITE_FRACTION = 0.05

# Physically implausible values indicate a unit change or a decode failure
# upstream, which is exactly the sort of silent breakage a canary exists for.
PLAUSIBLE_RANGES = {
    # Kelvin. Vostok's record low is 184 K; Death Valley's high is 331 K.
    "temperature_2m": (180.0, 340.0),
    # Pascals. Sea level is ~101325; the top of Everest is ~33700.
    "surface_pressure": (30000.0, 110000.0),
    # Metres of water equivalent per day. The 24h world record is ~1.8 m.
    "total_precipitation": (0.0, 3.0),
}

requires_gcsfs = pytest.mark.skipif(
    hres_module.gcsfs is None,
    reason="gcsfs is not installed; cloud sources cannot be reached",
)


def _finite_fraction(values: np.ndarray) -> float:
  """Returns the fraction of ``values`` that are finite."""
  return float(np.isfinite(values).mean())


def _assert_field_is_live(name: str, values: np.ndarray) -> None:
  """Asserts an extracted field contains real, physically plausible data."""
  finite = _finite_fraction(values)
  assert finite >= MIN_FINITE_FRACTION, (
      f"{name}: only {finite:.1%} of cells are finite. The upstream source "
      f"is probably returning nothing and the builder would silently write "
      f"an all-NaN slice."
  )

  if name in PLAUSIBLE_RANGES:
    low, high = PLAUSIBLE_RANGES[name]
    finite_values = values[np.isfinite(values)]
    observed_low = float(finite_values.min())
    observed_high = float(finite_values.max())
    assert low <= observed_low and observed_high <= high, (
        f"{name}: observed range [{observed_low:.4g}, {observed_high:.4g}] "
        f"falls outside the plausible range [{low:g}, {high:g}]. The upstream "
        f"units or encoding may have changed."
    )


class TestNoaaPslCpc:
  """Canaries for the NOAA PSL CPC precipitation feed."""

  def test_yearly_file_is_published(self) -> None:
    """The per-year NetCDF is reachable and large enough to be real."""
    year = datetime.date.today().year - 1
    url = cpc_module.NOAA_PSL_URL_TEMPLATE.format(year=year)
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "OpenMultiMet/1.1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
      assert response.status == 200
      size = int(response.headers.get("Content-Length", 0))

    # ensure_psl_cpc_netcdf treats a cached file of 1 MiB or less as a failed
    # download and re-fetches it, so a genuine file must comfortably exceed it.
    assert size > 1024 * 1024, (
        f"{url} is only {size} bytes. ensure_psl_cpc_netcdf would treat a "
        f"file this small as a truncated download and retry forever."
    )

  def test_yearly_netcdf_still_matches_expected_layout(self, tmp_path) -> None:
    """A real PSL file still parses onto the MultiMet grid with live values."""
    year = datetime.date.today().year - 1
    path = cpc_module.ensure_psl_cpc_netcdf(year, cache_dir=str(tmp_path))

    dataset = cpc_module.process_cpc_netcdf_to_dataset(
        path,
        target_start_date=pd.Timestamp(f"{year}-06-01"),
        target_end_date=pd.Timestamp(f"{year}-06-03"),
    )
    assert dataset is not None, "no dates survived the filter"

    assert cpc_module.CPC_VARIABLE in dataset
    assert dataset.sizes["latitude"] == len(cpc_module.CPC_LATS)
    assert dataset.sizes["longitude"] == len(cpc_module.CPC_LONS)

    latitudes = dataset["latitude"].values
    longitudes = dataset["longitude"].values
    assert np.all(np.diff(latitudes) > 0), "latitude is no longer ascending"
    assert longitudes.min() >= -180.0 and longitudes.max() < 180.0, (
        "longitude is no longer on the signed [-180, 180) grid"
    )

    _assert_field_is_live(
        "cpc_precipitation", dataset[cpc_module.CPC_VARIABLE].values
    )


@requires_gcsfs
class TestWeatherBench2:
  """Canaries for the WeatherBench 2 HRES archive (dates <= 2023-01-10)."""

  @pytest.fixture(scope="class")
  def source(self) -> hres_module.WeatherBench2Source:
    """Opens the remote archive once for the whole class."""
    return hres_module.WeatherBench2Source()

  def test_archive_is_openable(
      self, source: hres_module.WeatherBench2Source
  ) -> None:
    """The remote Zarr still opens anonymously and exposes its coordinates."""
    assert len(source.latitudes) == len(hres_module.HRES_LATS)
    assert len(source.longitudes) == len(hres_module.HRES_LONS)

  def test_date_extraction_returns_live_data(
      self, source: hres_module.WeatherBench2Source
  ) -> None:
    """A date inside the WeatherBench 2 window yields real values."""
    extracted = source.extract_date(pd.Timestamp("2022-06-01"))

    assert extracted is not None, (
        "WeatherBench 2 returned nothing; the builder would write NaN here"
    )

    for name in hres_module.HRES_VARIABLES:
      assert name in extracted, f"{name} is missing from the extraction"
      assert extracted[name].shape == (
          hres_module.NUM_LEAD_DAYS,
          len(hres_module.HRES_LATS),
          len(hres_module.HRES_LONS),
      )

    # WeatherBench 2 does not archive the two radiation fields, so the builder
    # legitimately emits NaN for them. Only check the three it does provide.
    for name in ("temperature_2m", "surface_pressure", "total_precipitation"):
      _assert_field_is_live(name, extracted[name])


@requires_gcsfs
class TestEcmwfOpenData:
  """Canaries for the ECMWF open data feed (dates >= 2023-07-13)."""

  @pytest.fixture(scope="class")
  def source(self) -> hres_module.ECMWFOpenDataSource:
    """Builds one anonymous GCS client for the whole class."""
    return hres_module.ECMWFOpenDataSource()

  def _index_grid(
      self, source: hres_module.ECMWFOpenDataSource, date: pd.Timestamp
  ) -> str | None:
    """Returns the grid label published for ``date``, or None if absent."""
    stamp = date.strftime("%Y%m%d")
    candidates = {
        "0p25": f"ecmwf-open-data/{stamp}/00z/ifs/0p25/oper/{stamp}000000",
        "0p4-beta": f"ecmwf-open-data/{stamp}/00z/0p4-beta/oper/{stamp}000000",
    }
    for label, prefix in candidates.items():
      if source.fs.exists(f"{prefix}-24h-oper-fc.index"):
        return label
    return None

  def test_recent_forecast_is_published(
      self, source: hres_module.ECMWFOpenDataSource
  ) -> None:
    """At least one of the last few days is present under the expected path.

    This is the cheapest and most valuable canary in the file: it needs no
    GRIB decoder, and it fires if ECMWF changes the bucket layout or stops
    publishing, which is what would silently stall the daily build.
    """
    today = pd.Timestamp(datetime.date.today())
    found = {
        (today - pd.Timedelta(days=back)).strftime("%Y-%m-%d"): grid
        for back in range(5)
        if (grid := self._index_grid(source, today - pd.Timedelta(days=back)))
    }
    assert found, (
        "no ECMWF open data forecast found in the last 5 days under "
        "gs://ecmwf-open-data/{date}/00z/... -- the layout or the feed has "
        "changed"
    )

  def test_first_archived_date_is_still_available(
      self, source: hres_module.ECMWFOpenDataSource
  ) -> None:
    """OPEN_DATA_START_DATE is still the first date the feed offers.

    The builder hands every date before this constant to a NaN fill, so if
    ECMWF ever extends its retention backwards the constant is leaving real
    data on the table.
    """
    start = hres_module.OPEN_DATA_START_DATE
    assert self._index_grid(source, start) is not None, (
        f"{start:%Y-%m-%d} is no longer published; OPEN_DATA_START_DATE needs "
        f"to move forward or history has been withdrawn"
    )
    assert self._index_grid(source, start - pd.Timedelta(days=1)) is None, (
        f"{start - pd.Timedelta(days=1):%Y-%m-%d} is now published; "
        f"OPEN_DATA_START_DATE could move earlier and recover real data that "
        f"is currently written as NaN"
    )


@requires_gcsfs
@pytest.mark.slow
class TestEcmwfOpenDataDecoding:
  """GRIB decoding canaries. These download and decode real messages."""

  @pytest.fixture(autouse=True)
  def _require_eccodes(self) -> None:
    pytest.importorskip(
        "eccodes", reason="eccodes is required to decode ECMWF GRIB2"
    )

  @pytest.mark.parametrize(
      ("label", "date"),
      [
          # Both grids are live: 0p4-beta covers the early archive and 0p25
          # took over during 2024. Only the 0p4-beta path runs the scipy
          # upsample, so both need covering.
          ("0p4-beta", "2023-08-01"),
          ("0p25", "2024-06-01"),
      ],
  )
  def test_grib_decodes_to_live_fields(self, label: str, date: str) -> None:
    source = hres_module.ECMWFOpenDataSource()
    extracted = source.extract_date(
        pd.Timestamp(date), hres_module.HRES_LATS, hres_module.HRES_LONS
    )

    assert extracted is not None, (
        f"{label} extraction for {date} returned nothing; the builder would "
        f"write a NaN slice"
    )

    for name in hres_module.HRES_VARIABLES:
      assert extracted[name].shape == (
          hres_module.NUM_LEAD_DAYS,
          len(hres_module.HRES_LATS),
          len(hres_module.HRES_LONS),
      ), f"{name} was not regridded onto the 721 x 1440 target grid"

    for name in ("temperature_2m", "surface_pressure", "total_precipitation"):
      _assert_field_is_live(name, extracted[name])

    # Precipitation is de-accumulated per lead day, so it must never be
    # negative once clipped.
    precipitation = extracted["total_precipitation"]
    finite = precipitation[np.isfinite(precipitation)]
    assert finite.min() >= 0.0, "de-accumulated precipitation went negative"
