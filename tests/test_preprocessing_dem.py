#! /usr/bin/env python

"""Tests for DEM preprocessing utilities."""

from pathlib import Path

import numpy
import rasterio
from rasterio.transform import from_origin

from gclandspill._preprocessing import preprocess_dem_file


def test_preprocess_dem_file_with_buffer(tmp_path: Path):
    """It should clean nodata cells and crop around source with buffer."""

    src_path = tmp_path / "input.tif"
    dst_path = tmp_path / "processed.tif"

    dem = numpy.array(
        [
            [10.0, 11.0, 12.0, 13.0],
            [14.0, -9999.0, 16.0, 17.0],
            [18.0, 19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0, 25.0],
        ],
        dtype=numpy.float64,
    )

    profile = {
        "driver": "GTiff",
        "height": dem.shape[0],
        "width": dem.shape[1],
        "count": 1,
        "dtype": rasterio.float64,
        "crs": "EPSG:32612",
        "transform": from_origin(100.0, 200.0, 10.0, 10.0),
        "nodata": -9999.0,
    }

    with rasterio.open(src_path, "w", **profile) as dst:
        dst.write(dem, 1)

    summary = preprocess_dem_file(
        dem_path=src_path,
        output_path=dst_path,
        source_row=1,
        source_col=1,
        roi_buffer_m=15.0,
    )

    assert summary["dem_stats"]["invalid_total"] == 1
    assert summary["dem_stats"]["invalid_remaining"] == 0

    with rasterio.open(dst_path) as processed:
        out = processed.read(1)
        assert out.shape == (4, 4)
        assert out[1, 1] > 0.0


def test_preprocess_dem_file_decimate(tmp_path: Path):
    """It should support DEM decimation and still write output."""

    src_path = tmp_path / "input_decimate.tif"
    dst_path = tmp_path / "processed_decimate.tif"

    dem = numpy.arange(64, dtype=numpy.float64).reshape(8, 8)

    profile = {
        "driver": "GTiff",
        "height": dem.shape[0],
        "width": dem.shape[1],
        "count": 1,
        "dtype": rasterio.float64,
        "crs": "EPSG:32612",
        "transform": from_origin(0.0, 80.0, 10.0, 10.0),
        "nodata": None,
    }

    with rasterio.open(src_path, "w", **profile) as dst:
        dst.write(dem, 1)

    preprocess_dem_file(
        dem_path=src_path,
        output_path=dst_path,
        decimate=2,
        source_row=1,
        source_col=1,
        roi_buffer_m=0.0,
    )

    with rasterio.open(dst_path) as processed:
        out = processed.read(1)
        assert out.shape == (4, 4)
