#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020-2021 Pi-Yueh Chuang and Lorena A. Barba
#
# Distributed under terms of the BSD 3-Clause license.

""" Functions related to creating data for a simulation.
"""

import os
import time
import pathlib
import shutil
import glob
import json
from typing import Tuple

import urllib3
import requests
import numpy
import rasterio
import rasterio.features
import rasterio.transform
from rasterio.enums import Resampling
from gclandspill import _misc
from gclandspill import clawutil


def create_data(
    case_dir: os.PathLike, log_level: int = None,
    out_dir: os.PathLike = "_output", overwrite: bool = False
):
    """Create *.data files (and topography & hydrological files) in case folder.

    Arguments
    ---------
    case_dir : PathLike
        The (absolute) path to the target case folder, where setrun.py can be found.
    log_level : int or None
        To overwrite the log verbosity in the original config file.
    out_dir : str or PathLike
        Folder to put output files for this particular run. If not an absolute path, assume it's
        relative to `case_dir`
    overwrite : bool
        Whether or not to force overwrite `out_dir` if it exists. If False (default) and if
        `out_dir` exists, copy to a new folder with current time appended.
    """

    # let pathlib handle path-related stuff
    case_dir = pathlib.Path(case_dir).expanduser().resolve()

    if not case_dir.is_dir():
        raise FileNotFoundError("{} does not exist or is not a folder".format(case_dir))

    # import setrun.py
    setrun = _misc.import_setrun(case_dir)
    rundata = setrun.setrun()  # get ClawRunData object

    # if we need to overwrite the log verbosity
    if log_level is not None:
        rundata.clawdata.verbosity = log_level
        rundata.clawdata.verbosity_regrid = log_level
        rundata.amrdata.verbosity_regrid = log_level

    # geoclaw' `TopographyData` assumes topography filenames are relative to the output folder,
    # while we assuem the filenames are always relative to the case folder. This forces us to write
    # *.data to the case folder first and move to our target output folder
    rundata.write(str(case_dir))  # write *.data to the case folder

    # get a list of *.data files just outputed
    data_files = glob.glob(str(case_dir.joinpath("*.data")))  # glob doesn't like PathLike...

    # prepare the true output folder so we can move *.data to there
    out_dir = pathlib.Path(out_dir).expanduser()

    if not out_dir.is_absolute():
        out_dir = case_dir.joinpath(out_dir)

    if out_dir.is_dir():
        if not overwrite:  # make a backup if out_dir exists and overwriting not allowed
            shutil.move(out_dir, str(out_dir)+"."+time.strftime("%Y%m%dT%H%M%S%Z"))
        else:  # just delete the old out_dir
            shutil.rmtree(out_dir)

    # create the folder regardless
    os.makedirs(out_dir)

    # move *.data to the true output folder
    for data_file in data_files:
        shutil.move(data_file, out_dir)

    # check if topo file exists. Download it if not exist
    check_download_topo(case_dir, rundata)

    # check if hudro file exists. Download it if not exist
    check_download_hydro(case_dir, rundata)


def prepare_dem(dem_path: os.PathLike, decimate: int = 1):
    """Read a DEM from raster, optionally decimate it, and return DEM/profile/cell sizes."""

    dem_path = pathlib.Path(dem_path).expanduser().resolve()

    with rasterio.open(dem_path) as src:
        if decimate <= 1:
            dem = src.read(1)
            profile = src.profile.copy()
            transform = src.transform
        else:
            out_rows = max(1, src.height // decimate)
            out_cols = max(1, src.width // decimate)
            dem = src.read(1, out_shape=(out_rows, out_cols), resampling=Resampling.average)
            transform = src.transform * src.transform.scale(src.width / out_cols, src.height / out_rows)
            profile = src.profile.copy()
            profile.update(height=out_rows, width=out_cols, transform=transform)

    dx = abs(transform.a)
    dy = abs(transform.e)
    return dem.astype(float), profile, dx, dy


def fill_nodata_local_average(dem, invalid, max_iters=8):
    """Fill nodata using local 8-neighbor average."""

    work = dem.copy()
    rem = invalid.copy()
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for _ in range(max_iters):
        if not rem.any():
            break

        sums = numpy.zeros_like(work)
        cnt = numpy.zeros_like(work, dtype=int)

        for dr, dc in neighbors:
            src = work[max(0, dr): work.shape[0] + min(0, dr), max(0, dc): work.shape[1] + min(0, dc)]
            src_mask = rem[max(0, dr): work.shape[0] + min(0, dr), max(0, dc): work.shape[1] + min(0, dc)]
            dst_r0, dst_r1 = max(0, -dr), work.shape[0] - max(0, dr)
            dst_c0, dst_c1 = max(0, -dc), work.shape[1] - max(0, dc)
            valid = ~src_mask
            sums[dst_r0:dst_r1, dst_c0:dst_c1] += src * valid
            cnt[dst_r0:dst_r1, dst_c0:dst_c1] += valid.astype(int)

        fillable = rem & (cnt > 0)

        if not fillable.any():
            break

        work[fillable] = sums[fillable] / cnt[fillable]
        rem[fillable] = False

    return work, rem


def validate_dem(dem, nodata=None):
    """Validate and patch invalid cells in DEM."""

    invalid = numpy.isnan(dem)

    if nodata is not None:
        invalid = invalid | (dem == nodata)

    total = int(invalid.sum())
    clean, rem = fill_nodata_local_average(dem, invalid)
    stats = {
        "invalid_total": total,
        "invalid_remaining": int(rem.sum()),
        "z_min": float(numpy.nanmin(clean)),
        "z_max": float(numpy.nanmax(clean))
    }
    return clean, stats


def crop_dem_with_buffer(dem, profile, source_row, source_col, dx, dy, buffer_m):
    """Crop DEM around a source point using a metric buffer distance."""

    if buffer_m <= 0:
        return dem, profile.copy(), {
            "roi_row_start": 0,
            "roi_row_end": int(dem.shape[0]),
            "roi_col_start": 0,
            "roi_col_end": int(dem.shape[1]),
            "roi_cells": int(dem.size),
            "buffer_m": float(buffer_m)
        }

    r_radius = int(numpy.ceil(buffer_m / max(dy, 1e-12)))
    c_radius = int(numpy.ceil(buffer_m / max(dx, 1e-12)))

    r0 = max(0, source_row - r_radius)
    r1 = min(dem.shape[0], source_row + r_radius + 1)
    c0 = max(0, source_col - c_radius)
    c1 = min(dem.shape[1], source_col + c_radius + 1)

    cropped = dem[r0:r1, c0:c1]
    transform = profile["transform"]
    cropped_transform = rasterio.transform.Affine(
        transform.a, transform.b, transform.c + c0 * transform.a + r0 * transform.b,
        transform.d, transform.e, transform.f + c0 * transform.d + r0 * transform.e
    )

    updated = profile.copy()
    updated.update(height=cropped.shape[0], width=cropped.shape[1], transform=cropped_transform)
    stats = {
        "roi_row_start": int(r0),
        "roi_row_end": int(r1),
        "roi_col_start": int(c0),
        "roi_col_end": int(c1),
        "roi_cells": int(cropped.size),
        "buffer_m": float(buffer_m)
    }
    return cropped, updated, stats


def preprocess_dem_file(
        dem_path: os.PathLike,
        output_path: os.PathLike,
        decimate: int = 1,
        source_row: int = None,
        source_col: int = None,
        roi_buffer_m: float = 0.0,
        summary_path: os.PathLike = None):
    """Preprocess a DEM from a local path and write processed raster and summary."""

    dem, profile, dx, dy = prepare_dem(dem_path, decimate)
    dem, dem_stats = validate_dem(dem, profile.get("nodata"))

    ny, nx = dem.shape
    source_row = ny // 2 if source_row is None else source_row
    source_col = nx // 2 if source_col is None else source_col

    if not (0 <= source_row < ny and 0 <= source_col < nx):
        raise ValueError("Source index out of DEM bounds")

    dem, profile, roi_stats = crop_dem_with_buffer(
        dem, profile, source_row, source_col, dx, dy, roi_buffer_m)

    output_path = pathlib.Path(output_path).expanduser().resolve()
    os.makedirs(output_path.parent, exist_ok=True)
    profile.update(dtype=rasterio.float64, count=1)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(dem.astype(numpy.float64), 1)

    summary = {
        "dem_path": str(pathlib.Path(dem_path).expanduser().resolve()),
        "output_path": str(output_path),
        "decimate": int(decimate),
        "cell_size": {"dx": float(dx), "dy": float(dy)},
        "source": {"row": int(source_row), "col": int(source_col)},
        "dem_stats": dem_stats,
        "roi_stats": roi_stats
    }

    if summary_path is None:
        summary_path = output_path.with_suffix(".summary.json")

    summary_path = pathlib.Path(summary_path).expanduser().resolve()

    with open(summary_path, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    return summary


def check_download_topo(case_dir: os.PathLike, rundata: clawutil.data.ClawRunData):
    """Check topo file and download it if it does not exist.

    Arguments
    ---------
    case_dir : PathLike
        Path to the directory of a case (where setrun.py can be found).
    rundata : ClawRunData
        An instance of `ClawRunData`.
    """

    # let pathlib handle paht-related stuff
    case_dir = pathlib.Path(case_dir).expanduser().resolve()

    # calculate extend and resolution required by the setrun.py
    ext = [rundata.clawdata.lower[0], rundata.clawdata.lower[1],
           rundata.clawdata.upper[0], rundata.clawdata.upper[1]]

    n_x = rundata.clawdata.num_cells[0]
    for i in range(rundata.amrdata.amr_levels_max-1):
        n_x *= rundata.amrdata.refinement_ratios_x[i]

    n_y = rundata.clawdata.num_cells[1]
    for i in range(rundata.amrdata.amr_levels_max-1):
        n_y *= rundata.amrdata.refinement_ratios_y[i]

    res = min((ext[2]-ext[0])/n_x, (ext[3]-ext[1])/n_y)

    # make the extent of the topo file a little bit larger than comp. domain
    ext[0] -= res
    ext[1] -= res
    ext[2] += res
    ext[3] += res

    # to indicate we've already download the topo once
    downloaded = None

    # check and download each topography file
    for topo in rundata.topo_data.topofiles:  # each topofile is represented as a list
        if os.path.isabs(topo[-1]):  # the topo filename is at the last element of the list
            topo_file = pathlib.Path(topo[-1])
        else:
            topo_file = case_dir.joinpath(topo[-1]).resolve()

        if topo_file.is_file():
            print("Topo file {} found. ".format(topo_file) + "Re-use it.")
            continue  # skip downloading

        if downloaded is None:
            print("Topo file {} not found. ".format(topo_file) + "Download it now.")
            downloaded = download_topo_single(topo_file, ext, res)
        else:  # already downloaded once, just check and copy that file
            print("Topo file {} not found. ".format(topo_file) + "Copy from {}.".format(downloaded))
            shutil.copyfile(downloaded, topo_file)


def download_topo_single(
        topo_file: os.PathLike,
        ext: Tuple[float, float, float, float],
        res: float):
    """Download a topo file.

    Arguments
    ---------
    topo_file : PathLike
        Path to the topo file.
    ext : tuple
        Extent of the topo, i.e., [x_min, y_min, x_max, y_max]
    res : float
        Resolution of the topo file.

    Returns
    -------
    The path to the downloaded file. Should be the same as the provided `topo_file`.
    """

    topo_file = pathlib.Path(topo_file).resolve()

    # prepare the folders
    os.makedirs(topo_file.parent, exist_ok=True)

    # download a GeoTiff file
    print("Downloading {}".format(topo_file.with_suffix(".tif")))
    obtain_geotiff(ext, topo_file.with_suffix(".tif"), res)
    print("Done downloading {}".format(topo_file.with_suffix(".tif")))

    # convert to Esri ASCII
    print("Converting to ESRI ASCII file")
    geotiff_2_esri_ascii(topo_file.with_suffix(".tif"), topo_file)
    print("Done converting to {}".format(topo_file))

    # remove the GeoTiff
    os.remove(topo_file.with_suffix(".tif"))

    return topo_file


def check_download_hydro(case_dir: os.PathLike, rundata: clawutil.data.ClawRunData):
    """Check hydro file and download it if it does not exist.

    Arguments
    ---------
    case_dir : PathLike
        Path to the directory of a case (where setrun.py can be found).
    rundata : ClawRunData
        An instance of `ClawRunData`.
    """

    if not rundata.landspill_data.hydro_features.files:
        return

    case_dir = pathlib.Path(case_dir).expanduser().resolve()

    if os.path.isabs(rundata.landspill_data.hydro_features.files[0]):
        hydro_file = pathlib.Path(rundata.landspill_data.hydro_features.files[0])
    else:
        hydro_file = case_dir.joinpath(rundata.landspill_data.hydro_features.files[0]).resolve()

    if os.path.isfile(hydro_file):
        print("Hydro file {} found. ".format(hydro_file) + "Re-use it.")
        return

    print("Hydro file {} not found. ".format(hydro_file) + "Download it now.")

    ext = [rundata.clawdata.lower[0], rundata.clawdata.lower[1],
           rundata.clawdata.upper[0], rundata.clawdata.upper[1]]

    n_x = rundata.clawdata.num_cells[0]
    for i in range(rundata.amrdata.amr_levels_max-1):
        n_x *= rundata.amrdata.refinement_ratios_x[i]

    n_y = rundata.clawdata.num_cells[1]
    for i in range(rundata.amrdata.amr_levels_max-1):
        n_y *= rundata.amrdata.refinement_ratios_y[i]

    res = min((ext[2]-ext[0])/n_x, (ext[3]-ext[1])/n_y)

    # make the extent of the hydro file a little bit larger than comp. domain
    ext[0] -= res
    ext[1] -= res
    ext[2] += res
    ext[3] += res

    os.makedirs(hydro_file.parent, exist_ok=True)

    print("Obtaining GeoJson from NHD high resolution dataset server.")
    feats = obtain_NHD_geojson(ext)

    print("Write GeoJson data to raster file {}".format(hydro_file))
    convert_geojson_2_raster(feats, hydro_file, ext, res)

    print("Done writing to {}".format(hydro_file))


def request_arcgis_token(
        username, password,
        token_server="https://www.arcgis.com/sharing/rest/generateToken",
        exp=5):
    """Request a token from ArcGIS token server

    Request a user token from ArcGIS. Only required when the source of elevation
    is chosen to be ESRI World Elevation.

    Return: a str, the token.
    """

    # information that will post to token server to obtain a token
    token_applicant = {
        "f": "json",
        "username": username,
        "password": password,
        "client": "referer",
        "expiration": str(exp),
        "referer": "https://www.arcgis.com"
    }

    # request a token
    token_response = requests.post(token_server, token_applicant)

    # try to raise an error if the server does not return success signal
    token_response.raise_for_status()

    # if execution comes to this point, we've got the token from the server
    token = token_response.json()["token"]

    return token


def obtain_geotiff(extent, filename, res=1, source="3DEP", token=None):
    """Grab the GeoTiff file for the elevation of a region.

    The region is defined by the argument extent. extent is a list with 4
    floating point numbers. Its format is extent = [Xmin, Ymin, Xmax, Ymax].

    Available elevation sources are 3DEP and ESRI. If using ESRI data, then
    token must present. Token can be obtained from the function
    request_arcgis_token. If using 3DEP, then remember that 3DEP only has data
    for North America.

    Args:
        extent [in]: a list with format [Xmin, Ymin, Xmax, Ymax]
        filename [in]: output GeoTiff filename.
        res [in]: output resolution. Default: 1 meter.
        source [in]: either 3DEP or ESRI.
        token [in]: if using ESRI source, the token must be provided.
    """

    # the REST endpoint of exportImage of the elevation server
    if source == "ESRI":
        dem_server = \
            "https://elevation.arcgis.com/arcgis/rest/services/" + \
            "WorldElevation/Terrain/ImageServer/exportImage"

        assert token is not None, \
            "Token cannot be None when using ESRI data source"

    elif source == "3DEP":
        dem_server = \
            "https://elevation.nationalmap.gov/arcgis/rest/services/" + \
            "3DEPElevation/ImageServer/exportImage"
    else:
        raise ValueError("Invalid elevation source: {}".format(source))

    # calculate number of cells
    x_size = int((extent[2]-extent[0])/res+0.5)
    y_size = int((extent[3]-extent[1])/res+0.5)

    # adjust North and East boundary to match x_size and y_size
    extent[2] = extent[0] + x_size * res
    extent[3] = extent[1] + y_size * res

    # parameters used to get response from the REST endpoint
    dem_query = {
        "f": "json",
        "bbox": "{},{},{},{}".format(extent[0], extent[1], extent[2], extent[3]),
        "size": "{},{}".format(x_size, y_size),
        "imageSR": "3857",
        "bboxSr": "3857",
        "format": "tiff",
        "pixelType": "F32",
        "noData": "-9999",
        "interpolation": "RSP_BilinearInterpolation",
    }

    # add token to parameters if using ESRI
    if source == "ESRI":
        dem_query["token"] = token
    else:
        dem_query["mosaicRule"] = \
            "{\"mosaicMethod\":\"esriMosaicAttribute\",\"sortField\":\"AcquisitionDate\"}"

    # create a HTTP session that can retry 5 times if 500, 502, 503, 504 happens
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(
        max_retries=urllib3.util.retry.Retry(
            total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])))

    # use GET to get response
    dem_response = session.get(dem_server, stream=True, params=dem_query)

    # try to raise an error if the server does not return success signal
    dem_response.raise_for_status()

    # close the session
    session.close()

    # if execution comes to this point, we've got the GeoTiff from the server
    tif_url = dem_response.json()["href"]

    # download the GeoTiff file, retry unitl success or timeout
    count = 0
    while True:

        rspnd = requests.get(tif_url, stream=True, allow_redirects=True)

        if rspnd.status_code == requests.codes.ok:  # pylint: disable=no-member
            break

        time.sleep(3)
        count += 3
        if count > 300:
            rspnd.raise_for_status()

    with open(os.path.abspath(filename), "wb") as file_obj:
        file_obj.write(rspnd.content)


def geotiff_2_esri_ascii(in_file, out_file):
    """Convert a GeoTiff to an ESRI ASCII file."""

    geotiff = rasterio.open(in_file, "r")

    # a workaround to ignore ERROR 4 message; we create a new Tiff and close it
    rasterio.open(
        os.path.abspath(out_file), mode="w", driver="GTiff",
        width=1, height=1, count=1, crs=rasterio.crs.CRS.from_epsg(3857),
        transform=geotiff.transform, dtype=rasterio.int8, nodata=0
    ).close()

    dst = rasterio.open(
        os.path.abspath(out_file), mode="w", driver="AAIGrid",
        width=geotiff.width, height=geotiff.height, count=geotiff.count,
        crs=rasterio.crs.CRS.from_epsg(3857),
        transform=geotiff.transform, dtype=rasterio.float32,
        nodata=geotiff.nodata)

    dst.write_band(1, geotiff.read(1))
    dst.close()

    geotiff.close()


def obtain_NHD_geojson(extent):  # pylint: disable=invalid-name
    """Obtain features from NHD high resolution dataset MapServer

    Retrun:
        A list: [<flowline>, <area>, <water body>]. The data types are GeoJson.
    """

    servers = []

    # flowline
    servers.append(
        "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/6/query")

    # area
    servers.append(
        "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/8/query")

    # waterbody
    servers.append(
        "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/10/query")

    queries = []

    # flowline
    queries.append({
        "where": "1=1",  # in the future, use this to filter FCode(s)
        "f": "geojson",
        "geometry": "{},{},{},{}".format(extent[0], extent[1], extent[2], extent[3]),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "3857",
        "spatialRel": "esriSpatialRelIntersects",
        "returnGeometry": "true",
        "outSR": "3857"
    })

    # area
    queries.append({
        "where": "1=1",  # in the future, use this to filter FCode(s)
        "f": "geojson",
        "geometry": "{},{},{},{}".format(extent[0], extent[1], extent[2], extent[3]),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "3857",
        "spatialRel": "esriSpatialRelIntersects",
        "returnGeometry": "true",
        "outSR": "3857"
    })

    # waterbody
    queries.append({
        "where": "1=1",  # in the future, use this to filter FCode(s)
        "f": "geojson",
        "geometry": "{},{},{},{}".format(extent[0], extent[1], extent[2], extent[3]),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "3857",
        "spatialRel": "esriSpatialRelIntersects",
        "returnGeometry": "true",
        "outSR": "3857"
    })

    geoms = []

    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(
        max_retries=urllib3.util.retry.Retry(
            total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])))

    for i, server in enumerate(servers):
        response = session.get(server, stream=True, params=queries[i])
        response.raise_for_status()
        geoms.append(response.json())

    session.close()

    return geoms


def convert_geojson_2_raster(feat_layers, filename, extent, res, crs=3857):
    """Convert a list of GeoTiff dict to raster (ESRI ASCII files)."""

    if crs != 3857:
        raise NotImplementedError("crs other than 3857 are not implemented yet")

    width = int((extent[2]-extent[0])/res+0.5)
    height = int((extent[3]-extent[1])/res+0.5)
    transform = rasterio.transform.from_origin(extent[0], extent[3], res, res)

    shapes = []
    for feat_layer in feat_layers:
        for geo in feat_layer["features"]:
            if not rasterio.features.is_valid_geom(geo["geometry"]):
                raise ValueError("Not a valid GeoJson gemoetry data")
            shapes.append((geo["geometry"], 10))

    # if no geometry exists in the list
    if not shapes:
        image = numpy.ones((height, width), dtype=rasterio.float32) * -9999.
    # else if there's any geometry
    else:
        image = rasterio.features.rasterize(
            shapes=shapes, out_shape=(width, height), fill=-9999.,
            transform=transform, all_touched=True, dtype=rasterio.float32)

    # a workaround to ignore ERROR 4 message
    rasterio.open(
        os.path.abspath(filename), mode="w", driver="GTiff",
        width=1, height=1, count=1, crs=rasterio.crs.CRS.from_epsg(3857),
        transform=transform, dtype=rasterio.int8, nodata=0
    ).close()

    dst = rasterio.open(
        os.path.abspath(filename), mode="w", driver="AAIGrid",
        width=width, height=height, count=1,
        crs=rasterio.crs.CRS.from_epsg(crs), transform=transform,
        dtype=rasterio.float32, nodata=-9999.)

    dst.write(image, indexes=1)

    dst.close()


def download_satellite_image(extent, filepath, force=False):
    """Download a setellite image of the given extent.

    Arguments
    ---------
        extent: a list of [xmin, ymin, xmax, ymax]
            The bound of the domain.
        filepath: os.PathLike
            Where to save the image.
        force: bool
            Download regardless of if the file already exists.

    Returns
    -------
        The extent of the saved image. The server does not always return the
        image within exactly the extent. Sometimes the extent of the image is
        larger, so we need to know.
    """

    filepath = pathlib.Path(filepath)

    api_url = "http://server.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"
    extent_file = filepath.with_suffix(filepath.suffix+".extent")

    # always with extra 5 pixels outside each boundary
    extent[0] = int(extent[0]) - 5
    extent[1] = int(extent[1]) - 5
    extent[2] = int(extent[2]) + 5
    extent[3] = int(extent[3]) + 5
    width = extent[2] - extent[0]
    height = extent[3] - extent[1]

    # if image and extent info already exists, we may stop downloading and leave
    if extent_file.is_file() and filepath.is_file() and not force:
        with open(extent_file, "r") as fileobj:
            img_extent = fileobj.readline()

        img_extent = [float(i) for i in img_extent.strip().split()]
        return img_extent

    # REST API parameters
    params = {
        "bbox": "{},{},{},{}".format(*extent),
        "bbSR": "3857",
        "size": "{},{}".format(width, height),
        "imageSR": "3857",
        "format": "png",
        "f": "json"
    }

    # create a HTTP session that can retry 5 times if 500, 502, 503, 504 happens
    session = requests.Session()
    session.mount("https://", requests.adapters.HTTPAdapter(
        max_retries=requests.packages.urllib3.util.retry.Retry(  # pylint: disable=no-member
            total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504]
        )
    ))

    # use GET to get response
    respns = session.get(api_url, params=params)
    respns.raise_for_status()  # raise an error if not success
    respns = respns.json()  # convert to a dictionary
    assert "href" in respns  # make sure the image's url is in the response

    # download the file, retry unitl success or timeout
    respns2 = session.get(respns["href"], stream=True, allow_redirects=True)
    respns2.raise_for_status()

    with open(filepath, "wb") as fileobj:
        fileobj.write(respns2.content)

    # close the session
    session.close()

    # write image extent to a text file
    with open(extent_file, "w") as fileobj:
        fileobj.write("{} {} {} {}".format(
            respns["extent"]["xmin"], respns["extent"]["ymin"],
            respns["extent"]["xmax"], respns["extent"]["ymax"]
        ))

    return [respns["extent"]["xmin"], respns["extent"]["ymin"],
            respns["extent"]["xmax"], respns["extent"]["ymax"]]
