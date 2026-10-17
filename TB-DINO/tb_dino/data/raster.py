from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import reproject


def ensure_chw(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"只支持二维或三维数组，实际形状为 {array.shape}")
    # Explicitly detect common HWC arrays returned by Pillow. Ambiguous cubes
    # such as 64x64x64 default to CHW because remote-sensing embeddings are
    # commonly stored with channels first.
    if (
        array.shape[-1] <= 16
        and array.shape[0] > array.shape[-1]
        and array.shape[1] > array.shape[-1]
    ):
        return np.moveaxis(array, -1, 0)
    if array.shape[0] <= 256:
        return array
    if array.shape[-1] <= 256:
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"无法判断三维数组的通道维: {array.shape}")


def read_raster(
    path: str | Path,
    bands: list[int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = ensure_chw(np.load(path))
        return array, {}
    if suffix == ".npz":
        archive = np.load(path)
        key = "array" if "array" in archive else archive.files[0]
        return ensure_chw(archive[key]), {}
    try:
        with rasterio.open(path) as source:
            indexes = bands or list(range(1, source.count + 1))
            array = source.read(indexes)
            profile = source.profile.copy()
            profile["transform"] = source.transform
            profile["crs"] = source.crs
            profile["nodata"] = source.nodata
            return array, profile
    except rasterio.errors.RasterioIOError:
        with Image.open(path) as image:
            array = ensure_chw(np.asarray(image))
        if bands:
            array = array[[index - 1 for index in bands]]
        return array, {}


def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    reference_profile: dict[str, Any] | None = None,
    *,
    dtype: str | None = None,
    nodata: float | int | None = None,
    transform: Any | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = ensure_chw(np.asarray(array))
    output_dtype = dtype or str(array.dtype)
    profile = dict(reference_profile or {})
    original_count = profile.get("count")
    profile.update(
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=array.shape[0],
        dtype=output_dtype,
        compress="deflate",
    )
    if transform is not None:
        profile["transform"] = transform
    if nodata is not None:
        profile["nodata"] = nodata
    if original_count is not None and original_count != array.shape[0]:
        profile.pop("photometric", None)
        profile.pop("interleave", None)
    block_x = profile.get("blockxsize")
    block_y = profile.get("blockysize")
    invalid_blocks = (
        (block_x is not None and (block_x > array.shape[2] or block_x % 16))
        or (block_y is not None and (block_y > array.shape[1] or block_y % 16))
        or array.shape[1] < 16
        or array.shape[2] < 16
    )
    if invalid_blocks:
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("tiled", None)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(array.astype(output_dtype, copy=False))


def grids_match(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if not first or not second:
        return True
    return (
        first.get("width") == second.get("width")
        and first.get("height") == second.get("height")
        and first.get("crs") == second.get("crs")
        and first.get("transform") == second.get("transform")
    )


def align_to_reference(
    source_path: str | Path,
    reference_path: str | Path,
    *,
    bands: list[int] | None = None,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(reference_path) as reference:
        reference_profile = reference.profile.copy()
        reference_profile["transform"] = reference.transform
        reference_profile["crs"] = reference.crs
        with rasterio.open(source_path) as source:
            indexes = bands or list(range(1, source.count + 1))
            destination = np.zeros(
                (len(indexes), reference.height, reference.width), dtype=np.float32
            )
            for output_index, source_index in enumerate(indexes):
                reproject(
                    source=source.read(source_index),
                    destination=destination[output_index],
                    src_transform=source.transform,
                    src_crs=source.crs,
                    dst_transform=reference.transform,
                    dst_crs=reference.crs,
                    resampling=resampling,
                )
    return destination, reference_profile


def l2_normalize_channels(array: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    norm = np.linalg.norm(array, axis=0, keepdims=True)
    return array / np.maximum(norm, epsilon)
