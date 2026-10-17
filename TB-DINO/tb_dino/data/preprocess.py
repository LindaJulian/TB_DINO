from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.windows import Window, transform as window_transform

from .manifest import ManifestRecord, read_manifest, write_manifest
from .raster import (
    align_to_reference,
    grids_match,
    l2_normalize_channels,
    read_raster,
    write_geotiff,
)


RASTER_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".npy", ".npz"}


def patch_starts(length: int, size: int, stride: int) -> list[int]:
    if size <= 0 or stride <= 0:
        raise ValueError("patch_size 和 stride 必须大于 0")
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, stride))
    final = length - size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _pad_patch(array: np.ndarray, height: int, width: int) -> np.ndarray:
    pad_height = max(0, height - array.shape[-2])
    pad_width = max(0, width - array.shape[-1])
    if pad_height == 0 and pad_width == 0:
        return array
    return np.pad(
        array,
        ((0, 0), (0, pad_height), (0, pad_width)),
        mode="constant",
    )


def _patch_profile(
    profile: dict[str, Any],
    x: int,
    y: int,
    size: int,
) -> tuple[dict[str, Any], Any | None]:
    result = profile.copy()
    result["height"] = size
    result["width"] = size
    transform = None
    if profile.get("transform") is not None:
        transform = window_transform(Window(x, y, size, size), profile["transform"])
        result["transform"] = transform
    return result, transform


def _files_by_stem(directory: str | Path) -> dict[str, Path]:
    directory = Path(directory).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    result = {
        path.stem: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in RASTER_SUFFIXES
    }
    if not result:
        raise ValueError(f"目录中未找到支持的影像文件: {directory}")
    return result


def _split_ids(
    ids: list[str],
    seed: int,
    ratios: dict[str, float],
) -> dict[str, str]:
    names = ("train", "val", "test")
    values = [float(ratios.get(name, 0.0)) for name in names]
    if abs(sum(values) - 1.0) > 1e-6:
        raise ValueError(f"train/val/test 划分比例之和必须为 1，实际为 {sum(values)}")
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    train_end = round(len(shuffled) * values[0])
    val_end = train_end + round(len(shuffled) * values[1])
    assignment: dict[str, str] = {}
    for scene_id in shuffled[:train_end]:
        assignment[scene_id] = "train"
    for scene_id in shuffled[train_end:val_end]:
        assignment[scene_id] = "val"
    for scene_id in shuffled[val_end:]:
        assignment[scene_id] = "test"
    return assignment


def _read_split_files(
    split_files: dict[str, str] | None,
) -> dict[str, str]:
    if not split_files:
        return {}
    assignment: dict[str, str] = {}
    for split, path in split_files.items():
        with Path(path).expanduser().resolve().open("r", encoding="utf-8") as file:
            for line in file:
                scene_id = line.strip()
                if scene_id:
                    assignment[Path(scene_id).stem] = split
    return assignment


def preprocess_bitemporal(config: dict[str, Any]) -> dict[str, int]:
    image_a = _files_by_stem(config["image_a_dir"])
    image_b = _files_by_stem(config["image_b_dir"])
    masks = _files_by_stem(config["mask_dir"])
    common = sorted(set(image_a) & set(image_b) & set(masks))
    if not common:
        raise ValueError("A/B/标签目录中没有同名场景")
    all_ids = set(image_a) | set(image_b) | set(masks)
    missing = {
        "A": sorted(all_ids - set(image_a)),
        "B": sorted(all_ids - set(image_b)),
        "mask": sorted(all_ids - set(masks)),
    }
    if any(missing.values()) and config.get("strict_pairing", True):
        raise ValueError(f"存在无法配对的场景: {missing}")

    split_assignment = _read_split_files(config.get("split_files"))
    unsplit = [scene_id for scene_id in common if scene_id not in split_assignment]
    split_assignment.update(
        _split_ids(
            unsplit,
            int(config.get("seed", 42)),
            config.get("split", {"train": 0.6, "val": 0.2, "test": 0.2}),
        )
    )
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    patch_size = int(config.get("patch_size", 256))
    stride = int(config.get("stride", patch_size))
    bands = config.get("bands")
    records_by_split: dict[str, list[ManifestRecord]] = defaultdict(list)

    for scene_id in common:
        first, first_profile = read_raster(image_a[scene_id], bands)
        second, second_profile = read_raster(image_b[scene_id], bands)
        mask, mask_profile = read_raster(masks[scene_id], [1])
        if first.shape != second.shape:
            raise ValueError(f"场景 {scene_id} 的双时相影像形状不一致")
        if first.shape[-2:] != mask.shape[-2:]:
            raise ValueError(f"场景 {scene_id} 的标签与影像形状不一致")
        if not grids_match(first_profile, second_profile):
            raise ValueError(f"场景 {scene_id} 的双时相 CRS 或像素网格不一致")
        if not grids_match(first_profile, mask_profile):
            raise ValueError(f"场景 {scene_id} 的标签网格与影像不一致")
        height, width = first.shape[-2:]
        split = split_assignment[scene_id]
        for y in patch_starts(height, patch_size, stride):
            for x in patch_starts(width, patch_size, stride):
                patch_id = f"{scene_id}_y{y:06d}_x{x:06d}"
                relative = Path("patches") / split / patch_id
                a_path = output_dir / relative / "t0.tif"
                b_path = output_dir / relative / "t1.tif"
                mask_path = output_dir / relative / "mask.tif"
                valid_path = output_dir / relative / "valid.tif"
                first_patch = _pad_patch(
                    first[:, y : y + patch_size, x : x + patch_size],
                    patch_size,
                    patch_size,
                )
                second_patch = _pad_patch(
                    second[:, y : y + patch_size, x : x + patch_size],
                    patch_size,
                    patch_size,
                )
                raw_mask = _pad_patch(
                    mask[:, y : y + patch_size, x : x + patch_size],
                    patch_size,
                    patch_size,
                )
                valid = np.zeros((1, patch_size, patch_size), dtype=np.uint8)
                valid[:, : min(patch_size, height - y), : min(patch_size, width - x)] = 1
                configured_nodata = config.get("mask_nodata")
                if configured_nodata is not None:
                    valid &= raw_mask != configured_nodata
                binary_mask = (raw_mask > 0).astype(np.uint8)
                first_patch_profile, first_transform = _patch_profile(
                    first_profile, x, y, patch_size
                )
                second_patch_profile, second_transform = _patch_profile(
                    second_profile, x, y, patch_size
                )
                mask_patch_profile, mask_transform = _patch_profile(
                    mask_profile or first_profile, x, y, patch_size
                )
                write_geotiff(
                    a_path,
                    first_patch,
                    first_patch_profile,
                    transform=first_transform,
                )
                write_geotiff(
                    b_path,
                    second_patch,
                    second_patch_profile,
                    transform=second_transform,
                )
                write_geotiff(
                    mask_path,
                    binary_mask,
                    mask_patch_profile,
                    dtype="uint8",
                    nodata=255,
                    transform=mask_transform,
                )
                write_geotiff(
                    valid_path,
                    valid,
                    mask_patch_profile,
                    dtype="uint8",
                    nodata=0,
                    transform=mask_transform,
                )
                records_by_split[split].append(
                    ManifestRecord(
                        id=patch_id,
                        images=[str(a_path), str(b_path)],
                        timestamps=[
                            str(config.get("timestamp_a", "t0")),
                            str(config.get("timestamp_b", "t1")),
                        ],
                        mask=str(mask_path),
                        valid_mask=str(valid_path),
                        bands=list(range(1, first.shape[0] + 1)),
                        metadata={"source_scene": scene_id},
                    )
                )

    counts = {}
    for split in ("train", "val", "test"):
        records = records_by_split.get(split, [])
        if records:
            write_manifest(records, output_dir / f"{split}.jsonl")
        counts[split] = len(records)
    return counts


def _load_raw_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            base = path.parent
            for key in ("images", "labels"):
                if key in value:
                    value[key] = [
                        str((base / item).resolve())
                        if not Path(item).is_absolute()
                        else str(Path(item).resolve())
                        for item in value[key]
                    ]
            for key in ("first_label", "last_label"):
                if value.get(key) and not Path(value[key]).is_absolute():
                    value[key] = str((base / value[key]).resolve())
            value["alpha_by_year"] = {
                str(year): (
                    str((base / item).resolve())
                    if not Path(item).is_absolute()
                    else str(Path(item).resolve())
                )
                for year, item in (value.get("alpha_by_year") or {}).items()
            }
            if "id" not in value or "images" not in value or "timestamps" not in value:
                raise ValueError(f"{path}:{line_number} 缺少 id/images/timestamps")
            if len(value["images"]) != len(value["timestamps"]):
                raise ValueError(f"{path}:{line_number} 影像与时间戳数量不一致")
            records.append(value)
    return records


def _resize_channels(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(array.astype(np.float32, copy=False))[None]
    result = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return result[0].numpy()


def _aligned_alpha(
    alpha_path: str,
    reference_path: str,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    alpha_raw, alpha_profile = read_raster(alpha_path)
    _, reference_profile = read_raster(reference_path)
    if (
        alpha_profile.get("crs") is not None
        and reference_profile.get("crs") is not None
        and alpha_profile.get("transform") is not None
        and reference_profile.get("transform") is not None
    ):
        alpha, profile = align_to_reference(
            alpha_path,
            reference_path,
            resampling=Resampling.bilinear,
        )
    else:
        alpha = alpha_raw
        if alpha.shape[-2:] != expected_shape:
            alpha = _resize_channels(alpha, expected_shape)
        profile = reference_profile
    return l2_normalize_channels(alpha), profile


def preprocess_timeseries(config: dict[str, Any]) -> dict[str, int]:
    raw_records = _load_raw_records(config["sequences_file"])
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    patch_size = int(config.get("patch_size", 64))
    stride = int(config.get("stride", patch_size))
    ignore_index = int(config.get("ignore_index", 255))
    bands = config.get("bands")

    explicit = {
        str(record["id"]): str(record["split"])
        for record in raw_records
        if record.get("split") in {"train", "val", "test"}
    }
    unsplit = [str(record["id"]) for record in raw_records if str(record["id"]) not in explicit]
    explicit.update(
        _split_ids(
            unsplit,
            int(config.get("seed", 42)),
            config.get("split", {"train": 0.6, "val": 0.2, "test": 0.2}),
        )
    )
    records_by_split: dict[str, list[ManifestRecord]] = defaultdict(list)

    for raw in raw_records:
        scene_id = str(raw["id"])
        image_arrays = []
        profiles = []
        for image_path in raw["images"]:
            array, profile = read_raster(image_path, bands)
            image_arrays.append(array)
            profiles.append(profile)
        shape = image_arrays[0].shape
        if any(array.shape != shape for array in image_arrays):
            raise ValueError(f"序列 {scene_id} 的影像形状不一致，请先完成配准")
        if any(not grids_match(profiles[0], profile) for profile in profiles[1:]):
            raise ValueError(f"序列 {scene_id} 的 CRS 或像素网格不一致，请先完成配准")
        target_mode = str(
            config.get("target_mode", "cumulative_monthly")
        ).lower()
        if target_mode not in {"cumulative_monthly", "endpoint"}:
            raise ValueError(
                "target_mode 必须为 cumulative_monthly 或 endpoint"
            )

        label_paths = raw.get("labels")
        if label_paths is not None:
            if len(label_paths) != len(image_arrays):
                raise ValueError(f"序列 {scene_id} 的标签与影像数量不一致")
            labels = []
            label_profiles = []
            for label_path in label_paths:
                label, label_profile = read_raster(label_path, [1])
                if label.shape[-2:] != shape[-2:]:
                    raise ValueError(f"序列 {scene_id} 的标签尺寸与影像不一致")
                if not grids_match(profiles[0], label_profile):
                    raise ValueError(f"序列 {scene_id} 的标签网格与影像不一致")
                labels.append(label)
                label_profiles.append(label_profile)
        else:
            if target_mode == "cumulative_monthly":
                raise ValueError(
                    f"序列 {scene_id} 的 cumulative_monthly target 必须提供完整 labels"
                )
            first_label_path = raw.get("first_label")
            last_label_path = raw.get("last_label")
            if not first_label_path or not last_label_path:
                raise ValueError(
                    f"序列 {scene_id} 缺少 labels，无法生成累积月度变化 target"
                )
            first_label, first_profile = read_raster(first_label_path, [1])
            last_label, last_profile = read_raster(last_label_path, [1])
            if first_label.shape[-2:] != shape[-2:] or last_label.shape[-2:] != shape[-2:]:
                raise ValueError(f"序列 {scene_id} 的标签尺寸与影像不一致")
            if not grids_match(profiles[0], first_profile) or not grids_match(
                profiles[0], last_profile
            ):
                raise ValueError(f"序列 {scene_id} 的标签网格与影像不一致")
            labels = [first_label, last_label]
            label_profiles = [first_profile, last_profile]

        if target_mode == "cumulative_monthly" and len(labels) < 2:
            raise ValueError(
                f"序列 {scene_id} 至少需要两个连续月份标签才能生成累积变化 target"
            )

        configured_nodata = config.get("label_nodata")

        def label_valid(label: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
            valid = np.isfinite(label) & (label != ignore_index)
            if configured_nodata is not None:
                valid &= label != configured_nodata
            elif profile.get("nodata") is not None:
                valid &= label != profile["nodata"]
            return valid

        if target_mode == "endpoint":
            first_label, last_label = labels[0], labels[-1]
            valid_full = (
                label_valid(first_label, label_profiles[0])
                & label_valid(last_label, label_profiles[-1])
            ).astype(np.uint8)
            mask_full = (
                (first_label != last_label) & (valid_full > 0)
            ).astype(np.uint8)
        else:
            # DynamicEarthNet uses cumulative monthly transitions: a pixel is
            # positive if its semantic label changes in any adjacent month.
            mask_full = np.zeros_like(labels[0], dtype=np.uint8)
            valid_full = np.zeros_like(labels[0], dtype=np.uint8)
            for previous, current, previous_profile, current_profile in zip(
                labels[:-1],
                labels[1:],
                label_profiles[:-1],
                label_profiles[1:],
                strict=True,
            ):
                transition_valid = (
                    label_valid(previous, previous_profile)
                    & label_valid(current, current_profile)
                )
                mask_full |= (
                    (previous != current) & transition_valid
                ).astype(np.uint8)
                valid_full |= transition_valid.astype(np.uint8)

        alpha_arrays: dict[str, np.ndarray] = {}
        alpha_profiles: dict[str, dict[str, Any]] = {}
        for year, alpha_path in raw.get("alpha_by_year", {}).items():
            alpha, alpha_profile = _aligned_alpha(
                alpha_path,
                raw["images"][0],
                shape[-2:],
            )
            alpha_arrays[str(year)] = alpha
            alpha_profiles[str(year)] = alpha_profile

        height, width = shape[-2:]
        split = explicit[scene_id]
        for y in patch_starts(height, patch_size, stride):
            for x in patch_starts(width, patch_size, stride):
                patch_id = f"{scene_id}_y{y:06d}_x{x:06d}"
                relative = Path("patches") / split / patch_id
                image_paths: list[str] = []
                for time_index, (array, profile) in enumerate(
                    zip(image_arrays, profiles, strict=True)
                ):
                    image_path = output_dir / relative / f"t{time_index:03d}.tif"
                    patch = _pad_patch(
                        array[:, y : y + patch_size, x : x + patch_size],
                        patch_size,
                        patch_size,
                    )
                    patch_profile, transform = _patch_profile(profile, x, y, patch_size)
                    write_geotiff(
                        image_path,
                        patch,
                        patch_profile,
                        transform=transform,
                    )
                    image_paths.append(str(image_path))

                mask_path = output_dir / relative / "mask.tif"
                valid_path = output_dir / relative / "valid.tif"
                mask_patch = _pad_patch(
                    mask_full[:, y : y + patch_size, x : x + patch_size],
                    patch_size,
                    patch_size,
                )
                valid_patch = _pad_patch(
                    valid_full[:, y : y + patch_size, x : x + patch_size],
                    patch_size,
                    patch_size,
                )
                mask_patch_profile, mask_transform = _patch_profile(
                    label_profiles[0] if label_profiles else profiles[0],
                    x,
                    y,
                    patch_size,
                )
                write_geotiff(
                    mask_path,
                    mask_patch,
                    mask_patch_profile,
                    dtype="uint8",
                    nodata=255,
                    transform=mask_transform,
                )
                write_geotiff(
                    valid_path,
                    valid_patch,
                    mask_patch_profile,
                    dtype="uint8",
                    nodata=0,
                    transform=mask_transform,
                )

                alpha_paths: dict[str, str] = {}
                for year, alpha in alpha_arrays.items():
                    alpha_path = output_dir / relative / f"alpha_{year}.tif"
                    alpha_patch = _pad_patch(
                        alpha[:, y : y + patch_size, x : x + patch_size],
                        patch_size,
                        patch_size,
                    )
                    alpha_profile, alpha_transform = _patch_profile(
                        alpha_profiles[year], x, y, patch_size
                    )
                    write_geotiff(
                        alpha_path,
                        alpha_patch,
                        alpha_profile,
                        dtype="float32",
                        transform=alpha_transform,
                    )
                    alpha_paths[year] = str(alpha_path)

                records_by_split[split].append(
                    ManifestRecord(
                        id=patch_id,
                        images=image_paths,
                        timestamps=[str(item) for item in raw["timestamps"]],
                        mask=str(mask_path),
                        alpha_by_year=alpha_paths,
                        valid_mask=str(valid_path),
                        bands=list(range(1, shape[0] + 1)),
                        metadata={
                            "source_scene": scene_id,
                            "target_mode": target_mode,
                            "transition_count": len(labels) - 1,
                        },
                    )
                )

    counts = {}
    for split in ("train", "val", "test"):
        records = records_by_split.get(split, [])
        if records:
            write_manifest(records, output_dir / f"{split}.jsonl")
        counts[split] = len(records)
    return counts


def validate_existing_manifest(config: dict[str, Any]) -> dict[str, int]:
    records = read_manifest(
        config["manifest"],
        validate_files=True,
        require_alpha=bool(config.get("require_alpha", False)),
    )
    return {"records": len(records)}


def run_preprocessing(config: dict[str, Any]) -> dict[str, int]:
    mode = str(config.get("mode", "")).lower()
    if mode in {"bitemporal", "levir", "whu"}:
        return preprocess_bitemporal(config)
    if mode in {"timeseries", "dynamic", "dynamicearthnet"}:
        return preprocess_timeseries(config)
    if mode == "validate_manifest":
        return validate_existing_manifest(config)
    raise ValueError(
        "未知预处理模式。可选: bitemporal/levir/whu/timeseries/dynamic/validate_manifest"
    )
