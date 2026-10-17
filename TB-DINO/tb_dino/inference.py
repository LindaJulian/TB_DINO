from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.windows import Window
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import (
    ChangeDetectionDataset,
    collate_change_batch,
    normalize_image,
)
from .data.preprocess import patch_starts
from .data.raster import read_raster, write_geotiff
from .engine import load_checkpoint, move_batch
from .metrics import BinaryConfusion
from .models import build_model
from .utils import choose_device, write_json


def _hann_weight(height: int, width: int) -> np.ndarray:
    vertical = np.hanning(height) if height > 1 else np.ones(1)
    horizontal = np.hanning(width) if width > 1 else np.ones(1)
    weight = np.outer(vertical, horizontal).astype(np.float32)
    return np.maximum(weight, 0.05)


def _save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


@torch.no_grad()
def infer_manifest(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    inference = config.get("inference", {})
    device = choose_device(str(inference.get("device", "auto")))
    model = build_model(config.get("model", {})).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    configured_threshold = inference.get("threshold")
    threshold = float(
        checkpoint.get("threshold", 0.5)
        if configured_threshold is None
        else configured_threshold
    )
    data_config = dict(config.get("data", {}))
    data_config["use_alphaearth"] = bool(
        config.get("model", {}).get("use_alphaearth", False)
    )
    dataset = ChangeDetectionDataset(manifest_path, data_config, training=False)
    loader = DataLoader(
        dataset,
        batch_size=int(inference.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(data_config.get("num_workers", 0)),
        collate_fn=collate_change_batch,
    )
    output_dir = Path(output_dir).expanduser().resolve()
    tracker = BinaryConfusion(threshold)
    for batch in tqdm(loader, desc="推理"):
        device_batch = move_batch(batch, device)
        output = model(
            device_batch["images"],
            device_batch["temporal_mask"],
            alpha=device_batch.get("alpha"),
            valid_mask=device_batch.get("valid_mask"),
        )
        probabilities = torch.sigmoid(output.logits).cpu()
        tracker.update(probabilities, batch["mask"], batch["valid_mask"])
        for index, record in enumerate(batch["records"]):
            probability = probabilities[index, 0].numpy().astype(np.float32)
            binary = (probability >= threshold).astype(np.uint8)
            _, profile = read_raster(record.images[0])
            sample_dir = output_dir / record.id
            write_geotiff(
                sample_dir / "probability.tif",
                probability,
                profile,
                dtype="float32",
                nodata=-1,
            )
            write_geotiff(
                sample_dir / "change.tif",
                binary,
                profile,
                dtype="uint8",
                nodata=255,
            )
            if inference.get("write_png", True):
                _save_png(sample_dir / "change.png", binary * 255)
    metrics = tracker.compute()
    write_json(metrics, output_dir / "metrics.json")
    return metrics


def _check_source_grids(sources: list[rasterio.io.DatasetReader]) -> None:
    first = sources[0]
    for source in sources[1:]:
        if (
            source.width != first.width
            or source.height != first.height
            or source.crs != first.crs
            or source.transform != first.transform
        ):
            raise ValueError("多时相影像未处于同一像素网格，请先运行预处理对齐")


def _read_normalized_window(
    source: rasterio.io.DatasetReader,
    window: Window,
    bands: list[int],
    tile_size: int,
    data_config: dict[str, Any],
) -> torch.Tensor:
    array = source.read(bands, window=window)
    if array.shape[-2:] != (tile_size, tile_size):
        padded = np.zeros((array.shape[0], tile_size, tile_size), dtype=array.dtype)
        padded[:, : array.shape[-2], : array.shape[-1]] = array
        array = padded
    return normalize_image(
        array,
        data_config.get("scale_divisor", 255.0),
        data_config.get("mean"),
        data_config.get("std"),
    )


@torch.no_grad()
def infer_rasters(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, str | float]:
    inference = config.get("inference", {})
    image_paths = [str(path) for path in inference.get("images", [])]
    timestamps = [str(item) for item in inference.get("timestamps", [])]
    if len(image_paths) < 2 or len(image_paths) != len(timestamps):
        raise ValueError("inference.images 与 timestamps 必须等长且至少包含两个时相")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    device = choose_device(str(inference.get("device", "auto")))
    model = build_model(config.get("model", {})).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    configured_threshold = inference.get("threshold")
    threshold = float(
        checkpoint.get("threshold", 0.5)
        if configured_threshold is None
        else configured_threshold
    )
    data_config = config.get("data", {})
    bands = list(data_config.get("bands", [1, 2, 3]))
    tile_size = int(inference.get("tile_size", 256))
    stride = int(inference.get("stride", tile_size // 2))
    if tile_size % int(getattr(model, "patch_size", 16)):
        raise ValueError("inference.tile_size 必须是模型 patch_size 的倍数")

    image_sources = [rasterio.open(path) for path in image_paths]
    alpha_sources: dict[str, rasterio.io.DatasetReader] = {}
    try:
        _check_source_grids(image_sources)
        reference = image_sources[0]
        alpha_paths = {
            str(year): str(path)
            for year, path in (inference.get("alpha_by_year") or {}).items()
        }
        if config.get("model", {}).get("use_alphaearth", False):
            missing = sorted({timestamp[:4] for timestamp in timestamps} - set(alpha_paths))
            if missing:
                raise ValueError(f"推理缺少年度 AlphaEarth: {missing}")
            alpha_sources = {
                year: rasterio.open(path) for year, path in alpha_paths.items()
            }
            _check_source_grids([reference, *alpha_sources.values()])
        raster_shape = (reference.height, reference.width)
        if inference.get("use_memmap", True):
            probability_sum_path = output_dir / ".probability_sum.tmp"
            weight_sum_path = output_dir / ".weight_sum.tmp"
            temporary_paths.extend([probability_sum_path, weight_sum_path])
            probability_sum = np.memmap(
                probability_sum_path,
                mode="w+",
                dtype=np.float32,
                shape=raster_shape,
            )
            weight_sum = np.memmap(
                weight_sum_path,
                mode="w+",
                dtype=np.float32,
                shape=raster_shape,
            )
            probability_sum[:] = 0
            weight_sum[:] = 0
        else:
            probability_sum = np.zeros(raster_shape, np.float32)
            weight_sum = np.zeros_like(probability_sum)
        full_weight = _hann_weight(tile_size, tile_size)
        temporal_mask = torch.ones(1, len(image_sources), dtype=torch.bool, device=device)

        for y in tqdm(
            patch_starts(reference.height, tile_size, stride),
            desc="滑窗推理",
        ):
            for x in patch_starts(reference.width, tile_size, stride):
                actual_height = min(tile_size, reference.height - y)
                actual_width = min(tile_size, reference.width - x)
                window = Window(x, y, actual_width, actual_height)
                images = torch.stack(
                    [
                        _read_normalized_window(
                            source, window, bands, tile_size, data_config
                        )
                        for source in image_sources
                    ]
                )[None].to(device)
                alpha = None
                if alpha_sources:
                    alpha_tensors = []
                    for timestamp in timestamps:
                        source = alpha_sources[timestamp[:4]]
                        array = source.read(window=window).astype(np.float32)
                        if array.shape[-2:] != (tile_size, tile_size):
                            padded = np.zeros(
                                (array.shape[0], tile_size, tile_size), np.float32
                            )
                            padded[:, :actual_height, :actual_width] = array
                            array = padded
                        tensor = torch.from_numpy(array)
                        tensor = torch.nn.functional.normalize(
                            tensor, p=2, dim=0, eps=1e-6
                        )
                        alpha_tensors.append(tensor)
                    alpha = torch.stack(alpha_tensors)[None].to(device)
                output = model(
                    images,
                    temporal_mask,
                    alpha=alpha,
                )
                probability = torch.sigmoid(output.logits)[0, 0].cpu().numpy()
                probability = probability[:actual_height, :actual_width]
                weight = full_weight[:actual_height, :actual_width]
                probability_sum[
                    y : y + actual_height, x : x + actual_width
                ] += probability * weight
                weight_sum[y : y + actual_height, x : x + actual_width] += weight

        if inference.get("use_memmap", True):
            probability_tmp = output_dir / ".probability.tmp"
            binary_tmp = output_dir / ".binary.tmp"
            temporary_paths.extend([probability_tmp, binary_tmp])
            probability = np.memmap(
                probability_tmp,
                mode="w+",
                dtype=np.float32,
                shape=raster_shape,
            )
            binary = np.memmap(
                binary_tmp,
                mode="w+",
                dtype=np.uint8,
                shape=raster_shape,
            )
            block_rows = int(inference.get("memmap_block_rows", 1024))
            for row in range(0, reference.height, block_rows):
                stop = min(reference.height, row + block_rows)
                denominator = np.maximum(weight_sum[row:stop], 1e-6)
                np.divide(
                    probability_sum[row:stop],
                    denominator,
                    out=probability[row:stop],
                )
                binary[row:stop] = probability[row:stop] >= threshold
        else:
            probability = probability_sum / np.maximum(weight_sum, 1e-6)
            binary = (probability >= threshold).astype(np.uint8)
        profile = reference.profile.copy()
        probability_path = output_dir / "probability.tif"
        mask_path = output_dir / "change.tif"
        write_geotiff(
            probability_path,
            probability,
            profile,
            dtype="float32",
            nodata=-1,
        )
        write_geotiff(mask_path, binary, profile, dtype="uint8", nodata=255)
        if inference.get("write_png", True):
            _save_png(output_dir / "change.png", binary * 255)
        result = {
            "probability": str(probability_path),
            "mask": str(mask_path),
            "threshold": threshold,
        }
        write_json(result, output_dir / "inference.json")
        if isinstance(probability_sum, np.memmap):
            probability_sum.flush()
            weight_sum.flush()
            probability.flush()
            binary.flush()
            del probability_sum, weight_sum, probability, binary
            for temporary_path in temporary_paths:
                temporary_path.unlink(missing_ok=True)
            temporary_paths.clear()
        return result
    finally:
        for source in image_sources:
            source.close()
        for source in alpha_sources.values():
            source.close()
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def run_inference(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    manifest: str | Path | None = None,
) -> dict[str, Any]:
    inference = config.get("inference", {})
    mode = str(inference.get("mode", "manifest"))
    output = output_dir or inference.get("output_dir") or Path(
        config.get("output_dir", "outputs/run")
    ) / "predictions"
    if mode == "manifest":
        target_manifest = manifest or inference.get("manifest") or config.get(
            "data", {}
        ).get("test_manifest")
        if not target_manifest:
            raise ValueError("manifest 推理需要指定输入 manifest")
        return infer_manifest(config, checkpoint_path, target_manifest, output)
    if mode == "raster":
        return infer_rasters(config, checkpoint_path, output)
    raise ValueError("inference.mode 必须是 manifest 或 raster")
