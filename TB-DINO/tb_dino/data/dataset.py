from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .manifest import ManifestRecord, read_manifest
from .raster import read_raster


def normalize_image(
    array: np.ndarray,
    scale_divisor: float | list[float],
    mean: list[float] | None,
    std: list[float] | None,
) -> torch.Tensor:
    tensor = torch.from_numpy(array.astype(np.float32, copy=False))
    if isinstance(scale_divisor, list):
        scale = torch.tensor(scale_divisor, dtype=tensor.dtype)[:, None, None]
        tensor = tensor / scale.clamp_min(1e-8)
    elif scale_divisor:
        tensor = tensor / float(scale_divisor)
    if mean is not None:
        if len(mean) != tensor.shape[0]:
            raise ValueError(
                f"mean 长度 {len(mean)} 与影像通道数 {tensor.shape[0]} 不一致"
            )
        tensor = tensor - torch.tensor(mean, dtype=tensor.dtype)[:, None, None]
    if std is not None:
        if len(std) != tensor.shape[0]:
            raise ValueError(
                f"std 长度 {len(std)} 与影像通道数 {tensor.shape[0]} 不一致"
            )
        tensor = tensor / torch.tensor(std, dtype=tensor.dtype)[:, None, None].clamp_min(
            1e-8
        )
    return tensor


def _apply_geometry(
    tensors: list[torch.Tensor],
    horizontal: bool,
    vertical: bool,
    rotations: int,
) -> list[torch.Tensor]:
    output = tensors
    if horizontal:
        output = [torch.flip(tensor, dims=(-1,)) for tensor in output]
    if vertical:
        output = [torch.flip(tensor, dims=(-2,)) for tensor in output]
    if rotations:
        output = [torch.rot90(tensor, rotations, dims=(-2, -1)) for tensor in output]
    return output


class ChangeDetectionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str | Path | list[ManifestRecord],
        config: dict[str, Any],
        *,
        training: bool = False,
        validate_files: bool = True,
    ) -> None:
        self.config = config
        self.training = training
        self.use_alpha = bool(config.get("use_alphaearth", False))
        if isinstance(manifest, list):
            self.records = manifest
        else:
            self.records = read_manifest(
                manifest,
                validate_files=validate_files,
                require_alpha=self.use_alpha,
            )
        self.bands = config.get("bands")
        self.scale_divisor = config.get("scale_divisor", 255.0)
        self.mean = config.get("mean")
        self.std = config.get("std")
        augmentation = config.get("augmentation", {})
        self.geometry_probability = float(augmentation.get("geometry_probability", 0.5))
        self.spectral_jitter = float(augmentation.get("spectral_jitter", 0.0))

    def __len__(self) -> int:
        return len(self.records)

    def _load_alpha(self, record: ManifestRecord) -> list[torch.Tensor]:
        cache: dict[str, torch.Tensor] = {}
        result: list[torch.Tensor] = []
        for timestamp in record.timestamps:
            year = timestamp[:4]
            if year not in record.alpha_by_year:
                raise ValueError(f"记录 {record.id} 缺少 {year} 年 AlphaEarth")
            if year not in cache:
                alpha, _ = read_raster(record.alpha_by_year[year])
                tensor = torch.from_numpy(alpha.astype(np.float32, copy=False))
                tensor = F.normalize(tensor, p=2, dim=0, eps=1e-6)
                cache[year] = tensor
            result.append(cache[year].clone())
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        selected_bands = record.bands or self.bands
        images = [
            normalize_image(
                read_raster(path, selected_bands)[0],
                self.scale_divisor,
                self.mean,
                self.std,
            )
            for path in record.images
        ]
        expected_shape = images[0].shape[-2:]
        if any(image.shape[-2:] != expected_shape for image in images):
            raise ValueError(f"记录 {record.id} 的多时相影像尺寸不一致")

        mask_array, _ = read_raster(record.mask, [1])
        mask = torch.from_numpy((mask_array[:1] > 0).astype(np.float32))
        if mask.shape[-2:] != expected_shape:
            raise ValueError(f"记录 {record.id} 的标签尺寸与影像不一致")
        if record.valid_mask:
            valid_array, _ = read_raster(record.valid_mask, [1])
            valid = torch.from_numpy((valid_array[:1] > 0).astype(np.float32))
        else:
            valid = torch.ones_like(mask)

        alpha = self._load_alpha(record) if self.use_alpha else []
        if self.training:
            horizontal = random.random() < self.geometry_probability
            vertical = random.random() < self.geometry_probability
            rotations = random.randrange(4)
            combined = [*images, mask, valid, *alpha]
            combined = _apply_geometry(combined, horizontal, vertical, rotations)
            image_count = len(images)
            images = combined[:image_count]
            mask = combined[image_count]
            valid = combined[image_count + 1]
            alpha = combined[image_count + 2 :]
            if self.spectral_jitter > 0:
                jittered = []
                for image in images:
                    brightness = 1.0 + random.uniform(
                        -self.spectral_jitter, self.spectral_jitter
                    )
                    offset = random.uniform(
                        -self.spectral_jitter, self.spectral_jitter
                    )
                    jittered.append(image * brightness + offset)
                images = jittered

        return {
            "id": record.id,
            "images": torch.stack(images),
            "timestamps": record.timestamps,
            "mask": mask,
            "valid_mask": valid,
            "temporal_mask": torch.ones(len(images), dtype=torch.bool),
            "alpha": torch.stack(alpha) if alpha else None,
            "record": record,
        }


def collate_change_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("无法拼接空批次")
    batch_size = len(samples)
    max_time = max(sample["images"].shape[0] for sample in samples)
    channels = samples[0]["images"].shape[1]
    height, width = samples[0]["images"].shape[-2:]
    images = torch.zeros(batch_size, max_time, channels, height, width)
    temporal_mask = torch.zeros(batch_size, max_time, dtype=torch.bool)
    has_alpha = samples[0]["alpha"] is not None
    alpha = None
    if has_alpha:
        alpha_channels = samples[0]["alpha"].shape[1]
        alpha_height, alpha_width = samples[0]["alpha"].shape[-2:]
        alpha = torch.zeros(
            batch_size,
            max_time,
            alpha_channels,
            alpha_height,
            alpha_width,
        )

    for batch_index, sample in enumerate(samples):
        current = sample["images"]
        if current.shape[1:] != (channels, height, width):
            raise ValueError("同一批次中的影像通道或空间尺寸不一致")
        time_steps = current.shape[0]
        images[batch_index, :time_steps] = current
        temporal_mask[batch_index, :time_steps] = True
        if has_alpha:
            if sample["alpha"] is None:
                raise ValueError("同一批次不能混合有/无 AlphaEarth 的样本")
            if sample["alpha"].shape[0] != time_steps:
                raise ValueError("AlphaEarth 时相数量与影像不一致")
            alpha[batch_index, :time_steps] = sample["alpha"]
        elif sample["alpha"] is not None:
            raise ValueError("同一批次不能混合有/无 AlphaEarth 的样本")

    return {
        "id": [sample["id"] for sample in samples],
        "images": images,
        "mask": torch.stack([sample["mask"] for sample in samples]),
        "valid_mask": torch.stack([sample["valid_mask"] for sample in samples]),
        "temporal_mask": temporal_mask,
        "timestamps": [sample["timestamps"] for sample in samples],
        "alpha": alpha,
        "records": [sample["record"] for sample in samples],
    }
