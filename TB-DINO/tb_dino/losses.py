from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .models.network import TBDINOOutput


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    if valid_mask is None:
        valid_mask = torch.ones_like(target)
    probabilities = probabilities * valid_mask
    target = target * valid_mask
    dimensions = tuple(range(1, target.ndim))
    intersection = (probabilities * target).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + target.sum(dim=dimensions)
    score = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - score.mean()


def _resolve_pos_weight(
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    value: float | str | None,
    maximum: float,
) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() != "auto":
            raise ValueError(f"未知 pos_weight 配置: {value}")
        positives = (target * valid_mask).sum()
        negatives = ((1.0 - target) * valid_mask).sum()
        return (negatives / positives.clamp_min(1.0)).clamp(1.0, maximum)
    return target.new_tensor(float(value))


def compute_loss(
    output: TBDINOOutput,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    pos_weight = _resolve_pos_weight(
        target,
        valid_mask,
        config.get("pos_weight", "auto"),
        float(config.get("max_pos_weight", 20.0)),
    )
    bce = F.binary_cross_entropy_with_logits(
        output.logits,
        target,
        reduction="none",
        pos_weight=pos_weight,
    )
    bce = (bce * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    dice = dice_loss(output.logits, target, valid_mask)
    dice_weight = float(config.get("dice_weight", 1.0))
    sca_weight = float(config.get("sca_weight", 0.1))
    segmentation = bce + dice_weight * dice
    total = segmentation + sca_weight * output.sca_loss
    values = {
        "loss": float(total.detach()),
        "bce": float(bce.detach()),
        "dice": float(dice.detach()),
        "sca_positive": float(output.sca_positive_loss.detach()),
        "sca_negative": float(output.sca_negative_loss.detach()),
    }
    return total, values

