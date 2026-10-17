from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SCAResult:
    fused: torch.Tensor
    alpha_features: torch.Tensor
    positive_loss: torch.Tensor
    negative_loss: torch.Tensor

    @property
    def loss(self) -> torch.Tensor:
        return self.positive_loss + self.negative_loss


class SimilarityGuidedContrastiveAlignment(nn.Module):
    def __init__(
        self,
        dim: int,
        alpha_channels: int = 64,
        margin: float = 0.2,
        max_samples: int = 512,
        hard_negatives: int = 256,
        negative_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.dino_projection = nn.Conv2d(dim, dim, kernel_size=1)
        self.alpha_projection = nn.Conv2d(alpha_channels, dim, kernel_size=1)
        self.gate = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.margin = margin
        self.max_samples = max_samples
        self.hard_negatives = hard_negatives
        self.negative_weight = negative_weight

    def _sample(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] <= self.max_samples:
            return values
        indexes = torch.randperm(values.shape[0], device=values.device)[
            : self.max_samples
        ]
        return values[indexes]

    def _hard_negative_loss(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        if first.numel() == 0 or second.numel() == 0:
            return first.new_zeros(())
        first = self._sample(first)
        second = self._sample(second)
        similarities = first @ second.transpose(0, 1)
        count = min(self.hard_negatives, similarities.numel())
        if count == 0:
            return similarities.new_zeros(())
        hardest = torch.topk(similarities.flatten(), count).values
        return F.relu(hardest - self.margin).mean()

    def forward(
        self,
        dino_features: torch.Tensor,
        alpha: torch.Tensor,
        temporal_mask: torch.Tensor,
        change_mask: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> SCAResult:
        batch, time, channels, height, width = dino_features.shape
        if alpha.shape[:2] != (batch, time):
            raise ValueError("AlphaEarth 与 DINO 时相维度不一致")
        flat_dino = dino_features.reshape(batch * time, channels, height, width)
        flat_alpha = alpha.reshape(
            batch * time, alpha.shape[2], alpha.shape[3], alpha.shape[4]
        )
        if flat_alpha.shape[-2:] != (height, width):
            flat_alpha = F.interpolate(
                flat_alpha,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        projected_dino = self.dino_projection(flat_dino)
        projected_alpha = self.alpha_projection(flat_alpha)
        normalized_dino = F.normalize(projected_dino, p=2, dim=1, eps=1e-6)
        normalized_alpha = F.normalize(projected_alpha, p=2, dim=1, eps=1e-6)

        normalized_dino = normalized_dino.reshape(
            batch, time, channels, height, width
        )
        normalized_alpha = normalized_alpha.reshape(
            batch, time, channels, height, width
        )
        time_valid = temporal_mask[:, :, None, None].expand(
            batch, time, height, width
        )
        if valid_mask is not None:
            spatial_valid = F.interpolate(
                valid_mask.float(), size=(height, width), mode="nearest"
            )[:, 0].bool()
            pair_valid = time_valid & spatial_valid[:, None]
        else:
            spatial_valid = torch.ones(
                batch, height, width, dtype=torch.bool, device=dino_features.device
            )
            pair_valid = time_valid

        cosine = (normalized_dino * normalized_alpha).sum(dim=2)
        positive_loss = (
            (1.0 - cosine)[pair_valid].mean()
            if pair_valid.any()
            else cosine.new_zeros(())
        )

        change = F.interpolate(
            change_mask.float(), size=(height, width), mode="nearest"
        )[:, 0].bool()
        change = change[:, None].expand(batch, time, height, width)
        dino_points = normalized_dino.permute(0, 1, 3, 4, 2)[pair_valid]
        alpha_points = normalized_alpha.permute(0, 1, 3, 4, 2)[pair_valid]
        states = change[pair_valid]
        changed_dino = dino_points[states]
        unchanged_dino = dino_points[~states]
        changed_alpha = alpha_points[states]
        unchanged_alpha = alpha_points[~states]
        negative_loss = 0.5 * (
            self._hard_negative_loss(changed_dino, unchanged_alpha)
            + self._hard_negative_loss(unchanged_dino, changed_alpha)
        )
        negative_loss = negative_loss * self.negative_weight

        gate = torch.sigmoid(
            self.gate(torch.cat([projected_dino, projected_alpha], dim=1))
        )
        fused = flat_dino + gate * projected_alpha
        fused = fused.reshape(batch, time, channels, height, width)
        alpha_features = projected_alpha.reshape(
            batch, time, channels, height, width
        )
        valid_values = temporal_mask[:, :, None, None, None].to(fused.dtype)
        return SCAResult(
            fused=fused * valid_values,
            alpha_features=alpha_features * valid_values,
            positive_loss=positive_loss,
            negative_loss=negative_loss,
        )

