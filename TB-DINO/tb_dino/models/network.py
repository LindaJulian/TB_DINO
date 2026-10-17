from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .atim import AdaptiveTemporalInteraction
from .backbone import build_backbone
from .sca import SimilarityGuidedContrastiveAlignment


def _group_count(channels: int) -> int:
    for value in (32, 16, 8, 4, 2):
        if channels % value == 0:
            return value
    return 1


@dataclass
class TBDINOOutput:
    logits: torch.Tensor
    sca_positive_loss: torch.Tensor
    sca_negative_loss: torch.Tensor

    @property
    def sca_loss(self) -> torch.Tensor:
        return self.sca_positive_loss + self.sca_negative_loss


class TBDINO(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        input_channels = int(config.get("input_channels", 3))
        self.input_adapter: nn.Module
        if input_channels == 3:
            self.input_adapter = nn.Identity()
        else:
            self.input_adapter = nn.Conv2d(input_channels, 3, kernel_size=1)

        self.backbone = build_backbone(config)
        feature_dim = int(getattr(self.backbone, "output_dim"))
        self.patch_size = int(getattr(self.backbone, "patch_size"))
        hidden_dim = int(config.get("hidden_dim", 128))
        self.hidden_dim = hidden_dim
        self.feature_projection = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
        )
        self.atim = AdaptiveTemporalInteraction(
            hidden_dim,
            num_heads=int(config.get("atim_heads", 2)),
            dropout=float(config.get("atim_dropout", 0.1)),
            mlp_ratio=float(config.get("atim_mlp_ratio", 2.0)),
        )
        self.use_alpha = bool(config.get("use_alphaearth", False))
        if self.use_alpha:
            sca_config = config.get("sca", {})
            self.sca = SimilarityGuidedContrastiveAlignment(
                hidden_dim,
                alpha_channels=int(config.get("alpha_channels", 64)),
                margin=float(sca_config.get("margin", 0.2)),
                max_samples=int(sca_config.get("max_samples", 512)),
                hard_negatives=int(sca_config.get("hard_negatives", 256)),
                negative_weight=float(sca_config.get("negative_weight", 1.0)),
            )
            change_channels = hidden_dim * 5
        else:
            self.sca = None
            change_channels = hidden_dim * 4
        self.change_reduction = nn.Sequential(
            nn.Conv2d(change_channels, hidden_dim, kernel_size=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
        )
        decoder_dim = max(hidden_dim // 2, 32)
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, decoder_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(decoder_dim), decoder_dim),
            nn.GELU(),
            nn.Conv2d(decoder_dim, 1, kernel_size=1),
        )
        self.encoder_chunk_size = int(config.get("encoder_chunk_size", 0))

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = images.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"输入尺寸 {(height, width)} 必须是 {self.patch_size} 的倍数"
            )
        flattened = images.reshape(batch * time, channels, height, width)
        flattened = self.input_adapter(flattened)
        chunk_size = self.encoder_chunk_size or flattened.shape[0]
        encoded_chunks = []
        for start in range(0, flattened.shape[0], chunk_size):
            encoded_chunks.append(self.backbone(flattened[start : start + chunk_size]))
        encoded = torch.cat(encoded_chunks, dim=0)
        encoded = self.feature_projection(encoded)
        return encoded.reshape(
            batch, time, self.hidden_dim, encoded.shape[-2], encoded.shape[-1]
        )

    @staticmethod
    def _endpoints(
        features: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = features.shape[0]
        valid_lengths = temporal_mask.sum(dim=1)
        if (valid_lengths < 2).any():
            raise ValueError("每个变化检测样本至少需要两个有效时相")
        batch_indexes = torch.arange(batch, device=features.device)
        first = features[:, 0]
        last = features[batch_indexes, valid_lengths - 1]
        weights = temporal_mask[:, :, None, None, None].to(features.dtype)
        mean = (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return first, last, mean

    def forward(
        self,
        images: torch.Tensor,
        temporal_mask: torch.Tensor,
        *,
        alpha: torch.Tensor | None = None,
        change_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> TBDINOOutput:
        input_size = images.shape[-2:]
        features = self._encode(images)
        temporal_values = temporal_mask[:, :, None, None, None].to(features.dtype)
        features = features * temporal_values
        zero = features.new_zeros(())
        alpha_features = None
        positive_loss = zero
        negative_loss = zero

        if self.use_alpha:
            if alpha is None:
                raise ValueError("当前模型配置启用了 AlphaEarth，但输入中没有 alpha")
            if change_mask is None:
                change_mask = torch.zeros(
                    images.shape[0],
                    1,
                    images.shape[-2],
                    images.shape[-1],
                    device=images.device,
                )
            assert self.sca is not None
            alignment = self.sca(
                features,
                alpha,
                temporal_mask,
                change_mask,
                valid_mask,
            )
            features = alignment.fused
            alpha_features = alignment.alpha_features
            positive_loss = alignment.positive_loss
            negative_loss = alignment.negative_loss

        contextual = self.atim(features, temporal_mask)
        first, last, mean = self._endpoints(contextual, temporal_mask)
        parts = [first, last, torch.abs(last - first), mean]
        if alpha_features is not None:
            alpha_first, alpha_last, _ = self._endpoints(
                alpha_features, temporal_mask
            )
            parts.append(torch.abs(alpha_last - alpha_first))
        change = self.change_reduction(torch.cat(parts, dim=1))
        logits = self.decoder(change)
        if logits.shape[-2:] != input_size:
            logits = F.interpolate(
                logits, size=input_size, mode="bilinear", align_corners=False
            )
        return TBDINOOutput(
            logits=logits,
            sca_positive_loss=positive_loss,
            sca_negative_loss=negative_loss,
        )


def build_model(config: dict[str, Any]) -> TBDINO:
    return TBDINO(config)

