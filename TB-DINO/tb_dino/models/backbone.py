from __future__ import annotations

from typing import Any

import torch
from torch import nn


class TinyPatchBackbone(nn.Module):
    """Small offline backbone used by tests and synthetic smoke runs."""

    def __init__(self, output_dim: int = 64, patch_size: int = 16) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            3, output_dim, kernel_size=patch_size, stride=patch_size
        )
        self.refine = nn.Sequential(
            nn.GroupNorm(8 if output_dim % 8 == 0 else 1, output_dim),
            nn.GELU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.refine(self.projection(images))


class DINOv3Encoder(nn.Module):
    """Hugging Face DINOv3 wrapper that returns a dense patch feature map."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "使用真实 DINOv3 需要安装 transformers。请执行 pip install -e ."
            ) from exc

        source = config.get(
            "backbone", "facebook/dinov3-vitb16-pretrain-lvd1689m"
        )
        kwargs = {
            "local_files_only": bool(config.get("local_files_only", False)),
            "trust_remote_code": bool(config.get("trust_remote_code", False)),
        }
        revision = config.get("revision")
        if revision:
            kwargs["revision"] = revision
        self.model = AutoModel.from_pretrained(source, **kwargs)
        model_config = self.model.config
        self.output_dim = int(
            getattr(model_config, "hidden_size", config.get("feature_dim", 384))
        )
        raw_patch_size = getattr(model_config, "patch_size", config.get("patch_size", 16))
        if isinstance(raw_patch_size, (tuple, list)):
            if raw_patch_size[0] != raw_patch_size[1]:
                raise ValueError("当前实现只支持正方形 DINOv3 patch")
            raw_patch_size = raw_patch_size[0]
        self.patch_size = int(raw_patch_size)

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        if config.get("gradient_checkpointing", False):
            enable = getattr(self.model, "gradient_checkpointing_enable", None)
            if callable(enable):
                enable()

        if config.get("use_lora", True):
            self._inject_lora(config.get("lora", {}))
            if config.get("gradient_checkpointing", False):
                enable_inputs = getattr(self.model, "enable_input_require_grads", None)
                if callable(enable_inputs):
                    enable_inputs()

    def _inject_lora(self, config: dict[str, Any]) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "LoRA 需要安装 peft。请执行 pip install -e ."
            ) from exc
        requested = list(
            config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"],
            )
        )
        module_names = [name for name, _ in self.model.named_modules()]
        targets = [
            target
            for target in requested
            if any(
                name == target or name.endswith("." + target) for name in module_names
            )
        ]
        if not targets:
            for fallback in (["qkv"], ["query", "key", "value"]):
                discovered = [
                    target
                    for target in fallback
                    if any(
                        name == target or name.endswith("." + target)
                        for name in module_names
                    )
                ]
                if discovered:
                    targets = discovered
                    break
        if not targets:
            candidates = sorted(
                {
                    name.rsplit(".", 1)[-1]
                    for name, module in self.model.named_modules()
                    if isinstance(module, nn.Linear)
                }
            )
            raise RuntimeError(
                "未找到任何 LoRA 目标层。"
                f"请求={requested}，可用线性层后缀={candidates}"
            )
        lora_config = LoraConfig(
            r=int(config.get("rank", 8)),
            lora_alpha=int(config.get("alpha", 16)),
            lora_dropout=float(config.get("dropout", 0.05)),
            target_modules=targets,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        trainable_lora = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
        if not trainable_lora:
            raise RuntimeError("LoRA 注入后没有可训练适配器参数")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"输入尺寸 {(height, width)} 必须是 patch_size={self.patch_size} 的倍数"
            )
        outputs = self.model(pixel_values=images, return_dict=True)
        tokens = outputs.last_hidden_state
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        patch_count = patch_height * patch_width
        if tokens.shape[1] < patch_count:
            raise RuntimeError(
                f"DINOv3 输出 token 数 {tokens.shape[1]} 小于空间 token 数 {patch_count}"
            )
        # DINOv3 special tokens precede patch tokens; selecting the tail is robust
        # to different numbers of CLS/register tokens.
        patch_tokens = tokens[:, -patch_count:, :]
        return patch_tokens.transpose(1, 2).reshape(
            images.shape[0], self.output_dim, patch_height, patch_width
        )


def build_backbone(config: dict[str, Any]) -> nn.Module:
    if config.get("fake_backbone", False):
        return TinyPatchBackbone(
            output_dim=int(config.get("fake_dim", 64)),
            patch_size=int(config.get("patch_size", 16)),
        )
    return DINOv3Encoder(config)
