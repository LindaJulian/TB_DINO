from __future__ import annotations

import os

import pytest
import torch

from tb_dino.models import build_model


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("TB_DINO_RUN_DINOV3") != "1",
    reason="设置 TB_DINO_RUN_DINOV3=1 后才下载并测试真实 DINOv3",
)
def test_real_dinov3_lora_forward() -> None:
    model = build_model(
        {
            "backbone": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "input_channels": 3,
            "hidden_dim": 64,
            "use_lora": True,
            "gradient_checkpointing": False,
            "lora": {
                "rank": 4,
                "alpha": 8,
                "dropout": 0.0,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "use_alphaearth": False,
        }
    )
    trainable_backbone = [
        name
        for name, parameter in model.backbone.named_parameters()
        if parameter.requires_grad
    ]
    assert trainable_backbone
    assert all("lora_" in name for name in trainable_backbone)
    output = model(
        torch.randn(1, 2, 3, 64, 64),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert output.logits.shape == (1, 1, 64, 64)

