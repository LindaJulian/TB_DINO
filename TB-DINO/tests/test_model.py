from __future__ import annotations

import torch

from tb_dino.data.dataset import collate_change_batch
from tb_dino.losses import compute_loss
from tb_dino.models import build_model


def model_config(use_alpha: bool) -> dict:
    return {
        "fake_backbone": True,
        "fake_dim": 32,
        "patch_size": 16,
        "input_channels": 3,
        "hidden_dim": 32,
        "use_alphaearth": use_alpha,
        "alpha_channels": 64,
        "atim_heads": 2,
        "sca": {"max_samples": 32, "hard_negatives": 16},
    }


def test_variable_time_collate() -> None:
    samples = []
    for time in (2, 5):
        samples.append(
            {
                "id": str(time),
                "images": torch.randn(time, 3, 32, 32),
                "mask": torch.zeros(1, 32, 32),
                "valid_mask": torch.ones(1, 32, 32),
                "temporal_mask": torch.ones(time, dtype=torch.bool),
                "timestamps": [str(index) for index in range(time)],
                "alpha": None,
                "record": None,
            }
        )
    batch = collate_change_batch(samples)
    assert batch["images"].shape == (2, 5, 3, 32, 32)
    assert batch["temporal_mask"].tolist() == [
        [True, True, False, False, False],
        [True, True, True, True, True],
    ]


def test_forward_without_alpha() -> None:
    model = build_model(model_config(False))
    images = torch.randn(2, 3, 3, 32, 32)
    temporal_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    output = model(images, temporal_mask)
    assert output.logits.shape == (2, 1, 32, 32)
    output.logits.mean().backward()
    assert any(
        parameter.grad is not None
        for parameter in model.atim.parameters()
        if parameter.requires_grad
    )


def test_forward_with_alpha_and_sca_edge_cases() -> None:
    model = build_model(model_config(True))
    images = torch.randn(1, 4, 3, 32, 32)
    alpha = torch.randn(1, 4, 64, 16, 16)
    temporal_mask = torch.ones(1, 4, dtype=torch.bool)
    for mask_value in (0.0, 1.0):
        mask = torch.full((1, 1, 32, 32), mask_value)
        valid = torch.ones_like(mask)
        output = model(
            images,
            temporal_mask,
            alpha=alpha,
            change_mask=mask,
            valid_mask=valid,
        )
        loss, values = compute_loss(
            output,
            mask,
            valid,
            {"pos_weight": "auto", "dice_weight": 1.0, "sca_weight": 0.1},
        )
        assert torch.isfinite(loss)
        assert values["sca_negative"] == 0.0

