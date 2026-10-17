from __future__ import annotations

from pathlib import Path

import rasterio

from scripts.make_synthetic_data import create_synthetic_dataset
from tb_dino.data.manifest import read_manifest
from tb_dino.engine import run_training
from tb_dino.inference import run_inference


def test_synthetic_train_and_infer(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    create_synthetic_dataset(
        data_dir,
        time_steps=3,
        image_size=32,
        use_alpha=True,
        seed=7,
    )
    output_dir = tmp_path / "outputs"
    config = {
        "seed": 7,
        "deterministic": True,
        "output_dir": str(output_dir),
        "data": {
            "train_manifest": str(data_dir / "train.jsonl"),
            "val_manifest": str(data_dir / "val.jsonl"),
            "test_manifest": str(data_dir / "test.jsonl"),
            "bands": [1, 2, 3],
            "scale_divisor": 255.0,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.25, 0.25, 0.25],
            "batch_size": 2,
            "eval_batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "augmentation": {
                "geometry_probability": 0.0,
                "spectral_jitter": 0.0,
            },
        },
        "model": {
            "fake_backbone": True,
            "fake_dim": 16,
            "patch_size": 16,
            "input_channels": 3,
            "hidden_dim": 16,
            "use_alphaearth": True,
            "alpha_channels": 64,
            "atim_heads": 2,
            "sca": {"max_samples": 16, "hard_negatives": 8},
        },
        "loss": {
            "pos_weight": "auto",
            "dice_weight": 1.0,
            "sca_weight": 0.1,
        },
        "training": {
            "device": "cpu",
            "epochs": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_accumulation": 1,
            "gradient_clip": 1.0,
            "amp": False,
            "threshold": 0.5,
        },
        "inference": {
            "mode": "manifest",
            "device": "cpu",
            "batch_size": 1,
            "write_png": True,
        },
    }
    metrics = run_training(config)
    assert "f1" in metrics
    checkpoint = output_dir / "best.pt"
    assert checkpoint.exists()
    prediction_dir = tmp_path / "predictions"
    inference_metrics = run_inference(
        config,
        checkpoint,
        output_dir=prediction_dir,
        manifest=data_dir / "test.jsonl",
    )
    assert "f1" in inference_metrics
    assert list(prediction_dir.glob("*/change.tif"))

    record = read_manifest(data_dir / "test.jsonl")[0]
    raster_config = {
        **config,
        "inference": {
            "mode": "raster",
            "device": "cpu",
            "images": record.images,
            "timestamps": record.timestamps,
            "alpha_by_year": record.alpha_by_year,
            "tile_size": 32,
            "stride": 16,
            "write_png": True,
        },
    }
    raster_output = tmp_path / "raster_prediction"
    result = run_inference(raster_config, checkpoint, output_dir=raster_output)
    assert Path(result["mask"]).exists()
    with rasterio.open(record.images[0]) as source, rasterio.open(
        result["mask"]
    ) as prediction:
        assert prediction.shape == source.shape
        assert prediction.crs == source.crs
        assert prediction.transform == source.transform
