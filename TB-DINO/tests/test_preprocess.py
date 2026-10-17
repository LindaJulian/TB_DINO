from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tb_dino.data.manifest import read_manifest
from tb_dino.data.preprocess import preprocess_bitemporal, preprocess_timeseries
from tb_dino.data.raster import read_raster, write_geotiff


def test_bitemporal_preprocessing(tmp_path: Path) -> None:
    directories = {
        name: tmp_path / name for name in ("A", "B", "label")
    }
    for directory in directories.values():
        directory.mkdir()
    profile = {
        "driver": "GTiff",
        "height": 40,
        "width": 40,
        "count": 3,
        "dtype": "uint8",
    }
    for index in range(5):
        name = f"scene_{index}.tif"
        first = np.zeros((3, 40, 40), np.uint8)
        second = first.copy()
        second[:, 10:20, 10:20] = 255
        label = np.zeros((1, 40, 40), np.uint8)
        label[:, 10:20, 10:20] = 1
        write_geotiff(directories["A"] / name, first, profile)
        write_geotiff(directories["B"] / name, second, profile)
        write_geotiff(
            directories["label"] / name,
            label,
            {**profile, "count": 1},
        )
    output = tmp_path / "prepared"
    counts = preprocess_bitemporal(
        {
            "image_a_dir": str(directories["A"]),
            "image_b_dir": str(directories["B"]),
            "mask_dir": str(directories["label"]),
            "output_dir": str(output),
            "patch_size": 32,
            "stride": 32,
            "seed": 1,
            "split": {"train": 0.6, "val": 0.2, "test": 0.2},
        }
    )
    assert sum(counts.values()) == 20
    records = read_manifest(output / "train.jsonl", validate_files=True)
    assert all(len(record.images) == 2 for record in records)


def test_timeseries_cumulative_monthly_target(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    image_paths = []
    label_paths = []
    profile = {
        "driver": "GTiff",
        "height": 16,
        "width": 16,
        "count": 3,
        "dtype": "uint8",
    }
    label_profile = {**profile, "count": 1}
    labels = []
    for month in range(3):
        image_path = scene_dir / f"image_{month}.tif"
        label_path = scene_dir / f"label_{month}.tif"
        image = np.zeros((3, 16, 16), dtype=np.uint8)
        label = np.zeros((1, 16, 16), dtype=np.uint8)
        if month == 1:
            label[:, 4:8, 4:8] = 1
        write_geotiff(image_path, image, profile)
        write_geotiff(label_path, label, label_profile)
        image_paths.append(str(image_path))
        label_paths.append(str(label_path))
        labels.append(label)

    sequences = tmp_path / "sequences.jsonl"
    sequences.write_text(
        json.dumps(
            {
                "id": "scene",
                "split": "train",
                "images": image_paths,
                "timestamps": ["2020-01-01", "2020-02-01", "2020-03-01"],
                "labels": label_paths,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "prepared"
    counts = preprocess_timeseries(
        {
            "sequences_file": str(sequences),
            "output_dir": str(output),
            "patch_size": 16,
            "stride": 16,
            "target_mode": "cumulative_monthly",
            "split": {"train": 0.6, "val": 0.2, "test": 0.2},
            "seed": 1,
        }
    )
    assert counts == {"train": 1, "val": 0, "test": 0}
    records = read_manifest(output / "train.jsonl", validate_files=True)
    assert records[0].metadata["target_mode"] == "cumulative_monthly"
    assert records[0].metadata["transition_count"] == 2

    mask, _ = read_raster(records[0].mask)
    # The middle-month change must remain positive even though labels[0]==labels[2].
    assert mask[0, 4:8, 4:8].all()

