from __future__ import annotations

from pathlib import Path

import numpy as np

from tb_dino.config import load_config
from tb_dino.data.manifest import ManifestRecord, read_manifest, write_manifest
from tb_dino.data.raster import write_geotiff


def test_config_inheritance_and_override(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "model:\n  hidden_dim: 64\ntraining:\n  epochs: 2\n",
        encoding="utf-8",
    )
    (tmp_path / "child.yaml").write_text(
        "_base_: base.yaml\nmodel:\n  use_alphaearth: true\n",
        encoding="utf-8",
    )
    config = load_config(
        tmp_path / "child.yaml",
        ["model.hidden_dim=32", "training.amp=false"],
    )
    assert config["model"]["hidden_dim"] == 32
    assert config["model"]["use_alphaearth"] is True
    assert config["training"]["epochs"] == 2
    assert config["training"]["amp"] is False


def test_manifest_relative_paths(tmp_path: Path) -> None:
    profile = {
        "driver": "GTiff",
        "height": 16,
        "width": 16,
        "count": 3,
        "dtype": "uint8",
    }
    image0 = tmp_path / "t0.tif"
    image1 = tmp_path / "t1.tif"
    mask = tmp_path / "mask.tif"
    write_geotiff(image0, np.zeros((3, 16, 16), np.uint8), profile)
    write_geotiff(image1, np.zeros((3, 16, 16), np.uint8), profile)
    write_geotiff(mask, np.zeros((1, 16, 16), np.uint8), {**profile, "count": 1})
    record = ManifestRecord(
        id="sample",
        images=[str(image0), str(image1)],
        timestamps=["t0", "t1"],
        mask=str(mask),
    )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest([record], manifest)
    loaded = read_manifest(manifest, validate_files=True)
    assert loaded[0].id == "sample"
    assert loaded[0].images == [str(image0.resolve()), str(image1.resolve())]

