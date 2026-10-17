from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tb_dino.data.manifest import ManifestRecord, write_manifest
from tb_dino.data.raster import l2_normalize_channels, write_geotiff


def create_synthetic_dataset(
    output_dir: str | Path,
    *,
    time_steps: int = 6,
    image_size: int = 64,
    use_alpha: bool = True,
    seed: int = 42,
) -> dict[str, int]:
    output_dir = Path(output_dir).expanduser().resolve()
    rng = np.random.default_rng(seed)
    split_counts = {"train": 6, "val": 2, "test": 2}
    transform = from_origin(100.0, 50.0, 10.0, 10.0)
    profile = {
        "driver": "GTiff",
        "height": image_size,
        "width": image_size,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": transform,
    }
    timestamps = [
        f"{2018 + index // 12}-{index % 12 + 1:02d}-01"
        for index in range(time_steps)
    ]
    for split, count in split_counts.items():
        records = []
        for sample_index in range(count):
            sample_id = f"{split}_{sample_index:03d}"
            sample_dir = output_dir / "samples" / sample_id
            base = rng.integers(
                20, 180, size=(3, image_size, image_size), dtype=np.uint8
            )
            square_size = max(8, image_size // 4)
            y = int(rng.integers(0, image_size - square_size + 1))
            x = int(rng.integers(0, image_size - square_size + 1))
            image_paths = []
            for time_index in range(time_steps):
                image = base.astype(np.float32)
                image += rng.normal(0, 3, size=image.shape)
                if time_index == time_steps - 1:
                    image[:, y : y + square_size, x : x + square_size] += 60
                image = np.clip(image, 0, 255).astype(np.uint8)
                path = sample_dir / f"t{time_index:03d}.tif"
                write_geotiff(path, image, profile)
                image_paths.append(str(path))
            mask = np.zeros((1, image_size, image_size), dtype=np.uint8)
            mask[:, y : y + square_size, x : x + square_size] = 1
            valid = np.ones_like(mask, dtype=np.uint8)
            mask_profile = dict(profile)
            mask_profile["count"] = 1
            mask_profile["dtype"] = "uint8"
            mask_path = sample_dir / "mask.tif"
            valid_path = sample_dir / "valid.tif"
            write_geotiff(mask_path, mask, mask_profile, nodata=255)
            write_geotiff(valid_path, valid, mask_profile, nodata=0)
            alpha_by_year = {}
            if use_alpha:
                for year in sorted({timestamp[:4] for timestamp in timestamps}):
                    alpha = rng.normal(
                        size=(64, image_size, image_size)
                    ).astype(np.float32)
                    alpha[:, y : y + square_size, x : x + square_size] += 0.25
                    alpha = l2_normalize_channels(alpha)
                    alpha_profile = dict(profile)
                    alpha_profile["count"] = 64
                    alpha_profile["dtype"] = "float32"
                    alpha_path = sample_dir / f"alpha_{year}.tif"
                    write_geotiff(alpha_path, alpha, alpha_profile)
                    alpha_by_year[year] = str(alpha_path)
            records.append(
                ManifestRecord(
                    id=sample_id,
                    images=image_paths,
                    timestamps=timestamps,
                    mask=str(mask_path),
                    valid_mask=str(valid_path),
                    alpha_by_year=alpha_by_year,
                    bands=[1, 2, 3],
                )
            )
        write_manifest(records, output_dir / f"{split}.jsonl")
    return split_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="TB-DINO")
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--time-steps", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--no-alpha", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = create_synthetic_dataset(
        args.output_dir,
        time_steps=args.time_steps,
        image_size=args.image_size,
        use_alpha=not args.no_alpha,
        seed=args.seed,
    )
    print("complete:", counts)


if __name__ == "__main__":
    main()

