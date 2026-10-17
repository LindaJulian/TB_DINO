from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tb_dino.config import load_config
from tb_dino.data.raster import (
    align_to_reference,
    l2_normalize_channels,
    write_geotiff,
)
from tb_dino.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproject and align AlphaEarth onto the grid of the remote sensing image"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    align = config.get("alphaearth_align", config)
    reference = Path(align["reference"]).expanduser().resolve()
    output_dir = Path(align["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for year, source in align["inputs"].items():
        array, profile = align_to_reference(source, reference)
        array = l2_normalize_channels(array)
        output = output_dir / f"alphaearth_{year}_aligned.tif"
        write_geotiff(output, array, profile, dtype="float32")
        outputs[str(year)] = str(output)
        print(f"{year}: {output}")
    write_json(outputs, output_dir / "alphaearth_aligned.json")


if __name__ == "__main__":
    main()

