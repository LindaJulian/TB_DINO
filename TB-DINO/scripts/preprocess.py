from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tb_dino.config import load_config
from tb_dino.data.preprocess import run_preprocessing


def main() -> None:
    parser = argparse.ArgumentParser(description="TB-DINO data preprocessing")
    parser.add_argument("--config", required=True, help="YAML Configuration path")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        help="preprocess.patch_size=128",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    preprocess_config = config.get("preprocess", config)
    result = run_preprocessing(preprocess_config)
    print("finish:", result)


if __name__ == "__main__":
    main()

