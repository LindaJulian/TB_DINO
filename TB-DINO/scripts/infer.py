from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tb_dino.config import load_config
from tb_dino.inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="TB-DINO infer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    result = run_inference(
        config,
        args.checkpoint,
        output_dir=args.output_dir,
        manifest=args.manifest,
    )
    print("complete:", result)


if __name__ == "__main__":
    main()

