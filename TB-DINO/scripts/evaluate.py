from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tb_dino.config import load_config
from tb_dino.engine import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation TB-DINO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--fixed-threshold", action="store_true")
    parser.add_argument("--write-threshold", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    metrics = run_evaluation(
        config,
        args.checkpoint,
        manifest=args.manifest,
        search_threshold=not args.fixed_threshold,
        write_threshold=args.write_threshold,
    )
    print("Evaluation completed:", metrics)


if __name__ == "__main__":
    main()

