from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tb_dino.config import load_config
from tb_dino.engine import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="train TB-DINO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--resume", help="path")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if args.resume:
        config.setdefault("training", {})["resume"] = args.resume
    metrics = run_training(config)
    print("finish:", metrics)


if __name__ == "__main__":
    main()

