"""Zero-shot baseline: evaluate the pretrained DimeNet++ IS2RE-100k checkpoint on
the held-out test split without any fine-tuning.

Usage (from the project root):
    python scripts/run_zero_shot_eval.py \
        --test-lmdb data/subsample/test.lmdb \
        --cache-dir .cache/checkpoints \
        --out results/zero_shot.txt
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import common


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot IS2RE evaluation")
    p.add_argument("--test-lmdb", required=True, help="Path to held-out test LMDB")
    p.add_argument("--cache-dir", default=".cache/checkpoints")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="Optional path to write results")

    # W&B
    p.add_argument("--wandb-project", default="is2re-finetune")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    common.set_seed(args.seed)
    device = common.get_device(args.cpu)

    ckpt_path = common.download_checkpoint(args.cache_dir)
    model = common.build_model(freeze_backbone=False)
    model, normalizer = common.load_pretrained(model, ckpt_path, device)

    loader = common.make_loader(
        args.test_lmdb, args.batch_size, shuffle=False, split="test"
    )
    metrics = common.evaluate(model, loader, normalizer, device)

    logger = common.WandBLogger(
        project=args.wandb_project,
        name="zero_shot",
        tags=["zero_shot"],
        config={"variant": "zero_shot"},
        enabled=not args.no_wandb,
    )
    logger.log(
        {
            "test_energy_mae": metrics["energy_mae"],
            "test_energy_mse": metrics["energy_mse"],
            "test_energy_within_threshold": metrics["energy_within_threshold"],
        }
    )
    logger.finish()

    lines = [
        f"zero_shot test MAE                = {metrics['energy_mae']:.6f} eV",
        f"zero_shot test MSE                = {metrics['energy_mse']:.6f} eV^2",
        f"zero_shot test EwT (0.02 eV)      = {metrics['energy_within_threshold']:.6f}",
    ]
    for line in lines:
        print(line)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
