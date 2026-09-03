"""Evaluate a saved fine-tuned checkpoint on the held-out test split.

This is the shared evaluation used after each fine-tuning variant (frozen-backbone
and full). It loads a checkpoint produced by ``scripts/common.save_checkpoint``
(which stores ``state_dict``, ``normalizers``, ``arch`` and ``freeze_backbone``)
and reports MAE / MSE / energy-within-threshold.

Usage (from the project root):
    python scripts/eval_test.py \
        --ckpt runs/frozen_head/best_checkpoint.pt \
        --test-lmdb data/subsample/test.lmdb \
        --out results/frozen_head_test.txt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import common


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned checkpoint on test")
    p.add_argument("--ckpt", required=True, help="Path to best_checkpoint.pt")
    p.add_argument("--test-lmdb", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)

    # W&B
    p.add_argument("--wandb-project", default="is2re-finetune")
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    common.set_seed(args.seed)
    device = common.get_device(args.cpu)

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    arch = checkpoint["arch"]
    freeze_backbone = checkpoint["freeze_backbone"]

    model = common.build_model(freeze_backbone=freeze_backbone)
    common.load_state_dict(model, checkpoint["state_dict"], strict=True)
    normalizer = common.create_normalizer(
        state_dict=checkpoint["normalizers"]["target"]
    )
    model = model.to(device)
    normalizer = normalizer.to(device)

    loader = common.make_loader(
        args.test_lmdb, args.batch_size, shuffle=False, split="test"
    )
    metrics = common.evaluate(model, loader, normalizer, device)

    tag = "full" if not freeze_backbone else "frozen_backbone"
    logger = common.WandBLogger(
        project=args.wandb_project,
        name=f"{tag}_finetune_test",
        tags=[tag],
        config={"variant": f"{tag}_finetune_test"},
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
        f"{tag} test MAE                = {metrics['energy_mae']:.6f} eV",
        f"{tag} test MSE                = {metrics['energy_mse']:.6f} eV^2",
        f"{tag} test EwT (0.02 eV)      = {metrics['energy_within_threshold']:.6f}",
    ]
    for line in lines:
        print(line)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
