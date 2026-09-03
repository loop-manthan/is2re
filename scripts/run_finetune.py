"""Fine-tune the pretrained DimeNet++ IS2RE checkpoint on a subsampled, system-level
split of ``val_ood_both``. Supports both the frozen-backbone and full fine-tuning
variants.

Usage (from the project root):
    python scripts/run_finetune.py \
        --config configs/frozen_head_finetune.yml \
        --train-lmdb data/subsample/train \
        --val-lmdb   data/subsample/val \
        --out-dir    runs/frozen_head

    # or override individual knobs on the command line:
    python scripts/run_finetune.py --full --lr 3e-5 ...
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import common


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune DimeNet++ on IS2RE OOD data")
    p.add_argument("--config", default=None, help="YAML config with training hyperparameters")
    p.add_argument("--train-lmdb", default=None)
    p.add_argument("--val-lmdb", default=None)
    p.add_argument("--cache-dir", default=".cache/checkpoints")
    p.add_argument("--out-dir", default="runs/finetune")
    p.add_argument("--cpu", action="store_true")

    # Freezing / schedule / optimizer
    p.add_argument("--full", dest="full", action="store_true",
                   help="Fine-tune all parameters (default is frozen backbone)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--warmup-epochs", type=int, default=None)
    p.add_argument("--warmup-factor", type=float, default=None)
    p.add_argument("--milestones", type=int, nargs="+", default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)

    # W&B
    p.add_argument("--wandb-project", default="is2re-finetune",
                   help="W&B project (shared across all variants)")
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable W&B logging for this run")
    return p.parse_args()


DEFAULTS = {
    "lr": 1e-4,
    "epochs": 20,
    "batch_size": 2,
    "warmup_epochs": 2,
    "warmup_factor": 0.2,
    "milestones": [4, 8, 12],
    "gamma": 0.1,
    "weight_decay": 0.0,
    "patience": 8,
    "seed": 0,
}


def main() -> None:
    args = parse_args()

    cfg = dict(DEFAULTS)
    file_cfg = {}
    if args.config:
        with open(args.config) as f:
            file_cfg = yaml.safe_load(f) or {}
        # YAML may hand back strings for things like `1e-4`; coerce to numbers.
        for key, cast in {
            "lr": float,
            "epochs": int,
            "batch_size": int,
            "warmup_epochs": int,
            "warmup_factor": float,
            "milestones": lambda v: [int(x) for x in v],
            "gamma": float,
            "weight_decay": float,
            "patience": int,
            "seed": int,
        }.items():
            if key in file_cfg and file_cfg[key] is not None:
                file_cfg[key] = cast(file_cfg[key])
        cfg.update(file_cfg)

    # Command-line overrides win over the config file.
    for key in (
        "lr",
        "epochs",
        "batch_size",
        "warmup_epochs",
        "warmup_factor",
        "milestones",
        "gamma",
        "weight_decay",
        "patience",
        "seed",
    ):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    # --full (CLI) overrides the config file's freeze_backbone setting.
    if args.full:
        freeze_backbone = False
    else:
        freeze_backbone = bool(file_cfg.get("freeze_backbone", True))

    train_lmdb = args.train_lmdb or cfg.get("train_lmdb")
    val_lmdb = args.val_lmdb or cfg.get("val_lmdb")
    if not train_lmdb or not val_lmdb:
        raise SystemExit("--train-lmdb and --val-lmdb are required (or set in config).")

    common.set_seed(cfg["seed"])
    device = common.get_device(args.cpu)

    ckpt_path = common.download_checkpoint(args.cache_dir)
    model = common.build_model(freeze_backbone=freeze_backbone)
    model, normalizer = common.load_pretrained(model, ckpt_path, device)

    train_loader = common.make_loader(
        train_lmdb, cfg["batch_size"], shuffle=True, split="train"
    )
    val_loader = common.make_loader(
        val_lmdb, cfg["batch_size"], shuffle=False, split="val"
    )

    logging.info("Variant: %s", "full" if not freeze_backbone else "frozen_backbone")
    logging.info("Config: %s", cfg)

    variant = "full_finetune" if not freeze_backbone else "frozen_head_finetune"
    logger = common.WandBLogger(
        project=args.wandb_project,
        name=variant,
        tags=[variant],
        config={**cfg, "freeze_backbone": freeze_backbone, "variant": variant},
        enabled=not args.no_wandb,
    )

    try:
        best_mae, best_epoch = common.train(
            model,
            normalizer,
            train_loader,
            val_loader,
            device,
            lr=cfg["lr"],
            epochs=cfg["epochs"],
            warmup_epochs=cfg["warmup_epochs"],
            warmup_factor=cfg["warmup_factor"],
            milestones=cfg["milestones"],
            gamma=cfg["gamma"],
            weight_decay=cfg["weight_decay"],
            patience=cfg["patience"],
            out_dir=args.out_dir,
            logger=logger,
        )
    finally:
        logger.finish()

    logging.info("Finished. Best val MAE %.4f at epoch %d.", best_mae, best_epoch)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
