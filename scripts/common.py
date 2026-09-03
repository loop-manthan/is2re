"""Shared utilities for the IS2RE fine-tuning experiments.

Everything here is intentionally self-contained: it builds the model and normalizer
from the pretrained OC20 checkpoint, loads the subsampled LMDBs, and provides a
transparent train/eval loop so the frozen-backbone vs. full fine-tuning variants are
controlled by a single flag rather than separate code paths.

We reuse fairchem-core's building blocks (``Evaluator`` for MAE / energy-within-
threshold, ``Normalizer`` for the pretrained target statistics, ``LmdbDataset`` for
loading) but keep our own training loop so freezing, the LR schedule and early
stopping are explicit.
"""

from __future__ import annotations

import logging
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import fairchem.core.datasets.lmdb_dataset  # noqa: F401  (registers "lmdb" dataset)
import fairchem.core.models.dimenet_plus_plus  # noqa: F401  (registers "dimenetplusplus")
import models.frozen_dimenetpp  # noqa: F401  (registers "dimenetplusplus_frozen_backbone")

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import load_state_dict, match_state_dict
from fairchem.core.datasets.base_dataset import create_dataset
from fairchem.core.datasets.lmdb_dataset import data_list_collater
from fairchem.core.modules.evaluator import Evaluator
from fairchem.core.modules.normalization.normalizer import create_normalizer
from fairchem.core.models.model_registry import model_name_to_local_file

log = logging.getLogger("is2re")


class WandBLogger:
    """Thin wrapper around W&B.

    One shared project with runs tagged per variant. If no W&B credentials are
    available the run is recorded in offline mode (nothing is lost) so a later
    ``wandb sync`` uploads it; W&B is never silently skipped.
    """

    def __init__(
        self,
        project: str,
        name: str,
        tags: list[str] | None = None,
        config: dict | None = None,
        enabled: bool = True,
    ) -> None:
        self._run = None
        if not enabled:
            log.info("W&B logging disabled for run %s.", name)
            return
        try:
            import wandb

            settings = None
            if not os.environ.get("WANDB_API_KEY") and not wandb.api.api_key:
                log.warning(
                    "No W&B API key found; run will be logged in OFFLINE mode. "
                    "Set WANDB_API_KEY (or run `wandb login`) to sync online."
                )
                settings = wandb.Settings(mode="offline")
            self._run = wandb.init(
                project=project,
                name=name,
                tags=tags or [],
                config=config,
                settings=settings,
                reinit=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("W&B init failed (%s); continuing without W&B.", exc)
            self._run = None

    @property
    def active(self) -> bool:
        return self._run is not None

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()

PRETRAINED_NAME = "DimeNet++-IS2RE-OC20-100k"

# Architecture read from the checkpoint's embedded ``model_attributes``.
# ``int_emb_size`` / ``basis_emb_size`` are absent there and fall back to the
# ``DimeNetPlusPlusWrap`` defaults.
ARCH = {
    "cutoff": 6.0,
    "hidden_channels": 256,
    "num_after_skip": 2,
    "num_before_skip": 1,
    "num_blocks": 3,
    "num_output_layers": 3,
    "num_radial": 6,
    "num_spherical": 7,
    "out_emb_channels": 192,
    "regress_forces": False,
    "use_pbc": True,
    "int_emb_size": 64,
    "basis_emb_size": 8,
}

# OC20 IS2RE LMDB data objects store the relaxed energy under ``y_relaxed``
# (scalar float), with the initial energy under ``y_init``.
KEY_MAPPING = {"y_relaxed": "energy"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cpu: bool = False) -> torch.device:
    if cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def download_checkpoint(cache_dir: str) -> str:
    """Download the pretrained checkpoint once and return its local path."""
    os.makedirs(cache_dir, exist_ok=True)
    return model_name_to_local_file(PRETRAINED_NAME, cache_dir)


def build_model(freeze_backbone: bool):
    """Instantiate the DimeNet++ model, optionally with a frozen backbone."""
    name = (
        "dimenetplusplus_frozen_backbone" if freeze_backbone else "dimenetplusplus"
    )
    model = registry.get_model_class(name)(**ARCH)
    log.info(
        "Built %s with %d params (%d trainable).",
        name,
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    return model


def load_pretrained(model, checkpoint_path: str, device: torch.device):
    """Load only ``state_dict`` + target normalizer from the pretrained checkpoint.

    Optimizer / scheduler / epoch / step are deliberately ignored so a fresh
    optimizer and LR schedule apply and the frozen variant has no shape mismatch.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    new_dict = match_state_dict(model.state_dict(), checkpoint["state_dict"])
    load_state_dict(model, new_dict, strict=True)
    normalizer = create_normalizer(state_dict=checkpoint["normalizers"]["target"])
    return model.to(device), normalizer.to(device)


def make_loader(
    src: str,
    batch_size: int,
    shuffle: bool,
    split: str = "train",
    key_mapping: dict | None = None,
) -> DataLoader:
    """Build a DataLoader over an LMDB directory/file of OC20 IS2RE graphs."""
    config = {
        "format": "lmdb",
        "src": src,
        "key_mapping": key_mapping or KEY_MAPPING,
    }
    dataset = create_dataset(config, split)
    collater = partial(data_list_collater, otf_graph=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collater,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(model, loader: DataLoader, normalizer, device: torch.device) -> dict:
    """Return {"energy_mae", "energy_mse", "energy_within_threshold"} floats."""
    model.eval()
    evaluator = Evaluator(task="is2re")
    metrics: dict = {}

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)["energy"].view(-1)
        pred = normalizer.denorm(pred)
        target = batch.energy.view(-1)
        natoms = batch.natoms.to(device)

        metrics = evaluator.eval(
            {"energy": pred, "natoms": natoms},
            {"energy": target, "natoms": natoms},
            prev_metrics=metrics,
        )

    return {k: metrics[k]["metric"] for k in metrics}


def compute_lr(
    base_lr: float,
    epoch: int,
    warmup_epochs: int,
    warmup_factor: float,
    milestones: list[int],
    gamma: float,
) -> float:
    """Linear warmup followed by multiplicative step decay at ``milestones``."""
    if epoch < warmup_epochs:
        alpha = (epoch + 1) / warmup_epochs
        return base_lr * (warmup_factor + alpha * (1.0 - warmup_factor))
    lr = base_lr
    for m in milestones:
        if epoch >= m:
            lr *= gamma
    return lr


def save_checkpoint(model, normalizer, out_dir: str, arch: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "best_checkpoint.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "normalizers": {"target": normalizer.state_dict()},
            "arch": arch,
            "freeze_backbone": isinstance(
                model, registry.get_model_class("dimenetplusplus_frozen_backbone")
            ),
        },
        path,
    )
    return path


def train(
    model,
    normalizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    lr: float,
    epochs: int,
    warmup_epochs: int,
    warmup_factor: float,
    milestones: list[int],
    gamma: float,
    weight_decay: float,
    patience: int,
    out_dir: str,
    logger: WandBLogger | None = None,
) -> tuple[float, int]:
    """Run fine-tuning with early stopping on val energy MAE. Returns (best_mae, best_epoch)."""
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_mae = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(epochs):
        lr_epoch = compute_lr(
            lr, epoch, warmup_epochs, warmup_factor, milestones, gamma
        )
        for group in optimizer.param_groups:
            group["lr"] = lr_epoch

        model.train()
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)["energy"].view(-1)
            target = normalizer.norm(batch.energy.view(-1))
            loss = F.l1_loss(pred, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 500 == 0 and batch_idx > 0:
                log.info(
                    "  epoch %d batch %d/%d running_loss %.4f",
                    epoch, batch_idx, len(train_loader),
                    running_loss / (batch_idx + 1),
                )

        train_loss = running_loss / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, normalizer, device)
        val_mae = val_metrics["energy_mae"]

        log.info(
            "epoch %3d | lr %.2e | train_loss %.4f | val_mae %.4f | val_ewt %.4f",
            epoch,
            lr_epoch,
            train_loss,
            val_mae,
            val_metrics["energy_within_threshold"],
        )
        if logger is not None:
            logger.log(
                {
                    "lr": lr_epoch,
                    "train_loss": train_loss,
                    "val_energy_mae": val_mae,
                    "val_energy_mse": val_metrics["energy_mse"],
                    "val_energy_within_threshold": val_metrics[
                        "energy_within_threshold"
                    ],
                },
                step=epoch,
            )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(model, normalizer, out_dir, ARCH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                log.info("Early stopping after %d epochs without improvement.", patience)
                break

    log.info("Best val MAE %.4f at epoch %d.", best_val_mae, best_epoch)
    return best_val_mae, best_epoch
