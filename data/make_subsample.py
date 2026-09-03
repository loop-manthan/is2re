"""Subsample + system-level split of an OC20 IS2RE LMDB split (e.g. ``val_ood_both``).

Reads the LMDB read-only, groups frames by ``sid`` (system = adsorbate + catalyst
combination), subsamples to ``--n-samples`` systems, and writes three new LMDBs
(train / val / test) plus per-split ``metadata.npz`` files.

When ``--mapping`` (the official ``oc20_data_mapping.pkl``) is provided, sampling is
**stratified** by the (adsorbate, catalyst) tuple looked up via ``random{sid}``: the
subsample is allocated proportionally across strata (largest-remainder method) and each
stratum is split 75/10/15 at the system level. The split is always at the **system**
level so no ``sid`` appears in more than one split, keeping the held-out test a genuine
out-of-distribution evaluation. Without ``--mapping`` it falls back to uniform random
system-level sampling.

The pickled data objects are copied verbatim (edge_index / cell_offsets / tags / y are
preserved), so the downstream training code uses the exact same graph structure.
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import pickle

import lmdb
import numpy as np
import torch

import fairchem.core.datasets.lmdb_dataset  # noqa: F401
from fairchem.core.datasets.lmdb_dataset import LmdbDataset

log = logging.getLogger("make_subsample")


def scalar_int(v) -> int:
    if torch.is_tensor(v):
        return int(v.item())
    return int(v)


def scalar_float(v) -> float:
    if torch.is_tensor(v):
        return float(v.item())
    return float(v)


def load_mapping(path: str | None) -> dict | None:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def stratum_of_sid(mapping: dict | None, sid: int) -> tuple[str, str]:
    """Return (ads_symbols, bulk_symbols) for a system id, or a placeholder."""
    if mapping is None:
        return ("*", "*")
    entry = mapping.get(f"random{sid}")
    if entry is None:
        return ("<missing>", "<missing>")
    return (entry["ads_symbols"], entry["bulk_symbols"])


def allocate_quotas(sizes: dict, total: int) -> dict:
    """Proportional allocation of ``total`` across groups (largest-remainder)."""
    if total <= 0:
        return {k: 0 for k in sizes}
    total_size = sum(sizes.values())
    if total_size <= 0:
        return {k: 0 for k in sizes}
    quotas = {k: sizes[k] * total / total_size for k in sizes}
    floors = {k: int(q) for k, q in quotas.items()}
    rem = total - sum(floors.values())
    for k in sorted(sizes, key=lambda k: quotas[k] - floors[k], reverse=True)[:rem]:
        floors[k] += 1
    return floors


def target_energy(d) -> float:
    """Return the IS2RE target energy (relaxed energy) of a data object."""
    for key in ("y_relaxed", "y", "energy"):
        if hasattr(d, key) and getattr(d, key) is not None:
            return scalar_float(getattr(d, key))
    raise ValueError(f"No target energy found in data object with keys {sorted(d.keys())}")


class LmdbWriter:
    """Writes a single-file LMDB (``<split>/data.lmdb``) with ``length`` key."""

    def __init__(self, split_dir: str) -> None:
        self.split_dir = split_dir
        os.makedirs(split_dir, exist_ok=True)
        self.path = os.path.join(split_dir, "data.lmdb")
        self.env = lmdb.open(
            self.path,
            map_size=int(2e11),
            subdir=False,
            meminit=False,
            map_async=True,
        )
        self._count = 0

    def write(self, data_object) -> int:
        idx = self._count
        self._count += 1
        with self.env.begin(write=True) as txn:
            txn.put(
                str(idx).encode("ascii"),
                pickle.dumps(data_object, protocol=-1),
            )
        return idx

    def close(self) -> None:
        with self.env.begin(write=True) as txn:
            txn.put(
                "length".encode("ascii"),
                pickle.dumps(self._count, protocol=-1),
            )
        self.env.sync()
        self.env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source LMDB (file or sharded dir)")
    p.add_argument("--out-dir", required=True, help="Dir to write train/val/test LMDBs")
    p.add_argument("--mapping", default=None,
                   help="Path to oc20_data_mapping.pkl for stratified sampling")
    p.add_argument("--n-samples", type=int, default=11000,
                   help="Number of systems to keep in total (None/0 = all)")
    p.add_argument("--train-frac", type=float, default=0.75)
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    mapping = load_mapping(args.mapping)
    if mapping is not None:
        log.info("Loaded mapping with %d systems.", len(mapping))
    else:
        log.warning(
            "No mapping provided; falling back to uniform (non-stratified) sampling."
        )

    dataset = LmdbDataset({"src": args.src})
    n = len(dataset)
    log.info("Source has %d frames.", n)

    # Pass 1: gather metadata and group by sid.
    sids = np.empty(n, dtype=np.int64)
    natoms = np.empty(n, dtype=np.int64)
    ys = np.empty(n, dtype=np.float64)
    for i in range(n):
        d = dataset[i]
        sids[i] = scalar_int(d.sid)
        natoms[i] = scalar_int(d.natoms)
        ys[i] = target_energy(d)

    unique_sids, counts = np.unique(sids, return_counts=True)
    log.info("Unique systems (sids): %d (min/max frames per system: %d/%d).",
             len(unique_sids), counts.min(), counts.max())

    if mapping is not None:
        covered = sum(1 for sid in unique_sids if f"random{sid}" in mapping)
        log.info("Mapping coverage: %d/%d sids (%.2f%%).",
                 covered, len(unique_sids), 100.0 * covered / len(unique_sids))
        split_field = collections.Counter(
            mapping[f"random{sid}"].get("split")
            for sid in unique_sids
            if f"random{sid}" in mapping
        )
        log.info("Mapping 'split' field among source sids: %s", dict(split_field))

    # Group unique systems by stratum (adsorbate, catalyst).
    strata: dict[tuple[str, str], list[int]] = collections.defaultdict(list)
    for sid in unique_sids:
        strata[stratum_of_sid(mapping, int(sid))].append(int(sid))

    # Stratified subsample: proportional quotas per stratum.
    n_sel = len(unique_sids) if not args.n_samples else min(
        args.n_samples, len(unique_sids)
    )
    quotas = allocate_quotas({s: len(v) for s, v in strata.items()}, n_sel)

    selected_by_stratum: dict[tuple[str, str], list[int]] = {}
    for s, sids in strata.items():
        perm = rng.permutation(len(sids))
        selected_by_stratum[s] = [sids[i] for i in perm[: quotas[s]]]
    selected_sids = [sid for sids in selected_by_stratum.values() for sid in sids]
    log.info("Selected %d systems across %d strata (target %d).",
             len(selected_sids), len(selected_by_stratum), n_sel)

    # System-level split (stratified at the subsample stage only): no sid spans splits.
    rng.shuffle(selected_sids)
    n_sel = len(selected_sids)
    if n_sel >= 3:
        n_train = max(round(n_sel * args.train_frac), 1)
        n_val = max(round(n_sel * args.val_frac), 1)
        n_test = n_sel - n_train - n_val
        while n_test < 1:
            if n_train > 1:
                n_train -= 1
            else:
                n_val -= 1
            n_test = n_sel - n_train - n_val
    else:
        n_train = max(round(n_sel * args.train_frac), 1)
        n_val = max(round(n_sel * args.val_frac), 0)
        n_test = max(n_sel - n_train - n_val, 0)
    assert n_train + n_val + n_test == n_sel

    split_of_sid: dict[int, str] = {}
    for i, sid in enumerate(selected_sids):
        if i < n_train:
            split_of_sid[sid] = "train"
        elif i < n_train + n_val:
            split_of_sid[sid] = "val"
        else:
            split_of_sid[sid] = "test"

    # Pass 2: write frames into their split's LMDB.
    writers = {
        split: LmdbWriter(os.path.join(args.out_dir, split))
        for split in ("train", "val", "test")
    }
    meta = {
        split: {"natoms": [], "y": [], "sid": []} for split in ("train", "val", "test")
    }
    for i in range(n):
        d = dataset[i]
        sid = scalar_int(d.sid)
        split = split_of_sid.get(sid)
        if split is None:
            continue
        writers[split].write(d)
        meta[split]["natoms"].append(scalar_int(d.natoms))
        meta[split]["y"].append(target_energy(d))
        meta[split]["sid"].append(sid)

    for split in ("train", "val", "test"):
        writers[split].close()
        split_dir = os.path.join(args.out_dir, split)
        np.savez_compressed(
            os.path.join(split_dir, "metadata.npz"),
            natoms=np.asarray(meta[split]["natoms"], dtype=np.int64),
            y=np.asarray(meta[split]["y"], dtype=np.float64),
            sid=np.asarray(meta[split]["sid"], dtype=np.int64),
        )
        log.info("Wrote %s: %d frames -> %s.", split, len(meta[split]["natoms"]), split_dir)

    if mapping is not None:
        for split in ("train", "val", "test"):
            split_sids = meta[split]["sid"]
            ads = {stratum_of_sid(mapping, s)[0] for s in split_sids}
            cat = {stratum_of_sid(mapping, s)[1] for s in split_sids}
            log.info("%s split: %d systems, %d distinct adsorbates, %d distinct catalysts.",
                     split, len(split_sids), len(ads), len(cat))

    log.info(
        "Split summary (systems): train=%d val=%d test=%d (seed=%d).",
        n_train, n_val, n_test, args.seed,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()