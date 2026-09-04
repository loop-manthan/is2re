"""Backing service for the IS2RE demo API.

Reads only existing artifacts: the test LMDB (``data/subsample/test``), the
``oc20_data_mapping.pkl`` join table, the three model checkpoints (pretrained +
frozen-backbone + full fine-tune), and the results files. Nothing is retrained or
regenerated. Heavy imports (torch / fairchem) are deferred so the module can be
imported cheaply by tests that use a fake service.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import sys
import threading
from pathlib import Path

log = logging.getLogger("is2re.demo")

VARIANT_ORDER = ("zero_shot", "frozen_backbone", "full")

# Parameters trained during fine-tuning for each variant (matches the README table).
TRAINABLE_PARAMS = {"zero_shot": 0, "frozen_backbone": 648960, "full": 2755462}

# Fallback aggregate results, used only if the results/ files are missing.
FALLBACK_RESULTS = {
    "zero_shot": {"test_mae": 0.6908, "test_ewt": 0.0285},
    "frozen_backbone": {"test_mae": 0.5517, "test_ewt": 0.0309},
    "full": {"test_mae": 0.5411, "test_ewt": 0.0315},
}

_COMMON = None


def _common():
    """Lazily import fairchem plumbing from the existing training code."""
    global _COMMON
    if _COMMON is None:
        root = Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import scripts.common  # noqa: PLC0415

        _COMMON = scripts.common
    return _COMMON


def load_mapping(path: str | None) -> dict:
    if not path or not os.path.isfile(path):
        log.warning("Mapping file not found at %s; adsorbate/catalyst will be 'unknown'.", path)
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_results(results_dir: str) -> dict[str, dict]:
    """Read test MAE / EwT from the three results files, else fall back."""
    out: dict[str, dict] = {}
    pattern_mae = re.compile(r"MAE\s*=\s*([0-9.]+)")
    pattern_ewt = re.compile(r"EwT \(0\.02 eV\)\s*=\s*([0-9.]+)")
    file_for = {
        "zero_shot": os.path.join(results_dir, "zero_shot.txt"),
        "frozen_backbone": os.path.join(results_dir, "frozen_head_test.txt"),
        "full": os.path.join(results_dir, "full_test.txt"),
    }
    for variant, path in file_for.items():
        if os.path.isfile(path):
            with open(path) as f:
                text = f.read()
            mae = pattern_mae.search(text)
            ewt = pattern_ewt.search(text)
            if mae and ewt:
                out[variant] = {"test_mae": float(mae.group(1)), "test_ewt": float(ewt.group(1))}
                continue
        log.warning("Results file %s missing or unparseable; using fallback for %s.", path, variant)
        out[variant] = dict(FALLBACK_RESULTS[variant])
    return out


def _curate(entries: list[dict], errors: dict, limit: int, ordered: int, random: int, seed: int) -> list[dict]:
    """Mostly structures with the clean error ordering, plus a few random ones."""
    import numpy as np

    ordered_list: list[dict] = []
    rest: list[dict] = []
    for e in entries:
        err = errors.get(e["sid"])
        if (
            err
            and err.get("full") is not None
            and err.get("frozen_backbone") is not None
            and err.get("zero_shot") is not None
            and err["full"] < err["frozen_backbone"] < err["zero_shot"]
        ):
            ordered_list.append(e)
        else:
            rest.append(e)

    ordered_list.sort(
        key=lambda e: errors[e["sid"]]["zero_shot"] - errors[e["sid"]]["full"],
        reverse=True,
    )
    picked = ordered_list[:ordered]
    leftover_rest = rest
    if len(picked) < ordered:
        needed = ordered - len(picked)
        picked = picked + leftover_rest[:needed]
        leftover_rest = leftover_rest[needed:]

    rng = np.random.default_rng(seed)
    n_random = min(random, len(leftover_rest))
    random_pick = (
        [leftover_rest[i] for i in rng.choice(len(leftover_rest), size=n_random, replace=False)]
        if n_random
        else []
    )

    final = (picked + random_pick)[:limit]
    rng.shuffle(final)
    return final


class DemoService:
    """Holds all models + data in memory and serves structure/prediction queries."""

    def __init__(
        self,
        test_lmdb: str,
        mapping: dict,
        models: dict,
        device,
        cutoff: float,
        model_info: list[dict],
        dataset,
        metadata: dict,
        curation_cache: str | None = None,
    ) -> None:
        self.test_lmdb = test_lmdb
        self.mapping = mapping
        self.models = models
        self.device = device
        self.cutoff = cutoff
        self.model_info = model_info
        self._dataset = dataset
        self._sids = [int(s) for s in metadata["sid"]]
        self._ys = [float(y) for y in metadata["y"]]
        self._natoms = [int(n) for n in metadata["natoms"]]
        self._sid_to_idx = {sid: i for i, sid in enumerate(self._sids)}
        self._infer_lock = threading.Lock()
        self._curation_cache = curation_cache
        self._entries = self._build_entries()
        self._curated = self._load_curated_cache() or self._compute_and_curate()

    # --- data helpers ----------------------------------------------------

    def _ads_catalyst(self, sid: int) -> tuple[str, str]:
        entry = self.mapping.get(f"random{sid}")
        if entry is None:
            return "unknown", "unknown"
        return str(entry.get("ads_symbols", "unknown")), str(entry.get("bulk_symbols", "unknown"))

    def _build_entries(self) -> list[dict]:
        entries = []
        for i, sid in enumerate(self._sids):
            ads, cat = self._ads_catalyst(sid)
            entries.append(
                {
                    "idx": i,
                    "sid": sid,
                    "adsorbate": ads,
                    "catalyst": cat,
                    "natoms": self._natoms[i],
                    "ground_truth_energy": self._ys[i],
                }
            )
        return entries

    # --- curation --------------------------------------------------------

    def _load_curated_cache(self) -> list[dict] | None:
        if not self._curation_cache or not os.path.isfile(self._curation_cache):
            return None
        try:
            import json  # noqa: PLC0415

            with open(self._curation_cache) as f:
                sids = json.load(f)
            by_sid = {e["sid"]: e for e in self._entries}
            curated = [by_sid[s] for s in sids if s in by_sid]
            if len(curated) == len(sids) and curated:
                log.info("Loaded curated structure list from %s", self._curation_cache)
                return curated
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to load curation cache (%s); recomputing.", exc)
        return None

    def _save_curated_cache(self, curated: list[dict]) -> None:
        if not self._curation_cache:
            return
        try:
            import json  # noqa: PLC0415

            with open(self._curation_cache, "w") as f:
                json.dump([e["sid"] for e in curated], f)
            log.info("Saved curated structure list to %s", self._curation_cache)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to save curation cache (%s).", exc)

    def _compute_and_curate(self) -> list[dict]:
        errors = self._compute_all_errors()
        curated = _curate(
            self._entries, errors, limit=24, ordered=20, random=4, seed=0
        )
        self._save_curated_cache(curated)
        return curated

    def _compute_all_errors(self, batch_size: int = 4) -> dict[int, dict[str, float]]:
        import torch  # noqa: PLC0415
        from fairchem.core.datasets.lmdb_dataset import data_list_collater  # noqa: PLC0415

        log.info("Computing per-structure prediction errors (%d structures)...", len(self._sids))
        errors: dict[int, dict[str, float]] = {}
        n = len(self._sids)
        for start in range(0, n, batch_size):
            idxs = list(range(start, min(start + batch_size, n)))
            data_list = [self._dataset[i] for i in idxs]
            batch = data_list_collater(data_list, otf_graph=False).to(self.device)
            gt = torch.tensor([self._ys[i] for i in idxs], device=self.device)
            with self._infer_lock:
                for variant, (model, normalizer) in self.models.items():
                    model.eval()
                    with torch.no_grad():
                        pred = model(batch)["energy"].view(-1)
                        pred = normalizer.denorm(pred)
                    for k, i in enumerate(idxs):
                        err = float(abs(pred[k].item() - self._ys[i]))
                        errors.setdefault(self._sids[i], {})[variant] = err
        log.info("Computed errors for %d structures.", len(errors))
        return errors

    def _data_for(self, sid: int):
        idx = self._sid_to_idx[sid]
        return self._dataset[idx]

    # --- public API ------------------------------------------------------

    def list_structures(self) -> list[dict]:
        return [
            {
                "sid": e["sid"],
                "adsorbate": e["adsorbate"],
                "catalyst": e["catalyst"],
                "natoms": e["natoms"],
                "ground_truth_energy": e["ground_truth_energy"],
            }
            for e in self._curated
        ]

    def get_structure(self, sid: int) -> dict | None:
        if sid not in self._sid_to_idx:
            return None
        data = self._data_for(sid)
        ads, cat = self._ads_catalyst(sid)
        return {
            "sid": sid,
            "adsorbate": ads,
            "catalyst": cat,
            "natoms": int(data.natoms),
            "ground_truth_energy": self._ys[self._sid_to_idx[sid]],
            "atomic_numbers": [int(x) for x in data.atomic_numbers.tolist()],
            "positions": [list(map(float, p)) for p in data.pos.tolist()],
            "tags": [int(x) for x in data.tags.tolist()],
            "cell": [list(map(float, row)) for row in data.cell[0].tolist()],
            "cutoff": self.cutoff,
        }

    def get_predictions(self, sid: int) -> dict | None:
        if sid not in self._sid_to_idx:
            return None
        data = self._data_for(sid)
        gt = self._ys[self._sid_to_idx[sid]]
        preds = self._predict(data)
        return {
            "sid": sid,
            "ground_truth_energy": gt,
            "predictions": [
                {"variant": v, "energy": preds[v], "error": abs(preds[v] - gt)}
                for v in VARIANT_ORDER
                if v in preds
            ],
        }

    def results_table(self) -> list[dict]:
        return [
            {
                "variant": v,
                "trainable_params": TRAINABLE_PARAMS[v],
                "test_mae": self.model_info[v]["test_mae"],
                "test_ewt": self.model_info[v]["test_ewt"],
            }
            for v in VARIANT_ORDER
        ]

    # --- inference -------------------------------------------------------

    def _predict(self, data) -> dict[str, float]:
        import torch  # noqa: PLC0415

        from fairchem.core.datasets.lmdb_dataset import data_list_collater  # noqa: PLC0415

        batch = data_list_collater([data], otf_graph=False).to(self.device)
        out: dict[str, float] = {}
        with self._infer_lock:
            for variant, (model, normalizer) in self.models.items():
                model.eval()
                with torch.no_grad():
                    pred = model(batch)["energy"].view(-1)
                    pred = normalizer.denorm(pred)
                out[variant] = float(pred[0].cpu())
        return out


def build_service(settings: dict | None = None) -> DemoService:
    """Load all artifacts and construct the production service."""
    import torch  # noqa: PLC0415

    settings = settings or {}
    root = Path(__file__).resolve().parent.parent

    test_lmdb = settings.get("test_lmdb") or str(root / "data" / "subsample" / "test")
    mapping_path = settings.get("mapping") or str(root / "data" / "raw" / "oc20_data_mapping.pkl")
    cache_dir = settings.get("cache_dir") or str(root / ".cache" / "checkpoints")
    frozen_ckpt = settings.get("frozen_ckpt") or str(root / "runs" / "frozen_head" / "best_checkpoint.pt")
    full_ckpt = settings.get("full_ckpt") or str(root / "runs" / "full" / "best_checkpoint.pt")
    results_dir = settings.get("results_dir") or str(root / "results")
    curation_cache = settings.get("curation_cache") or str(
        root / "data" / "subsample" / "curated_sids.json"
    )
    device_name = settings.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    common = _common()
    log.info("Loading mapping from %s", mapping_path)
    mapping = load_mapping(mapping_path)

    log.info("Building models on %s ...", device)
    pretrained = common.download_checkpoint(cache_dir)
    models = _load_models(common, pretrained, frozen_ckpt, full_ckpt, device)

    log.info("Loading test LMDB %s", test_lmdb)
    from fairchem.core.datasets.lmdb_dataset import LmdbDataset  # noqa: PLC0415

    dataset = LmdbDataset({"src": test_lmdb, "key_mapping": common.KEY_MAPPING})
    meta_path = os.path.join(test_lmdb, "metadata.npz")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"metadata.npz not found at {meta_path}")
    import numpy as np  # noqa: PLC0415

    metadata = np.load(meta_path, allow_pickle=True)

    model_info = parse_results(results_dir)
    cutoff = float(common.ARCH["cutoff"])

    return DemoService(
        test_lmdb=test_lmdb,
        mapping=mapping,
        models=models,
        device=device,
        cutoff=cutoff,
        model_info=model_info,
        dataset=dataset,
        metadata=metadata,
        curation_cache=curation_cache,
    )


def _load_models(common, pretrained_ckpt: str, frozen_ckpt: str, full_ckpt: str, device):
    import torch  # noqa: PLC0415

    models = {}

    model = common.build_model(freeze_backbone=False)
    model, normalizer = common.load_pretrained(model, pretrained_ckpt, device)
    models["zero_shot"] = (model, normalizer)

    model = common.build_model(freeze_backbone=True)
    checkpoint = torch.load(frozen_ckpt, map_location="cpu", weights_only=False)
    common.load_state_dict(model, checkpoint["state_dict"], strict=True)
    normalizer = common.create_normalizer(state_dict=checkpoint["normalizers"]["target"]).to(device)
    models["frozen_backbone"] = (model.to(device), normalizer)

    model = common.build_model(freeze_backbone=False)
    checkpoint = torch.load(full_ckpt, map_location="cpu", weights_only=False)
    common.load_state_dict(model, checkpoint["state_dict"], strict=True)
    normalizer = common.create_normalizer(state_dict=checkpoint["normalizers"]["target"]).to(device)
    models["full"] = (model.to(device), normalizer)

    log.info("Loaded %d variants.", len(models))
    return models