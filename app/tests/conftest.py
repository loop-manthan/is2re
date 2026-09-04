"""Shared fixtures: a lightweight fake service so the API tests run fast and
never load the real models / LMDB. The fake mirrors the public surface of
``app.service.DemoService``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

_VARIANTS = ("zero_shot", "frozen_backbone", "full")


class FakeService:
    def __init__(self) -> None:
        self._structs = [
            {"sid": 1001, "adsorbate": "*O", "catalyst": "Si", "natoms": 12,
             "ground_truth_energy": -1.5},
            {"sid": 1002, "adsorbate": "*N2", "catalyst": "FeTi", "natoms": 40,
             "ground_truth_energy": 0.25},
            {"sid": 1003, "adsorbate": "*OH", "catalyst": "Au", "natoms": 25,
             "ground_truth_energy": -0.8},
        ]

    def list_structures(self):
        return [dict(s) for s in self._structs]

    def get_structure(self, sid: int):
        for s in self._structs:
            if s["sid"] == sid:
                n = s["natoms"]
                return {
                    **s,
                    "atomic_numbers": [6] * n,
                    "positions": [[float(i), 0.0, 0.0] for i in range(n)],
                    "tags": [0] * (n // 2) + [1] * (n - n // 2 - 1) + [2],
                    "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                    "cutoff": 6.0,
                }
        return None

    def get_predictions(self, sid: int):
        for s in self._structs:
            if s["sid"] == sid:
                gt = s["ground_truth_energy"]
                energies = {"zero_shot": gt + 0.35, "frozen_backbone": gt + 0.15,
                            "full": gt + 0.08}
                return {
                    "sid": sid,
                    "ground_truth_energy": gt,
                    "predictions": [
                        {"variant": v, "energy": energies[v],
                         "error": abs(energies[v] - gt)}
                        for v in _VARIANTS
                    ],
                }
        return None

    def results_table(self):
        return [
            {"variant": v, "trainable_params": 0 if v == "zero_shot" else 1,
             "test_mae": 0.7 if v == "zero_shot" else 0.5, "test_ewt": 0.03}
            for v in _VARIANTS
        ]


@pytest.fixture
def client():
    service = FakeService()
    app = create_app(service=service)
    return TestClient(app)