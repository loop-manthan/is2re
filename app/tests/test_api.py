"""API tests for the IS2RE demo backend.

Runs against a fake service (see conftest) so the suite is fast and has no heavy
dependencies. Endpoint shapes and basic semantics are validated, including a smoke
test asserting predictions are sane numeric energies.
"""

from __future__ import annotations

import math

_EXPECTED_SUMMARY_KEYS = {"sid", "adsorbate", "catalyst", "natoms", "ground_truth_energy"}
_EXPECTED_DETAIL_KEYS = _EXPECTED_SUMMARY_KEYS | {
    "atomic_numbers",
    "positions",
    "tags",
    "cell",
    "cutoff",
}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_structure_list_shape(client):
    resp = client.get("/structures")
    assert resp.status_code == 200
    structures = resp.json()
    assert isinstance(structures, list) and len(structures) == 3
    for item in structures:
        assert set(item.keys()) == _EXPECTED_SUMMARY_KEYS
        assert isinstance(item["sid"], int)
        assert isinstance(item["adsorbate"], str)
        assert isinstance(item["natoms"], int) and item["natoms"] > 0
        assert isinstance(item["ground_truth_energy"], (int, float))


def test_structure_detail_shape(client):
    resp = client.get("/structures/1001")
    assert resp.status_code == 200
    detail = resp.json()
    assert set(detail.keys()) == _EXPECTED_DETAIL_KEYS
    n = detail["natoms"]
    assert len(detail["atomic_numbers"]) == n
    assert len(detail["positions"]) == n
    assert all(len(p) == 3 for p in detail["positions"])
    assert len(detail["tags"]) == n
    assert len(detail["cell"]) == 3 and all(len(row) == 3 for row in detail["cell"])
    assert detail["cutoff"] == 6.0


def test_structure_detail_not_found(client):
    resp = client.get("/structures/999999")
    assert resp.status_code == 404


def test_predictions_smoke(client):
    resp = client.get("/structures/1002/predictions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sid"] == 1002
    assert isinstance(body["ground_truth_energy"], float)
    assert len(body["predictions"]) == 3
    variants = [p["variant"] for p in body["predictions"]]
    assert variants == ["zero_shot", "frozen_backbone", "full"]
    for pred in body["predictions"]:
        energy = pred["energy"]
        assert isinstance(energy, float) and math.isfinite(energy)
        assert -20.0 <= energy <= 20.0, f"energy out of sane range: {energy}"
        assert pred["error"] >= 0.0


def test_predictions_not_found(client):
    resp = client.get("/structures/999999/predictions")
    assert resp.status_code == 404


def test_model_info(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    variants = resp.json()["variants"]
    assert [v["variant"] for v in variants] == ["zero_shot", "frozen_backbone", "full"]
    for v in variants:
        assert {"variant", "trainable_params", "test_mae", "test_ewt"} <= set(v.keys())
        assert v["test_mae"] > 0.0 and 0.0 <= v["test_ewt"] <= 1.0


def test_static_index_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]