"""Optional integration test against the real artifacts (models, LMDB, mapping).

Skipped unless RUN_INTEGRATION=1 because it loads the three checkpoints and the
test LMDB (tens of seconds, needs a working GPU/CPU + the project artifacts).
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 to run against real artifacts",
)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402
from app.service import build_service  # noqa: E402


def test_real_predictions_are_sane():
    service = build_service()
    app = create_app(service=service)
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}

    structures = client.get("/structures").json()
    assert len(structures) > 0

    sid = structures[0]["sid"]
    detail = client.get(f"/structures/{sid}").json()
    assert detail["cutoff"] == 6.0
    assert len(detail["positions"]) == detail["natoms"]

    pred = client.get(f"/structures/{sid}/predictions").json()
    assert {p["variant"] for p in pred["predictions"]} == {
        "zero_shot", "frozen_backbone", "full",
    }
    for p in pred["predictions"]:
        assert -20.0 <= p["energy"] <= 20.0
        assert p["error"] >= 0.0

    info = client.get("/model-info").json()["variants"]
    assert len(info) == 3