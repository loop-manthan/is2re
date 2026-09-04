"""Pydantic response schemas for the IS2RE demo backend."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StructureSummary(BaseModel):
    """One entry in the curated structure list."""

    sid: int
    adsorbate: str = Field(description="Adsorbate formula from oc20_data_mapping.pkl")
    catalyst: str = Field(description="Catalyst composition from oc20_data_mapping.pkl")
    natoms: int
    ground_truth_energy: float = Field(description="Relaxed adsorption energy (eV)")


class StructureDetail(StructureSummary):
    """Full geometry + graph metadata for a single structure."""

    atomic_numbers: list[int]
    positions: list[list[float]] = Field(description="3D atomic positions (Angstrom)")
    tags: list[int] = Field(description="OC20 tags: 0=bulk, 1=surface, 2=adsorbate")
    cell: list[list[float]] = Field(description="3x3 periodic cell (Angstrom)")
    cutoff: float = Field(description="Graph edge cutoff radius, from the model config (Angstrom)")


class PredictionResult(BaseModel):
    variant: str
    energy: float
    error: float = Field(description="|prediction - ground truth| (eV)")


class PredictionResponse(BaseModel):
    sid: int
    ground_truth_energy: float
    predictions: list[PredictionResult]


class ModelInfoEntry(BaseModel):
    variant: str
    trainable_params: int
    test_mae: float = Field(description="Held-out test MAE (eV)")
    test_ewt: float = Field(description="Held-out test energy-within-threshold (0.02 eV)")


class ModelInfoResponse(BaseModel):
    variants: list[ModelInfoEntry]


class HealthResponse(BaseModel):
    status: str