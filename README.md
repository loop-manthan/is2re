# IS2RE: Fine-Tuning a Pretrained Catalyst GNN

A graph neural network project built on the Open Catalyst 2020 (OC20) dataset,
exploring how well a pretrained model transfers to unseen catalyst chemistry
through fine-tuning.

## Overview

This project fine-tunes a pretrained Graph Neural Network on a held-out,
out-of-distribution subset of OC20 and compares multiple fine-tuning
strategies against a zero-shot baseline. The goal is to measure how much of a
pretrained model's performance carries over to unseen adsorbate and catalyst
combinations, and how much of the network actually needs to be retrained to
recover that performance.

## Task

The project targets **IS2RE (Initial Structure to Relaxed Energy)**, one of
the core tasks defined in the OC20 benchmark.

**Input**
- Atomic numbers for every atom in the system
- 3D atomic positions (the unrelaxed, initial structure)
- Atom tags indicating bulk, surface, or adsorbate atoms
- Periodic cell information

**Output**
- A single scalar value: the predicted relaxed adsorption energy (eV)

The model never sees a relaxation trajectory or intermediate structures. It
predicts the final relaxed energy directly from the initial geometry.

## Approach

- **Base model**: DimeNet++, pretrained on 100k OC20 IS2RE structures
- **Fine-tuning data**: a stratified, held-out subset of OC20's `val_ood_both`
  split, containing adsorbates and catalyst compositions not seen during
  pretraining
- **Variants compared**:
  - Zero-shot: the pretrained model, evaluated with no fine-tuning
  - Frozen-backbone fine-tune: only the final output layers are retrained,
    the rest of the network stays frozen
  - Full fine-tune: every parameter in the network is retrained

## Results

Evaluated on a held-out test split of 1,650 systems, never used during
training or model selection.

| Variant | Trainable params | Test MAE (eV) | Test EwT (0.02 eV) |
|---|---|---|---|
| Zero-shot | 0 / 2,755,462 | 0.6908 | 2.85% |
| Frozen-backbone fine-tune | 648,960 / 2,755,462 | 0.5517 | 3.09% |
| Full fine-tune | 2,755,462 / 2,755,462 | 0.5411 | 3.15% |

Fine-tuning reduces test MAE by 21.7% relative to the zero-shot baseline.
Frozen-backbone fine-tuning, retraining only 23.5% of the network's
parameters, recovers most of that improvement. Full fine-tuning adds a
further, smaller gain on top.

### Data split

| Split | Systems |
|---|---|
| Train | 8,250 |
| Validation | 1,100 |
| Test | 1,650 |

All splits are drawn from OC20's `val_ood_both` set and split at the system
level, so no structure appears in more than one split.

## Tech Stack

- PyTorch
- fairchem-core (OC20 model implementations and pretrained checkpoints)
- PyTorch Geometric
- Weights and Biases (experiment tracking)
- Conda (environment management)

## Project Structure

```
configs/      model and training configs
data/         download and preprocessing scripts
models/       model definitions and fine-tuning logic
scripts/      training, evaluation, and experiment orchestration
results/      final metrics and comparisons
```

## Status

Core experiment complete. Documentation, setup instructions, and additional
tooling in progress.