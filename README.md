# IS2RE: Fine-tuning DimeNet++ for Out-of-Distribution Catalysis

**Where the OC20 100k champion learns to handle the chemistry it never saw.**

A small, self-contained, fully reproducible study: take the pretrained
DimeNet++ IS2RE model, fine-tune it on a **stratified, system-level** subsample of
OC20 `val_ood_both` — the toughest out-of-distribution split (unseen adsorbate +
unseen catalyst) — and measure just how much a little fine-tuning closes the gap.

## The payoff

Fine-tuning on 8,250 OOD systems cuts held-out MAE by ~21%:

| Variant | test MAE (eV) | EwT (0.02 eV) |
| --- | --- | --- |
| zero-shot (pretrained) | 0.6908 | 2.85% |
| frozen-backbone | 0.5517 | 3.09% |
| full fine-tune | 0.5411 | 3.15% |

Notably, freezing the whole message-passing backbone and training only the 4
energy readout layers recovers almost all of the gain — the head does the heavy
lifting.

## What's inside

- `scripts/` — zero-shot eval, fine-tune (frozen or full), shared test eval, and an
  end-to-end runner. W&B logging on (offline-friendly).
- `models/` — the registered frozen-backbone DimeNet++ variant.
- `data/` — download + stratified subsampling (joins `oc20_data_mapping.pkl` for
  real adsorbate/catalyst stratification; no `sid` leaks across splits).
- `configs/` — the two fine-tuning recipes.

## Run it

```bash
bash scripts/run_experiments.sh          # download -> subsample -> train -> eval
```

Everything is pinned and reproducible: one shared W&B project
(`is2re-finetune`), fixed seed, and results in `results/comparison.txt`.