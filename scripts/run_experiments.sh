#!/usr/bin/env bash
# End-to-end reproduction of the IS2RE fine-tuning experiments:
#   download val_ood_both -> subsample + system-level split -> zero-shot baseline
#   -> frozen-backbone fine-tune -> full fine-tune -> held-out test evals.
#
# Usage:
#   bash scripts/run_experiments.sh [--val-ood-both PATH] [--n-samples 11000]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VAL_OOD_BOTH=""
N_SAMPLES=11000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --val-ood-both) VAL_OOD_BOTH="$2"; shift 2 ;;
        --n-samples) N_SAMPLES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$VAL_OOD_BOTH" ]]; then
    echo "Downloading val_ood_both (~8.6 GB) ..."
    bash data/download_val_ood_both.sh data/raw
    VAL_OOD_BOTH="$(find data/raw -type d -name 'val_ood_both' | head -1)"
    echo "val_ood_both at: $VAL_OOD_BOTH"
fi

echo "=== 1. Subsampling + system-level split ==="
python data/make_subsample.py \
    --src "$VAL_OOD_BOTH" \
    --out-dir data/subsample \
    --mapping data/raw/oc20_data_mapping.pkl \
    --n-samples "$N_SAMPLES" \
    --seed 0

echo "=== 2. Zero-shot baseline (pretrained, no fine-tuning) ==="
python scripts/run_zero_shot_eval.py \
    --test-lmdb data/subsample/test \
    --cache-dir .cache/checkpoints \
    --out results/zero_shot.txt

echo "=== 3. Frozen-backbone fine-tune ==="
python scripts/run_finetune.py \
    --config configs/frozen_head_finetune.yml \
    --train-lmdb data/subsample/train \
    --val-lmdb data/subsample/val \
    --cache-dir .cache/checkpoints \
    --out-dir runs/frozen_head

echo "=== 4. Eval frozen-backbone on held-out test ==="
python scripts/eval_test.py \
    --ckpt runs/frozen_head/best_checkpoint.pt \
    --test-lmdb data/subsample/test \
    --out results/frozen_head_test.txt

echo "=== 5. Full fine-tune ==="
python scripts/run_finetune.py \
    --config configs/full_finetune.yml \
    --train-lmdb data/subsample/train \
    --val-lmdb data/subsample/val \
    --cache-dir .cache/checkpoints \
    --out-dir runs/full

echo "=== 6. Eval full fine-tune on held-out test ==="
python scripts/eval_test.py \
    --ckpt runs/full/best_checkpoint.pt \
    --test-lmdb data/subsample/test \
    --out results/full_test.txt

echo "=== Done. Results: ==="
for f in results/zero_shot.txt results/frozen_head_test.txt results/full_test.txt; do
    echo "--- $f ---"
    cat "$f"
done