#!/usr/bin/env bash
# Download the OC20 IS2RE tarball and extract *only* the val_ood_both split.
# Also fetches oc20_data_mapping.pkl (adsorbate/catalyst labels keyed by random<sid>)
# used for stratified subsampling. The full tarball is ~8.6 GB and contains every
# IS2RE split; we only keep val_ood_both (the out-of-distribution adsorbate+catalyst
# validation split).
set -euo pipefail

URL="https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz"
MAPPING_URL="https://dl.fbaipublicfiles.com/opencatalystproject/data/oc20_data_mapping.pkl"
OUT_DIR="${1:-data}"
TARBALL="$OUT_DIR/is2res_train_val_test_lmdbs.tar.gz"

mkdir -p "$OUT_DIR"

echo "Downloading OC20 data mapping (~208 MB)"
wget -c "$MAPPING_URL" -O "$OUT_DIR/oc20_data_mapping.pkl"

echo "Downloading $URL"
wget -c "$URL" -O "$TARBALL"

echo "Extracting val_ood_both into $OUT_DIR"
tar -xzf "$TARBALL" -C "$OUT_DIR" --wildcards '*val_ood_both*'

echo "Removing tarball"
rm -f "$TARBALL"

echo "Done. val_ood_both located under:"
find "$OUT_DIR" -maxdepth 4 -type d -name 'val_ood_both' -print
