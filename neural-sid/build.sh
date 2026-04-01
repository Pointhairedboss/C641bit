#!/bin/bash
# Neural SID - Full build pipeline
# Generates synthetic corpus → trains model → exports weights → packs .d64

set -e

HIDDEN=128
EPOCHS=100
DURATION=120

echo "=== Neural SID Build Pipeline ==="
echo ""

# Step 1: Generate synthetic corpus
echo "[1/4] Generating synthetic corpus..."
cd tools
python synthetic_corpus.py --output ../data/synthetic.bin --duration $DURATION
cd ..

# Step 2: Train model
echo ""
echo "[2/4] Training model (hidden=$HIDDEN, epochs=$EPOCHS)..."
cd train
python train.py \
    --corpus ../data/synthetic.bin \
    --hidden $HIDDEN \
    --epochs $EPOCHS \
    --device cpu
cd ..

# Step 3: Export weights
echo ""
echo "[3/4] Exporting weights..."
cd train
python export.py \
    --model checkpoints/best.pt \
    --output ../data/weights.bin \
    --hidden $HIDDEN \
    --verify
cd ..

# Step 4: Pack .d64 (weights only for now, until .prg builds)
echo ""
echo "[4/4] Packing .d64..."
cd tools
python d64pack.py \
    --weights ../data/weights.bin \
    --output ../data/neural-sid.d64
cd ..

echo ""
echo "=== Done ==="
echo "Output: data/neural-sid.d64"
echo ""
echo "To test in VICE: x64 -autostart data/neural-sid.d64"
