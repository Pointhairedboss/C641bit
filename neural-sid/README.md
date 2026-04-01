# Neural SID 🎵⚡

**A 1-bit neural network that generates music natively on a Commodore 64.**

The C64's 6510 CPU runs a binary multilayer perceptron using XOR + popcount — no
multiplication, no floating point. Weights are trained offline in Python, quantised to
1-bit ({-1, +1}), packed 8 per byte, and loaded from a 1541 floppy disk. Output goes
straight to the SID sound chip. No screen. No ROM. Just inference and audio.

## How It Works

1. **Train** a small MLP on SID register frame sequences (Python/PyTorch)
2. **Quantise** weights to 1-bit and pack 8 per byte
3. **Export** to a .d64 floppy disk image
4. **Run** on a real C64 or VICE emulator — the network dreams SID music

## Quick Start

```bash
# Train on synthetic data
cd train && pip install -r requirements.txt
python train.py --corpus ../data/synthetic.bin --hidden 128 --epochs 100
python export.py --model checkpoints/best.pt --output ../data/weights.bin

# Build C64 binary (requires ACME assembler)
cd ../c64 && make

# Pack disk image
cd ../tools && python d64pack.py \
  --prg ../c64/neural-sid.prg \
  --weights ../data/weights.bin \
  --output ../data/neural-sid.d64

# Run in VICE
x64 -autostart ../data/neural-sid.d64
```

## Architecture

- **Model:** 200 → 128 → 128 → 200 (1-bit weights, ~8KB packed)
- **Input/Output:** 25 SID registers × 8 bits = 200 binary neurons
- **Inference:** XOR + popcount via lookup table, ~2-3 frames at 50Hz
- **Memory:** All ROMs banked out, ~54KB available for weights + code

## Project Structure

```text
neural-sid/
├── SPEC.md              # Full technical specification
├── CLAUDE.md            # Agent development instructions
├── train/               # Python training pipeline
│   ├── model.py         # BinaryMLP with straight-through estimator
│   ├── train.py         # Training loop
│   └── export.py        # Weight binarisation & packing
├── c64/                 # 6502 assembly (ACME syntax)
│   ├── main.asm         # Entry point
│   ├── inference.asm    # XOR+popcount forward pass
│   ├── sid_output.asm   # SID register writer
│   └── fastload.asm     # Minimal 1541 fast loader
├── tools/               # Build utilities
│   ├── d64pack.py       # .d64 disk image creator
│   ├── sid_dump.py      # HVSC → register dump converter
│   └── synthetic_corpus.py  # Test data generator
└── data/                # Training data & outputs
```

## Why?

Because BitNet proved 1-bit weights work. Because the 6502's XOR instruction
makes binary neural nets genuinely efficient. Because the SID chip deserves to
dream. And because "we scaled down" is a better story than "we scaled up."

## License

MIT
