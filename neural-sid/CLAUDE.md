# CLAUDE.md

## Project: Neural SID

A 1-bit neural network that runs native 6502 inference on a Commodore 64, generating
music through the SID chip.

## Key Files

- `SPEC.md` — Complete technical specification. READ THIS FIRST.
- `issues/` — Task breakdown. Work through these in order.
- `train/` — Python training pipeline (PyTorch)
- `c64/` — 6502 assembly for ACME cross-assembler
- `tools/` — Build tools (d64 packer, SID dumper, synthetic data generator)
- `data/` — Training data, weights, disk images (gitignored binaries)

## Architecture Decisions

- 1-bit weights packed 8 per byte. {-1,+1} stored as {0,1}.
- Forward pass is XOR + popcount. No multiplication needed.
- Default model: 200→128→128→200 (input/output = 25 SID registers × 8 bits)
- C64 runs with all ROMs banked out. No BASIC, no KERNAL, no screen.
- Output is 100% SID chip. No video output.
- PAL timing: 50Hz IRQ drives inference + SID register writes.
- Popcount via 256-byte lookup table.

## Development Order

1. `train/model.py` — BinaryMLP with straight-through estimator
2. `train/train.py` — Training loop
3. `train/export.py` — Weight binarisation and packing
4. `tools/synthetic_corpus.py` — Generate test data (arpeggio patterns, sweeps)
5. `tools/d64pack.py` — Create .d64 disk images
6. `c64/` assembly files — ACME syntax, target is .prg output
7. Integration: train → export → pack → run in VICE

## Coding Standards

- Python: 3.10+, type hints, minimal dependencies
- 6502 assembly: ACME syntax, heavy comments explaining every trick
- All magic numbers get named constants
- Weight file format is documented in SPEC.md — follow it exactly

## Testing

- Python tests: pytest, validate weight packing round-trips
- Synthetic corpus: patterns that are trivially learnable (constant, alternating)
- VICE emulator for C64 testing (not automated, manual verification)

## Constraints

- C64 has 64KB RAM total, ~54KB usable after banking out ROMs
- 1541 disk holds ~170KB
- 6510 CPU at 1MHz (985,248 cycles/sec PAL)
- No floating point. All inference is integer/bitwise operations.
- No screen output. The SID chip IS the output device.
