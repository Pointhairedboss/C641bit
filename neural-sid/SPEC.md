# Neural SID: 1-Bit Neural Network Music Generator for Commodore 64

## Overview

A 1-bit quantised multilayer perceptron that runs native inference on a Commodore 64,
generating music by predicting SID chip register states frame-by-frame. Weights are
trained offline in Python, packed 8-per-byte, stored on a 1541 floppy disk, and paged
into RAM layer-by-layer during inference.

The C64 runs with BASIC ROM, KERNAL ROM, and screen RAM all reclaimed for weight
storage. Output is driven entirely through the SID chip via a 50Hz PAL IRQ.

## Architecture

### Neural Network

- **Type:** Multilayer Perceptron (MLP), fully connected
- **Quantisation:** 1-bit weights ({-1, +1}), stored as {0, 1} packed 8 per byte
- **Activation:** Binary sign function (sign of accumulated sum)
- **Task:** Autoregressive SID register frame prediction
- **Input:** Current SID register state (25 bytes = 200 bits after bit-decomposition)
- **Output:** Next SID register state (25 bytes = 200 bits)
- **Hidden layers:** 2× 256 neurons (configurable)
- **Total weights:** 200×256 + 256×256 + 256×200 = 117,600
- **Packed size:** 117,600 / 8 = 14,700 bytes (~14.4 KB)

This fits entirely in RAM with room to spare. Larger models can page from disk.

### SID Register Map (25 bytes)

The SID chip (6581/8580) is memory-mapped at $D400-$D418:

| Offset | Register          | Voice |
|--------|-------------------|-------|
| $00-04 | Freq, PW, Control | 1     |
| $07-0B | Freq, PW, Control | 2     |
| $0E-12 | Freq, PW, Control | 3     |
| $05-06 | ADSR              | 1     |
| $0C-0D | ADSR              | 2     |
| $13-14 | ADSR              | 3     |
| $15-18 | Filter + Volume   | -     |

Total: 25 writable registers per frame.

### Input Encoding

Each of the 25 SID register bytes is bit-decomposed into 8 binary values,
giving a 200-bit input vector. This maps naturally to the 1-bit weight scheme:
the dot product of a binary input with binary weights is XOR + popcount.

### Inference Pipeline (per frame, 50Hz)

1. Read current SID state (25 bytes) → bit-decompose → 200-bit input vector
2. Layer 1: input(200) × weights(200×256) → 256 activations (sign function)
3. Layer 2: hidden(256) × weights(256×256) → 256 activations (sign function)
4. Layer 3: hidden(256) × weights(256×200) → 200-bit output
5. Pack 200-bit output → 25 bytes
6. Write to SID registers $D400-$D418

### Timing Budget

At 1MHz PAL (985,248 cycles/sec), 50Hz gives ~19,705 cycles per frame.

Per-neuron cost (256 inputs): 32 XOR + 32 popcount ≈ 320 cycles
Layer 1 (256 neurons × 200 inputs): ~256 × 250 cycles ≈ 64,000 cycles
Layer 2 (256 neurons × 256 inputs): ~256 × 320 cycles ≈ 81,920 cycles
Layer 3 (200 neurons × 256 inputs): ~200 × 320 cycles ≈ 64,000 cycles

Total: ~210,000 cycles ≈ ~10.7 frames

**This means inference takes ~10 frames at 50Hz, giving ~5Hz output rate.**

Options to improve:
- Reduce hidden layer size to 128 neurons → ~4× faster → ~20Hz
- Run at lower frame rate (acceptable for ambient/generative music)
- Use 128-neuron hidden layers: total ~30,000 weights = 3.75KB, ~52K cycles = ~2.6 frames

**Recommended default: 200→128→128→200, giving ~2-3 frame inference = 17-25Hz**

## Memory Map (C64, all ROMs banked out)

```text
$0000-$00FF  Zero page: inner loop variables, pointers, accumulators
$0100-$01FF  Stack
$0200-$03FF  Code: inference engine, IRQ handler, SID write routine
$0400-$07FF  Reclaimed screen RAM: overflow weights or activation buffers
$0800-$9FFF  Main weight storage (~38 KB)
$A000-$BFFF  Reclaimed BASIC ROM: weight storage (+8 KB)
$C000-$CFFF  Activation buffers, frame state (4 KB)
$D000-$D3FF  I/O area (VIC, SID, CIA - keep mapped)
$D400-$D418  SID registers (output target)
$D800-$DBFF  Color RAM (can be ignored)
$DC00-$DFFF  CIA (need for IRQ timing)
$E000-$FFFF  Reclaimed KERNAL ROM: weight storage or code (+8 KB)
```

Usable RAM for weights: ~54 KB = 432,000 bits = enough for 200→128→128→200 many times over.

## C64 Implementation

### Language & Assembler

6502 assembly, targeting the ACME cross-assembler (widely available, simple syntax).
Output: .prg file (with 2-byte load address header).

### Core Routines

1. **init.asm** - Bank out ROMs, set up IRQ vector, init SID
2. **inference.asm** - The XOR+popcount forward pass
3. **sid_output.asm** - Write prediction to SID registers
4. **fastload.asm** - Minimal 1541 fast loader (for disk-paged models)
5. **main.asm** - Master file, includes all modules

### XOR + Popcount Inner Loop (Zero Page)

```text
; Compute dot product of 256-bit input with 256-bit weight row
; Input at $10-$2F (32 bytes = 256 bits)
; Weights loaded at $30-$4F (32 bytes)
; Accumulator in X register

    LDX #0          ; dot product accumulator
    LDY #31         ; byte counter (32 bytes = 256 bits)
.loop:
    LDA $10,Y       ; input byte
    EOR $30,Y       ; XOR with weight byte
    TAX
    LDA popcount_table,X  ; lookup popcount
    ; popcount gives number of 1s = number of mismatches
    ; dot product = total_bits - 2 * mismatches
    ; but for sign activation we only need: if popcount > 128 → -1, else → +1
    CLC
    ADC accumulator
    STA accumulator
    DEY
    BPL .loop
    ; Now: if accumulator > half_total → output bit = 0, else = 1
```

### Popcount Lookup Table

256-byte table mapping each byte value to its popcount (0-8).
Stored in a fixed page for fast indexed access.

## Training Pipeline (Python)

### Dependencies

- PyTorch (training)
- NumPy (data manipulation)
- struct/bitarray (weight packing)

### Data Preparation

1. Source: HVSC SID collection, dumped via libsidplayfp to raw register writes
2. Alternative: synthetic corpus generator for testing
3. Format: sequence of 25-byte frames at 50Hz
4. Bit-decompose each frame → 200-bit vectors
5. Training pairs: (frame_t, frame_t+1)

### Model

```python
class BinaryMLP(nn.Module):
    def __init__(self, input_size=200, hidden_size=128, output_size=200):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc3 = nn.Linear(hidden_size, output_size, bias=False)
    
    def forward(self, x):
        x = torch.sign(self.fc1(x))
        x = torch.sign(self.fc2(x))
        x = self.fc3(x)  # final layer: continuous for loss, binarised at export
        return x
```

### Training

- Loss: Binary cross-entropy (per-bit prediction)
- Optimiser: Adam with straight-through estimator for sign activation gradient
- Binarisation: weights snapped to {-1, +1} at export time via sign()
- Epochs: until convergence on validation set

### Export

1. Binarise all weights: w → 0 if w < 0, 1 if w ≥ 0
2. Pack 8 weights per byte (MSB first)
3. Write layer-by-layer as raw binary
4. Header: layer count, dimensions per layer (for loader)

### Weight File Format

```text
Offset  Size     Description
0x00    1 byte   Number of layers (N)
0x01    N×4      Layer dimensions: [in_size:u16_le, out_size:u16_le] × N
0x01+N×4  ...    Packed weight data, layer by layer, row-major
                  Each row = ceil(in_size / 8) bytes
```

## .d64 Disk Image

Use a Python tool to create a standard CBM DOS .d64 image containing:
- The weight file as a PRG or SEQ file
- Optionally: the C64 inference program as a PRG

Standard .d64 = 35 tracks, 683 blocks, 174,848 bytes usable.

## Synthetic Test Corpus

For development without HVSC, generate synthetic SID register sequences:
- Simple arpeggios (cycling through frequency values)
- Pulse width sweeps
- Filter sweeps
- Random walks with momentum on frequency registers
- Known patterns that a small MLP should be able to learn

## Build & Run

### Training
```bash
cd train/
pip install -r requirements.txt
python train.py --corpus data/synthetic.bin --epochs 100 --hidden 128
python export.py --model checkpoints/best.pt --output ../data/weights.bin
```

### C64 Build
```bash
cd c64/
make                    # requires ACME assembler
# produces: neural-sid.prg
```

### Disk Image
```bash
cd tools/
python d64pack.py --prg ../c64/neural-sid.prg --weights ../data/weights.bin --output ../data/neural-sid.d64
```

### Test
```bash
# In VICE emulator:
x64 -autostart data/neural-sid.d64
```

## Development Issues

See issues/ directory for individual task breakdowns suitable for
Claude Code or other agent-driven development.

## References

- HVSC: https://www.hvsc.c64.org/
- libsidplayfp: https://github.com/libsidplayfp/libsidplayfp
- ACME assembler: https://sourceforge.net/projects/acme-crossass/
- SID register map: https://www.c64-wiki.com/wiki/SID
- VICE emulator: https://vice-emu.sourceforge.io/
- BitNet: https://arxiv.org/abs/2310.11453
