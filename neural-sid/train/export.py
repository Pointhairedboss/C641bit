"""
Export trained BinaryMLP weights to packed binary format for the C64.

Weight file format:
  Offset 0x00:       1 byte   — number of layers (N)
  Offset 0x01:       N×4 bytes — layer dims [in:u16le, out:u16le] per layer
  Offset 0x01+N*4:   ...      — packed weight data, layer by layer, row-major
                                 Each row = ceil(in_size / 8) bytes
                                 Weights: 0 = -1, 1 = +1

Packing: MSB first within each byte. 8 weights per byte.
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from model import BinaryMLP


def binarise_weights(model: BinaryMLP) -> list[np.ndarray]:
    """Extract and binarise weights from all layers.
    
    Returns list of uint8 arrays, one per layer.
    Shape per layer: (out_features, in_features) with values in {0, 1}.
    Convention: 0 represents weight -1, 1 represents weight +1.
    """
    binary_layers = []
    for layer in model.layers:
        w = layer.weight.detach().cpu().numpy()
        # sign: negative → 0, non-negative → 1
        binary = (w >= 0).astype(np.uint8)
        binary_layers.append(binary)
    return binary_layers


def pack_layer(binary_weights: np.ndarray) -> bytes:
    """Pack a binary weight matrix into bytes, 8 weights per byte, MSB first.
    
    Input: (out_features, in_features) uint8 array with values in {0, 1}
    Output: bytes, row-major, each row padded to ceil(in_features/8) bytes
    """
    out_features, in_features = binary_weights.shape
    row_bytes = (in_features + 7) // 8
    packed = bytearray()
    
    for row in range(out_features):
        # Pad row to multiple of 8
        bits = binary_weights[row]
        if len(bits) % 8 != 0:
            bits = np.concatenate([bits, np.zeros(8 - len(bits) % 8, dtype=np.uint8)])
        
        # Pack 8 bits per byte, MSB first
        for byte_idx in range(row_bytes):
            byte_val = 0
            for bit_idx in range(8):
                byte_val = (byte_val << 1) | int(bits[byte_idx * 8 + bit_idx])
            packed.append(byte_val)
    
    return bytes(packed)


def export_weights(model: BinaryMLP, output_path: Path) -> None:
    """Export model weights to the C64 binary format."""
    binary_layers = binarise_weights(model)
    dims = model.layer_dims()
    n_layers = len(dims)
    
    # Build header
    header = struct.pack("B", n_layers)
    for in_size, out_size in dims:
        header += struct.pack("<HH", in_size, out_size)
    
    # Pack all layers
    weight_data = b""
    for i, (layer_weights, (in_size, out_size)) in enumerate(zip(binary_layers, dims)):
        packed = pack_layer(layer_weights)
        row_bytes = (in_size + 7) // 8
        expected = row_bytes * out_size
        assert len(packed) == expected, (
            f"Layer {i}: expected {expected} bytes, got {len(packed)}"
        )
        weight_data += packed
        print(f"Layer {i}: {in_size}×{out_size} → {len(packed)} bytes")
    
    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header)
        f.write(weight_data)
    
    total = len(header) + len(weight_data)
    print(f"\nExported {total} bytes ({total / 1024:.1f} KB) to {output_path}")
    print(f"Header: {len(header)} bytes")
    print(f"Weights: {len(weight_data)} bytes")


def verify_export(model: BinaryMLP, export_path: Path) -> None:
    """Verify exported weights by reading them back and comparing."""
    with open(export_path, "rb") as f:
        data = f.read()
    
    # Parse header
    n_layers = data[0]
    offset = 1
    dims = []
    for _ in range(n_layers):
        in_size, out_size = struct.unpack_from("<HH", data, offset)
        dims.append((in_size, out_size))
        offset += 4
    
    # Verify against model
    model_dims = model.layer_dims()
    assert dims == model_dims, f"Dimension mismatch: {dims} vs {model_dims}"
    
    # Read back and compare each layer
    binary_layers = binarise_weights(model)
    for i, (layer_weights, (in_size, out_size)) in enumerate(zip(binary_layers, dims)):
        row_bytes = (in_size + 7) // 8
        layer_size = row_bytes * out_size
        layer_data = data[offset : offset + layer_size]
        offset += layer_size
        
        # Unpack and compare
        for row in range(out_size):
            for byte_idx in range(row_bytes):
                byte_val = layer_data[row * row_bytes + byte_idx]
                for bit_idx in range(8):
                    col = byte_idx * 8 + bit_idx
                    if col >= in_size:
                        break
                    expected = layer_weights[row, col]
                    actual = (byte_val >> (7 - bit_idx)) & 1
                    assert actual == expected, (
                        f"Mismatch at layer {i}, row {row}, col {col}: "
                        f"expected {expected}, got {actual}"
                    )
    
    print("Verification passed: exported weights match model.")


def main():
    parser = argparse.ArgumentParser(description="Export Binary MLP weights for C64")
    parser.add_argument("--model", type=Path, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output .bin path")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden layer size (must match training)")
    parser.add_argument("--num-hidden", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--verify", action="store_true", help="Verify export after writing")
    args = parser.parse_args()
    
    # Recreate model architecture
    model = BinaryMLP(
        input_size=200,
        hidden_size=args.hidden,
        output_size=200,
        num_hidden=args.num_hidden,
    )
    
    # Load trained weights
    state_dict = torch.load(args.model, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded model from {args.model}")
    
    # Export
    export_weights(model, args.output)
    
    if args.verify:
        verify_export(model, args.output)


if __name__ == "__main__":
    main()
