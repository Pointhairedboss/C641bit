"""
Training loop for the Binary MLP SID predictor.

Trains on sequences of bit-decomposed SID register frames,
predicting frame_t+1 from frame_t.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import BinaryMLP


def load_corpus(path: Path) -> np.ndarray:
    """Load a SID register corpus file.
    
    Format: raw sequence of 25-byte frames (one per SID state snapshot).
    Returns: array of shape (n_frames, 25) with uint8 values.
    """
    data = np.fromfile(path, dtype=np.uint8)
    n_frames = len(data) // 25
    if n_frames < 2:
        raise ValueError(f"Corpus too small: {len(data)} bytes, need at least 50")
    return data[: n_frames * 25].reshape(n_frames, 25)


def bit_decompose(frames: np.ndarray) -> np.ndarray:
    """Decompose each byte into 8 bits.
    
    Input:  (n_frames, 25) uint8
    Output: (n_frames, 200) float32 in {0, 1}
    """
    # Unpack each byte into 8 bits, MSB first
    bits = np.unpackbits(frames, axis=1).astype(np.float32)
    return bits


def make_pairs(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create (input, target) pairs from sequential frames.
    
    Input:  frame_t
    Target: frame_t+1
    """
    x = bits[:-1]
    y = bits[1:]
    return x, y


def train(
    corpus_path: Path,
    hidden_size: int = 128,
    num_hidden: int = 2,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    checkpoint_dir: Path = Path("checkpoints"),
    device: str = "cpu",
) -> BinaryMLP:
    """Train the BinaryMLP on a SID register corpus."""
    
    print(f"Loading corpus from {corpus_path}")
    frames = load_corpus(corpus_path)
    print(f"Loaded {len(frames)} frames ({len(frames) / 50:.1f}s at 50Hz)")
    
    bits = bit_decompose(frames)
    x_data, y_data = make_pairs(bits)
    print(f"Training pairs: {len(x_data)}")
    
    # Train/val split (90/10)
    split = int(len(x_data) * 0.9)
    x_train, x_val = x_data[:split], x_data[split:]
    y_train, y_val = y_data[:split], y_data[split:]
    
    train_ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val),
        torch.from_numpy(y_val),
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Model
    model = BinaryMLP(
        input_size=200,
        hidden_size=hidden_size,
        output_size=200,
        num_hidden=num_hidden,
    ).to(device)
    
    print(f"\nModel: {model.layer_dims()}")
    print(f"Weights: {model.total_weights():,} ({model.packed_size_bytes():,} bytes packed)")
    
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            train_loss += loss.item() * len(x_batch)
            preds = (logits > 0).float()
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.numel()
        
        train_loss /= len(train_ds)
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(x_batch)
                preds = (logits > 0).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.numel()
        
        val_loss /= len(val_ds)
        val_acc = val_correct / val_total
        
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | "
                f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f} acc: {val_acc:.4f}"
            )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
    
    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Model saved to {checkpoint_dir / 'best.pt'}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Binary MLP for SID prediction")
    parser.add_argument("--corpus", type=Path, required=True, help="Path to SID corpus .bin")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--num-hidden", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()
    
    train(
        corpus_path=args.corpus,
        hidden_size=args.hidden,
        num_hidden=args.num_hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
