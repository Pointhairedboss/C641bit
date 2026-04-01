"""
Binary MLP for SID register prediction.

1-bit weights ({-1, +1}) with straight-through estimator for training.
At inference time on the C64, the forward pass becomes XOR + popcount.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SignSTE(torch.autograd.Function):
    """Sign activation with straight-through estimator gradient."""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.sign(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        # Straight-through: pass gradient where |x| <= 1
        grad_input = grad_output.clone()
        grad_input[x.abs() > 1.0] = 0
        return grad_input


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    """Apply sign function with straight-through estimator."""
    return SignSTE.apply(x)


class BinaryLinear(nn.Module):
    """Linear layer that binarises weights during forward pass.
    
    Maintains full-precision weights for gradient updates,
    but snaps to {-1, +1} for the actual computation.
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binary_weight = sign_ste(self.weight)
        return F.linear(x, binary_weight)
    
    def binarised_weight(self) -> torch.Tensor:
        """Return the binarised weight matrix for export."""
        with torch.no_grad():
            return torch.sign(self.weight).clamp(min=0).to(torch.uint8)


class BinaryMLP(nn.Module):
    """Multilayer perceptron with 1-bit weights for SID prediction.
    
    Input:  200-bit vector (25 SID registers × 8 bits)
    Output: 200-bit vector (predicted next frame)
    Hidden: configurable size, default 128 neurons
    """
    
    def __init__(
        self,
        input_size: int = 200,
        hidden_size: int = 128,
        output_size: int = 200,
        num_hidden: int = 2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        layers: list[nn.Module] = []
        
        # Input layer
        layers.append(BinaryLinear(input_size, hidden_size))
        
        # Hidden layers
        for _ in range(num_hidden - 1):
            layers.append(BinaryLinear(hidden_size, hidden_size))
        
        # Output layer
        layers.append(BinaryLinear(hidden_size, output_size))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert {0,1} input to {-1,+1} for computation
        x = x * 2.0 - 1.0
        
        # Hidden layers with sign activation
        for layer in self.layers[:-1]:
            x = sign_ste(layer(x))
        
        # Output layer: raw logits for loss computation
        x = self.layers[-1](x)
        return x
    
    def layer_dims(self) -> list[tuple[int, int]]:
        """Return (in_size, out_size) for each layer."""
        return [
            (layer.in_features, layer.out_features)
            for layer in self.layers
        ]
    
    def total_weights(self) -> int:
        """Total number of weight parameters."""
        return sum(l.in_features * l.out_features for l in self.layers)
    
    def packed_size_bytes(self) -> int:
        """Size of packed 1-bit weights in bytes."""
        total_bits = self.total_weights()
        return (total_bits + 7) // 8


if __name__ == "__main__":
    model = BinaryMLP(input_size=200, hidden_size=128, output_size=200, num_hidden=2)
    print(f"Layer dimensions: {model.layer_dims()}")
    print(f"Total weights:    {model.total_weights():,}")
    print(f"Packed size:      {model.packed_size_bytes():,} bytes ({model.packed_size_bytes() / 1024:.1f} KB)")
    
    # Test forward pass
    x = torch.randint(0, 2, (4, 200), dtype=torch.float32)
    y = model(x)
    print(f"Input shape:      {x.shape}")
    print(f"Output shape:     {y.shape}")
