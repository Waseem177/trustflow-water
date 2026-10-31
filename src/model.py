"""
Reconstruction-based anomaly detector: a symmetric dense autoencoder.

Why an autoencoder?
-------------------
Trained only on normal data, the AE learns to compress (encoder) and rebuild
(decoder) normal sensor vectors through a narrow bottleneck. It can only do this
well for patterns similar to what it saw in training. An attack pushes sensors
into correlations the AE never learned, so it reconstructs them poorly -> high
reconstruction error -> anomaly.

The bottleneck (default 8 << 43) is the whole point: it forces the model to learn
the *physical correlations* between sensors (e.g. pump ON => flow up => tank fills)
rather than memorizing each channel independently.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, n_features: int = 43, hidden=(32, 16), bottleneck: int = 8):
        super().__init__()
        self.n_features = n_features
        dims = [n_features, *hidden, bottleneck]

        enc = []
        for a, b in zip(dims[:-1], dims[1:]):
            enc += [nn.Linear(a, b), nn.ReLU()]
        self.encoder = nn.Sequential(*enc)

        dec = []
        rev = list(reversed(dims))
        for i, (a, b) in enumerate(zip(rev[:-1], rev[1:])):
            dec.append(nn.Linear(a, b))
            if i < len(rev) - 2:          # ReLU on hidden layers only
                dec.append(nn.ReLU())
        # No output activation: standardized inputs are unbounded (can be negative).
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean squared error across features -> shape (N,)."""
        recon = self.forward(x)
        return ((recon - x) ** 2).mean(dim=1)


if __name__ == "__main__":
    m = Autoencoder()
    n = sum(p.numel() for p in m.parameters())
    print(m)
    print(f"trainable params: {n:,}")
    x = torch.randn(4, 43)
    print("forward:", m(x).shape, "recon_err:", m.reconstruction_error(x).shape)
