import torch
import torch.nn as nn
from core.config import K


class EarmuffPINN(nn.Module):
    def __init__(self, layers=6, width=128):
        super().__init__()
        sizes = [3] + [width] * layers + [1]
        self.linears = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        ])
        self.act = nn.SiLU()

        for lin in self.linears:
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)

        self.register_buffer("x_mean", torch.zeros(3))
        self.register_buffer("x_std", torch.ones(3))

    def forward(self, xyz):
        x = (xyz - self.x_mean) / (self.x_std + 1e-8)
        for lin in self.linears[:-1]:
            x = self.act(lin(x))
        return self.linears[-1](x)

    def set_normalization(self, pts_np):
        import numpy as np
        self.x_mean.copy_(torch.tensor(pts_np.mean(0), dtype=torch.float32))
        self.x_std.copy_(torch.tensor(pts_np.std(0), dtype=torch.float32))


def helmholtz_residual(model, pts):
    pts = pts.clone().requires_grad_(True)
    u = model(pts)

    grad_u = torch.autograd.grad(
        u, pts, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]

    laplacian = sum(
        torch.autograd.grad(
            grad_u[:, i : i + 1],
            pts,
            torch.ones(len(pts), 1, device=pts.device),
            create_graph=True,
            retain_graph=True,
        )[0][:, i : i + 1]
        for i in range(3)
    )

    return laplacian + (K**2) * u
