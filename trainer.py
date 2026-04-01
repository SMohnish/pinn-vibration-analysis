import torch
from core.model import helmholtz_residual
from core.config import LEAKAGE_FACTOR, EPOCHS, LR, DEVICE


def get_batch(tensor, size):
    idx = torch.randint(0, len(tensor), (min(size, len(tensor)),))
    return tensor[idx].to(DEVICE)


def train(model, pts_interior, pts_surface, pts_seal, pts_leakage):
    t_int = torch.tensor(pts_interior, dtype=torch.float32)
    t_surf = torch.tensor(pts_surface, dtype=torch.float32)
    t_seal = torch.tensor(pts_seal, dtype=torch.float32)
    t_leak = torch.tensor(pts_leakage, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    history = {"total": [], "pde": [], "surf": [], "seal": [], "leak": []}

    batch_sizes = {"int": 1500, "surf": 800, "seal": 400, "leak": 300}

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()

        b_int = get_batch(t_int, batch_sizes["int"])
        b_surf = get_batch(t_surf, batch_sizes["surf"])
        b_seal = get_batch(t_seal, batch_sizes["seal"])
        b_leak = get_batch(t_leak, batch_sizes["leak"])

        loss_pde = 1.0 * helmholtz_residual(model, b_int).pow(2).mean()
        loss_surf = 10.0 * (model(b_surf) - 1.0).pow(2).mean()
        loss_seal = 8.0 * (model(b_seal) - 0.0).pow(2).mean()
        loss_leak = 5.0 * (model(b_leak) - LEAKAGE_FACTOR).pow(2).mean()

        loss = loss_pde + loss_surf + loss_seal + loss_leak
        loss.backward()
        optimizer.step()
        scheduler.step()

        del b_int, b_surf, b_seal, b_leak

        if epoch % 10 == 0:
            history["total"].append(loss.item())
            history["pde"].append(loss_pde.item())
            history["surf"].append(loss_surf.item())
            history["seal"].append(loss_seal.item())
            history["leak"].append(loss_leak.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{EPOCHS} — loss: {loss.item():.5f}")

    return history
