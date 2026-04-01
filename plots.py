import gc
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0
from core.config import C_ABS, F_NOM, LEAKAGE_DB, LEAKAGE_FACTOR, K


def plot_contours(U_vol, S_vol, xs, ys, zs, mesh, stl_name, out_path):
    cx, cy, cz = mesh.centroid
    ix = int(np.argmin(np.abs(xs - cx)))
    iy = int(np.argmin(np.abs(ys - cy)))
    iz = int(np.argmin(np.abs(zs - cz)))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    def voxel_slice(ax, data2d, xvals, yvals, title, xl, yl, clabel):
        d = data2d.T.copy().astype(float)
        d[d == 0] = np.nan
        vmin = np.nanpercentile(d, 2)
        vmax = np.nanpercentile(d, 98)
        im = ax.imshow(
            d, origin="lower", aspect="auto", cmap="jet",
            vmin=vmin, vmax=vmax,
            extent=[xvals[0] * 100, xvals[-1] * 100, yvals[0] * 100, yvals[-1] * 100],
        )
        fig.colorbar(im, ax=ax, label=clabel, shrink=0.85)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel(f"{xl} [cm]")
        ax.set_ylabel(f"{yl} [cm]")

    voxel_slice(axes[0, 0], U_vol[ix, :, :], ys, zs, f"Displacement — X-slice (x={cx*100:.1f}cm)", "y", "z", "|u|")
    voxel_slice(axes[0, 1], U_vol[:, iy, :], xs, zs, f"Displacement — Y-slice (y={cy*100:.1f}cm)", "x", "z", "|u|")
    voxel_slice(axes[0, 2], U_vol[:, :, iz], xs, ys, f"Displacement — Z-slice (z={cz*100:.1f}cm)", "x", "y", "|u|")
    voxel_slice(axes[1, 0], S_vol[ix, :, :], ys, zs, "Stress |∇u| — X-slice", "y", "z", "[1/m]")
    voxel_slice(axes[1, 1], S_vol[:, iy, :], xs, zs, "Stress |∇u| — Y-slice", "x", "z", "[1/m]")
    voxel_slice(axes[1, 2], S_vol[:, :, iz], xs, ys, "Stress |∇u| — Z-slice", "x", "y", "[1/m]")

    plt.suptitle(
        f"{stl_name}  |  ABS Plastic  |  f = {F_NOM/1000:.1f} kHz\n"
        f"RED = max vibration  |  WHITE = outside mesh  |  Leakage = −{LEAKAGE_DB} dB",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    gc.collect()
    print(f"[✓] Saved: {out_path}")


def plot_loss(history, epochs, out_path):
    ep = list(range(10, epochs + 1, 10))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(ep, history["total"], "k-",  lw=2,   label="Total")
    ax.semilogy(ep, history["pde"],   "b--", lw=1.5, label="PDE")
    ax.semilogy(ep, history["surf"],  "r-",  lw=1.5, label="Outer BC")
    ax.semilogy(ep, history["seal"],  "g-",  lw=1.5, label="Seal BC")
    ax.semilogy(ep, history["leak"],  "m-",  lw=1.5, label="Leakage BC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Training Loss", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.show()
    gc.collect()
    print(f"[✓] Saved: {out_path}")


def plot_frequency_sweep(U_vol, xs, ys, zs, mesh, stl_name, out_path):
    freqs = [1000, 1500, 2000, 3000, 3500, 4000]
    cx = mesh.centroid[0]
    ix = int(np.argmin(np.abs(xs - cx)))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for idx, f in enumerate(freqs):
        ax = axes[idx // 3][idx % 3]
        k_f = 2 * np.pi * f / C_ABS
        scale = (k_f / K) ** 2
        U_f = U_vol * scale

        d = U_f[ix, :, :].T.copy().astype(float)
        d[d == 0] = np.nan

        im = ax.imshow(
            d, origin="lower", aspect="auto", cmap="jet",
            extent=[ys[0] * 100, ys[-1] * 100, zs[0] * 100, zs[-1] * 100],
        )
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{f/1000:.1f} kHz  (k={k_f:.2f} rad/m)", fontweight="bold")
        ax.set_xlabel("y [cm]")
        ax.set_ylabel("z [cm]")

    plt.suptitle(
        f"Frequency Sweep 1–4 kHz — {stl_name}\n(X centroid slice)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    gc.collect()
    print(f"[✓] Saved: {out_path}")
