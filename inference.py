import gc
import numpy as np
import torch
from core.config import DEVICE
from geometry.mesh_utils import is_inside


def run_inference(model, dom_min, dom_max, occ_grid, occ_xs, occ_ys, occ_zs,
                  bbox_min, grid_res=40, batch_size=512):

    xs = np.linspace(dom_min[0], dom_max[0], grid_res)
    ys = np.linspace(dom_min[1], dom_max[1], grid_res)
    zs = np.linspace(dom_min[2], dom_max[2], grid_res)
    dx, dy, dz = xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1).astype(np.float32)
    del XX, YY, ZZ
    gc.collect()

    inside_mask = is_inside(occ_grid, occ_xs, occ_ys, occ_zs, bbox_min, grid_pts)

    model.eval()
    u_pred = np.zeros(len(grid_pts), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(grid_pts), batch_size):
            b = torch.tensor(grid_pts[i : i + batch_size], device=DEVICE)
            u_pred[i : i + batch_size] = model(b).squeeze().cpu().numpy()
            del b

    u_pred[~inside_mask] = 0.0
    U_vol = u_pred.reshape(grid_res, grid_res, grid_res)
    del grid_pts, u_pred
    gc.collect()

    S_vol = np.sqrt(
        np.gradient(U_vol, dx, axis=0) ** 2
        + np.gradient(U_vol, dy, axis=1) ** 2
        + np.gradient(U_vol, dz, axis=2) ** 2
    )
    inside_3d = inside_mask.reshape(grid_res, grid_res, grid_res)
    S_vol[~inside_3d] = 0.0
    del inside_mask
    gc.collect()

    return U_vol, S_vol, inside_3d, xs, ys, zs
