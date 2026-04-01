import gc
import numpy as np
import trimesh
from scipy.ndimage import zoom


def load_mesh(stl_path):
    mesh = trimesh.load(stl_path, force="mesh")

    # auto-scale mm to metres
    if (mesh.bounds[1] - mesh.bounds[0]).max() > 1.0:
        mesh.apply_scale(1e-3)

    mesh.apply_translation(-mesh.bounds.mean(axis=0))

    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)

    return mesh


def build_occupancy_grid(mesh, res=64):
    bbox_min = mesh.bounds[0]
    bbox_max = mesh.bounds[1]
    extents = bbox_max - bbox_min

    pitch = extents.max() / res
    vox = trimesh.voxel.creation.voxelize(mesh, pitch=pitch)
    raw = vox.matrix.astype(bool)

    if raw.shape != (res, res, res):
        factors = (res / raw.shape[0], res / raw.shape[1], res / raw.shape[2])
        raw = zoom(raw.astype(np.float32), factors, order=0).astype(bool)

    xs = np.linspace(bbox_min[0], bbox_max[0], res)
    ys = np.linspace(bbox_min[1], bbox_max[1], res)
    zs = np.linspace(bbox_min[2], bbox_max[2], res)

    return raw, xs, ys, zs, bbox_min


def is_inside(occ_grid, occ_xs, occ_ys, occ_zs, bbox_min, pts):
    res = occ_grid.shape[0]
    dx = occ_xs[1] - occ_xs[0]
    dy = occ_ys[1] - occ_ys[0]
    dz = occ_zs[1] - occ_zs[0]

    ix = np.clip(((pts[:, 0] - bbox_min[0]) / dx).astype(int), 0, res - 1)
    iy = np.clip(((pts[:, 1] - bbox_min[1]) / dy).astype(int), 0, res - 1)
    iz = np.clip(((pts[:, 2] - bbox_min[2]) / dz).astype(int), 0, res - 1)

    return occ_grid[ix, iy, iz]
