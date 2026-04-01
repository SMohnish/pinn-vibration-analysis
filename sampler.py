import numpy as np
import trimesh


def sample_interior(mesh, n, dom_min, dom_max, chunk=2000):
    pts = []
    while len(pts) < n:
        cands = np.random.uniform(dom_min, dom_max, (chunk, 3)).astype(np.float32)
        inside = mesh.contains(cands)
        pts.extend(cands[inside].tolist())
    return np.array(pts[:n], dtype=np.float32)


def sample_surface(mesh, n):
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts.astype(np.float32)


def sample_seal(mesh, n, bbox_min, bbox_extents):
    pts, _ = trimesh.sample.sample_surface(mesh, n * 10)

    z_thresh = bbox_min[2] + 0.15 * bbox_extents[2]
    bottom = pts[pts[:, 2] < z_thresh]

    if len(bottom) > 0:
        cx = (mesh.bounds[0][0] + mesh.bounds[1][0]) / 2
        cy = (mesh.bounds[0][1] + mesh.bounds[1][1]) / 2
        r = np.sqrt((bottom[:, 0] - cx) ** 2 + (bottom[:, 1] - cy) ** 2)
        rim = bottom[r > 0.60 * r.max()]
    else:
        rim = bottom

    if len(rim) < n:
        idx = np.random.choice(max(len(rim), 1), n, replace=True)
        rim = rim[idx]

    return rim[:n].astype(np.float32)


def sample_leakage(mesh, n, bbox_min, bbox_extents):
    pts, _ = trimesh.sample.sample_surface(mesh, n * 8)

    y_hi = bbox_min[1] + 0.70 * bbox_extents[1]
    z_lo = bbox_min[2] + 0.25 * bbox_extents[2]
    leak = pts[(pts[:, 1] > y_hi) & (pts[:, 2] < z_lo)]

    if len(leak) == 0:
        leak = pts[pts[:, 1] > y_hi]

    if len(leak) < n:
        idx = np.random.choice(max(len(leak), 1), n, replace=True)
        leak = leak[idx]

    return leak[:n].astype(np.float32)
