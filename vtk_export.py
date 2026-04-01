import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def _make_image_data(u_vol, s_vol, xs, ys, zs):
    img = vtk.vtkImageData()
    img.SetDimensions(len(xs), len(ys), len(zs))
    img.SetOrigin(float(xs[0]), float(ys[0]), float(zs[0]))
    img.SetSpacing(float(xs[1] - xs[0]), float(ys[1] - ys[0]), float(zs[1] - zs[0]))

    for arr, name in [(u_vol, "displacement_u"), (s_vol, "stress_magnitude")]:
        va = numpy_to_vtk(
            arr.ravel(order="F").astype(np.float32),
            deep=True,
            array_type=vtk.VTK_FLOAT,
        )
        va.SetName(name)
        img.GetPointData().AddArray(va)

    img.GetPointData().SetActiveScalars("displacement_u")
    return img


def export_vti(u_vol, s_vol, xs, ys, zs, path):
    img = _make_image_data(u_vol, s_vol, xs, ys, zs)
    w = vtk.vtkXMLImageDataWriter()
    w.SetFileName(path)
    w.SetInputData(img)
    w.Write()
    print(f"[✓] Saved: {path}")


def export_vtk_legacy(u_vol, s_vol, xs, ys, zs, path):
    img = _make_image_data(u_vol, s_vol, xs, ys, zs)
    w = vtk.vtkDataSetWriter()
    w.SetFileName(path)
    w.SetInputData(img)
    w.Write()
    print(f"[✓] Saved: {path}")
