# Earmuff 3D Vibration Analysis — PINN

Physics-Informed Neural Network for passive noise cancellation earmuff analysis.

## What it does

Solves the 3D Helmholtz wave equation on your earmuff STL geometry:

```
∇²u + k²u = 0
```

Material: ABS Plastic (c = 2250 m/s, ρ = 1050 kg/m³)  
Frequency range: 1–4 kHz  
Output: Displacement and stress contour maps + `.vti` files for ParaView

## Repo structure

```
earmuff-pinn/
├── core/
│   ├── config.py          # physics constants, training hyperparameters
│   └── model.py           # PINN network + Helmholtz residual
├── geometry/
│   ├── mesh_utils.py      # STL loading, occupancy grid
│   └── sampler.py         # interior/surface/seal/leakage point sampling
├── training/
│   ├── trainer.py         # training loop
│   └── inference.py       # volumetric grid inference
├── visualization/
│   ├── plots.py           # contour maps, loss curves, frequency sweep
│   └── vtk_export.py      # .vti and .vtk export for ParaView
├── notebooks/
│   └── earmuff_main.ipynb # main Colab notebook
└── requirements.txt
```

## Run on Google Colab

1. Upload `earmuff_main.ipynb` to Colab
2. Set runtime to **T4 GPU**
3. Run Cell 1 — it clones this repo automatically
4. Run Cell 2 — upload your STL file
5. Run remaining cells in order

## Boundary conditions

| Zone | Value | Meaning |
|------|-------|---------|
| Outer shell | u = 1.0 | Incident noise |
| Seal ring | u = 0.0 | Rigid ear contact |
| Spectacle gap | u = 0.562 | 5 dB leakage penalty |

## ParaView

Open `earmuff_displacement.vti`:
- Color by `displacement_u` → Representation → Volume
- Color map: Rainbow Desaturated
- Red = max vibration, Blue = attenuated
