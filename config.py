import torch
import numpy as np

# ABS Plastic
C_ABS = 2250.0
RHO_ABS = 1050.0

# Frequency
F_NOM = 2500.0
K = 2 * np.pi * F_NOM / C_ABS

# Spectacle leakage
LEAKAGE_DB = 5.0
LEAKAGE_FACTOR = 10 ** (-LEAKAGE_DB / 20.0)

# Training
N_INTERIOR = 12000
N_BOUNDARY = 4000
N_SEAL = 2000
N_LEAKAGE = 1500
EPOCHS = 6000
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
