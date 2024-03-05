import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model parameters
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20

# Training parameters
NUM_EPOCHS = 20
BATCH_SIZE = 128
LR_RATE = 1e-5
NUM_WORKERS = 4

# Dataset path
root = "dataset/"
