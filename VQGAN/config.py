import torch

# Decoder configuration
DECODER_CONFIG = {
    "latent_dim": 512,
    "image_channels": 3,
    "channels": [512, 256, 256, 128, 128],  # Number of channels in each layer
    "attn_resolution": [16],  # Resolution for attention blocks
    "num_res_blocks": 3,  # Number of residual blocks
    "resolution": 16,  # Initial resolution
}

# Encoder configuration
ENCODER_CONFIG = {
    "latent_dim": 512,
    "image_channels": 3,
    "channels": [128, 128, 128, 256, 256, 512],
    "num_res_block": 2,
    "attn_resolution": [16],
    "resolution": 256,
}

# Configuration parameters for CodeBook module
CODEBOOK_CONFIG = {
    "num_codebook_vectors": 64,  # Number of vectors in the codebook
    "latent_dim": 32,  # Dimension of the latent space
    "beta": 0.1,  # Coefficient for the commitment loss
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Configuration parameters for VQGAN
VQGAN_CONFIG = {
    "latent_dim": 256,
    "image_channels": 3,
    "channels": [128, 128, 256, 512],
    "num_res_blocks": 2,
    "attn_resolution": [16],
    "device": (
        "cuda" if torch.cuda.is_available() else "cpu"
    ),  # Add device parameter
}
