import torch

# Common parameters
latent_dim = 256  # Dimension of the latent space
image_size = 256  # Size of the input images
num_codebook_vectors = 128  # Number of vectors in the codebook
attn_resolution = [16]  # Resolution for attention blocks
beta = 0.25  # Coefficient for the commitment loss
image_channels = 3  # Number of channels in the input images
dataset_path = "dataset\\oxford_17flowers"  # Path to the dataset
device = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Device for training: cuda if available, else cpu
batch_size = 5  # Batch size for training
epochs = 50  # Number of epochs for training
learning_rate = 2.25e-05  # Learning rate for training
beta1 = 0.5  # Beta1 value for Adam optimizer
beta2 = 0.9  # Beta2 value for Adam optimizer
disc_start = 0  # Epoch to start training the discriminator
disc_factor = 1.0  # Factor for adjusting discriminator loss
rec_loss_factor = 1.0  # Factor for adjusting reconstruction loss
perceptual_loss_factor = 1.0  # Factor for adjusting perceptual loss
num_workers = 8

# Decoder parameters
decoder_channels = [
    512,
    256,
    256,
    128,
    128,
]  # Number of channels in each layer of the decoder
decoder_num_res_blocks = 3  # Number of residual blocks in the decoder
decoder_resolution = 16  # Initial resolution for the decoder

# Encoder parameters
encoder_channels = [
    128,
    128,
    128,
    256,
    256,
    512,
]  # Number of channels in each layer of the encoder
encoder_num_res_block = 2  # Number of residual blocks in the encoder
encoder_resolution = 256  # Initial resolution for the encoder

# Training parameters
training_params = {
    "latent_dim": latent_dim,
    "image_size": image_size,
    "num_codebook_vectors": num_codebook_vectors,
    "beta": beta,
    "image_channels": image_channels,
    "dataset_path": dataset_path,
    "device": device,
    "batch_size": batch_size,
    "epochs": epochs,
    "learning_rate": learning_rate,
    "beta1": beta1,
    "beta2": beta2,
    "disc_start": disc_start,
    "disc_factor": disc_factor,
    "rec_loss_factor": rec_loss_factor,
    "perceptual_loss_factor": perceptual_loss_factor,
}

# Transformer parameters
pkeep = 0.5  # Probability of keeping a neuron active during training
sos_token = 0  # Start-of-sequence token
