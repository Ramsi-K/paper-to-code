import torch

# Common parameters
latent_dim = 32
image_size = 32
num_codebook_vectors = 128
beta = 0.25
image_channels = 3
dataset_path = "dataset\\oxford_17flowers"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 6
epochs = 1
learning_rate = 2.25e-05
beta1 = 0.5
beta2 = 0.9
disc_start = 10000
disc_factor = 1.0
rec_loss_factor = 1.0
perceptual_loss_factor = 1.0

# Decoder paramters
decoder_channels = [512, 256, 256, 128, 128]
attn_resolution = [16]
decoder_num_res_blocks = 3
decoder_resolution = 16

# Encoder parameters
encoder_channels = [128, 128, 128, 256, 256, 512]
encoder_num_res_block = 2
encoder_resolution = 256

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
