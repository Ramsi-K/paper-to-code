import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from codebook import CodeBook
import config


class VQGAN(nn.Module):
    """
    Main VQGAN module for image generation.

    Args:
        config (dict): A dictionary containing configuration parameters.

    Attributes:
        encoder (Encoder): Encoder module.
        decoder (Decoder): Decoder module.
        codebook (CodeBook): CodeBook module.
        quant_conv (nn.Conv2d): Quantization convolution layer.
        post_quant_conv (nn.Conv2d): Post-quantization convolution layer.
    """

    def __init__(self, config):
        """
        Initialize the VQGAN model.

        Args:
            config (dict): Configuration parameters.
        """
        super(VQGAN, self).__init__()
        self.encoder = Encoder(config).to(device=config.device)
        self.decoder = Decoder(config).to(device=config.device)
        self.codebook = CodeBook(config).to(device=config.device)
        self.quant_conv = nn.Conv2d(
            config.latent_dim, config.latent_dim, 1
        ).to(device=config.device)
        self.post_quant_conv = nn.Conv2d(
            config.latent_dim, config.latent_dim, 1
        ).to(device=config.device)

    def forward(self, imgs):
        """
        Forward pass of the VQGAN model.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            tuple: Decoded images, codebook indices, and quantization loss.
        """
        encoded_images = self.encoder(imgs)
        quantized_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(
            quantized_encoded_images
        )
        quantized_codebook_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(quantized_codebook_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, imgs):
        """
        Encode input images.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            tuple: Codebook mapping, codebook indices, and quantization loss.
        """
        encoded_images = self.encoder(imgs)
        quantized_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(
            quantized_encoded_images
        )
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        """
        Decode latent space vectors.

        Args:
            z (torch.Tensor): Latent space vectors.

        Returns:
            torch.Tensor: Decoded images.
        """
        quantized_encoded_images = self.post_quant_conv(z)
        decoded_images = self.decoder(quantized_encoded_images)
        return decoded_images

    def calc_lambda(self, perceptual_loss, gan_loss):
        """
        Calculate lambda for weight adaptation.

        Args:
            perceptual_loss (torch.Tensor): Perceptual loss.
            gan_loss (torch.Tensor): GAN loss.

        Returns:
            torch.Tensor: Lambda value.
        """
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(
            perceptual_loss, last_layer_weight, retain_graph=True
        )[0]
        gan_loss_grads = torch.autograd.grad(
            gan_loss, last_layer_weight, retain_graph=True
        )[0]

        lambda_ = torch.norm(perceptual_loss_grads) / (
            torch.norm(gan_loss_grads) + 1e-4
        )
        lambda_ = torch.clamp(lambda_, 0, 1e4).detach()
        return 0.8 * lambda_

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, val=0.0):
        """
        Adopt weight based on the iteration.

        Args:
            disc_factor (float): Discriminator factor.
            i (int): Iteration index.
            threshold (int): Threshold for adoption.
            val (float): Value to set if condition is met.

        Returns:
            float: Updated discriminator factor.
        """
        if i < threshold:
            disc_factor = val
        return disc_factor

    def load_checkpoint(self, path):
        """
        Load model checkpoint.

        Args:
            path (str): Path to the checkpoint file.
        """
        self.load_state_dict(torch.load(path))


def test_vqgan():
    """
    Test function to check the VQGAN model.
    """
    # Create an instance of the VQGAN model using configuration from config.py
    vqgan = VQGAN(config)

    # Generate some dummy input data
    batch_size = 4
    channels = 3
    height = 128
    width = 128
    dummy_input = torch.randn(batch_size, channels, height, width).to(
        device=config.device
    )

    # Forward pass through the VQGAN model
    decoded_images, codebook_indices, q_loss = vqgan(dummy_input)

    # Print information about the output
    print("Decoded Images Shape:", decoded_images.shape)
    print("Codebook Indices Shape:", codebook_indices.shape)
    print("Quantization Loss:", q_loss.item())


if __name__ == "__main__":
    test_vqgan()
