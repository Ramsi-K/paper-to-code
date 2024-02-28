import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from codebook import CodeBook
from config import (
    VQGAN_CONFIG,
    ENCODER_CONFIG,
    DECODER_CONFIG,
    CODEBOOK_CONFIG,
    DEVICE,
)


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

    def __init__(
        self, VQGAN_CONFIG, ENCODER_CONFIG, DECODER_CONFIG, CODEBOOK_CONFIG
    ):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(ENCODER_CONFIG).to(
            device=VQGAN_CONFIG["device"]
        )
        self.decoder = Decoder(DECODER_CONFIG).to(
            device=VQGAN_CONFIG["device"]
        )
        self.codebook = CodeBook(CODEBOOK_CONFIG).to(
            device=VQGAN_CONFIG["device"]
        )
        self.quant_conv = nn.Conv2d(
            VQGAN_CONFIG["latent_dim"], VQGAN_CONFIG["latent_dim"], 1
        ).to(device=VQGAN_CONFIG["device"])
        self.post_quant_conv = nn.Conv2d(
            VQGAN_CONFIG["latent_dim"], VQGAN_CONFIG["latent_dim"], 1
        ).to(device=VQGAN_CONFIG["device"])

    def forward(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(
            quant_conv_encoded_images
        )
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(
            quant_conv_encoded_images
        )
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

    def calc_lambda(self, perceptual_loss, gan_loss):
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
        if i < threshold:
            disc_factor = val
            return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))


def test_vqgan():
    # Create an instance of the VQGAN model using configuration from config.py
    vqgan = VQGAN(
        VQGAN_CONFIG, ENCODER_CONFIG, DECODER_CONFIG, CODEBOOK_CONFIG
    )

    # Generate some dummy input data
    batch_size = 4
    channels = 3
    height = 128
    width = 128
    dummy_input = torch.randn(batch_size, channels, height, width)

    # Forward pass through the VQGAN model
    decoded_images, codebook_indices, q_loss = vqgan(dummy_input)

    # Print information about the output
    print("Decoded Images Shape:", decoded_images.shape)
    print("Codebook Indices Shape:", codebook_indices.shape)
    print("Quantization Loss:", q_loss.item())


if __name__ == "__main__":
    test_vqgan()
