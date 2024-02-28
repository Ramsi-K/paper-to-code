import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from codebook import CodeBook
from config import ENCODER_CONFIG, DECODER_CONFIG, DEVICE, CODEBOOK_CONFIG


class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(args).to(device=args.device)
        self.decoder = Decoder(args).to(device=args.device)
        self.codebook = CodeBook(args).to(device=args.device)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(
            device=args.device
        )
        self.post_quant_conv = nn.Conv2d(
            args.latent_dim, args.latent_dim, 1
        ).to(device=args.device)

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
    """
    Test function to create and run the VQGAN model.

    This function creates an instance of the VQGAN model using the provided
    configuration arguments and performs a forward pass on a dummy input tensor.
    It prints information about the model's output, including decoded images,
    codebook indices, and quantization loss.
    """
    # Create an instance of the VQGAN model
    args = {}  # Add your configuration arguments here
    vqgan = VQGAN(args)

    # Generate a dummy input tensor
    batch_size = 4
    channels = 3
    height = 256
    width = 256
    dummy_input = torch.randn(batch_size, channels, height, width)

    # Perform a forward pass on the dummy input
    decoded_images, codebook_indices, q_loss = vqgan(dummy_input)

    # Print information about the model's output
    print("Decoded Images Shape:", decoded_images.shape)
    print("Codebook Indices Shape:", codebook_indices.shape)
    print("Quantization Loss:", q_loss.item())


if __name__ == "__main__":
    test_vqgan()
