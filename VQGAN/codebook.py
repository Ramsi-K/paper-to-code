import torch
import torch.nn as nn
import config


class CodeBook(nn.Module):
    """
    CodeBook module for vector quantization.

    Args:
        config (dict): Dictionary containing configuration parameters.

    Attributes:
        Embedding (nn.Embedding): Embedding layer representing the codebook vectors.

    """

    def __init__(self, config):
        super(CodeBook, self).__init__()
        self.num_codebook_vectors = config.num_codebook_vectors
        self.latent_dim = config.latent_dim
        self.beta = config.beta

        # Initialize the codebook vectors with uniform distribution
        self.Embedding = nn.Embedding(
            self.num_codebook_vectors, self.latent_dim
        )
        self.Embedding.weight.data.uniform_(
            -1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors
        )

    def forward(self, x):
        """
        Forward pass of the CodeBook module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Quantized tensor after encoding.
            torch.Tensor: Indices of the closest codebook vectors.
            torch.Tensor: Total loss computed for the quantization.

        """
        # Reshape input tensor for computation
        x_flattened = (
            x.permute(0, 2, 3, 1).contiguous().view(-1, self.latent_dim)
        )

        # Compute distances between input vectors and codebook vectors
        d = (
            torch.sum(x_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.Embedding.weight**2, dim=1)
            - 2 * (torch.matmul(x_flattened, self.Embedding.weight.t()))
        )

        # Find indices of the closest codebook vectors
        min_encoding_indices = torch.argmin(d, dim=1)

        # Quantize the input vectors using the codebook vectors
        x_q = self.Embedding(min_encoding_indices).view(x.shape)

        # Compute commitment loss
        loss = torch.mean((x_q.detach() - x) ** 2) + self.beta * torch.mean(
            (x_q - x.detach()) ** 2
        )

        # Update the quantized vectors with the commitment loss
        x_q = x + (x_q - x).detach()

        return x_q, min_encoding_indices, loss


def test_codebook():
    # Define configuration parameters
    # Create an instance of the CodeBook module
    codebook = CodeBook(config)

    # Generate some dummy input data
    batch_size = 4
    channels = 3
    height = 32
    width = 32
    dummy_input = torch.randn(batch_size, channels, height, width)

    # Forward pass through the CodeBook module
    quantized_output, indices, loss = codebook(dummy_input)

    # Print information about the output
    print("Quantized Output Shape:", quantized_output.shape)
    print("Indices Shape:", indices.shape)
    print("Loss Value:", loss.item())


if __name__ == "__main__":
    test_codebook()
