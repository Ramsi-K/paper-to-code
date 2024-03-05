# import torch
# import torch.nn as nn


# class VariationalAutoEncoder(nn.Module):
#     """
#     Variational Autoencoder (VAE) model.

#     Args:
#         input_dim (int): Dimensionality of input data.
#         h_dim (int): Dimensionality of hidden layers. Defaults to 200.
#         z_dim (int): Dimensionality of latent space. Defaults to 20.

#     Attributes:
#         img_2hid (nn.Conv2d): Convolutional layer for mapping input to hidden layer.
#         hid2mu (nn.Conv2d): Convolutional layer for mapping hidden layer to mean of latent space.
#         hid2sigma (nn.Conv2d): Convolutional layer for mapping hidden layer to standard deviation of latent space.
#         z_2hid (nn.ConvTranspose2d): Transposed convolutional layer for mapping latent space to hidden layer.
#         hid_2img (nn.ConvTranspose2d): Transposed convolutional layer for mapping hidden layer to output.
#         relu (nn.ReLU): ReLU activation function.
#     """

#     def __init__(self, input_dim, h_dim=200, z_dim=20):
#         super().__init__()
#         # Encoder
#         self.img_2hid = nn.Conv2d(input_dim, h_dim, 3, 2, 1)
#         self.hid2mu = nn.Conv2d(h_dim, z_dim, 3, 2, 1)
#         self.hid2sigma = nn.Conv2d(h_dim, z_dim, 3, 2, 1)  # standard gaussian

#         # Decoder
#         self.z_2hid = nn.ConvTranspose2d(
#             z_dim, h_dim, 3, 2, 1, output_padding=1
#         )
#         self.hid_2img = nn.ConvTranspose2d(
#             h_dim, input_dim, 3, 2, 1, output_padding=1
#         )

#         self.relu = nn.ReLU()

#     def encode(self, x):
#         """
#         Encodes input data to obtain mean and standard deviation of the latent space.

#         Args:
#             x (torch.Tensor): Input data.

#         Returns:
#             torch.Tensor: Mean of the latent space.
#             torch.Tensor: Standard deviation of the latent space.
#         """
#         # q_phi(z|x)
#         h = self.relu(self.img_2hid(x))
#         mu, sigma = self.hid2mu(h), self.hid2sigma(h)

#         return mu, sigma

#     def decode(self, z):
#         """
#         Decodes latent space to reconstruct input data.

#         Args:
#             z (torch.Tensor): Latent space representation.

#         Returns:
#             torch.Tensor: Reconstructed output.
#         """
#         # p_theta(x|z)
#         h = self.relu(self.z_2hid(z))
#         return torch.sigmoid(self.hid_2img(h))

#     def forward(self, x):
#         """
#         Forward pass of the VAE.

#         Args:
#             x (torch.Tensor): Input data.

#         Returns:
#             torch.Tensor: Reconstructed output.
#             torch.Tensor: Mean of the latent space.
#             torch.Tensor: Standard deviation of the latent space.
#         """
#         mu, sigma = self.encode(x)
#         epsilon = torch.randn_like(sigma)
#         z_reparam = mu + sigma * epsilon
#         x_recons = self.decode(z_reparam)
#         return x_recons, mu, sigma


# def test_vae():
#     # Define input dimensions
#     input_dim = 1
#     height = 28
#     width = 28
#     # Generate random input data
#     x = torch.randn(4, input_dim, height, width)  # Batch size of 4

#     # Create an instance of the VAE model
#     vae = VariationalAutoEncoder(input_dim)

#     # Perform a forward pass
#     x_reconstructed, mu, sigma = vae(x)

#     # Print information about the output
#     print("Reconstructed Images Shape:", x_reconstructed.shape)
#     print("Mean Shape:", mu.shape)
#     print("Standard Deviation Shape:", sigma.shape)


# # Test the VAE model
# if __name__ == "__main__":
#     test_vae()
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim=1, h_dim=200, z_dim=20):
        super().__init__()
        self.z_dim = z_dim
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(
                input_dim, h_dim, kernel_size=3, stride=2, padding=1
            ),  # -> N, h_dim, 14, 14
            nn.ReLU(),
            nn.Conv2d(
                h_dim, 2 * z_dim, kernel_size=3, stride=2, padding=1
            ),  # -> N, 2*z_dim, 7, 7
            nn.ReLU(),
            nn.Conv2d(
                2 * z_dim, 4 * z_dim, kernel_size=7
            ),  # -> N, 4*z_dim, 1, 1
        )

        # Fully connected layers for mean and standard deviation
        self.fc_mu = nn.Linear(4 * z_dim, z_dim)
        self.fc_sigma = nn.Linear(4 * z_dim, z_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                4 * z_dim, 2 * z_dim, kernel_size=7
            ),  # -> N, 2*z_dim, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(
                2 * z_dim,
                h_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # N, h_dim, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(
                h_dim,
                input_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # N, input_dim, 28, 28
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        flattened = encoded.view(encoded.size(0), -1)
        mu = self.fc_mu(flattened)
        log_var = self.fc_sigma(flattened)
        sigma = torch.exp(0.5 * log_var)
        # z = mu + sigma * torch.randn_like(sigma)
        decoded = self.decoder(
            encoded
        )  # Pass the encoded tensor to the decoder
        return decoded, mu, sigma


def test_autoencoder(autoencoder):
    """
    Test function for the autoencoder model.

    Args:
        autoencoder (Autoencoder): The autoencoder model to be tested.

    Returns:
        None
    """
    # Generate random input data
    input_data = torch.randn(10, 1, 28, 28)  # Batch size of 10

    # Pass the input data through the autoencoder
    with torch.no_grad():
        reconstructed, mu, sigma = autoencoder(input_data)

    # Assert shapes
    assert (
        input_data.shape == reconstructed.shape
    ), "Output shape doesn't match input shape"
    assert mu.shape == (10, autoencoder.z_dim), "Mu shape is incorrect"
    assert sigma.shape == (10, autoencoder.z_dim), "Sigma shape is incorrect"

    print("All assertions passed.")


# Example usage:
autoencoder = Autoencoder(input_dim=1, h_dim=200, z_dim=20)
test_autoencoder(autoencoder)
