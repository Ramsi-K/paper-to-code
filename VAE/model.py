import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoEncoder(nn.Module):
    """
    Variational Autoencoder (VAE) model.

    Args:
        input_dim (int): Dimensionality of input data.
        h_dim (int): Dimensionality of hidden layers. Defaults to 200.
        z_dim (int): Dimensionality of latent space. Defaults to 20.

    Attributes:
        img_2hid (nn.Linear): Linear layer for mapping input to hidden layer.
        hid2mu (nn.Linear): Linear layer for mapping hidden layer to mean of latent space.
        hid2sigma (nn.Linear): Linear layer for mapping hidden layer to standard deviation of latent space.
        z_2hid (nn.Linear): Linear layer for mapping latent space to hidden layer.
        hid_2img (nn.Linear): Linear layer for mapping hidden layer to output.
        relu (nn.ReLU): ReLU activation function.
    """

    def __init__(self, input_dim, h_dim=200, z_dim=20):
        super().__init__()
        # Encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        self.hid2mu = nn.Linear(h_dim, z_dim)
        self.hid2sigma = nn.Linear(h_dim, z_dim)  # standard gaussian

        # Decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        """
        Encodes input data to obtain mean and standard deviation of the latent space.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Mean of the latent space.
            torch.Tensor: Standard deviation of the latent space.
        """
        # q_phi(z|x)
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid2mu(h), self.hid2sigma(h)

        return mu, sigma

    def decode(self, z):
        """
        Decodes latent space to reconstruct input data.

        Args:
            z (torch.Tensor): Latent space representation.

        Returns:
            torch.Tensor: Reconstructed output.
        """
        # p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        """
        Forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Reconstructed output.
            torch.Tensor: Mean of the latent space.
            torch.Tensor: Standard deviation of the latent space.
        """
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparam = mu + sigma * epsilon
        x_recons = self.decode(z_reparam)
        return x_recons, mu, sigma


def test_vae():
    """
    Function to test the Variational Autoencoder (VAE) model.
    """
    x = torch.randn(4, 28 * 28)  # Create random input data
    vae = VariationalAutoEncoder(input_dim=784)  # Instantiate the VAE model
    x_reconstructed, mu, sigma = vae(x)  # Perform forward pass

    # Print the shape of the reconstructed output
    print("Reconstructed Output Shape:", x_reconstructed.shape)
    print("Mu Output Shape:", mu.shape)
    print("Sigma Output Shape:", sigma.shape)

    # Print mean and standard deviation of the latent space
    print("Mean of Latent Space:")
    print(mu)

    print("Standard Deviation of Latent Space:")
    print(sigma)


# Test the VAE model
if __name__ == "__main__":
    test_vae()
