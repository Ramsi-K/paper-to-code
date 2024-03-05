# import torch
# import config
# from tqdm import tqdm
# from torchinfo import summary
# from torch import nn, optim
# from conv_model import Autoencoder
# import torchvision.datasets as datasets
# from torch.utils.data import DataLoader
# from torchvision.utils import save_image
# from torchvision import transforms


# def train(model, train_loader, optimizer, loss_fn):
#     """
#     Train the Convolutional Autoencoder (CAE) model.

#     Args:
#         model (nn.Module): CAE model.
#         train_loader (DataLoader): DataLoader for training data.
#         optimizer (Optimizer): Optimizer for model parameters.
#         loss_fn (callable): Loss function.

#     Returns:
#         None
#     """

#     for epoch in range(config.NUM_EPOCHS):
#         loop = tqdm(enumerate(train_loader))

#         for i, (x, _) in loop:
#             x = x.to(config.DEVICE)
#             x_reconstructed = model(x)

#             # Compute loss
#             loss = loss_fn(x_reconstructed, x)

#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             loop.set_postfix(loss=loss.item())

#             # Save reconstructed images
#             if i % 100 == 0:
#                 save_image(
#                     x_reconstructed,
#                     f"VAE/conv_reconstructed_images_{epoch}_{i}.png",
#                     normalize=True,
#                 )


# if __name__ == "__main__":
#     # Load dataset
#     transform = transforms.Compose([transforms.ToTensor()])
#     train_dataset = datasets.MNIST(
#         root=config.root, train=True, transform=transform, download=True
#     )
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=True,
#         num_workers=config.NUM_WORKERS,
#     )

#     # Initialize model, optimizer and loss function
#     model = Autoencoder(
#         input_dim=config.INPUT_DIM,  # Provide input_dim parameter
#         h_dim=config.H_DIM,
#         z_dim=config.Z_DIM,
#     ).to(config.DEVICE)
#     summary(model)
#     optimizer = optim.Adam(model.parameters(), lr=config.LR_RATE)
#     loss_fn = nn.BCELoss(reduction="sum")

#     # Train the model
#     train(model, train_loader, optimizer, loss_fn)

import torch
import config
from tqdm import tqdm
from torchinfo import summary
from torch import nn, optim
from conv_model import Autoencoder
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms


def train(model, train_loader, optimizer, loss_fn):
    """
    Train the Convolutional Autoencoder (CAE) model.

    Args:
        model (nn.Module): CAE model.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer for model parameters.
        loss_fn (callable): Loss function.

    Returns:
        None
    """

    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(train_loader)

        for i, (x, _) in enumerate(loop):
            x = x.to(config.DEVICE)
            x_reconstructed, mu, sigma = model(x)

            # Compute loss
            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_div = torch.sum(
                1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)
            )
            loss = reconstruction_loss + kl_div

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

            # Save reconstructed images
            if i % 100 == 0:
                save_image(
                    x_reconstructed,
                    f"VAE/conv_reconstructed_images_{epoch}_{i}.png",
                    normalize=True,
                )


if __name__ == "__main__":
    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root=config.root, train=True, transform=transform, download=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        drop_last=True,  # Add this line to drop last incomplete batch
    )
    # Initialize model, optimizer and loss function
    model = Autoencoder(
        input_dim=1,  # Adjust input_dim to match MNIST images (single channel)
        h_dim=config.H_DIM,
        z_dim=config.Z_DIM,
    ).to(config.DEVICE)
    summary(model)
    optimizer = optim.Adam(model.parameters(), lr=config.LR_RATE)
    loss_fn = nn.MSELoss(reduction="sum")

    # Train the model
    train(model, train_loader, optimizer, loss_fn)
