import torch
import config
from tqdm import tqdm
from torchinfo import summary
from torch import nn, optim
from model import VariationalAutoEncoder
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms


def train(model, train_loader, optimizer, loss_fn):
    """
    Train the VAE model.

    Args:
        model (nn.Module): VAE model.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer for model parameters.
        loss_fn (callable): Loss function.

    Returns:
        None
    """

    for epoch in range(config.NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader))

        for i, (x, _) in loop:
            x = x.to(config.DEVICE).view(x.shape[0], config.INPUT_DIM)
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

            save_image(
                x_reconstructed.view(x.size(0), 1, 28, 28),
                f"VAE/reconstructed_images_{epoch}.png",
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
    )

    # Initialize model, optimizer and loss function
    model = VariationalAutoEncoder(
        config.INPUT_DIM, config.H_DIM, config.Z_DIM
    ).to(config.DEVICE)
    summary(model)
    optimizer = optim.Adam(model.parameters(), lr=config.LR_RATE)
    loss_fn = nn.BCELoss(reduction="sum")

    # Train the model
    train(model, train_loader, optimizer, loss_fn)
