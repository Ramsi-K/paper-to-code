import torch


def accuracy(outputs, labels):
    """
    Calculate the accuracy of model predictions.

    Args:
        outputs (torch.Tensor): Predicted outputs from the model.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        float: Accuracy percentage.
    """
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total


def save_checkpoint(model, optimizer, epoch, filepath):
    """
    Save model checkpoint to file.

    Args:
        model (torch.nn.Module): Model to be saved.
        optimizer (torch.optim.Optimizer): Optimizer state.
        epoch (int): Current epoch.
        filepath (str): Path to save the checkpoint.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath):
    """
    Load model checkpoint from file.

    Args:
        model (torch.nn.Module): Model to load the checkpoint into.
        optimizer (torch.optim.Optimizer): Optimizer to load state into.
        filepath (str): Path to the checkpoint file.

    Returns:
        int: Epoch from the loaded checkpoint.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]
