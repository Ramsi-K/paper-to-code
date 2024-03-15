import torch
import config
from torch import nn, optim
from utils import (
    gradient_penalty,
    load_checkpoint,
    save_checkpoint,
    plot_examples,
)
from loss import VGGLoss
from torch.utils.data import DataLoader
from model import Generator, Discriminator, initialize_weights
from tqdm import tqdm
from dataset import MyImageFolder
from torch.utils.tensorboard import SummaryWriter

torch.backends.cudnn.benchmark = True


def train_fn(
    loader,
    disc,
    gen,
    opt_gen,
    opt_disc,
    l1_loss,
    vgg_loss,
    g_scaler,
    d_scaler,
    writer,
    tb_step,
):
    """
    Training function for ESRGAN.

    Args:
        loader (DataLoader): DataLoader for the training dataset.
        disc (Discriminator): Discriminator model.
        gen (Generator): Generator model.
        opt_gen (Optimizer): Optimizer for Generator parameters.
        opt_disc (Optimizer): Optimizer for Discriminator parameters.
        l1_loss (nn.L1Loss): L1 loss function.
        vgg_loss (VGGLoss): VGG loss function.
        g_scaler (GradScaler): Gradient scaler for Generator.
        d_scaler (GradScaler): Gradient scaler for Discriminator.
        writer (SummaryWriter): Tensorboard SummaryWriter.
        tb_step (int): Current tensorboard step.

    Returns:
        int: Updated tensorboard step.
    """
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(config.DEVICE)
        low_res = low_res.to(config.DEVICE)

        # Train Discriminator
        opt_disc.zero_grad()
        with torch.cuda.amp.autocast():
            fake = gen(low_res).detach()
            critic_real = disc(high_res)
            critic_fake = disc(fake)
            gp = gradient_penalty(disc, high_res, fake, device=config.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
            )
        d_scaler.scale(loss_critic).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        opt_gen.zero_grad()
        with torch.cuda.amp.autocast():
            fake = gen(low_res)
            critic_fake = disc(fake)
            adversarial_loss = -torch.mean(critic_fake)
            l1_loss_val = l1_loss(fake, high_res)
            vgg_loss_val = vgg_loss(fake, high_res)
            gen_loss = (
                config.LAMBDA_ADV * adversarial_loss
                + config.LAMBDA_L1 * l1_loss_val
                + config.LAMBDA_VGG * vgg_loss_val
            )
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Write losses to Tensorboard
        writer.add_scalar(
            "Critic loss", loss_critic.item(), global_step=tb_step
        )
        writer.add_scalar(
            "Generator loss", gen_loss.item(), global_step=tb_step
        )
        writer.add_scalar("L1 loss", l1_loss_val.item(), global_step=tb_step)
        writer.add_scalar("VGG loss", vgg_loss_val.item(), global_step=tb_step)
        writer.add_scalar(
            "Adversarial loss", adversarial_loss.item(), global_step=tb_step
        )
        tb_step += 1

        loop.set_postfix(
            gp=gp.item(),
            critic=loss_critic.item(),
            l1=l1_loss_val.item(),
            vgg=vgg_loss_val.item(),
            adversarial=adversarial_loss.item(),
        )

    return tb_step
