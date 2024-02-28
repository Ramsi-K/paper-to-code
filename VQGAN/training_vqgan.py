import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import utils as vutils
from discriminator import Discriminator
from lpips import LPIPS
from vqgan import VQGAN
from utils import load_data, weights_init
import config


class TrainVQGAN:
    """
    Class to train the VQGAN model.

    Args:
        args (dict): Configuration parameters for training.

    Attributes:
        vqgan (VQGAN): Instance of the VQGAN model.
        discriminator (Discriminator): Instance of the discriminator model.
        perceptual_loss (LPIPS): Instance of the perceptual loss model.
        opt_vq (torch.optim.Adam): Optimizer for VQGAN.
        opt_disc (torch.optim.Adam): Optimizer for the discriminator.
    """

    def __init__(self, args):
        # Initialize VQGAN, discriminator, perceptual loss, and optimizers
        self.vqgan = VQGAN(args).to(device=args.device)
        self.discriminator = Discriminator(args).to(device=args.device)
        self.discriminator.apply(weights_init)
        self.perceptual_loss = LPIPS().eval().to(device=args.device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(args)

        self.prepare_training()  # Prepare directories for saving results and checkpoints

        self.train(args)  # Start training

    def configure_optimizers(self, args):
        """
        Configure optimizers for VQGAN and discriminator.

        Args:
            args (dict): Configuration parameters for training.

        Returns:
            tuple: Optimizers for VQGAN and discriminator.
        """
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters())
            + list(self.vqgan.decoder.parameters())
            + list(self.vqgan.codebook.parameters())
            + list(self.vqgan.quant_conv.parameters())
            + list(self.vqgan.post_quant_conv.parameters()),
            lr=lr,
            eps=1e-08,
            betas=(args.beta1, args.beta2),
        )
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            eps=1e-08,
            betas=(args.beta1, args.beta2),
        )

        return opt_vq, opt_disc

    @staticmethod
    def prepare_training():
        """Create directories for saving results and checkpoints."""
        os.makedirs("results", exist_ok=True)
        os.makedirs("checkpoints", exist_ok=True)

    def train(self, args):
        """Train the VQGAN model."""
        train_dataset = load_data(args)
        steps_per_epoch = len(train_dataset)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    imgs = imgs.to(device=args.device)
                    decoded_images, _, q_loss = self.vqgan(imgs)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.vqgan.adopt_weight(
                        args.disc_factor,
                        epoch * steps_per_epoch + i,
                        threshold=args.disc_start,
                    )

                    perceptual_loss = self.perceptual_loss(
                        imgs, decoded_images
                    )
                    rec_loss = torch.abs(imgs - decoded_images)
                    perceptual_rec_loss = (
                        args.perceptual_loss_factor * perceptual_loss
                        + args.rec_loss_factor * rec_loss
                    )
                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.calculate_lambda(
                        perceptual_rec_loss, g_loss
                    )
                    vq_loss = (
                        perceptual_rec_loss + q_loss + disc_factor * λ * g_loss
                    )

                    d_loss_real = torch.mean(F.relu(1.0 - disc_real))
                    d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
                    gan_loss = disc_factor * 0.5 * (d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    if i % 10 == 0:
                        with torch.no_grad():
                            real_fake_images = torch.cat(
                                (
                                    imgs.add(1).mul(0.5)[:4],
                                    decoded_images.add(1).mul(0.5)[:4],
                                )
                            )
                            vutils.save_image(
                                real_fake_images,
                                os.path.join("results", f"{epoch}_{i}.jpg"),
                                nrow=4,
                            )

                    pbar.set_postfix(
                        VQ_Loss=np.round(
                            vq_loss.cpu().detach().numpy().item(), 5
                        ),
                        GAN_Loss=np.round(
                            gan_loss.cpu().detach().numpy().item(), 3
                        ),
                    )
                    pbar.update(0)
                torch.save(
                    self.vqgan.state_dict(),
                    os.path.join("checkpoints", f"vqgan_epoch_{epoch}.pt"),
                )


if __name__ == "__main__":
    # Start training with configuration from config.py
    train_vqgan = TrainVQGAN(config.CONFIG["training_params"])
