import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import config
from utils import save_checkpoint, load_checkpoint
from generator import Generator
from discriminator import Discriminator
from A2B_dataset import A2BDataset


def train_fn(
    disc_H,
    disc_Z,
    gen_Z,
    gen_H,
    loader,
    opt_disc,
    opt_gen,
    l1,
    mse,
    d_scaler,
    g_scaler,
):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (A, B) in enumerate(loop):
        A = A.to(config.DEVICE)
        B = B.to(config.DEVICE)

        # Train Discriminators A and B
        with torch.cuda.amp.autocast():
            fake_B = gen_H(A)
            D_H_real = disc_H(B)
            D_H_fake = disc_H(fake_B.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_A = gen_Z(B)
            D_Z_real = disc_Z(A)
            D_Z_fake = disc_Z(fake_A.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            # put it together
            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators A and B
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_B)
            D_Z_fake = disc_Z(fake_A)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_A = gen_Z(fake_B)
            cycle_B = gen_H(fake_A)
            cycle_A_loss = l1(A, cycle_A)
            cycle_B_loss = l1(B, cycle_B)

            ## identity loss set at 0
            # identity_A = gen_Z(A)
            # identity_B = gen_H(B)
            # identity_A_loss = l1(A, identity_A)
            # identity_B_loss = l1(B, identity_B)

            # add all together
            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_A_loss * config.LAMBDA_CYCLE
                + cycle_B_loss * config.LAMBDA_CYCLE
                # + identity_A_loss * config.LAMBDA_IDENTITY
                # + identity_B_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(
                fake_A * 0.5 + 0.5, f"A2B/cardamage/results_a2b/A_{idx}.png"
            )
            save_image(
                fake_B * 0.5 + 0.5, f"A2B/cardamage/results_a2b/B_{idx}.png"
            )

        loop.set_postfix(
            H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1)
        )


def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H,
            gen_H,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z,
            gen_Z,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_H,
            disc_H,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_Z,
            disc_Z,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = A2BDataset(
        root_B=config.ROOT_DIR + "trainA",
        root_A=config.ROOT_DIR + "trainB",
        transform=config.transforms,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_H,
            disc_Z,
            gen_Z,
            gen_H,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(
                disc_H, opt_disc, filename=config.CHECKPOINT_DISC_H
            )
            save_checkpoint(
                disc_Z, opt_disc, filename=config.CHECKPOINT_DISC_Z
            )


if __name__ == "__main__":
    main()
