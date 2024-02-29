import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from transformer import VQGANTransformer
from utils import load_data, plot_images
import config


class TrainTransformer:
    """
    Trainer class for the VQGANTransformer model.
    """

    def __init__(self, config):
        """
        Initialize the Trainer.

        Args:
            config (dict): Configuration parameters.
        """
        self.model = VQGANTransformer(config).to(device=config.device)
        self.optim = self.configure_optimizers()

        self.train(config)

    def configure_optimizers(self):
        """
        Configure optimizer with different weight decays for different parameter groups.

        Returns:
            torch.optim.AdamW: AdamW optimizer.
        """
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(
                    m, whitelist_weight_modules
                ):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(
                    m, blacklist_weight_modules
                ):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {
            pn: p for pn, p in self.model.transformer.named_parameters()
        }

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": 0.01,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=4.5e-06, betas=(0.9, 0.95)
        )
        return optimizer

    def train(self, args):
        """
        Train the model.

        Args:
            args (dict): Configuration parameters.
        """
        train_dataset = load_data(args)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for i, imgs in zip(pbar, train_dataset):
                    self.optim.zero_grad()
                    imgs = imgs.to(device=args.device)
                    logits, targets = self.model(imgs)
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        targets.reshape(-1),
                    )
                    loss.backward()
                    self.optim.step()
                    pbar.set_postfix(
                        Transformer_Loss=loss.cpu().detach().numpy().item()
                    )
                    pbar.update(0)
            log, sampled_imgs = self.model.log_images(imgs[0][None])
            vutils.save_image(
                sampled_imgs,
                os.path.join("VQGAN\\results", f"transformer_{epoch}.jpg"),
                nrow=4,
            )
            plot_images(log)
            torch.save(
                self.model.state_dict(),
                os.path.join("VQGAN\\checkpoints", f"transformer_{epoch}.pt"),
            )


if __name__ == "__main__":
    train_transformer = TrainTransformer(config)
