import torch
import torch.nn as nn
import torch.nn.functional as F
from mingpt import GPT
from vqgan import VQGAN


class VQGANTransformer(nn.Module):
    """
    VQGAN Transformer model combining VQGAN and GPT.
    """

    def __init__(self, config):
        """
        Initialize the VQGANTransformer model.

        Args:
            config (dict): Configuration parameters.
        """
        super(VQGANTransformer, self).__init__()

        self.sos_token = config.sos_token

        self.vqgan = self.load_vqgan(config)

        transformer_config = {
            "vocab_size": config.num_codebook_vectors,
            "block_size": 512,
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 1024,
        }
        self.transformer = GPT(**transformer_config)

        self.pkeep = config.pkeep

    @staticmethod
    def load_vqgan(config):
        """
        Load the pre-trained VQGAN model.

        Args:
            config (dict): Configuration parameters.

        Returns:
            VQGAN: Loaded VQGAN model.
        """
        model = VQGAN(config)
        model.load_checkpoint(config.checkpoint_path)
        model = model.eval()
        return model

    @torch.inference_mode()
    def encode_to_z(self, x):
        """
        Encode input images to latent space vectors.

        Args:
            x (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Quantized latent space vectors.
            torch.Tensor: Quantized indices.
        """
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.inference_mode()
    def z_to_image(self, indices, p1=16, p2=16):
        """
        Decode latent space indices to images.

        Args:
            indices (torch.Tensor): Quantized indices.
            p1 (int): First dimension size for reshaping.
            p2 (int): Second dimension size for reshaping.

        Returns:
            torch.Tensor: Decoded images.
        """
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(
            indices.shape[0], p1, p2, 256
        )
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors)
        return image

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Output logits.
            torch.Tensor: Target indices.
        """
        _, indices = self.encode_to_z(x)

        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        mask = torch.bernoulli(
            self.pkeep * torch.ones(indices.shape, device=indices.device)
        )
        mask = mask.round().to(dtype=torch.int64)
        random_indices = torch.randint_like(
            indices, self.transformer.config.vocab_size
        )
        new_indices = mask * indices + (1 - mask) * random_indices

        new_indices = torch.cat((sos_tokens, new_indices), dim=1)

        target = indices

        logits, _ = self.transformer(new_indices[:, :-1])

        return logits, target

    def top_k_logits(self, logits, k):
        """
        Compute top-k logits.

        Args:
            logits (torch.Tensor): Input logits.
            k (int): Number of top logits to keep.

        Returns:
            torch.Tensor: Top-k logits.
        """
        v, _ = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float("inf")
        return out

    @torch.inference_mode()
    def sample(self, x, c, steps, temperature=1.0, top_k=100):
        """
        Sample sequences from the model.

        Args:
            x (torch.Tensor): Input sequence.
            c (torch.Tensor): Context sequence.
            steps (int): Number of steps for sampling.
            temperature (float): Sampling temperature.
            top_k (int): Number of top-k logits to consider.

        Returns:
            torch.Tensor: Sampled sequence.
        """
        self.transformer.eval()
        x = torch.cat((c, x), dim=1)
        for k in range(steps):
            logits, _ = self.transformer(x)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            probs = F.softmax(logits, dim=-1)

            ix = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, ix), dim=1)

        x = x[:, c.shape[1] :]
        self.transformer.train()
        return x

    @torch.inference_mode()
    def log_images(self, x):
        """
        Log images for visualization.

        Args:
            x (torch.Tensor): Input images.

        Returns:
            dict: Dictionary containing input, reconstructed, and sampled images.
            torch.Tensor: Concatenated images.
        """
        log = dict()

        _, indices = self.encode_to_z(x)
        sos_tokens = torch.ones(x.shape[0], 1) * self.sos_token
        sos_tokens = sos_tokens.long().to("cuda")

        start_indices = indices[:, : indices.shape[1] // 2]
        sample_indices = self.sample(
            start_indices,
            sos_tokens,
            steps=indices.shape[1] - start_indices.shape[1],
        )
        half_sample = self.z_to_image(sample_indices)

        start_indices = indices[:, :0]
        sample_indices = self.sample(
            start_indices, sos_tokens, steps=indices.shape[1]
        )
        full_sample = self.z_to_image(sample_indices)

        x_rec = self.z_to_image(indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = half_sample
        log["full_sample"] = full_sample

        return log, torch.cat((x, x_rec, half_sample, full_sample))
