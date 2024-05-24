from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MultiresolutionalHashEncoding"]

"""
https://arxiv.org/abs/2201.05989
https://github.com/NVlabs/instant-ngp
"""

class MultiresolutionalHashEncoding(nn.Module):
    def __init__(
        self,
        level=16,
        mod_t=2**16,
        feature_dim=3,
        n_min: int = 4,
        n_max: int = 256,
        interpolation="bilinear",
    ):
        super().__init__()
        self.level = level
        self.mod_t = mod_t
        self.feature_dim = feature_dim

        self.n_max = torch.tensor(n_max)
        self.n_min = torch.tensor(n_min)

        self.b = torch.exp(
            (torch.log(self.n_max) - torch.log(self.n_min)) / (self.level - 1)
        )
        self.grid_sizes = [int(self.n_min * (self.b**i)) for i in range(self.level)]

        self.interpolation = interpolation

        self.register_buffer(
            "primes",
            torch.tensor(
                [
                    1,
                    2654435761,
                    805459861,
                    3674653429,
                    2097192037,
                    1434869437,
                    2165219737,
                ]
            ),
        )

        self.primes = cast(torch.Tensor, self.primes)[: self.feature_dim].view(
            [self.feature_dim, 1, 1]
        )
        self.hash_table = nn.Parameter(
            torch.rand([self.level, self.mod_t, self.feature_dim]) * 2e-4 - 1e-4,
            requires_grad=True,
        )  # [-1e-04 ~ 1e-04]

    @property
    def out_channel_size(self):
        return self.level * self.feature_dim

    def forward(self, images: torch.Tensor):
        """
        images: [B, C, H, W]
        return: [B, level * feature_dim, H, W]
        """
        b, c, h, w = images.shape
        fs = torch.empty((b, self.out_channel_size, h, w), device=images.device)
        for level, grid in enumerate(self.grid_sizes):
            grid = (
                F.max_pool2d(
                    images * grid,
                    kernel_size=(images.size(2) // grid, images.size(3) // grid),
                ).to(torch.long)
                * self.primes
            )
            g = grid[:, 0]
            for i in range(1, self.feature_dim):
                g = g ^ grid[:, i]
            g = g % self.mod_t
            f = self.hash_table[level, g].permute(0, 3, 1, 2)
            f = F.interpolate(f, (h, w), mode=self.interpolation)
            fs[:, level * self.feature_dim : (level + 1) * self.feature_dim] = f
        return fs