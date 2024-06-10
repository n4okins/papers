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


if __name__ == "__main__":
    from pathlib import Path

    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision.transforms as TF
    from PIL import Image
    from tqdm.auto import tqdm

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = MultiresolutionalHashEncoding(
                n_min=4, n_max=512, mod_t=2**14
            )
            self.decoder = nn.Sequential(
                nn.Conv2d(48, 48, 3, 1, 1),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.Conv2d(48, 48, 1),
                nn.ReLU(),
                nn.Conv2d(48, 3, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    im_dir = Path("/media/n4okins/WDC8TBNTFS/ImageNet/ILSVRC/Data/CLS-LOC/train")

    for im_name in [
        "n01796340/n01796340_218.JPEG",
        "n01871265/n01871265_116.JPEG",
    ]:
        im_path = im_dir / im_name

        fig_dir = Path(__file__).parent / "figs" / im_path.stem
        fig_dir.mkdir(exist_ok=True, parents=True)
        model = Model()
        transform = TF.Compose(
            [
                TF.Resize((512, 512)),
                TF.ToTensor(),
            ]
        )
        image = Image.open(im_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        origin = Image.fromarray(
            (image[0].permute(1, 2, 0).numpy() * 255).astype("uint8")
        )
        origin.save(fig_dir / "original.png")

        model = model.to("cuda")
        model.train()

        image = image.to("cuda")

        optimizer = torch.optim.Adam(
            [
                {"params": model.encoder.parameters()},
                {"params": model.decoder.parameters(), "weight_decay": 1e-6},
            ],
            lr=0.01,
            betas=(0.9, 0.99),
            eps=1e-15,
        )

        for e in tqdm(range(100)):
            optimizer.zero_grad()
            out = model(image)
            loss = F.mse_loss(out, image)
            loss.backward()
            optimizer.step()
            TF.ToPILImage()(out[0].cpu()).save(fig_dir / f"{e:04}.png")

    for im_name in [
        "n01796340/n01796340_218.JPEG",
        "n01871265/n01871265_116.JPEG",
    ]:
        im_path = im_dir / im_name

        fig_dir = Path(__file__).parent / "figs" / f"{im_path.stem}_noise"
        fig_dir.mkdir(exist_ok=True, parents=True)
        model = Model()
        transform = TF.Compose(
            [
                TF.Resize((512, 512)),
                TF.ToTensor(),
            ]
        )
        image = Image.open(im_path).convert("RGB")
        image = transform(image).unsqueeze(0)
        mask = torch.rand((512, 512)) < 0.1
        origin = Image.fromarray(
            ((image * mask)[0].permute(1, 2, 0).numpy() * 255).astype("uint8")
        )
        origin.save(fig_dir / "original.png")

        model = model.to("cuda")
        mask = mask.to("cuda")
        image = image.to("cuda")

        model.train()

        optimizer = torch.optim.Adam(
            [
                {"params": model.encoder.parameters()},
                {"params": model.decoder.parameters(), "weight_decay": 1e-6},
            ],
            lr=0.01,
            betas=(0.9, 0.99),
            eps=1e-15,
        )

        for e in tqdm(range(100)):
            optimizer.zero_grad()
            out = model(image)
            loss = F.mse_loss(out * mask, image * mask)
            loss.backward()
            optimizer.step()
            TF.ToPILImage()(out[0].cpu()).save(fig_dir / f"{e:04}.png")
