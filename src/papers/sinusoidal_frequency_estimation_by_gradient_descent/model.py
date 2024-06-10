"""
https://arxiv.org/abs/2210.14476
https://qiita.com/xiao_ming/items/37303e1a5a8c26ff5088
https://tam5917.hatenablog.com/entry/2022/11/19/171228
https://gist.github.com/tam17aki/b60313df3f438bad80e77cc1055bd311
"""

# %%
from multiprocessing import Pool
from pathlib import Path

import cv2
import librosa
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from IPython.display import Audio, display
from tqdm.auto import tqdm

assets_dir = Path("assets")
fig_dir = Path("figs")


class SinFreqEstimator(nn.Module):
    def __init__(
        self,
        n: int = 4,
        sampling_rate: int = 16000,
        frame_length: int = 1024,
        hop_length: int = 80,
    ):
        super().__init__()
        self.n = n
        self.freq = torch.rand(n, 1, requires_grad=False) * torch.pi
        self.phase = torch.rand(n, 1, requires_grad=False) * torch.pi

        self._omega = nn.Parameter(torch.exp(self.freq * 1.0j), requires_grad=True)
        self._phi = nn.Parameter(torch.exp(self.phase * 1.0j), requires_grad=True)

        self.array_n = torch.arange(frame_length)

    @property
    def omega(self):
        return self._omega

    @property
    def phi(self):
        return self._phi

    @property
    def magnitude(self):
        return self._omega.abs().pow(self.array_n) * self._phi.abs()

    @property
    def sinusoid(self):
        return torch.cos(self._omega.angle() * self.array_n + self._phi.angle())

    def forward(self):
        return (self.magnitude * self.sinusoid).sum(dim=0)


def get_power_max_frame(audio, frame_length, hop_length):
    """Extract the frame with maximum power."""
    frames = librosa.util.frame(
        audio.numpy(), frame_length=frame_length, hop_length=hop_length
    ).T
    frames = torch.from_numpy(frames)
    max_ind = torch.argmax(torch.sum(frames * frames, axis=1))
    max_frame = frames[max_ind, :]
    max_frame = max_frame / torch.abs(torch.max(max_frame))
    return max_frame


name = "a_1"
resample = torchaudio.transforms.Resample(44100, 16000)

for name in ["ka_1", "ki_1", "ku_1", "ke_1", "ko_1"]:
    fig_dir = Path("figs")
    target_wave, sampling_rate = torchaudio.load(assets_dir / f"{name}.wav")
    fig_dir = fig_dir / name
    fig_dir.mkdir(exist_ok=True, parents=True)
    length = 2048
    target_wave = resample(target_wave)
    sampling_rate = 16000
    target_wave = get_power_max_frame(target_wave[0, :], length, 80)
    plt.plot(target_wave)
    plt.title("Target Waveform")
    plt.show()
    display(Audio(target_wave, rate=sampling_rate))

    k = 16

    def plot(args):
        sinusoid, freq, phase, mag, n, i, est_wave, target_wave, loss = args
        fig = plt.figure(figsize=(k * 6, k * 3))
        gs = gridspec.GridSpec(k, n // k + k)
        ax = fig.add_subplot(gs[:, :k])
        ax.set_xlim(-1, length + 1)
        ax.set_ylim(-1, 1)
        ax.plot(target_wave, label="Target", lw=5)
        ax.plot(est_wave, label="Estimation", lw=5)
        ax.set_title("Waveform", fontsize=k * 2)
        ax.legend(loc="upper right", fontsize=k * 3)

        for j in range(n):
            ax = fig.add_subplot(gs[j // k, j % k + k])
            ax.set_xlim(-1, length + 1)
            ax.set_ylim(-1, 1)
            ax.plot(sinusoid[j, :][: length + 1])
            ax.set_title(f"Wave_{j} f={freq[j].item() * 1000:.4f}")
        fig.suptitle(f"Epoch: {i:08}, Loss: {loss:.8f}", fontsize=k * 4)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(fig_dir / f"epoch_{i:08}.png")

    sin = SinFreqEstimator(
        n=k**2, sampling_rate=sampling_rate, frame_length=target_wave.shape[-1]
    )
    sin.load_state_dict(torch.load(fig_dir.parent / name[1:] / "sin_freq_est.pth"))
    optimizer = optim.Adam(sin.parameters(), lr=0.0001)

    size = 1000
    h, w = (2560, 5120)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        str(fig_dir.parent / f"waves_{name}.mp4"), fourcc, 15, (w, h)
    )
    sin.train()

    try:
        for i in tqdm(range(100 * size)):
            optimizer.zero_grad()
            est_wave = sin()
            loss = F.mse_loss(est_wave, target_wave) / sin.n
            loss.backward()
            optimizer.step()

            if (
                (loss > 0.01 and i % (size // 10) == 0)
                or (loss > 0.001 and i % (size // 2) == 0)
                or (loss > 0.0001 and i % size == 0)
                or (i % (size * 5) == 0)
            ):
                tqdm.write(f"Epoch: {i}, Loss: {loss.item()}")

                with Pool(1) as p:
                    p.map(
                        plot,
                        [
                            (
                                sin.sinusoid.detach().numpy(),
                                sin.omega.angle().detach().numpy(),
                                sin.phase,
                                sin.phi.abs().detach().numpy(),
                                sin.n,
                                i,
                                est_wave.detach().numpy(),
                                target_wave.detach().numpy(),
                                loss.item(),
                            )
                        ],
                    )

                im = cv2.imread(str(fig_dir / f"epoch_{i:08}.png"))
                im = cv2.resize(im, (w, h))
                video.write(im)

            if loss < 1e-5:
                break

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        video.release()
        with open(fig_dir / "sin_freq_est.pth", "wb") as f:
            torch.save(sin.state_dict(), f)

        torchaudio.save(
            uri=fig_dir / f"target_{name}.wav",
            src=target_wave.detach().unsqueeze(0),
            sample_rate=sampling_rate,
        )

        torchaudio.save(
            uri=fig_dir / f"est_{name}.wav",
            src=est_wave.detach().unsqueeze(0),
            sample_rate=sampling_rate,
        )

# %%
