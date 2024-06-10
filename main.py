# %%
from pathlib import Path

import cv2
from IPython.display import display
from tqdm.auto import tqdm

fig_dir = (
    Path("src")
    / "papers"
    / "sinusoidal_frequency_estimation_by_gradient_descent"
    / "figs"
)

k = 8
h, w = (480 * k, 960 * k)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(str(fig_dir.parent / "waves2.mp4"), fourcc, 30, (w, h))


for i, p in enumerate(tqdm(sorted(fig_dir.glob("*.png")))):
    im = cv2.imread(str(p))
    im = cv2.resize(im, (w, h))
    video.write(im)

video.release()

# ims[0].save(
#     fig_dir / "waves.gif",
#     save_all=True,
#     append_images=ims[1:],
#     duration=50,
# )
# %%
