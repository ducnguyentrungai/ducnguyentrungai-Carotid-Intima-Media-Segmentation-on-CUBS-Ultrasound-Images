from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


image_path = Path(r"C:\Users\Admin\Downloads\CUBS_DATSET\data\images\clin_0001_L.tiff")
mask_path = Path(r"C:\Users\Admin\Downloads\CUBS_DATSET\data\masks\clin_0001_L_mask.png")


def mask_to_li_ma(mask):
    mask = np.asarray(mask) > 0
    h, w = mask.shape

    li = np.full(w, np.nan, dtype=np.float32)
    ma = np.full(w, np.nan, dtype=np.float32)

    for x in range(w):
        ys = np.flatnonzero(mask[:, x])
        if ys.size > 0:
            li[x] = ys.min()
            ma[x] = ys.max()

    return li, ma


image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(image_path)
if mask is None:
    raise FileNotFoundError(mask_path)

mask_bin = mask > 127
li, ma = mask_to_li_ma(mask_bin)

x = np.arange(mask_bin.shape[1])
valid_li = ~np.isnan(li)
valid_ma = ~np.isnan(ma)

overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
overlay[mask_bin] = (
    0.65 * overlay[mask_bin] + 0.35 * np.array([255, 0, 0])
).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(image, cmap="gray")
axes[0].plot(x[valid_li], li[valid_li], color="cyan", linewidth=1.2, label="LI")
axes[0].plot(x[valid_ma], ma[valid_ma], color="yellow", linewidth=1.2, label="MA")
axes[0].set_title("Ultrasound + LI/MA boundaries", fontsize=25)
axes[0].legend(loc="lower right")
axes[0].axis("off")

axes[1].imshow(mask_bin, cmap="gray")
axes[1].set_title("Binary CIMT mask",  fontsize=25)
axes[1].axis("off")

axes[2].imshow(overlay)
axes[2].plot(x[valid_li], li[valid_li], color="cyan", linewidth=1.2, label="LI")
axes[2].plot(x[valid_ma], ma[valid_ma], color="yellow", linewidth=1.2, label="MA")
axes[2].set_title("Overlay mask + LI/MA",  fontsize=25)
axes[2].legend(loc="lower right")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(r"C:\Users\Admin\Downloads\CUBS_DATSET\im_mk_li_ma.png", bbox_inches="tight")
plt.show()
