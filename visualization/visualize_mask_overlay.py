from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


IMAGE_PATH = Path(r"C:\Users\Admin\Downloads\CUBS_DATSET\data\images\clin_0001_L.tiff")
MASK_PATH = Path(r"C:\Users\Admin\Downloads\CUBS_DATSET\data\masks\clin_0001_L_mask.png")
OUTPUT_PATH = Path(r"C:\Users\Admin\Downloads\CUBS_DATSET\overlay.png")


def main():
    image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {MASK_PATH}")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"Image shape {image.shape[:2]} != mask shape {mask.shape[:2]}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    mask_bool = mask > 127

    overlay = image_rgb.copy()
    red = np.array([255, 0, 0], dtype=np.uint8)
    alpha = 0.35
    overlay[mask_bool] = ((1 - alpha) * overlay[mask_bool] + alpha * red).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Image", fontsize=23)
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask", fontsize=23)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=23)
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved overlay to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
