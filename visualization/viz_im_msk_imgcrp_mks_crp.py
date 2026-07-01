from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


# =========================
# Paths
# =========================
ORIGINAL_IMAGE_PATH = Path(
    r"C:\Users\Admin\Downloads\CUBS_DATSET\data\images\clin_0001_L.tiff"
)
ORIGINAL_MASK_PATH = Path(
    r"C:\Users\Admin\Downloads\CUBS_DATSET\data\masks\clin_0001_L_mask.png"
)

CROPPED_IMAGE_PATH = Path(
    r"C:\Users\Admin\Downloads\CUBS_DATSET\data\data_cropped\images\clin_0001_L.tiff"
)
CROPPED_MASK_PATH = Path(
    r"C:\Users\Admin\Downloads\CUBS_DATSET\data\data_cropped\masks\clin_0001_L_mask.png"
)

OUTPUT_PATH = Path(
    r"C:\Users\Admin\Downloads\CUBS_DATSET\src\cubs_data_visualization.png"
)

RESIZE_SIZE = (256, 256)  # width, height


# =========================
# Helpers
# =========================
def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Cannot read file: {path}")

    return image


def binarize_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    return (mask > threshold).astype(np.uint8)


def resize_pair(image: np.ndarray, mask: np.ndarray, size=(256, 256)):
    image_resized = cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_LINEAR,
    )

    mask_01 = (mask > 0).astype(np.uint8)

    mask_resized = cv2.resize(
        mask_01,
        size,
        interpolation=cv2.INTER_NEAREST,
    )

    mask_resized = (mask_resized > 0).astype(np.uint8)

    return image_resized, mask_resized

def make_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color=(255, 0, 0),
    alpha: float = 0.45,
    dilate_for_display: bool = True,
) -> np.ndarray:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    mask_show = (mask > 0).astype(np.uint8)

    if dilate_for_display:
        kernel = np.ones((2, 2), np.uint8)
        mask_show = cv2.dilate(mask_show, kernel, iterations=1)

    mask_bool = mask_show > 0

    overlay = image_rgb.copy()
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool]
        + alpha * np.array(color, dtype=np.uint8)
    ).astype(np.uint8)

    return overlay

def show_triplet(axes, image, mask, overlay, row_title):
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"{row_title} - Ultrasound", fontsize=23)
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"{row_title} - Mask", fontsize=23)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"{row_title} - Overlay", fontsize=23)
    axes[2].axis("off")


# =========================
# Main
# =========================
def main():
    original_image = read_grayscale(ORIGINAL_IMAGE_PATH)
    original_mask = binarize_mask(read_grayscale(ORIGINAL_MASK_PATH))

    cropped_image = read_grayscale(CROPPED_IMAGE_PATH)
    cropped_mask = binarize_mask(read_grayscale(CROPPED_MASK_PATH))

    cropped_image_256, cropped_mask_256 = resize_pair(
        cropped_image,
        cropped_mask,
        size=RESIZE_SIZE,
    )

    original_overlay = make_overlay(original_image, original_mask)
    cropped_overlay_256 = make_overlay(cropped_image_256, cropped_mask_256)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    show_triplet(
        axes=axes[0],
        image=original_image,
        mask=original_mask,
        overlay=original_overlay,
        row_title="Original",
    )

    show_triplet(
        axes=axes[1],
        image=cropped_image_256,
        mask=cropped_mask_256,
        overlay=cropped_overlay_256,
        row_title="Image 256x256",
    )

    plt.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualization to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()