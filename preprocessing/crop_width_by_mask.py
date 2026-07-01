from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Sua cac bien duong dan o day neu ban muon chay voi folder khac.
DATASET_DIR = Path(r"C:\Users\Admin\Downloads\CUBS_DATSET\data")
IMAGES_DIR = DATASET_DIR / "images"
MASKS_DIR = DATASET_DIR / "masks"
OUTPUT_IMAGES_DIR = DATASET_DIR / "images_cropped"
OUTPUT_MASKS_DIR = DATASET_DIR / "masks_cropped"
LOG_FILE = DATASET_DIR / "crop_width_log.csv"

# Mask pixel > THRESHOLD se duoc xem la vung can giu lai.
THRESHOLD = 0

# So pixel giu them ben trai/phai sau khi tim bien mask.
PADDING = 0


def find_mask_x_bounds(mask: Image.Image, threshold: int) -> tuple[int, int] | None:
    """Return inclusive x bounds for non-background pixels in a mask."""
    arr = np.asarray(mask)

    if arr.ndim == 3:
        foreground = np.any(arr[..., :3] > threshold, axis=2)
    else:
        foreground = arr > threshold

    cols = np.where(np.any(foreground, axis=0))[0]
    if cols.size == 0:
        return None

    return int(cols.min()), int(cols.max())


def paired_mask_path(mask_dir: Path, image_path: Path) -> Path:
    return mask_dir / f"{image_path.stem}_mask.png"


def crop_pair(
    image_path: Path,
    mask_path: Path,
    output_image_dir: Path,
    output_mask_dir: Path,
    threshold: int,
    padding: int,
) -> dict[str, object]:
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        if image.size != mask.size:
            raise ValueError(
                f"Size mismatch: {image_path.name} {image.size} != "
                f"{mask_path.name} {mask.size}"
            )

        bounds = find_mask_x_bounds(mask, threshold)
        if bounds is None:
            raise ValueError(f"Mask has no foreground pixels: {mask_path.name}")

        width, height = image.size
        x_min, x_max = bounds
        left = max(0, x_min - padding)
        right = min(width, x_max + 1 + padding)
        crop_box = (left, 0, right, height)

        cropped_image = image.crop(crop_box)
        cropped_mask = mask.crop(crop_box)

        output_image_path = output_image_dir / image_path.name
        output_mask_path = output_mask_dir / mask_path.name
        cropped_image.save(output_image_path)
        cropped_mask.save(output_mask_path)

        return {
            "image": image_path.name,
            "mask": mask_path.name,
            "old_width": width,
            "old_height": height,
            "new_width": right - left,
            "new_height": height,
            "crop_left": left,
            "crop_right_exclusive": right,
        }


def main() -> None:
    images_dir = IMAGES_DIR.resolve()
    masks_dir = MASKS_DIR.resolve()
    output_image_dir = OUTPUT_IMAGES_DIR.resolve()
    output_mask_dir = OUTPUT_MASKS_DIR.resolve()
    log_file = LOG_FILE.resolve()

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Masks folder not found: {masks_dir}")
    if PADDING < 0:
        raise ValueError("PADDING must be >= 0")
    if not 0 <= THRESHOLD <= 255:
        raise ValueError("THRESHOLD must be in [0, 255]")

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No image files found in {images_dir}")

    rows: list[dict[str, object]] = []
    missing_masks: list[str] = []
    errors: list[str] = []

    for image_path in tqdm(image_paths, desc="Cropping image/mask pairs", unit="file"):
        mask_path = paired_mask_path(masks_dir, image_path)
        if not mask_path.is_file():
            missing_masks.append(mask_path.name)
            continue

        try:
            rows.append(
                crop_pair(
                    image_path=image_path,
                    mask_path=mask_path,
                    output_image_dir=output_image_dir,
                    output_mask_dir=output_mask_dir,
                    threshold=THRESHOLD,
                    padding=PADDING,
                )
            )
        except Exception as exc:
            errors.append(f"{image_path.name}: {exc}")

    fieldnames = [
        "image",
        "mask",
        "old_width",
        "old_height",
        "new_width",
        "new_height",
        "crop_left",
        "crop_right_exclusive",
    ]
    with log_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done: {len(rows)} pairs cropped")
    print(f"Images saved to: {output_image_dir}")
    print(f"Masks saved to: {output_mask_dir}")
    print(f"Log saved to: {log_file}")

    if missing_masks:
        print(f"Missing masks: {len(missing_masks)}")
        for name in missing_masks[:10]:
            print(f"  - {name}")
    if errors:
        print(f"Errors: {len(errors)}")
        for error in errors[:10]:
            print(f"  - {error}")

    if missing_masks or errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
