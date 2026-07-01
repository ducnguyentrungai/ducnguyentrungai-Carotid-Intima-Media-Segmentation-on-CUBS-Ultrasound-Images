import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# 1. Đường dẫn dữ liệu
# =========================

image_path = r"C:\Users\Admin\Downloads\CUBS_DATSET\data\images\clin_0001_L.tiff"
mask_path = r"C:\Users\Admin\Downloads\CUBS_DATSET\data\masks\clin_0001_L_mask.png"

output_path = Path(r"C:\Users\Admin\Desktop\Image Processing\model_input_output_pipeline.png")


# =========================
# 2. Đọc ảnh và mask
# =========================

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Không đọc được ảnh gốc: {image_path}")

if mask is None:
    raise FileNotFoundError(f"Không đọc được mask: {mask_path}")


# Mask nhị phân: vùng IMT = 1, nền = 0
mask_bin = (mask > 0).astype(np.uint8)


# =========================
# 3. Crop theo width có mask, giữ nguyên height
# =========================

ys, xs = np.where(mask_bin > 0)

if len(xs) == 0:
    raise ValueError("Mask không có vùng foreground.")

x_min, x_max = xs.min(), xs.max()

# Thêm padding theo chiều ngang
padding_x = 30

x_min = max(x_min - padding_x, 0)
x_max = min(x_max + padding_x, image.shape[1] - 1)

# Giữ nguyên toàn bộ chiều cao ảnh
y_min = 0
y_max = image.shape[0] - 1

image_crop = image[y_min:y_max + 1, x_min:x_max + 1]
mask_crop = mask_bin[y_min:y_max + 1, x_min:x_max + 1]

# =========================
# 4. Resize ảnh và mask về 256 x 256
# =========================

target_size = (256, 256)

image_resized = cv2.resize(
    image_crop,
    target_size,
    interpolation=cv2.INTER_LINEAR
)

mask_resized = cv2.resize(
    mask_crop,
    target_size,
    interpolation=cv2.INTER_NEAREST
)

mask_resized = (mask_resized > 0).astype(np.uint8)


# =========================
# 5. Minh họa output của mô hình
# =========================
# Trong thực tế:
#
# probability_map = model.predict(image_resized)
#
# Ở đây ta mô phỏng probability map từ mask resize
# bằng cách làm mờ mask để giống bản đồ xác suất.

probability_map = cv2.GaussianBlur(
    mask_resized.astype(np.float32),
    ksize=(21, 21),
    sigmaX=5
)

probability_map = probability_map / probability_map.max()


# =========================
# 6. Ngưỡng hóa bản đồ xác suất
# =========================

threshold = 0.5

pred_mask = (probability_map >= threshold).astype(np.uint8)


# =========================
# 7. Hiển thị quy trình
# =========================

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes[0, 0].imshow(image, cmap="gray")
axes[0, 0].set_title("Original ultrasound image", fontsize=12)
axes[0, 0].axis("off")

axes[0, 1].imshow(mask_bin, cmap="gray")
axes[0, 1].set_title("Ground-truth mask", fontsize=12)
axes[0, 1].axis("off")

axes[0, 2].imshow(image_crop, cmap="gray")
axes[0, 2].imshow(mask_crop, cmap="Reds", alpha=0.35)
axes[0, 2].set_title("Cropped region by mask", fontsize=12)
axes[0, 2].axis("off")

axes[1, 0].imshow(image_resized, cmap="gray")
axes[1, 0].imshow(mask_resized, cmap="Reds", alpha=0.35)
axes[1, 0].set_title("Image/mask resized to 256 x 256", fontsize=12)
axes[1, 0].axis("off")

im = axes[1, 1].imshow(probability_map, cmap="jet", vmin=0, vmax=1)
axes[1, 1].set_title("Output probability map", fontsize=12)
axes[1, 1].axis("off")

axes[1, 2].imshow(pred_mask, cmap="gray")
axes[1, 2].set_title(f"Binary mask after threshold = {threshold}", fontsize=12)
axes[1, 2].axis("off")

plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Saved to: {output_path}")

plt.show()