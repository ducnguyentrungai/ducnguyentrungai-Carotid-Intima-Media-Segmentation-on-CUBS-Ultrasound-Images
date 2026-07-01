import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

mask_path = r"C:\Users\Admin\Downloads\CUBS_DATSET\data\data_cropped\masks\tech_483_mask.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if mask is None:
    raise FileNotFoundError(f"Không đọc được ảnh mask: {mask_path}")


# =========================
# 1. Tạo mask nhị phân IMT
# =========================

imt_mask = (mask > 0).astype(np.uint8)


# =========================
# 2. Giữ vùng liên thông lớn nhất
# =========================

num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    imt_mask,
    connectivity=8
)

if num_labels <= 1:
    raise ValueError("Không tìm thấy vùng IMT trong mask.")

largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
imt_mask_clean = (labels == largest_label).astype(np.uint8)


# =========================
# 3. Trích xuất LI, MA theo từng cột
# =========================

height, width = imt_mask_clean.shape

li_points = np.full(width, np.nan)
ma_points = np.full(width, np.nan)
cimt_pixels = np.full(width, np.nan)

for x in range(width):
    y_indices = np.where(imt_mask_clean[:, x] > 0)[0]

    if len(y_indices) == 0:
        continue

    li_y = y_indices.min()
    ma_y = y_indices.max()

    li_points[x] = li_y
    ma_points[x] = ma_y
    cimt_pixels[x] = ma_y - li_y


mean_cimt_px = np.nanmean(cimt_pixels)

print(f"Mean CIMT: {mean_cimt_px:.2f} pixels")


# =========================
# 4. Chọn một cột x làm ví dụ
# =========================
# Có thể tự chọn, ví dụ:
# example_x = 300
#
# Ở đây chọn cột nằm giữa vùng có IMT hợp lệ.

valid_x = np.where(~np.isnan(cimt_pixels))[0]

if len(valid_x) == 0:
    raise ValueError("Không có cột hợp lệ để tính CIMT.")

example_x = valid_x[len(valid_x) // 2]

example_li = li_points[example_x]
example_ma = ma_points[example_x]
example_cimt = cimt_pixels[example_x]


# =========================
# 5. Hiển thị 3 hình trên subplot(3, 1)
# =========================

x_coords = np.arange(width)
valid = ~np.isnan(li_points)

fig, axes = plt.subplots(3, 1, figsize=(8, 14))


# -------------------------
# Hình 1: Mask + hai biên LI/MA
# -------------------------

axes[0].imshow(imt_mask_clean, cmap="gray")

axes[0].plot(
    x_coords[valid],
    li_points[valid],
    color="cyan",
    linewidth=1.5,
    label="LI"
)

axes[0].plot(
    x_coords[valid],
    ma_points[valid],
    color="red",
    linewidth=1.5,
    label="MA"
)

axes[0].set_title("LI and MA boundaries", fontsize=14)
axes[0].legend()
axes[0].axis("off")


# -------------------------
# Hình 2: Minh họa tại một cột x
# -------------------------

axes[1].imshow(imt_mask_clean, cmap="gray")

axes[1].plot(
    x_coords[valid],
    li_points[valid],
    color="cyan",
    linewidth=1.2,
    label="LI"
)

axes[1].plot(
    x_coords[valid],
    ma_points[valid],
    color="red",
    linewidth=1.2,
    label="MA"
)

axes[1].plot(
    [example_x, example_x],
    [example_li, example_ma],
    color="yellow",
    linewidth=3,
    label=f"CIMT at x={example_x}"
)

axes[1].scatter(
    example_x,
    example_li,
    color="cyan",
    s=60,
    edgecolor="black",
    zorder=5
)

axes[1].scatter(
    example_x,
    example_ma,
    color="red",
    s=60,
    edgecolor="black",
    zorder=5
)

axes[1].set_title(
    f"Example column x={example_x}\n"
    f"CIMT = {example_cimt:.1f} pixels",
    fontsize=14
)

axes[1].legend()
axes[1].axis("off")


# -------------------------
# Hình 3: CIMT theo từng cột ảnh
# -------------------------

axes[2].plot(
    x_coords,
    cimt_pixels,
    color="blue",
    linewidth=1.2
)

axes[2].scatter(
    example_x,
    example_cimt,
    color="red",
    s=60,
    zorder=5
)

axes[2].axvline(
    example_x,
    color="red",
    linestyle="--",
    linewidth=1
)

axes[2].set_title("Column-wise CIMT", fontsize=14)
axes[2].set_xlabel("Column index x")
axes[2].set_ylabel("CIMT (pixels)")
axes[2].grid(True, alpha=0.3)


plt.tight_layout()

output_path = Path(r"C:\Users\Admin\Desktop\Image Processing\ma_li_cal.png")
plt.savefig(output_path, dpi=200, bbox_inches="tight")

print(f"Saved to: {output_path}")

plt.show()