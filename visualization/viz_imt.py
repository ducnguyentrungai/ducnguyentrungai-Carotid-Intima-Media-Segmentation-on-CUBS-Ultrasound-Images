import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


image_path = r"C:\Users\Admin\Downloads\CUBS_DATSET\data\data_cropped\images\tech_001.tiff"
mask_path = r"C:\Users\Admin\Downloads\CUBS_DATSET\data\data_cropped\masks\tech_001_mask.png"

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Không đọc được ảnh gốc: {image_path}")

if mask is None:
    raise FileNotFoundError(f"Không đọc được mask: {mask_path}")


# =========================
# 1. Tạo mask nhị phân
# =========================

imt_mask = (mask > 0).astype(np.uint8)


# =========================
# 2. Giữ vùng IMT lớn nhất
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
# 3. Lấy biên LI và MA theo từng cột
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


# =========================
# 4. Chọn một cột x để minh họa CIMT
# =========================

valid_x = np.where(~np.isnan(cimt_pixels))[0]

if len(valid_x) == 0:
    raise ValueError("Không có cột hợp lệ để tính CIMT.")

# Chọn cột ở giữa vùng có IMT
example_x = valid_x[len(valid_x) // 2]

li_y = li_points[example_x]
ma_y = ma_points[example_x]
cimt = cimt_pixels[example_x]


# =========================
# 5. Vẽ kiểu minh họa CIMT
# =========================

x_coords = np.arange(width)
valid = ~np.isnan(li_points)

plt.figure(figsize=(10, 5))

# Hiển thị ảnh siêu âm gốc
plt.imshow(image, cmap="gray")

# Vẽ biên LI màu xanh
plt.plot(
    x_coords[valid],
    li_points[valid],
    color="lime",
    linewidth=2,
    label="LI"
)

# Vẽ biên MA màu đỏ
plt.plot(
    x_coords[valid],
    ma_points[valid],
    color="red",
    linewidth=2,
    label="MA"
)

# Vẽ đường CIMT tại một cột x
plt.plot(
    [example_x, example_x],
    [li_y, ma_y],
    color="yellow",
    linewidth=2.5,
    linestyle="--"
)

# Đánh dấu hai đầu LI và MA
plt.scatter(
    example_x,
    li_y,
    color="yellow",
    s=45,
    zorder=5
)

plt.scatter(
    example_x,
    ma_y,
    color="yellow",
    s=45,
    zorder=5
)

# Ghi chữ IMT bên cạnh đường đo
plt.text(
    example_x + 10,
    (li_y + ma_y) / 2,
    "IMT",
    color="yellow",
    fontsize=16,
    fontweight="bold",
    va="center"
)

plt.title(f"CIMT measurement at x={example_x}: {cimt:.1f} pixels")
plt.axis("off")
plt.tight_layout()

output_path = Path(r"C:\Users\Admin\Desktop\Image Processing\cimt_viz.png")
plt.savefig(output_path, dpi=200, bbox_inches="tight")

print(f"Saved to: {output_path}")
plt.show()