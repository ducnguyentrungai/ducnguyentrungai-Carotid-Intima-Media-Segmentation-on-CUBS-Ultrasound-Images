from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# image_paths = {
#     "U-Net": Path(r"C:\Users\Admin\Desktop\Image Processing\unet_loss_val_train.png"),
#     "U-Net++": Path(r"C:\Users\Admin\Desktop\Image Processing\unetplus_los_val_train.png"),
#     "U-Net++ + ASPP": Path(r"C:\Users\Admin\Desktop\Image Processing\unetplusaspp_los_val_train.png"),
#     "ReU-Net++ + ASPP": Path(r"C:\Users\Admin\Desktop\Image Processing\reunetplusaspp_los_val_train.png"),
# }


image_paths = {
    "U-Net": Path(r"C:\Users\Admin\Desktop\Image Processing\unet_dice_iou_hd95_val.png"),
    "U-Net++": Path(r"C:\Users\Admin\Desktop\Image Processing\unetplus_dice_iou_hd95_val.png"),
    "U-Net++ + ASPP": Path(r"C:\Users\Admin\Desktop\Image Processing\unetplusassp_dice_iou_hd95_val.png"),
    "ReU-Net++ + ASPP": Path(r"C:\Users\Admin\Desktop\Image Processing\reunetplusaspp_dice_iou_hd95_val.png"),
}
# fig, axes = plt.subplots(2, 2, figsize=(10, 15))
fig, axes = plt.subplots(4, 1, figsize=(10, 12))
axes = axes.ravel()

for ax, (title, path) in zip(axes, image_paths.items()):
    if not path.is_file():
        raise FileNotFoundError(path)

    img = mpimg.imread(path)

    ax.imshow(img)
    ax.set_title(title, fontsize=12)
    ax.axis("off")

output_path = Path(r"C:\Users\Admin\Desktop\Image Processing\all_model_metric_comparison.png")
fig.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Saved to: {output_path}")
plt.tight_layout()
plt.show()