import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_paths = [
    r"C:\Users\Admin\Desktop\Image Processing\bland_altman_imt_all_folds_reunetplus_aspp.png",
    r"C:\Users\Admin\Desktop\Image Processing\bland_altman_imt_all_folds_unet.png",
    r"C:\Users\Admin\Desktop\Image Processing\bland_altman_imt_all_folds_unetplus_aspp.png",
    r"C:\Users\Admin\Desktop\Image Processing\bland_altman_imt_all_folds_unetplusplus.png"
]

titles = [
    "REUNet++ ASPP",
    "UNet",
    "UNet+ ASPP",
    "UNet++"
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, img_path, title in zip(axes.ravel(), image_paths, titles):
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.set_title(title, fontsize=14)
    ax.axis("off")

plt.tight_layout()
plt.savefig(
    r"C:\Users\Admin\Desktop\Image Processing\bland_altman_subplot_2x2.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()