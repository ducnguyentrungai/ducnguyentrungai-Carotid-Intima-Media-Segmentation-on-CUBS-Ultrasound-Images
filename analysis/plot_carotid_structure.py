from pathlib import Path
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def boundary_from_mask(mask):
    mask = mask > 0
    xs, li, ma = [], [], []
    for x in range(mask.shape[1]):
        ys = np.where(mask[:, x])[0]
        if len(ys):
            xs.append(x)
            li.append(ys.min())
            ma.append(ys.max())
    return np.array(xs), np.array(li), np.array(ma)


def smooth(y, k=17):
    k = min(k, len(y) if len(y) % 2 else len(y) - 1)
    if k < 3:
        return y
    pad = k // 2
    y_pad = np.pad(y.astype(float), pad, mode="edge")
    return np.array([np.median(y_pad[i:i + k]) for i in range(len(y))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="tech_401", help="Image id, e.g. tech_401")
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    image_path = root / "images" / f"{args.case}.tiff"
    mask_path = root / "GT_masks" / f"{args.case}_mask.png"
    out_path = Path(args.out) if args.out else Path(__file__).with_name(f"carotid_structure_{args.case}.png")

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)
    if mask is None:
        raise FileNotFoundError(mask_path)

    xs, li, ma = boundary_from_mask(mask)
    li, ma = smooth(li), smooth(ma)

    x_mid = int(xs[len(xs) // 2])
    li_mid = int(li[len(li) // 2])
    ma_mid = int(ma[len(ma) // 2])

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
    ax.imshow(img, cmap="gray")
    ax.fill_between(xs, li, ma, color="#2ca25f", alpha=0.28, label="Vùng nội - trung mạc")
    ax.plot(xs, li, color="#00d5ff", lw=2.2, label="LI: Lumen-Intima")
    ax.plot(xs, ma, color="#ffb000", lw=2.2, label="MA: Media-Adventitia")

    ax.annotate("Lòng mạch", xy=(x_mid, li_mid - 8), xytext=(x_mid - 170, li_mid - 95),
                color="white", fontsize=11,
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", alpha=0.65))
    ax.annotate("Lớp nội mạc", xy=(x_mid, li_mid + 3), xytext=(x_mid + 70, li_mid - 65),
                color="#00d5ff", fontsize=11,
                arrowprops=dict(arrowstyle="->", color="#00d5ff", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="#00d5ff", alpha=0.65))
    ax.annotate("Lớp trung mạc", xy=(x_mid, ma_mid - 3), xytext=(x_mid + 85, ma_mid + 65),
                color="#ffb000", fontsize=11,
                arrowprops=dict(arrowstyle="->", color="#ffb000", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="#ffb000", alpha=0.65))
    ax.annotate("Lớp ngoại mạc", xy=(x_mid, ma_mid + 8), xytext=(x_mid - 180, ma_mid + 95),
                color="white", fontsize=11,
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", alpha=0.65))

    ax.set_title("Cấu trúc thành động mạch cảnh trên ảnh siêu âm B-mode", fontsize=14, weight="bold")
    ax.set_xlabel("Trục ngang ảnh (pixel)")
    ax.set_ylabel("Trục dọc ảnh (pixel)")
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim(max(xs.min() - 80, 0), min(xs.max() + 80, img.shape[1] - 1))
    ax.set_ylim(min(ma.max() + 130, img.shape[0] - 1), max(li.min() - 130, 0))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(out_path)


if __name__ == "__main__":
    main()
