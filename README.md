# CUBS Segmentation Source Guide

Thu muc `src/` chua code tien xu ly du lieu, phan tich CIMT/LI-MA, visualization va notebook huan luyen mo hinh segmentation tren CUBS.

## Tac gia

- Name/GitHub: ducnguyentrungai
- Email: ducnguyentrungai@gmail.com
- Project: Final project for Image Processing course

## Bai bao va bo du lieu tham khao

- 2021 - Carotid Ultrasound Boundary Study (CUBS): An Open Multicenter Analysis of Computerized Intima-Media Thickness Measurement Systems and Their Clinical Impact.
  - Article DOI: https://doi.org/10.1016/j.ultrasmedbio.2021.03.022
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/33941415/
  - Dataset DOI: https://doi.org/10.17632/fpv535fss7.1

- 2022 - Carotid Ultrasound Boundary Study (CUBS): Technical considerations on an open multi-center analysis of computerized measurement systems for intima-media thickness measurement on common carotid artery longitudinal B-mode ultrasound scans.
  - Article DOI: https://doi.org/10.1016/j.compbiomed.2022.105333
  - PubMed: https://pubmed.ncbi.nlm.nih.gov/35279425/
  - Dataset DOI: https://doi.org/10.17632/m7ndn58sv6.1
  - Mendeley Data: https://data.mendeley.com/datasets/m7ndn58sv6/1

## Cau truc thu muc

```text
src/
  analysis/        # CIMT, LI/MA, carotid structure plots
  configs/         # cau hinh duong dan cho preprocessing
  notebooks/       # notebook train/evaluate tren Kaggle
  preprocessing/   # tao mask, crop image/mask
  visualization/   # overlay, metric chart, split chart
```

Du lieu chinh cua repo dang nam o:

```text
C:\Users\Admin\Downloads\CUBS_DATSET\data
  images/
  masks/
  data_cropped/
  kfold_dataset.rar
```

## Cai dat moi truong local

Chay tren Windows PowerShell tu thu muc goc du an:

```powershell
cd C:\Users\Admin\Downloads\CUBS_DATSET
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pillow tqdm pyyaml opencv-python matplotlib seaborn pandas scikit-learn
```

Neu can chay notebook local:

```powershell
pip install jupyter albumentations monai segmentation-models-pytorch torch torchvision
```

## Chay preprocessing data

### 1. Tao mask tu profile LI/MA

Kiem tra va sua file cau hinh:

```text
src\configs\config.yaml
```

File nay can tro toi:

- `dataset_dir`: thu muc CUBS goc.
- `image_subdir`: thu muc anh goc, vi du `IMAGES`.
- `profile_subdir`: thu muc profile LI/MA.
- `output_dir`: noi luu mask sinh ra.
- `ext`: duoi anh, thuong la `.tiff`.

Sau do chay:

```powershell
python src\preprocessing\make_mask_from_profiles.py
```

Ket qua la cac file mask dang:

```text
<image_id>_mask.png
```

### 2. Crop anh va mask theo chieu rong mask

Script nay dang mac dinh doc:

```text
data\images
data\masks
```

va ghi ra:

```text
data\images_cropped
data\masks_cropped
data\crop_width_log.csv
```

Chay:

```powershell
python src\preprocessing\crop_width_by_mask.py
```

Neu muon doi folder input/output, sua cac bien o dau file:

```text
src\preprocessing\crop_width_by_mask.py
```

Cac bien quan trong:

- `DATASET_DIR`
- `IMAGES_DIR`
- `MASKS_DIR`
- `OUTPUT_IMAGES_DIR`
- `OUTPUT_MASKS_DIR`
- `LOG_FILE`
- `THRESHOLD`
- `PADDING`

## Chay code phan tich va visualization

Mot so script ve hinh dang dung duong dan tuyet doi tren may Windows. Neu chay may khac, can sua cac bien `image_path`, `mask_path`, `output_path` trong tung file.

Vi du:

```powershell
python src\visualization\visualize_mask_overlay.py
python src\visualization\visualize_cimt_mask_boundaries.py
python src\analysis\plot_carotid_structure.py --help
```

## Chay notebook tren Kaggle GPU T4

Moi truong khuyen nghi:

- Kaggle Notebook.
- Accelerator: `GPU T4`.
- Internet: bat `On` neu notebook can `pip install`.
- Dataset input: upload folder da preprocessing hoac file `.rar/.zip` len Kaggle Dataset.

### Cach dua data len Kaggle

Nen upload cau truc da crop san:

```text
cubs-k-fold-dataset/
  data_cropped/
    images/
    masks/
```

Sau khi add Kaggle Dataset vao notebook, duong dan se co dang:

```text
/kaggle/input/<dataset-slug>/data_cropped/images
/kaggle/input/<dataset-slug>/data_cropped/masks
```

Trong notebook, dat lai bien path theo dataset slug cua ban:

```python
from pathlib import Path

DATA_ROOT = Path("/kaggle/input/<dataset-slug>/data_cropped")
IMAGE_DIR = DATA_ROOT / "images"
MASK_DIR = DATA_ROOT / "masks"
WORK_DIR = Path("/kaggle/working")
```

Neu upload file nen:

```python
!mkdir -p /kaggle/working/cubs
!unrar x /kaggle/input/<dataset-slug>/kfold_dataset.rar /kaggle/working/cubs/
```

Sau khi giai nen, cap nhat lai:

```python
DATA_ROOT = Path("/kaggle/working/cubs/data_cropped")
```

### Cai package trong Kaggle

Chay cell dau notebook:

```python
!pip install -q albumentations monai segmentation-models-pytorch
```

Kiem tra GPU:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Notebook hien co

```text
src\notebooks\unet-segmentation-cubs.ipynb
src\notebooks\unetplusplus-cubs.ipynb
src\notebooks\unetpusplusaspp-cubs.ipynb
src\notebooks\reunetplusplussaspp.ipynb
```

Thu tu chay khuyen nghi:

1. Kiem tra path anh/mask bang mot cell visualize nhanh.
2. Chay cell cai package.
3. Chay cell import va config.
4. Chay split train/validation hoac k-fold.
5. Train model tren `DEVICE = "cuda"`.
6. Luu checkpoint vao `/kaggle/working`.
7. Download output/checkpoint tu tab Output cua Kaggle.

## Luu y

- `__pycache__/` la cache Python, co the xoa an toan.
- Khong nen hard-code path Windows trong notebook Kaggle; hay dung `/kaggle/input/...` va `/kaggle/working/...`.
- Neu notebook bao loi khong thay file, in `list(DATA_ROOT.iterdir())` de kiem tra dung slug va dung cap thu muc.
