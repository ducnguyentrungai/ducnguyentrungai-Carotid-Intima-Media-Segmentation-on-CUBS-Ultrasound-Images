from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "weight" / "model_unetplusplus.pt"
INPUT_SIZE = 256


@dataclass
class PredictionResult:
    original_image: Image.Image
    original_shape: tuple[int, int]
    resized_image: Image.Image
    prob_256: np.ndarray
    pred_mask_256: np.ndarray
    prob_orig: np.ndarray
    pred_mask_orig: np.ndarray
    mask_confidence: float
    global_confidence: float


@dataclass
class BoundaryResult:
    column_index: np.ndarray
    li_px: np.ndarray
    ma_px: np.ndarray
    li_um: np.ndarray
    ma_um: np.ndarray
    imt_um: np.ndarray


def configure_page() -> None:
    st.set_page_config(
        page_title="CIMT",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; max-width: 1360px; }
        section[data-testid="stSidebar"] { min-width: 330px; }
        .info-box {
            border: 1px solid #d5dde3;
            border-radius: 8px;
            padding: 14px 16px;
            background: #f7fafb;
        }
        .formula {
            border-left: 4px solid #0f766e;
            border-radius: 6px;
            background: #f2faf7;
            padding: 12px 14px;
            font-family: Consolas, "Courier New", monospace;
            font-size: 0.92rem;
        }
        .stage-label {
            color: #53656f;
            font-size: 0.92rem;
            margin-top: -0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_uploaded_image(uploaded_file) -> Image.Image:
    image = Image.open(uploaded_file)
    image.load()
    return image.convert("L")


def to_uint8_array(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("L"))
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    max_value = float(np.nanmax(arr)) if arr.size else 1.0
    if max_value <= 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip(arr / max_value * 255.0, 0, 255).astype(np.uint8)


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8,
    )
    if num_labels <= 1:
        return mask_uint8.astype(bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def load_model():
    try:
        import torch
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "Moi truong hien tai thieu torch hoac segmentation_models_pytorch. "
            "Hay chay app trong env conda `cubs`."
        ) from exc

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Khong tim thay weight: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=1,
        classes=2,
    ).to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned_state = {}
    for key, value in state_dict.items():
        cleaned_state[key.replace("module.", "", 1)] = value

    model.load_state_dict(cleaned_state, strict=True)
    model.eval()
    return model, device, torch


@st.cache_resource(show_spinner=False)
def cached_model():
    return load_model()


def predict_unetplusplus(
    image: Image.Image,
    threshold: float,
    keep_largest: bool,
) -> PredictionResult:
    model, device, torch = cached_model()

    original = image.convert("L")
    original_width, original_height = original.size
    original_shape = (original_height, original_width)

    original_arr = to_uint8_array(original)
    resized_arr = cv2.resize(
        original_arr,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    tensor_arr = resized_arr.astype(np.float32) / 255.0
    tensor = torch.from_numpy(tensor_arr)[None, None, :, :].to(device)

    with torch.no_grad():
        logits = model(tensor.float())
        if logits.shape[1] == 1:
            prob = torch.sigmoid(logits[:, 0])
        else:
            prob = torch.softmax(logits, dim=1)[:, 1]

    prob_256 = prob[0].detach().cpu().numpy().astype(np.float32)
    pred_mask_256 = prob_256 >= threshold
    if keep_largest:
        pred_mask_256 = largest_component(pred_mask_256)

    prob_orig = cv2.resize(
        prob_256,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pred_mask_orig = cv2.resize(
        pred_mask_256.astype(np.uint8),
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    if keep_largest:
        pred_mask_orig = largest_component(pred_mask_orig)

    if pred_mask_orig.any():
        mask_confidence = float(np.mean(prob_orig[pred_mask_orig]))
    else:
        mask_confidence = 0.0

    return PredictionResult(
        original_image=original,
        original_shape=original_shape,
        resized_image=Image.fromarray(resized_arr),
        prob_256=prob_256,
        pred_mask_256=pred_mask_256,
        prob_orig=prob_orig,
        pred_mask_orig=pred_mask_orig,
        mask_confidence=mask_confidence,
        global_confidence=float(np.mean(prob_orig)),
    )


def mask_to_li_ma(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.squeeze(mask)
    mask = np.asarray(mask) > 0
    _, width = mask.shape
    li = np.full(width, np.nan, dtype=np.float32)
    ma = np.full(width, np.nan, dtype=np.float32)

    for x in range(width):
        ys = np.flatnonzero(mask[:, x])
        if ys.size > 0:
            li[x] = ys.min()
            ma[x] = ys.max()

    return li, ma


def boundary_from_mask(mask: np.ndarray, um_per_pixel: float) -> BoundaryResult:
    li_px, ma_px = mask_to_li_ma(mask)
    valid = ~(np.isnan(li_px) | np.isnan(ma_px))
    return BoundaryResult(
        column_index=np.where(valid)[0],
        li_px=li_px,
        ma_px=ma_px,
        li_um=li_px[valid] * um_per_pixel,
        ma_um=ma_px[valid] * um_per_pixel,
        imt_um=(ma_px[valid] - li_px[valid]) * um_per_pixel,
    )


def boundary_errors_from_masks(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    um_per_pixel: float,
) -> pd.DataFrame:
    pred_li, pred_ma = mask_to_li_ma(pred_mask)
    gt_li, gt_ma = mask_to_li_ma(gt_mask)
    valid = ~(np.isnan(pred_li) | np.isnan(pred_ma) | np.isnan(gt_li) | np.isnan(gt_ma))

    if valid.sum() == 0:
        return pd.DataFrame()

    li_pred_um = pred_li[valid] * um_per_pixel
    li_gt_um = gt_li[valid] * um_per_pixel
    ma_pred_um = pred_ma[valid] * um_per_pixel
    ma_gt_um = gt_ma[valid] * um_per_pixel
    imt_pred_um = (pred_ma[valid] - pred_li[valid]) * um_per_pixel
    imt_gt_um = (gt_ma[valid] - gt_li[valid]) * um_per_pixel

    li_signed_error_um = li_pred_um - li_gt_um
    ma_signed_error_um = ma_pred_um - ma_gt_um
    imt_signed_error_um = imt_pred_um - imt_gt_um

    return pd.DataFrame(
        {
            "column_index": np.where(valid)[0],
            "li_pred_um": li_pred_um,
            "li_gt_um": li_gt_um,
            "ma_pred_um": ma_pred_um,
            "ma_gt_um": ma_gt_um,
            "imt_pred_um": imt_pred_um,
            "imt_gt_um": imt_gt_um,
            "li_abs_error_um": np.abs(li_signed_error_um),
            "ma_abs_error_um": np.abs(ma_signed_error_um),
            "imt_abs_error_um": np.abs(imt_signed_error_um),
        }
    )


def safe_mean(values) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    return float(series.mean()) if len(series) else float("nan")


def fmt_um(value: float) -> str:
    if np.isnan(value):
        return "N/A"
    return f"{value:,.2f} um"


def fmt_mm_from_um(value_um: float) -> str:
    if np.isnan(value_um):
        return "N/A"
    return f"{value_um / 1000.0:.4f} mm"


def mask_overlay(image: Image.Image, mask: np.ndarray, color=(255, 64, 32), alpha=0.35) -> Image.Image:
    base = image.convert("RGB")
    base_arr = np.asarray(base).astype(np.float32)
    overlay = np.zeros_like(base_arr)
    overlay[:, :] = color
    mask_bool = mask.astype(bool)
    base_arr[mask_bool] = (1.0 - alpha) * base_arr[mask_bool] + alpha * overlay[mask_bool]
    return Image.fromarray(np.clip(base_arr, 0, 255).astype(np.uint8))


def probability_heatmap(prob: np.ndarray) -> Image.Image:
    prob_uint8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heat)


def draw_boundaries(image: Image.Image, boundary: BoundaryResult) -> Image.Image:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    li_points = []
    ma_points = []

    for x, (li_y, ma_y) in enumerate(zip(boundary.li_px, boundary.ma_px)):
        if not np.isnan(li_y):
            li_points.append((int(x), int(li_y)))
        if not np.isnan(ma_y):
            ma_points.append((int(x), int(ma_y)))

    if len(li_points) > 1:
        draw.line(li_points, fill=(0, 220, 255), width=2)
    if len(ma_points) > 1:
        draw.line(ma_points, fill=(255, 184, 0), width=2)
    return canvas


def make_imt_chart(boundary: BoundaryResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(boundary.column_index, boundary.imt_um / 1000.0, color="#0f766e", linewidth=1.8)
    ax.set_xlabel("Column index")
    ax.set_ylabel("IMT (mm)")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def make_error_chart(errors: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(errors["column_index"], errors["li_abs_error_um"], label="MAE LI", color="#0284c7")
    ax.plot(errors["column_index"], errors["ma_abs_error_um"], label="MAE MA", color="#7c3aed")
    ax.plot(errors["column_index"], errors["imt_abs_error_um"], label="MAE IMT", color="#b45309")
    ax.set_xlabel("Column index")
    ax.set_ylabel("Absolute error (um)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncols=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def make_boundary_comparison_chart(
    errors: pd.DataFrame,
    predicted_column: str,
    ground_truth_column: str,
    title: str,
    y_label: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(
        errors["column_index"],
        errors[predicted_column],
        label="Predicted",
        color="#0284c7",
        linewidth=1.7,
    )
    ax.plot(
        errors["column_index"],
        errors[ground_truth_column],
        label="Ground truth",
        color="#b45309",
        linewidth=1.7,
        alpha=0.9,
    )
    ax.set_title(title)
    ax.set_xlabel("Column index")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def render_sidebar() -> tuple[float, float, bool]:
    st.sidebar.header("Pixel -> mm")
    st.sidebar.markdown(
        '<div class="info-box">Nhap truc tiep he so chuyen doi mm/pixel. Gia tri phai nho hon 1. '
        'Khi tinh mean absolute error theo micrometer: um/pixel = mm/pixel x 1000.</div>',
        unsafe_allow_html=True,
    )

    mm_per_pixel = st.sidebar.number_input(
        "He so chuyen doi (mm/pixel)",
        min_value=0.000000,
        max_value=0.999999,
        value=0.0,
        step=0.001000,
        format="%.6f",
    )
    um_per_pixel = mm_per_pixel * 1000.0
    st.sidebar.metric("Micromet / pixel", f"{um_per_pixel:.3f} um")

    st.sidebar.header("Unet++")
    threshold = st.sidebar.slider(
        "Nguong tao mask tu probability",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
    )
    keep_largest = st.sidebar.checkbox("Giu vung lien thong lon nhat", value=True)

    st.sidebar.caption(f"Weight: {MODEL_PATH.name if MODEL_PATH.is_file() else 'missing'}")
    return um_per_pixel, threshold, keep_largest


def render_pipeline_summary(result: PredictionResult) -> None:
    st.metric("Mask confidence", f"{result.mask_confidence * 100:.2f}%")


def render_prediction_views(result: PredictionResult, boundary: BoundaryResult) -> None:
    st.subheader("Ket qua du doan")
    mask_pred_orig_image = Image.fromarray((result.pred_mask_orig.astype(np.uint8) * 255))
    columns = st.columns(4)
    columns[0].image(
        result.original_image,
        caption="Anh goc",
        use_container_width=True,
    )
    columns[1].image(
        mask_pred_orig_image,
        caption="Anh mask da chuyen ve anh goc",
        use_container_width=True,
    )
    columns[2].image(
        mask_overlay(result.original_image, result.pred_mask_orig),
        caption="Overlay",
        use_container_width=True,
    )
    columns[3].image(
        draw_boundaries(result.original_image, boundary),
        caption="LI cyan, MA vang tren mask_pred_orig",
        use_container_width=True,
    )


def render_pred_imt_statistics(boundary: BoundaryResult) -> None:
    st.subheader("Thong so Carotid Intima Media Thickness cua prediction")
    imt_values_um = pd.to_numeric(pd.Series(boundary.imt_um), errors="coerce").dropna()

    if imt_values_um.empty:
        st.warning("Khong co gia tri Carotid Intima Media Thickness hop le de tinh thong so.")
        return

    imt_values_px = pd.Series(
        boundary.ma_px[boundary.column_index] - boundary.li_px[boundary.column_index],
        dtype="float32",
    ).dropna()

    stats_df = pd.DataFrame(
        [
            {
                "Thong so": "Mean Carotid Intima Media Thickness",
                "Pixel": imt_values_px.mean(),
                "Micrometer": imt_values_um.mean(),
                "Millimeter": imt_values_um.mean() / 1000.0,
            },
            {
                "Thong so": "Min Carotid Intima Media Thickness",
                "Pixel": imt_values_px.min(),
                "Micrometer": imt_values_um.min(),
                "Millimeter": imt_values_um.min() / 1000.0,
            },
            {
                "Thong so": "Max Carotid Intima Media Thickness",
                "Pixel": imt_values_px.max(),
                "Micrometer": imt_values_um.max(),
                "Millimeter": imt_values_um.max() / 1000.0,
            },
            {
                "Thong so": "Median Carotid Intima Media Thickness",
                "Pixel": imt_values_px.median(),
                "Micrometer": imt_values_um.median(),
                "Millimeter": imt_values_um.median() / 1000.0,
            },
        ]
    )
    st.table(
        stats_df.style.format(
            {
                "Pixel": "{:,.2f}",
                "Micrometer": "{:,.2f}",
                "Millimeter": "{:.4f}",
            }
        )
    )
    st.caption(
        "Gia tri micrometer = Pixel x he so mm/pixel x 1000. "
        "Neu Pixel lon, hay kiem tra lai mask prediction/threshold; neu Pixel hop ly nhung micrometer lon, hay kiem tra he so mm/pixel."
    )


def render_mae_section(result: PredictionResult, um_per_pixel: float) -> None:
    st.subheader("Mean absolute error")
    st.caption("Can upload ground truth mask de tinh sai so trung binh tuyet doi.")
    gt_file = st.file_uploader(
        "Upload mask ground truth cung anh goc (tuy chon)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="gt_mask_for_mae",
    )
    if gt_file is None:
        st.info("Chua co ground truth mask nen chua tinh MAE.")
        return

    gt_image = load_uploaded_image(gt_file)
    h, w = result.original_shape
    if gt_image.size != (w, h):
        st.warning("GT mask khac original shape, app se resize nearest ve original shape truoc khi tinh MAE.")
        gt_arr = cv2.resize(
            to_uint8_array(gt_image),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        gt_arr = to_uint8_array(gt_image)

    gt_mask = gt_arr > 0
    errors = boundary_errors_from_masks(result.pred_mask_orig, gt_mask, um_per_pixel)
    if errors.empty:
        st.error("Khong co cot hop le chung giua mask_pred_orig va ground truth.")
        return

    li_mae = safe_mean(errors["li_abs_error_um"])
    ma_mae = safe_mean(errors["ma_abs_error_um"])
    imt_mae = safe_mean(errors["imt_abs_error_um"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cot so sanh", f"{len(errors):,}")
    c2.metric("Mean absolute error Lumen Intima", fmt_um(li_mae))
    c3.metric("Mean absolute error Media Adventitia", fmt_um(ma_mae))
    c4.metric("Mean absolute error Carotid Intima Media Thickness", fmt_um(imt_mae))

    st.pyplot(
        make_boundary_comparison_chart(
            errors=errors,
            predicted_column="li_pred_um",
            ground_truth_column="li_gt_um",
            title="Lumen Intima predicted and Lumen Intima ground truth",
            y_label="Lumen Intima position (micrometer)",
        ),
        clear_figure=True,
    )
    st.pyplot(
        make_boundary_comparison_chart(
            errors=errors,
            predicted_column="ma_pred_um",
            ground_truth_column="ma_gt_um",
            title="Media Adventitia predicted and Media Adventitia ground truth",
            y_label="Media Adventitia position (micrometer)",
        ),
        clear_figure=True,
    )
    st.pyplot(
        make_boundary_comparison_chart(
            errors=errors,
            predicted_column="imt_pred_um",
            ground_truth_column="imt_gt_um",
            title="Carotid Intima Media Thickness predicted and Carotid Intima Media Thickness ground truth",
            y_label="Carotid Intima Media Thickness (micrometer)",
        ),
        clear_figure=True,
    )



def main() -> None:
    configure_page()
    um_per_pixel, threshold, keep_largest = render_sidebar()

    st.title("Do do day noi trung mac dong mach canh")
    
    image_file = st.file_uploader(
        "Upload anh noi soi/sieu am dong mach canh",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="original_image",
    )
    if image_file is None:
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("prediction_file_key", None)
        st.info("Hay upload anh de bat dau pipeline.")
        return
    
    file_key = f"{image_file.name}_{image_file.size}"
    if st.session_state.get("prediction_file_key") != file_key:
        st.session_state.pop("prediction_result", None)
        st.session_state["prediction_file_key"] = file_key

    if st.button("Thuc hien", type="primary", use_container_width=True):
        try:
            image = load_uploaded_image(image_file)
            with st.spinner("Dang chay Unet++ tren anh resize 256 x 256..."):
                st.session_state["prediction_result"] = predict_unetplusplus(
                    image,
                    threshold,
                    keep_largest,
                )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    if "prediction_result" not in st.session_state:
        st.info("Nhan nut Thuc hien de bat dau du doan.")
        return

    result = st.session_state["prediction_result"]

    boundary = boundary_from_mask(result.pred_mask_orig, um_per_pixel)
    render_pipeline_summary(result)

    if len(boundary.column_index) == 0:
        st.error("Mask du doan khong co cot hop le de tinh LI/MA/IMT.")
        render_prediction_views(result, boundary)
        st.stop()

    render_prediction_views(result, boundary)
    render_pred_imt_statistics(boundary)
    render_mae_section(result, um_per_pixel)


if __name__ == "__main__":
    main()
