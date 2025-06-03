import os
from typing import Tuple, List, Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile

from processing.color_config import Config
from processing.depth_config import DepthConfig
from processing.visual_config import VisualizationConfig

# 1. Сегментация одноканального изображения
def segment_gray(
    gray: np.ndarray,
    cfg: Any
) -> Dict[str, np.ndarray]:
    steps: Dict[str, np.ndarray] = {"original_gray": gray.copy()}

    if cfg.median_blur_size > 1:
        m = cv2.medianBlur(gray, cfg.median_blur_size)
    else:
        m = gray.copy()
    steps["median_blur"] = m

    if cfg.invert:
        m = cv2.bitwise_not(m)
    steps["inverted"] = m.copy()

    if getattr(cfg, "use_clahe", False):
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit, tileGridSize=cfg.clahe_tile_grid)
        m = clahe.apply(m)
        steps["clahe"] = m.copy()

    if cfg.equalize_hist:
        m = cv2.equalizeHist(m)
        steps["equalized"] = m.copy()

    if cfg.adaptive_thresh:
        b = cv2.adaptiveThreshold(
            m, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY if isinstance(cfg, Config) else cv2.THRESH_BINARY_INV,
            cfg.adaptive_block_size, cfg.adaptive_C
        )
    else:
        flag = cv2.THRESH_BINARY_INV if isinstance(cfg, DepthConfig) else cv2.THRESH_BINARY
        if cfg.use_otsu and cfg.binary_thresh == 0:
            flag |= cv2.THRESH_OTSU
        _, b = cv2.threshold(m, cfg.binary_thresh, 255, flag)
    steps["binary_mask"] = b

    shape_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS
    }
    kern = cv2.getStructuringElement(
        shape_map.get(cfg.morph_kernel_shape, cv2.MORPH_RECT),
        (cfg.morph_kernel_size, cfg.morph_kernel_size)
    )
    opened = cv2.morphologyEx(b, cv2.MORPH_OPEN, kern, iterations=cfg.morph_iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kern, iterations=cfg.dilate_iterations)
    dilated = cv2.dilate(closed, kern, iterations=cfg.dilate_iterations)

    steps["morph_opened"] = opened
    steps["morph_closed"] = closed
    steps["morph_dilated"] = dilated

    dt = cv2.distanceTransform(dilated, cv2.DIST_L2, cfg.distance_transform_mask)
    steps["distance_transform"] = dt
    dtn = cv2.normalize(dt, None, 0, 1.0, cv2.NORM_MINMAX)
    fg = (dtn > cfg.foreground_threshold_ratio).astype(np.uint8) * 255
    steps["foreground_mask"] = fg

    return steps

# 2. Анализ контуров
def analyze_gray(
    img_for_draw: np.ndarray,
    mask: np.ndarray,
    cfg: Any
) -> Tuple[np.ndarray, List[float], List[float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = img_for_draw.copy()
    diams: List[float] = []
    xs:   List[float] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < getattr(cfg, "min_contour_area", 0):
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), _ = rect
        d = float(np.hypot(w, h))
        if not (cfg.min_particle_size <= d <= cfg.max_particle_size):
            continue

        box = cv2.boxPoints(rect).astype(np.int32)
        diams.append(d)
        xs.append(cx)
        cv2.drawContours(out, [box], 0, cfg.bbox_color, cfg.bbox_thickness)
        cv2.putText(out, f"{d:.1f}", (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, cfg.bbox_color, cfg.font_thickness)
    return out, diams, xs

# 3a. Анализ цветного кадра
def analyze_color_frame(
    color_img: np.ndarray,
    cfg: Config,
    v_cfg: VisualizationConfig,
    label: str = "",
    output_dir: str = "",
    save_plots: bool = False
) -> Tuple[np.ndarray, List[float], List[float]]:
    img = (color_img*255).astype(np.uint8) if color_img.dtype != np.uint8 else color_img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps = segment_gray(gray, cfg)
    steps["binary_mask_color"] = steps["binary_mask"]

    final_mask = steps["foreground_mask"]
    res, diams, xs = analyze_gray(img, final_mask, cfg)
    steps["result"] = res

    if save_plots:
        _save_plots(steps, "color", label, output_dir, v_cfg.plot_figsize)
    return res, diams, xs

# 3b. Анализ depth-кадра
def analyze_depth_frame(
    depth_img: np.ndarray,
    cfg: DepthConfig,
    v_cfg: VisualizationConfig,
    label: str = "",
    output_dir: str = "",
    save_plots: bool = False
) -> Tuple[np.ndarray, List[float], List[float]]:
    norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    steps = segment_gray(norm, cfg)

    vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    res, diams, xs = analyze_gray(vis, steps["foreground_mask"], cfg)
    steps["result"] = res

    if save_plots:
        _save_plots(steps, "depth", label, output_dir, v_cfg.plot_figsize)
    return res, diams, xs

# Сохранение промежуточных результатов
def _save_plots(
    steps: Dict[str, np.ndarray],
    mode: str,
    label: str,
    output_dir: str,
    figsize: Tuple[float, float]
):
    last_title, last_img = list(steps.items())[-1]
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    plt.title(f"{mode.upper()}[{label}] {last_title}")
    plt.imshow(last_img, cmap='gray' if last_img.ndim==2 else None)
    plt.axis('off')
    path = os.path.join(output_dir, f"{mode}_{label}_result.png")
    fig.savefig(path); plt.close(fig)

    items = list(steps.items())[:-1]
    n = len(items)
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    for i,(t, im) in enumerate(items):
        ax = fig.add_subplot(n, 1, i+1)
        ax.set_title(f"{mode.upper()}[{label}] {t}")
        ax.imshow(im, cmap='gray' if im.ndim==2 else None)
        ax.axis('off')
    path = os.path.join(output_dir, f"{mode}_{label}_steps.png")
    fig.savefig(path); plt.close(fig)

if __name__ == "__main__":
    from config import OUTPUT_DIR, FRAME_NUMBER

    color_cfg = Config()
    depth_cfg = Config()
    vis_cfg   = VisualizationConfig(show_plots=True)

    img = cv2.imread(os.path.join(OUTPUT_DIR, f"frame_{FRAME_NUMBER}_color.png"))
    c_res, c_diams, c_xs = analyze_color_frame(img, color_cfg, vis_cfg,
                                               label="C1", output_dir=OUTPUT_DIR, save_plots=True)

    depth = tifffile.imread(os.path.join(OUTPUT_DIR, f"frame_{FRAME_NUMBER}_depth.tiff"))
    d_res, d_diams, d_xs = analyze_depth_frame(depth, depth_cfg, vis_cfg,
                                               label="D1", output_dir=OUTPUT_DIR, save_plots=True)
