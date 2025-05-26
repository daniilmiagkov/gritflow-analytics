from dataclasses import dataclass
from typing import Tuple, List

import cv2
import numpy as np
import matplotlib.pyplot as plt

from processing.color_config import ColorConfig
from processing.visual_config import VisualizationConfig


def analyze_mask(img: np.ndarray, mask: np.ndarray,
                 cfg: ColorConfig) -> Tuple[np.ndarray, List[float], List[float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = img.copy()
    diagonals, xs = [], []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # if area < cfg.min_contour_area:
        #     continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        diag = np.sqrt(w ** 2 + h ** 2)

        if not (cfg.min_particle_size <= diag <= cfg.max_particle_size):
            continue

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        diagonals.append(diag)
        xs.append(cx)

        cv2.drawContours(res, [box], 0, cfg.bbox_color, cfg.bbox_thickness)
        cv2.putText(res, f"{diag:.1f}", (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, cfg.bbox_color, cfg.font_thickness)

    return res, diagonals, xs


def segment_color(img: np.ndarray, cfg: ColorConfig) -> dict:
    steps = {"original": img.copy()}

    median = cv2.medianBlur(img, cfg.median_blur_size) if cfg.median_blur_size > 1 else img.copy()
    steps["median_blur"] = median.copy()

    gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    steps["gray"] = gray.copy()

    if cfg.adaptive_thresh:
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, cfg.adaptive_block_size, cfg.adaptive_C
        )
    else:
        flag = cv2.THRESH_BINARY
        if cfg.use_otsu and cfg.binary_thresh == 0:
            flag |= cv2.THRESH_OTSU
        _, binary = cv2.threshold(gray, cfg.binary_thresh, 255, flag)
    steps["binary_mask"] = binary.copy()

    # === Дополнительная логика с distance transform ===
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, cfg.distance_transform_mask)
    steps["distance_transform"] = dist_transform.copy()

    dt_norm = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    foreground = (dt_norm > cfg.foreground_threshold_ratio).astype(np.uint8) * 255
    steps["foreground_mask"] = foreground.copy()

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(foreground, kernel, iterations=cfg.dilate_iterations)
    steps["dilated_foreground"] = dilated.copy()

    return steps


def process_mask(mask: np.ndarray, cfg: ColorConfig) -> np.ndarray:
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS
    }
    kernel_shape = shape_map.get(cfg.morph_kernel_shape, cv2.MORPH_RECT)
    ker = cv2.getStructuringElement(kernel_shape, (cfg.morph_kernel_size, cfg.morph_kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker, iterations=cfg.morph_iterations)
    eroded = cv2.erode(opened, np.ones((2, 2), np.uint8), iterations=1)
    return eroded


def analyze_frame(color_img: np.ndarray,
                 c_cfg: ColorConfig,
                 v_cfg: VisualizationConfig):
    if color_img.dtype != np.uint8 and color_img.max() <= 1.0:
        color_img = (color_img * 255).astype(np.uint8)

    # === COLOR PIPELINE ===
    color_steps = segment_color(color_img, c_cfg)
    morphed_mask = process_mask(color_steps["dilated_foreground"], c_cfg)
    color_steps["morphed_mask"] = morphed_mask.copy()
    cres, cdi, cxs = analyze_mask(color_img, morphed_mask, c_cfg)
    color_steps["result"] = cres.copy()

    # === PLOTS ===
    if v_cfg.show_plots and len(color_steps) == 9:
        # Первое окно: 8 графиков (2 столбца × 4 строки)
        plt.figure(1, figsize=(v_cfg.plot_figsize[0] * 2, 
                             v_cfg.plot_figsize[1] * 4))
        
        for i in range(8):
            plt.subplot(4, 2, i+1)
            title, img = list(color_steps.items())[i]
            plt.title(f"COLOR: {title}", fontsize=10)
            cmap = 'gray' if img.ndim == 2 else None
            plt.imshow(img, cmap=cmap)
            plt.axis('off')
        
        plt.tight_layout()
        
        # Второе окно: 9-й график
        plt.figure(2, figsize=(v_cfg.plot_figsize[0] * 2.5,
                             v_cfg.plot_figsize[1] * 1.5))
        
        last_title, last_img = list(color_steps.items())[8]
        plt.title(f"COLOR: {last_title}", fontsize=12)
        cmap = 'gray' if last_img.ndim == 2 else None
        plt.imshow(last_img, cmap=cmap)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    return cres, cdi, cxs