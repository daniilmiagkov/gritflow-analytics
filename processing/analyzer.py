from dataclasses import dataclass
from typing import Tuple, List, Union

import cv2
import numpy as np
import matplotlib.pyplot as plt

from processing.color_config import ColorConfig
from processing.depth_config import DepthConfig
from processing.visual_config import VisualizationConfig

def correct_slope(depth_map: np.ndarray, cfg: DepthConfig) -> np.ndarray:
    h, w = depth_map.shape
    scale = cfg.slope_scale_large if depth_map.nbytes > cfg.slope_bytes_threshold else cfg.slope_scale_small
    small = cv2.resize(depth_map, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(small > cfg.depth_min)
    if len(xs) < cfg.svd_point_threshold:
        return depth_map
    zs = small[ys, xs]
    xs_f, ys_f = xs / scale, ys / scale
    pts = np.stack([xs_f, ys_f, zs], axis=1)
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = Vt[2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Zp = ((normal[0] * (X - centroid[0]) + normal[1] * (Y - centroid[1])) / normal[2]) + centroid[2]
    corr = np.where(depth_map > cfg.depth_min, depth_map - Zp, depth_map)
    return np.clip(corr, cfg.depth_min, cfg.depth_max).astype(np.float32)


def analyze_mask(img: np.ndarray, mask: np.ndarray,
                 cfg: Union[ColorConfig, DepthConfig]) -> Tuple[np.ndarray, List[float], List[float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = img.copy()
    diagonals, xs = [], []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg.min_contour_area:
            continue

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
    return steps


def segment_depth(dm: np.ndarray, cfg: DepthConfig) -> dict:
    steps = {"original_depth": dm.copy()}

    median = cv2.medianBlur(dm, cfg.median_blur_size) if cfg.median_blur_size > 1 else dm.copy()
    steps["median_blur"] = median.copy()

    normalized = cv2.normalize(median, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    steps["normalized"] = normalized.copy()

    if cfg.adaptive_thresh:
        binary = cv2.adaptiveThreshold(
            normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, cfg.adaptive_block_size, cfg.adaptive_C
        )
    else:
        flag = cv2.THRESH_BINARY
        if cfg.use_otsu and cfg.binary_thresh == 0:
            flag |= cv2.THRESH_OTSU
        _, binary = cv2.threshold(normalized, cfg.binary_thresh, 255, flag)
    steps["binary_mask"] = binary.copy()
    return steps


def process_mask(mask: np.ndarray, cfg: Union[ColorConfig, DepthConfig]) -> np.ndarray:
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


def analyze_frame(color_img: np.ndarray, depth_map: np.ndarray,
                  d_cfg: DepthConfig, c_cfg: ColorConfig,
                  v_cfg: VisualizationConfig,
                  use_depth: bool = True):
    if color_img.dtype != np.uint8 and color_img.max() <= 1.0:
        color_img = (color_img * 255).astype(np.uint8)
    if depth_map.dtype != np.float32:
        depth_map = depth_map.astype(np.float32)

    if d_cfg.slope_correction and use_depth:
        depth_map = correct_slope(depth_map, d_cfg)

    # === COLOR PIPELINE ===
    color_steps = segment_color(color_img, c_cfg)
    morphed_mask = process_mask(color_steps["binary_mask"], c_cfg)
    color_steps["morphed_mask"] = morphed_mask.copy()
    cres, cdi, cxs = analyze_mask(color_img, morphed_mask, c_cfg)
    color_steps["result"] = cres.copy()

    # === DEPTH PIPELINE ===
    depth_steps = None
    if use_depth:
        depth_steps = segment_depth(depth_map, d_cfg)
        dp = process_mask(depth_steps["binary_mask"], d_cfg)
        depth_steps["morphed_mask"] = dp.copy()
        dres, ddi, dxs = analyze_mask(color_img, dp, d_cfg)
        depth_steps["result"] = dres.copy()
    else:
        dres, ddi, dxs = None, None, None

    # === PLOTS ===
    if v_cfg.show_plots:
        plt.figure(figsize=(v_cfg.plot_figsize[0], v_cfg.plot_figsize[1] * len(color_steps)))
        for i, (title, img) in enumerate(color_steps.items()):
            plt.subplot(len(color_steps), 1, i + 1)
            plt.title(f"COLOR: {title}")
            cmap = 'gray' if img.ndim == 2 else None
            plt.imshow(img, cmap=cmap)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

        # Раскомментируй при необходимости визуализации глубины
        # if use_depth and depth_steps:
        #     plt.figure(figsize=(v_cfg.plot_figsize[0], v_cfg.plot_figsize[1] * len(depth_steps)))
        #     for i, (title, img) in enumerate(depth_steps.items()):
        #         plt.subplot(len(depth_steps), 1, i + 1)
        #         plt.title(f"DEPTH: {title}")
        #         cmap = 'gray' if img.ndim == 2 else None
        #         plt.imshow(img, cmap=cmap)
        #         plt.axis('off')
        #     plt.tight_layout()
        #     plt.show()

    return (cres, cdi, cxs), (dres, ddi, dxs)
