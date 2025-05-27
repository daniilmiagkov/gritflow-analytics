from dataclasses import dataclass
import os
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
        # cv2.putText(res, f"{diag:.1f}", (int(cx), int(cy)),
        #             cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, cfg.bbox_color, cfg.font_thickness)

    return res, diagonals, xs


def segment_color(img: np.ndarray, cfg: ColorConfig) -> dict:
    steps = {"original": img.copy()}

    # === Grayscale и контраст ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps["gray"] = gray.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)
    steps["clahe_gray"] = clahe_gray.copy()

    # === Шумоподавление ===
    median = cv2.medianBlur(clahe_gray, cfg.median_blur_size) if cfg.median_blur_size > 1 else clahe_gray.copy()
    steps["median_blur"] = median.copy()

    # === Пороговая бинаризация ===
    if cfg.adaptive_thresh:
        binary = cv2.adaptiveThreshold(
            median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, cfg.adaptive_block_size, cfg.adaptive_C
        )
    else:
        flag = cv2.THRESH_BINARY
        if cfg.use_otsu and cfg.binary_thresh == 0:
            flag |= cv2.THRESH_OTSU
        _, binary = cv2.threshold(median, cfg.binary_thresh, 255, flag)
    steps["binary_mask"] = binary.copy()

    # === Морфологическая фильтрация до distance transform ===
    shape_map = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross": cv2.MORPH_CROSS
    }
    kernel_shape = shape_map.get(cfg.morph_kernel_shape, cv2.MORPH_RECT)
    ker = cv2.getStructuringElement(kernel_shape, (cfg.morph_kernel_size, cfg.morph_kernel_size))
    
    # Открытие — удаляет мелкие шумы
    morph_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ker, iterations=cfg.morph_iterations)
    steps["morph_opened"] = morph_opened.copy()

    # Закрытие — соединяет разорванные объекты
    morph_closed = cv2.morphologyEx(morph_opened, cv2.MORPH_CLOSE, ker, iterations=cfg.dilate_iterations)
    steps["morph_closed"] = morph_closed.copy()

    # === Distance Transform по очищенной маске ===
    dist_transform = cv2.distanceTransform(morph_closed, cv2.DIST_L2, cfg.distance_transform_mask)
    steps["distance_transform"] = dist_transform.copy()

    # === Отделение переднего плана ===
    dt_norm = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
    foreground = (dt_norm > cfg.foreground_threshold_ratio).astype(np.uint8) * 255
    steps["foreground_mask"] = foreground.copy()

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
                  v_cfg: VisualizationConfig,
                  label: str = "",
                  output_dir: str = "",
                  save_plots: bool = True):
    ...
    if color_img.dtype != np.uint8 and color_img.max() <= 1.0:
        color_img = (color_img * 255).astype(np.uint8)

    color_steps = segment_color(color_img, c_cfg)
    morphed_mask = process_mask(color_steps["dilated_foreground"], c_cfg)
    color_steps["morphed_mask"] = morphed_mask.copy()
    cres, cdi, cxs = analyze_mask(color_img, morphed_mask, c_cfg)
    color_steps["result"] = cres.copy()

    if v_cfg.show_plots and save_plots:
        # === Первая фигура — итоговый результат отдельно ===
        fig1 = plt.figure(figsize=(v_cfg.plot_figsize[0] * 2.5,
                                    v_cfg.plot_figsize[1] * 1.5))
        last_title, last_img = list(color_steps.items())[-1]
        plt.title(f"COLOR: {last_title}", fontsize=12)
        cmap = 'gray' if last_img.ndim == 2 else None
        plt.imshow(last_img, cmap=cmap)
        plt.axis('off')
        plt.tight_layout()

        if output_dir and label:
            fig1_path = os.path.join(output_dir, f"color_steps_{label.lower()}_1_result.png")
            fig1.savefig(fig1_path)
            print(f"[{label}] Сохранён график результата: {fig1_path}")
        plt.close(fig1)

        # === Вторая фигура — все промежуточные шаги ===
        intermediate_items = list(color_steps.items())[:-1]
        num_steps = len(intermediate_items)

        fig2 = plt.figure(figsize=(v_cfg.plot_figsize[0] * 2,
                                    v_cfg.plot_figsize[1] * num_steps))

        for i, (title, img) in enumerate(intermediate_items):
            plt.subplot(num_steps, 1, i + 1)
            plt.title(f"COLOR: {title}", fontsize=14)
            cmap = 'gray' if img.ndim == 2 else None
            plt.imshow(img, cmap=cmap)
            plt.axis('off')
        plt.tight_layout()

        if output_dir and label:
            fig2_path = os.path.join(output_dir, f"color_steps_{label.lower()}_2_steps.png")
            fig2.savefig(fig2_path)
            print(f"[{label}] Сохранён график шагов: {fig2_path}")
        plt.close(fig2)
