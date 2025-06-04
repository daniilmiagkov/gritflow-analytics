import os
from typing import Tuple, List, Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile

from utils.convert import pixel_to_mm
from processing.config import Config
from processing.visual_config import VisualizationConfig

# --------------------------------------------------
# Вспомогательная функция: строит RGBA-изображение (H×W×4),
# где на прозрачном фоне отрисованы контуры из маски
# --------------------------------------------------
def contours_to_rgba(mask: np.ndarray, color_bgr: Tuple[int,int,int], thickness: int = 2) -> np.ndarray:
    """
    Преобразует бинарную маску (dtype=uint8 или bool, shape=(H,W)) в RGBA-изображение (H,W,4),
    где контуры объектов из mask отрисованы цветом color_bgr (B,G,R), альфа=255 (непрозрачно),
    остальной фон — полностью прозрачный (альфа=0).
    """
    # 1) Гарантируем uint8: фон=0, объекты=255
    if mask.dtype != np.uint8:
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
    else:
        mask_uint8 = mask.copy()

    # 2) Ищем внешние контуры
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask_uint8.shape

    # 3) Создаём отдельный трёхканальный BGR-слой для рисования контуров
    bgr_layer = np.zeros((h, w, 3), dtype=np.uint8)
    # Рисуем контуры на этом слое
    cv2.drawContours(bgr_layer, contours, -1, color_bgr, thickness=thickness)

    # 4) Формируем альфа-канал: там, где на bgr_layer != (0,0,0), ставим alpha=255
    alpha = np.any(bgr_layer != 0, axis=2).astype(np.uint8) * 255  # (H,W)

    # 5) Сливаем BGR и альфу в RGBA
    rgba = cv2.merge([bgr_layer, alpha])

    return rgba


# --------------------------------------------------
# Сегментация одноканального изображения
# --------------------------------------------------
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
        flag = cv2.THRESH_BINARY_INV if isinstance(cfg, Config) else cv2.THRESH_BINARY
        if getattr(cfg, "use_otsu", False) and cfg.binary_thresh == 0:
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


# --------------------------------------------------
# Анализ контуров + возвращение RGBA-контуров
# --------------------------------------------------
def analyze_gray(
    img_for_draw: np.ndarray,
    mask: np.ndarray,
    cfg: Any,
    depth_img: np.ndarray = None,
    fx: float = 580.0,
    fy: float = 580.0
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], List[float], List[float]]:
    """
    :param img_for_draw: BGR-изображение (H×W×3), на котором будем рисовать прямоугольники
    :param mask: бинарная маска (H×W), где 255/True — «объект»
    :param cfg: конфиг с полями min_contour_area, min_particle_size, max_particle_size,
               bbox_color (BGR-цвет для линий), bbox_thickness
    :param depth_img: (H×W) float32 — глубина для расчёта в миллиметрах
    :return:
        out_bgr:             BGR (H×W×3) с отрисованными контурами,
        rgba_contours:       RGBA (H×W×4) с контурами на прозрачном фоне,
        widths_px:           List[float] — ширины (в пикселях) ограничивающих прямоугольников,
        heights_px:          List[float] — высоты (в пикселях) ограничивающих прямоугольников,
        diagonals_mm:        List[float] — диагонали (в миллиметрах) каждого прямоугольника,
        xs:                  List[float] — X-координаты центров каждого прямоугольника.
    """
    # 1) Находим внешние контуры
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out_bgr = img_for_draw.copy()
    widths_px:    List[float] = []
    heights_px:   List[float] = []
    diagonals_px: List[float] = []
    diagonals_mm: List[float] = []
    xs:           List[float] = []

    h, w = mask.shape
    mask_boxes = np.zeros((h, w), dtype=np.uint8)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < getattr(cfg, "min_contour_area", 0):
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w_box, h_box), _ = rect

        # Отфильтровываем по диапазону: здесь проверяем диагональ (которую пока не считали)
        diag_px = float(np.hypot(w_box, h_box))
        if not (cfg.min_particle_size <= diag_px <= cfg.max_particle_size):
            continue

        box = cv2.boxPoints(rect).astype(np.int32)

        # Сохраняем ширину/высоту/диагональ
        widths_px.append(w_box)
        heights_px.append(h_box)
        diagonals_px.append(diag_px)
        xs.append(cx)

        # Переводим диагональ в миллиметры (по формуле)
        if depth_img is not None:
            cx_i, cy_i = int(round(cx)), int(round(cy))
            if 0 <= cx_i < depth_img.shape[1] and 0 <= cy_i < depth_img.shape[0]:
                depth_val = float(depth_img[cy_i, cx_i])
                diag_mm = pixel_to_mm(diag_px, depth_val, fx, fy)
            else:
                diag_mm = 0.0
        else:
            diag_mm = 0.0
        diagonals_mm.append(diag_mm)

        # 2) Рисуем бокс на цветном out_bgr
        cv2.drawContours(out_bgr, [box], 0, cfg.bbox_color, cfg.bbox_thickness)

        # 3) Заполняем этот прямоугольник в mask_boxes
        cv2.drawContours(mask_boxes, [box], 0, 255, thickness=-1)

    # 4) Построим RGBA-слой контуров из mask_boxes
    rgba_contours = contours_to_rgba(mask_boxes, color_bgr=cfg.bbox_color, thickness=cfg.bbox_thickness)

    return out_bgr, rgba_contours, widths_px, heights_px, diagonals_mm, xs


# --------------------------------------------------
# Анализ цветного кадра
# --------------------------------------------------
def analyze_color_frame(
    color_img: np.ndarray,
    cfg: Config,
    v_cfg: VisualizationConfig,
    label: str = "",
    output_dir: str = "",
    save_plots: bool = False,
    depth_img: np.ndarray = None,
    fx: float = 580.0,
    fy: float = 580.0
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], List[float], List[float]]:
    """
    :return: out_bgr, rgba_contours, widths_px, heights_px, diagonals_mm, xs
    """
    img = (color_img * 255).astype(np.uint8) if color_img.dtype != np.uint8 else color_img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps = segment_gray(gray, cfg)
    steps["binary_mask_color"] = steps["binary_mask"]

    final_mask = steps["foreground_mask"]
    out_bgr, rgba_contours, widths_px, heights_px, diagonals_mm, xs = analyze_gray(
        img, final_mask, cfg, depth_img=depth_img, fx=fx, fy=fy
    )
    steps["result"] = out_bgr

    if save_plots:
        _save_plots(steps, "color", label, output_dir, v_cfg.plot_figsize)
    return out_bgr, rgba_contours, widths_px, heights_px, diagonals_mm, xs


# --------------------------------------------------
# Анализ depth-кадра
# --------------------------------------------------
def analyze_depth_frame(
    depth_img: np.ndarray,
    cfg: Config,
    v_cfg: VisualizationConfig,
    label: str = "",
    output_dir: str = "",
    save_plots: bool = False,
    fx: float = 580.0,
    fy: float = 580.0
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float], List[float], List[float]]:
    """
    :return: out_bgr, rgba_contours, widths_px, heights_px, diagonals_mm, xs
    """
    norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    steps = segment_gray(norm, cfg)

    vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    out_bgr, rgba_contours, widths_px, heights_px, diagonals_mm, xs = analyze_gray(
        vis, steps["foreground_mask"], cfg, depth_img=depth_img, fx=fx, fy=fy
    )
    steps["result"] = out_bgr

    if save_plots:
        _save_plots(steps, "depth", label, output_dir, v_cfg.plot_figsize)
    return out_bgr, rgba_contours, widths_px, heights_px, diagonals_mm, xs


# --------------------------------------------------
# Сохранение промежуточных результатов (без изменений)
# --------------------------------------------------
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
    plt.imshow(last_img, cmap='gray' if last_img.ndim == 2 else None)
    plt.axis('off')
    path = os.path.join(output_dir, f"{mode}_{label}_result.png")
    fig.savefig(path); plt.close(fig)

    items = list(steps.items())[:-1]
    n = len(items)
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    for i, (t, im) in enumerate(items):
        ax = fig.add_subplot(n, 1, i + 1)
        ax.set_title(f"{mode.upper()}[{label}] {t}")
        ax.imshow(im, cmap='gray' if im.ndim == 2 else None)
        ax.axis('off')
    path = os.path.join(output_dir, f"{mode}_{label}_steps.png")
    fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    from config import OUTPUT_DIR, FRAME_NUMBER

    color_cfg = Config()
    depth_cfg = Config()
    vis_cfg   = VisualizationConfig(show_plots=True)

    img = cv2.imread(os.path.join(OUTPUT_DIR, f"frame_{FRAME_NUMBER}_color.png"))
    c_res, c_rgba, c_w_px, c_h_px, c_d_mm, c_xs = analyze_color_frame(
        img, color_cfg, vis_cfg,
        label="C1", output_dir=OUTPUT_DIR, save_plots=True
    )

    depth = tifffile.imread(os.path.join(OUTPUT_DIR, f"frame_{FRAME_NUMBER}_depth.tiff"))
    d_res, d_rgba, d_w_px, d_h_px, d_d_mm, d_xs = analyze_depth_frame(
        depth, depth_cfg, vis_cfg,
        label="D1", output_dir=OUTPUT_DIR, save_plots=True
    )
