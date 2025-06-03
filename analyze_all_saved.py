import os
import cv2
import tifffile
import numpy as np
import matplotlib.pyplot as plt

from processing.analyzer import analyze_depth_frame, analyze_color_frame
from processing.config import Config
from processing.visual_config import VisualizationConfig
from config import OUTPUT_DIR, FRAME_NUMBER, SVO_PATH
from zed.zed_loader import get_intrinsics_from_svo

def load_saved_depth(frame_number: int) -> np.ndarray:
    """Загружает сохранённую карту глубины (TIFF, float32)."""
    path = os.path.join(OUTPUT_DIR, f"frame_{frame_number}_depth.tiff")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Глубина не найдена: {path}")
    return tifffile.imread(path)

def load_saved_color(frame_number: int) -> np.ndarray:
    """Загружает сохранённое цветное изображение."""
    path = os.path.join(OUTPUT_DIR, f"frame_{frame_number}_color.png")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Цветное изображение не найдено: {path}")
    return img

def plot_diameters(diams: list, xs: list, frame_number: int, title: str):
    if not diams:
        print(f"[{title}] Нет данных для графика.")
        return
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8))
    ax1.hist(diams, bins=30, color='skyblue', edgecolor='black')
    ax1.set_title(f"{title} — Диаметры (кадр {frame_number})")
    ax1.set_xlabel("Диаметр, мм")
    ax1.set_ylabel("Частота")
    ax1.grid(True)

    ax2.scatter(xs, diams, c='green', edgecolors='black', alpha=0.7)
    ax2.set_title(f"{title} — Диаметры по X (кадр {frame_number})")
    ax2.set_xlabel("X-позиция (пиксели)")
    ax2.set_ylabel("Диаметр, мм")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    # --- Загружаем входные данные ---
    depth_np = load_saved_depth(FRAME_NUMBER)
    color_np = load_saved_color(FRAME_NUMBER)

    # --- Конфигурации для глубинного анализа ---
    depth_configs = [
        ("Depth-Big", Config(
            median_blur_size=3,
            adaptive_thresh=True,
            adaptive_block_size=101,
            adaptive_C=-15,
            morph_kernel_size=13,
            morph_iterations=1,
            dilate_iterations=2,
            min_particle_size=20,
            max_particle_size=200,
            distance_transform_mask=0,
            foreground_threshold_ratio=0.50,
            invert=True,
            equalize_hist=False,
            use_clahe=True,
        ))
    ]

    # --- Конфигурации для цвета ---
    color_configs = [
        ("Small", Config(
            # === Пороговая обработка ===
            adaptive_thresh=True,
            adaptive_block_size=13,
            adaptive_C=-7,

            # === Гладкость и шумоподавление ===
            median_blur_size=5,

            # === Морфология ===
            morph_kernel_shape='rect',
            morph_kernel_size=3,
            morph_iterations=1,
            dilate_iterations=0,

            # === Детекция контуров ===
            min_particle_size=40,
            max_particle_size=200,

            # === Аннотации (Bounding Boxes) ===
            bbox_color=(0, 255, 0),
            bbox_thickness=1,
            font_scale=0.5,
            font_thickness=0,

            # === Постобработка и выделение переднего плана ===
            distance_transform_mask=5,
            foreground_threshold_ratio=0.1,

            use_clahe=True,
            equalize_hist=True,
        )),
    ]

    vis_cfg = VisualizationConfig(show_plots=True)
    fx, fy, *_ = get_intrinsics_from_svo(SVO_PATH)

    for label, cfg in depth_configs:
            print(f"\n=== Анализ Depth: {label} ===")
            seg, diams_px, diams_mm, xs= analyze_depth_frame(
                depth_np, cfg, vis_cfg,
                label=label,
                output_dir=OUTPUT_DIR,
                save_plots=True,
                fx=fx, fy=fy
            )

            out_path = os.path.join(OUTPUT_DIR, f"depth_seg_{label.lower()}_{FRAME_NUMBER}.png")
            cv2.imwrite(out_path, seg)
            print(f"[{label}] Сохранено сегментированное (depth): {out_path}")
            if diams_mm:
                print(f"[{label}] Найдено объектов: {len(diams_mm)}  Диаметры: {min(diams_mm):.1f}–{max(diams_mm):.1f} мм")
                plot_diameters(diams_mm, xs, FRAME_NUMBER, label)
            else:
                print(f"[{label}] Объекты не найдены.")
    # === Анализ цвета ===
    for label, cfg in color_configs:
        print(f"\n=== Анализ Color: {label} ===")
        seg, diams_px, diams_mm, xs = analyze_color_frame(
            color_np, cfg, vis_cfg,
            label=label,
            output_dir=OUTPUT_DIR,
            save_plots=True,
            depth_img=depth_np,
            fx=fx,
            fy=fy,
        )
        out_path = os.path.join(OUTPUT_DIR, f"color_seg_{label.lower()}_{FRAME_NUMBER}.png")
        cv2.imwrite(out_path, seg)
        print(f"[{label}] Сохранено сегментированное (color): {out_path}")
        if diams_mm:
            print(f"[{label}] Найдено объектов: {len(diams_mm)}  Диаметры: {min(diams_mm):.1f}–{max(diams_mm):.1f} мм")
            plot_diameters(diams_mm, xs, FRAME_NUMBER, label)

if __name__ == "__main__":
    main()
