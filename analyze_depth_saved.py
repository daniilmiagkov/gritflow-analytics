import os
import cv2
import tifffile
import numpy as np
import matplotlib.pyplot as plt

# Импортируем наш единый пайплайн
from processing.analyzer import analyze_depth_frame, analyze_color_frame
# from processing.depth_config import DepthConfig
from processing.color_config import ColorConfig
from processing.visual_config import VisualizationConfig
from config import OUTPUT_DIR, FRAME_NUMBER

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

    # --- Конфигурации для малого и большого глубинного анализа ---
    depth_configs = [
        ("Depth-Small", ColorConfig(
            median_blur_size=1,
            adaptive_thresh=True,
            adaptive_block_size=41,
            adaptive_C=-10,
            use_otsu=False,
            morph_kernel_size=3,
            morph_iterations=1,
            dilate_iterations=0,
            min_contour_area=10,
            min_particle_size=5,
            max_particle_size=60,
            distance_transform_mask=0,
            foreground_threshold_ratio=0.3
        )),
        ("Depth-Big", ColorConfig(
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

    # --- Конфигурации для цвета (пример) ---
    color_configs = [
        # ("Color-Default", ColorConfig()),
        # ту же логику можно повторить для малого и большого
    ]

    vis_cfg = VisualizationConfig(show_plots=True)

    # === Анализ глубины ===
    for label, cfg in depth_configs:
        print(f"\n=== Анализ Depth: {label} ===")
        seg, diams, xs = analyze_depth_frame(
            depth_np, cfg, vis_cfg,
            label=label, output_dir=OUTPUT_DIR, save_plots=True
        )
        out_path = os.path.join(OUTPUT_DIR, f"depth_seg_{label.lower()}_{FRAME_NUMBER}.png")
        cv2.imwrite(out_path, seg)
        print(f"[{label}] Сохранено сегментированное (depth): {out_path}")
        # if diams:
        #     print(f"[{label}] Найдено объектов: {len(diams)}  Диаметры: {min(diams):.1f}–{max(diams):.1f}")
        #     plot_diameters(diams, xs, FRAME_NUMBER, label)

    # === Анализ цвета ===
    for label, cfg in color_configs:
        print(f"\n=== Анализ Color: {label} ===")
        seg, diams, xs = analyze_color_frame(
            color_np, cfg, vis_cfg,
            label=label, output_dir=OUTPUT_DIR, save_plots=True
        )
        out_path = os.path.join(OUTPUT_DIR, f"color_seg_{label.lower()}_{FRAME_NUMBER}.png")
        cv2.imwrite(out_path, seg)
        print(f"[{label}] Сохранено сегментированное (color): {out_path}")
        # if diams:
        #     print(f"[{label}] Найдено объектов: {len(diams)}  Диаметры: {min(diams):.1f}–{max(diams):.1f}")
        #     plot_diameters(diams, xs, FRAME_NUMBER, label)

if __name__ == "__main__":
    main()
