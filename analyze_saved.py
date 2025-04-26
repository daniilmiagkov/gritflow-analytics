import os
import cv2
import tifffile
import numpy as np
import matplotlib.pyplot as plt

from processing.analyzer import analyze_frame
from processing.depth_config import DepthConfig
from processing.color_config import ColorConfig
from processing.visual_config import VisualizationConfig
from config import OUTPUT_DIR, FRAME_NUMBER  # если эти константы у тебя в общем config

def load_saved_data(frame_number: int):
    """Загружает сохранённые изображения и карты глубины."""
    color_path = os.path.join(OUTPUT_DIR, f"frame_{frame_number}_color.png")
    depth_path = os.path.join(OUTPUT_DIR, f"frame_{frame_number}_depth.tiff")

    color = cv2.imread(color_path, cv2.IMREAD_COLOR)
    depth = tifffile.imread(depth_path)

    if color is None:
        raise FileNotFoundError(f"Не найден файл изображения: {color_path}")
    if depth is None:
        raise FileNotFoundError(f"Не найден файл глубины: {depth_path}")

    return color, depth

def plot_diameters(diameters: list, x_positions: list, frame_number: int, title_prefix: str):
    """Строит гистограмму и диаграмму размеров частиц по оси X изображения."""
    if not diameters or not x_positions:
        print(f"[{title_prefix}] Нет данных для построения графиков.")
        return

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    axs[0].hist(diameters, bins=30, color='skyblue', edgecolor='black')
    axs[0].set_title(f"{title_prefix} — Распределение диаметров (кадр {frame_number})")
    axs[0].set_xlabel("Диаметр, мм")
    axs[0].set_ylabel("Количество")
    axs[0].grid(True)

    axs[1].scatter(x_positions, diameters,
                   color='orange', edgecolors='black', alpha=0.7)
    axs[1].set_title(f"{title_prefix} — Размеры по X (кадр {frame_number})")
    axs[1].set_xlabel("X-позиция (пиксели)")
    axs[1].set_ylabel("Диаметр, мм")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def main():
    # Создаём конфиги
    depth_cfg = DepthConfig()
    color_cfg = ColorConfig()
    vis_cfg   = VisualizationConfig()

    # Загружаем данные
    color_np, depth_np = load_saved_data(FRAME_NUMBER)

    # Пайплайн анализа
    (c_res, c_diams, c_xs), (d_res, d_diams, d_xs) = analyze_frame(
        color_np, depth_np,
        depth_cfg, color_cfg, vis_cfg
    )

    # Сохраняем результаты
    seg_color_path = os.path.join(OUTPUT_DIR, f"segmented_color_{FRAME_NUMBER}.png")
    cv2.imwrite(seg_color_path, c_res)
    print(f"[Color] Сегментированное изображение сохранено: {seg_color_path}")

    seg_depth_path = os.path.join(OUTPUT_DIR, f"segmented_depth_{FRAME_NUMBER}.png")
    cv2.imwrite(seg_depth_path, d_res)
    print(f"[Depth] Сегментированное изображение сохранено: {seg_depth_path}")

    # Вывод и графики по цвету
    if c_diams:
        print(f"[Color] Обнаружено частиц: {len(c_diams)}")
        print(f"[Color] Диапазон диаметров: {min(c_diams):.2f}–{max(c_diams):.2f} мм")
        plot_diameters(c_diams, c_xs, FRAME_NUMBER, title_prefix="Color")
    else:
        print("[Color] Частицы не обнаружены.")

    # Вывод и графики по глубине
    if d_diams:
        print(f"[Depth] Обнаружено частиц: {len(d_diams)}")
        print(f"[Depth] Диапазон диаметров: {min(d_diams):.2f}–{max(d_diams):.2f} мм")
        # plot_diameters(d_diams, d_xs, FRAME_NUMBER, title_prefix="Depth")
    else:
        print("[Depth] Частицы не обнаружены.")

if __name__ == "__main__":
    main()
