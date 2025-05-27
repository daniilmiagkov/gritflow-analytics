from zed.zed_loader import init_camera, grab_frame
from processing.image_io import save_point_cloud
from processing.analyzer import analyze_depth_frame
from visualization.viewer import visualize_ply
from config import (
    SVO_PATH, OUTPUT_DIR, FRAME_NUMBER,
    CROP, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT,
    DEPTH_MIN, DEPTH_MAX
)
from processing.color_config import ColorConfig  # Можно заменить на DepthConfig при переименовании
from processing.visual_config import VisualizationConfig

import os
import cv2
import tifffile
import numpy as np

def apply_crop(image, x, y, width, height):
    h, w = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x1 + width)
    y2 = min(h, y1 + height)
    return image[y1:y2, x1:x2], (x1, y1, x2-x1, y2-y1)

def main():
    zed = init_camera(SVO_PATH)
    image, depth, depth_shading, point_cloud = grab_frame(zed, FRAME_NUMBER)

    if image is None or depth is None:
        print(f"Не удалось захватить кадр {FRAME_NUMBER}")
        return

    # Извлекаем numpy-данные
    color_np = image.get_data()
    depth_np = depth.get_data().astype(np.float32)  # предполагаем глубину в метрах
    depth_shading_np = depth_shading.get_data()

    # Применяем обрезку
    crop_params = None
    if CROP:
        color_np, crop_params = apply_crop(color_np, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)
        depth_np, _ = apply_crop(depth_np, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)
        depth_shading_np, _ = apply_crop(depth_shading_np, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)
        print(f"Область обрезки: x={crop_params[0]}, y={crop_params[1]}, width={crop_params[2]}, height={crop_params[3]}")

    # Сохранение изображений
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{FRAME_NUMBER}_color.png"), color_np)
    tifffile.imwrite(os.path.join(OUTPUT_DIR, f"frame_{FRAME_NUMBER}_depth.tiff"), depth_np)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{FRAME_NUMBER}_depth_shading.png"), depth_shading_np)

    # Сохранение облака точек
    ply_path = os.path.join(OUTPUT_DIR, f"point_cloud_{FRAME_NUMBER}.ply")
    save_point_cloud(point_cloud, ply_path)

    print(f"\nСохранено:")
    print(f"- Цвет: frame_{FRAME_NUMBER}_color.png")
    print(f"- Глубина: frame_{FRAME_NUMBER}_depth.tiff")
    print(f"- Depth Shading: frame_{FRAME_NUMBER}_depth_shading.png")
    print(f"- Облако точек: {ply_path}")

    # --- Анализ глубины ---
    print("\nЗапуск анализа карты глубины...")

    depth_cfg = ColorConfig(  # пока используем его как DepthConfig
        median_blur_size=3,
        adaptive_thresh=False,
        adaptive_block_size=11,
        adaptive_C=2,
        binary_thresh=40,   # экспериментируй, можно 10–100 в зависимости от нормализации
        use_otsu=True
    )

    vis_cfg = VisualizationConfig(
        show_plots=True,
        plot_figsize=(8, 4)
    )

    segmented, diameters, positions = analyze_depth_frame(
        depth_np, depth_cfg, vis_cfg,
        label=f"{FRAME_NUMBER}", output_dir=OUTPUT_DIR
    )

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"segmented_{FRAME_NUMBER}_depth.png"), segmented)

    print(f"\nРезультаты анализа по глубине:")
    print(f"Объектов: {len(diameters)}")
    if diameters:
        print(f"Диаметры: {round(min(diameters), 2)} – {round(max(diameters), 2)} мм")
    else:
        print("Объекты не найдены.")

    # Визуализация
    if input("Показать 3D визуализацию? (y/n): ").lower() == "y":
        visualize_ply(ply_path)

    zed.close()

if __name__ == "__main__":
    main()
