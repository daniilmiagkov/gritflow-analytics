import os
import cv2
import tifffile
import numpy as np

from zed.zed_loader import get_intrinsics_from_svo, init_camera, grab_frame
from processing.image_io import save_point_cloud
from processing.analyzer import analyze_color_frame, analyze_depth_frame
from visualization.viewer import visualize_ply

from config import (
    SVO_PATH,
    OUTPUT_DIR,
    FRAME_NUMBER,  # стартовый кадр
    CROP, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT,
)
from processing.config import Config
from processing.visual_config import VisualizationConfig

# -----------------------------------------
# Задаём, сколько кадров подряд обрабатываем
NUM_FRAMES = 20
# -----------------------------------------

# --- Конфигурации для глубинного анализа (пример) ---
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

# --- Конфигурации для цветового анализа (пример) ---
color_configs = [
    ("Small", Config(
        adaptive_thresh=True,
        adaptive_block_size=13,
        adaptive_C=-7,
        median_blur_size=5,
        morph_kernel_shape='rect',
        morph_kernel_size=3,
        morph_iterations=1,
        dilate_iterations=0,
        min_particle_size=10,
        max_particle_size=20,
        bbox_color=(0, 255, 0),
        bbox_thickness=1,
        font_scale=0.5,
        font_thickness=0,
        distance_transform_mask=5,
        foreground_threshold_ratio=0.1,
        use_clahe=True,
        equalize_hist=True,
    )),
]

def apply_crop(image, x, y, width, height):
    """
    Обрезает входное изображение по прямоугольнику (x, y, width, height).
    Возвращает (обрезанное_изображение, (новые_x, новые_y, new_width, new_height)).
    """
    h, w = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x1 + width)
    y2 = min(h, y1 + height)
    return image[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

def main():
    # 1) Убеждаемся, что директория OUTPUT_DIR существует
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[INFO] Результаты будут сохранены в: {OUTPUT_DIR}")

    # 2) Инициализируем ZED-камера (SVO) один раз
    print(f"[INFO] Инициализация камеры по пути: {SVO_PATH}")
    zed = init_camera(SVO_PATH)

    # 3) Получаем параметры камеры (fx, fy и пр.) один раз
    fx, fy, *_ = get_intrinsics_from_svo(SVO_PATH)
    vis_cfg = VisualizationConfig(
        show_plots=True,
        plot_figsize=(8, 4),
    )

    # Папка для необработанных кадров
    raw_dir = os.path.join(OUTPUT_DIR, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Папки под каждый depth config
    depth_dirs = {}
    for label, _ in depth_configs:
        d = os.path.join(OUTPUT_DIR, f"depth_{label.lower()}")
        os.makedirs(d, exist_ok=True)
        depth_dirs[label] = d

    # Папки под каждый color config
    color_dirs = {}
    for label, _ in color_configs:
        d = os.path.join(OUTPUT_DIR, f"color_{label.lower()}")
        os.makedirs(d, exist_ok=True)
        color_dirs[label] = d

    for i in range(NUM_FRAMES):
        current_frame = FRAME_NUMBER + i
        print(f"\n========== Обработка кадра #{current_frame} ==========")

        image, depth, depth_shading, point_cloud = grab_frame(zed, current_frame)
        if image is None or depth is None:
            print(f"[ОШИБКА] Кадр {current_frame} не получен. Пропуск.")
            continue

        color_np = image.get_data()
        depth_np = depth.get_data().astype(np.float32)
        depth_shading_np = depth_shading.get_data()

        crop_params = None
        if CROP:
            color_np, crop_params = apply_crop(color_np, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)
            depth_np, _ = apply_crop(depth_np, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)
            depth_shading_np, _ = apply_crop(depth_shading_np, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)
            print(f"[INFO] Обрезка кадра: x={crop_params[0]}, y={crop_params[1]}, "
                  f"width={crop_params[2]}, height={crop_params[3]}")

        # --- Сохраняем исходники в raw ---
        # color_path = os.path.join(raw_dir, f"color_frame_{current_frame}.png")
        depth_path = os.path.join(raw_dir, f"depth_frame_{current_frame}.tiff")
        # # shading_path = os.path.join(raw_dir, f"depth_shading_frame_{current_frame}.png")
        # # ply_path = os.path.join(raw_dir, f"point_cloud_{current_frame}.ply")

        # cv2.imwrite(color_path, color_np)
        # tifffile.imwrite(depth_path, depth_np)
        cv2.imwrite(depth_path, depth_np)

        # # cv2.imwrite(shading_path, depth_shading_np)
        # # save_point_cloud(point_cloud, ply_path)

        # print(f"[INFO] Сохранены raw-данные кадра {current_frame} в {raw_dir}")

        # --- Анализ Depth ---
        for label, cfg in depth_configs:
            subdir = depth_dirs[label]
            print(f"\n--- Анализ Depth: {label} ---")

            # раньше: seg, diams_px, diams_mm, xs = analyze_depth_frame(...)
            # теперь: out_bgr_d, rgba_d, widths_d_px, heights_d_px, diagonals_d_mm, xs_d
            seg, rgba_d, widths_d_px, heights_d_px, diagonals_d_mm, xs_d = analyze_depth_frame(
                depth_np, cfg, vis_cfg,
                label=label,
                output_dir=subdir,
                save_plots=True,
                fx=fx, fy=fy
            )

            seg_filename = f"depth_seg_{label.lower()}_frame_{current_frame}.png"
            seg_path = os.path.join(subdir, seg_filename)
            cv2.imwrite(seg_path, seg)
            print(f"[{label}] Сегментация сохранена: {seg_path}")

            if diagonals_d_mm:
                print(f"[{label}] Объектов: {len(diagonals_d_mm)} | "
                      f"Диагонали (мм): {min(diagonals_d_mm):.1f}–{max(diagonals_d_mm):.1f}")
            else:
                print(f"[{label}] Нет объектов.")

        # --- Анализ Color ---
        for label, cfg in color_configs:
            subdir = color_dirs[label]
            print(f"\n--- Анализ Color: {label} ---")

            # раньше: seg, diams_px, diams_mm, xs = analyze_color_frame(...)
            # теперь: out_bgr_c, rgba_c, widths_c_px, heights_c_px, diagonals_c_mm, xs_c
            seg, rgba_c, widths_c_px, heights_c_px, diagonals_c_mm, xs_c = analyze_color_frame(
                color_np, cfg, vis_cfg,
                label=label,
                output_dir=subdir,
                save_plots=True,
                depth_img=depth_np,
                fx=fx, fy=fy
            )

            seg_filename = f"color_seg_{label.lower()}_frame_{current_frame}.png"
            seg_path = os.path.join(subdir, seg_filename)
            cv2.imwrite(seg_path, seg)
            print(f"[{label}] Сегментация сохранена: {seg_path}")

            if diagonals_c_mm:
                print(f"[{label}] Объектов: {len(diagonals_c_mm)} | "
                      f"Диагонали (мм): {min(diagonals_c_mm):.1f}–{max(diagonals_c_mm):.1f}")
            else:
                print(f"[{label}] Нет объектов.")

    # 6) После обработки всех кадров закрываем камеру
    zed.close()
    print("\n[INFO] Обработка последовательности завершена.")

if __name__ == "__main__":
    main()
