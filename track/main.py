import os
from pathlib import Path
import sys
import cv2
import numpy as np
import tifffile

# Добавляем корень проекта в sys.path, чтобы найти модули
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from processing.analyzer import analyze_color_frame, analyze_depth_frame
from processing.config import Config
from processing.visual_config import VisualizationConfig
from config import (
    SVO_PATH, OUTPUT_DIR,
    CROP, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT,
)
from zed.zed_loader import get_intrinsics_from_svo, grab_frame, init_camera

def apply_crop(image, x, y, width, height):
    h, w = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x1 + width)
    y2 = min(h, y1 + height)
    return image[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

# Класс для отслеживания одного объекта
class Track:
    def __init__(self, center, size):
        self.center = center   # (x, y) в пикселях
        self.size = size       # диаметр в мм
        self.count = 1         # сколько кадров подряд обнаружен

def match_and_update(tracks, detections, max_dist=20):
    """
    Сопоставляет найденные объекты detections с существующими tracks.
    tracks: список текущих треков (из предыдущего кадра)
    detections: список кортежей (center, size) для новых объектов в текущем кадре
    max_dist: максимальное расстояние (px) для сопоставления центров
    Возвращает новый список треков (tracks) для следующего шага.
    """
    new_tracks = []
    used = set()
    for ctr, sz in detections:
        best = None
        best_dist = max_dist
        for i, tr in enumerate(tracks):
            if i in used:
                continue
            dx = tr.center[0] - ctr[0]
            dy = tr.center[1] - ctr[1]
            dist = np.hypot(dx, dy)
            if dist < best_dist:
                best = i
                best_dist = dist
        if best is not None:
            # Обновляем существующий трек
            tr = tracks[best]
            tr.center = ctr
            tr.size = sz
            tr.count += 1
            new_tracks.append(tr)
            used.add(best)
        else:
            # Новый трек
            new_tracks.append(Track(ctr, sz))
    return new_tracks

def main():
    # 1. Инициализируем ZED-камеру и получаем fx, fy
    zed = init_camera(SVO_PATH)
    fx, fy, *_ = get_intrinsics_from_svo(SVO_PATH)

    # 2. Конфигурации для глубины и цвета
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
            min_particle_size=40,
            max_particle_size=200,
            bbox_color=(0,255,0),
            bbox_thickness=1,
            font_scale=0.5,
            font_thickness=0,
            distance_transform_mask=5,
            foreground_threshold_ratio=0.1,
            use_clahe=True,
            equalize_hist=True,
        )),
    ]

    vis_cfg = VisualizationConfig(show_plots=False)

    # 3. Переберём несколько кадров подряд
    start_frame = 100  # номер первого кадра (можете изменить)
    num_frames = 5     # сколько кадров обрабатываем подряд
    frame_ids = list(range(start_frame, start_frame + num_frames))

    # 4. Инициализируем пустые списки треков
    tracks_depth = []
    tracks_color = []

    # 5. Папки для сохранения «стабильных» сегментированных кадров
    stable_depth_dir = os.path.join(OUTPUT_DIR, "stable_depth")
    stable_color_dir = os.path.join(OUTPUT_DIR, "stable_color")
    os.makedirs(stable_depth_dir, exist_ok=True)
    os.makedirs(stable_color_dir, exist_ok=True)

    for fid in frame_ids:
        # 6. Захват кадра из SVO
        zed.set_svo_position(fid)
        image, depth, depth_shading, point_cloud = grab_frame(zed, fid)
        if image is None or depth is None:
            continue

        color_np = image.get_data()
        depth_np = depth.get_data().astype(np.float32)

        # 7. Обрезка (если нужна)
        if CROP:
            color_np, _ = apply_crop(color_np, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)
            depth_np, _ = apply_crop(depth_np, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)

        # 8. Анализ глубины
        label_d, cfg_d = depth_configs[0]
        res_d, diams_px_d, diams_mm_d, xs_d = analyze_depth_frame(
            depth_np, cfg_d, vis_cfg,
            label=f"D{fid}", output_dir=OUTPUT_DIR, save_plots=False,
            fx=fx, fy=fy
        )
        # Сформируем список детекций: прикидываем Y как середину кадра
        det_depth = [((x, depth_np.shape[0]//2), size) for x, size in zip(xs_d, diams_mm_d)]
        tracks_depth = match_and_update(tracks_depth, det_depth)

        # Проверяем, у кого count == 3 — значит «стабильный» трек
        for tr in tracks_depth:
            if tr.count == 3:
                # Сохраняем сегментированное изображение глубины целиком
                fname = f"stable_depth_frame{fid}_x{int(tr.center[0])}_y{int(tr.center[1])}.png"
                cv2.imwrite(os.path.join(stable_depth_dir, fname), res_d)

        # 9. Анализ цвета
        label_c, cfg_c = color_configs[0]
        res_c, diams_px_c, diams_mm_c, xs_c = analyze_color_frame(
            color_np, cfg_c, vis_cfg,
            label=f"C{fid}", output_dir=OUTPUT_DIR, save_plots=False,
            depth_img=depth_np, fx=fx, fy=fy
        )
        det_color = [((x, color_np.shape[0]//2), size) for x, size in zip(xs_c, diams_mm_c)]
        tracks_color = match_and_update(tracks_color, det_color)

        # Аналогично: сохраняем, когда count == 3
        for tr in tracks_color:
            if tr.count == 3:
                fname = f"stable_color_frame{fid}_x{int(tr.center[0])}_y{int(tr.center[1])}.png"
                cv2.imwrite(os.path.join(stable_color_dir, fname), res_c)

    zed.close()

    # 10. Оставляем только стабильные (появившиеся ≥ 3 раз)
    stable_depth = [tr for tr in tracks_depth if tr.count >= 3]
    stable_color = [tr for tr in tracks_color if tr.count >= 3]

    print("Надёжные камни по глубине (count>=3):")
    for tr in stable_depth:
        print(f"  Центр: {tr.center}, Диаметр: {tr.size:.1f} мм, Появлений: {tr.count}")

    print("Надёжные камни по цвету (count>=3):")
    for tr in stable_color:
        print(f"  Центр: {tr.center}, Диаметр: {tr.size:.1f} мм, Появлений: {tr.count}")

if __name__ == "__main__":
    main()
