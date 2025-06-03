# server.py
import os
import cv2
import base64
import json
import asyncio

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from zed.zed_loader import get_intrinsics_from_svo, init_camera, grab_frame
# analyze_* → (out_bgr, rgba_contours, diams_px, diams_mm, xs)
from processing.analyzer import analyze_color_frame, analyze_depth_frame

from config import (
    SVO_PATH,
    FRAME_NUMBER,  # стартовый кадр
    CROP, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT,
)
from processing.config import Config
from processing.visual_config import VisualizationConfig

app = FastAPI()

# Раздаём статику из папки static/
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")


# ------------------------------------------------
# Конфигурации
# ------------------------------------------------
NUM_FRAMES = 40  # сколько кадров подряд обрабатываем

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
        bbox_color=(255, 0, 0),  # цвет контура глубины (BGR)
        invert=True,
        equalize_hist=False,
        use_clahe=True,
    ))
]

color_configs = [
    ("Color-Small", Config(
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
        bbox_color=(0, 255, 0),  # цвет контура для цветовой сегментации
        bbox_thickness=1,
        font_scale=0.5,
        font_thickness=0,
        distance_transform_mask=5,
        foreground_threshold_ratio=0.1,
        use_clahe=True,
        equalize_hist=True,
    )),
]

def apply_crop(image: np.ndarray, x, y, width, height):
    """
    Обрезает входное изображение по прямоугольнику (x, y, width, height).
    """
    h, w = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x1 + width)
    y2 = min(h, y1 + height)
    return image[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def alpha_composite(rgba_base: np.ndarray, rgba_overlay: np.ndarray) -> np.ndarray:
    """
    Классический альфа-композитинг (over).
    Оба изображения (base и overlay) должны быть RGBA (H,W,4), dtype=uint8.
    Возвращает новое RGBA (H,W,4).
    """
    base_rgb = rgba_base[..., :3].astype(np.float32) / 255.0
    base_a   = rgba_base[..., 3:].astype(np.float32) / 255.0  # shape (H,W,1)

    over_rgb = rgba_overlay[..., :3].astype(np.float32) / 255.0
    over_a   = rgba_overlay[..., 3:].astype(np.float32) / 255.0  # shape (H,W,1)

    comp_a   = over_a + base_a * (1.0 - over_a)  # (H,W,1)
    comp_rgb = over_rgb * over_a + base_rgb * base_a * (1.0 - over_a)  # (H,W,3)

    comp_rgba = np.zeros_like(rgba_base, dtype=np.uint8)
    # RGB
    comp_rgba[..., :3] = np.clip(comp_rgb * 255.0, 0, 255).astype(np.uint8)
    # Альфа: снимаем лишнюю размерность
    comp_rgba[...,  3] = np.clip(comp_a[..., 0] * 255.0, 0, 255).astype(np.uint8)

    return comp_rgba


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Клиент подключился по WebSocket!")

    try:
        # 1) Инициализируем камеру один раз
        zed = init_camera(SVO_PATH)
        fx, fy, *_ = get_intrinsics_from_svo(SVO_PATH)
        vis_cfg = VisualizationConfig(show_plots=False, plot_figsize=(4, 3))

        # 2) Проходим по кадрам
        for i in range(NUM_FRAMES):
            current_frame = FRAME_NUMBER + i
            print(f"[СЕРВЕР] Обработка кадра #{current_frame}")

            # 2.1) Захватим кадр
            image, depth, depth_shading, point_cloud = grab_frame(zed, current_frame)
            if image is None or depth is None:
                print(f"[СЕРВЕР] WARNING: не удалось получить кадр {current_frame}")
                continue

            # 2.2) Цвет (BGR) — убираем альфу, если есть
            color_np = image.get_data()
            if color_np.ndim == 3 and color_np.shape[2] == 4:
                color_np = color_np[:, :, :3]

            # 2.3) Глубина (float32)
            depth_np = depth.get_data().astype(np.float32)

            # 2.4) Обрезка, если CROP=True
            if CROP:
                color_np, _ = apply_crop(color_np, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)
                depth_np, _ = apply_crop(depth_np, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)

            # 3) Сегментация depth:
            #    analyze_depth_frame → (out_bgr_depth, rgba_depth_contours, diams_px, diams_mm, xs)
            out_bgr_d, rgba_depth_contours, diams_px, diams_mm, xs = analyze_depth_frame(
                depth_np, depth_configs[0][1], vis_cfg,
                label=depth_configs[0][0],
                output_dir=None,
                save_plots=False,
                fx=fx, fy=fy
            )

            # 4) Сегментация color:
            out_bgr_c, rgba_color_contours, diams_color_px, diams_color_mm, xs_color = analyze_color_frame(
                color_np, color_configs[0][1], vis_cfg,
                label=color_configs[0][0],
                output_dir=None,
                save_plots=False,
                depth_img=depth_np,
                fx=fx, fy=fy
            )

            # 5) Строим RGBA-фон из color_np
            h, w = color_np.shape[:2]
            rgba_base = np.zeros((h, w, 4), dtype=np.uint8)
            rgba_base[..., :3] = color_np
            rgba_base[...,  3] = 255  # полностью непрозрачный фон

            # 6) Альфа-композитинг: сначала depth-контуры, потом color-контуры
            comp1 = alpha_composite(rgba_base, rgba_depth_contours)
            comp2 = alpha_composite(comp1, rgba_color_contours)

            # 7) Кодируем comp2 (RGBA) → PNG → base64
            success, buffer = cv2.imencode('.png', comp2)
            if not success:
                print(f"[СЕРВЕР] Ошибка кодирования overlay_rgba кадра {current_frame}")
                continue
            overlay_b64 = base64.b64encode(buffer).decode('utf-8')

            # 8) Отправляем JSON
            payload = {
                "frame": current_frame,
                "overlay_b64": overlay_b64,
                "diams_mm": diams_px,          # глубинные диаметры
                "xs": xs,                      # глубинные X
                "diams_color_mm": diams_color_mm,  # цветовые диаметры
            }
            await websocket.send_text(json.dumps(payload))

            # 9) Лёгкая задержка
            await asyncio.sleep(0.1)

        # Закрываем камеру и WebSocket
        zed.close()
        print("[СЕРВЕР] Завершено, камера закрыта.")
        await websocket.close()

    except WebSocketDisconnect:
        print("Клиент отключился.")
    except Exception as e:
        print(f"[СЕРВЕР] Ошибка: {e}")
        try:
            await websocket.close()
        except:
            pass
