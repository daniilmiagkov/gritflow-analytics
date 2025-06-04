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
# analyze_* возвращают теперь 6 полей:
#   out_bgr, rgba_contours, widths_px, heights_px, diagonals_mm, xs
from processing.analyzer import analyze_color_frame, analyze_depth_frame

from config import (
    SVO_PATH,
    FRAME_NUMBER,  # стартовый кадр (будет игнорироваться, если позволяем клиенту задавать диапазон)
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
# Теперь диапазон кадров задаёт клиент, поэтому NUM_FRAMES не используется напрямую
# Но оставим его, если нужно ограничить максимальный длину
NUM_FRAMES = 1000  # просто потолок на число кадров, если потребуется

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
        bbox_color=(255, 0, 0),  # контуры глубины — красным
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
        bbox_color=(0, 255, 0),  # контуры цвета — зелёным
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
    Альфа-композитинг (over). base и overlay должны быть RGBA (H,W,4), dtype=uint8.
    Возвращает новое RGBA (H,W,4).
    """
    base_rgb = rgba_base[..., :3].astype(np.float32) / 255.0
    base_a   = rgba_base[..., 3:].astype(np.float32) / 255.0  # (H,W,1)

    over_rgb = rgba_overlay[..., :3].astype(np.float32) / 255.0
    over_a   = rgba_overlay[..., 3:].astype(np.float32) / 255.0  # (H,W,1)

    comp_a   = over_a + base_a * (1.0 - over_a)               # (H,W,1)
    comp_rgb = over_rgb * over_a + base_rgb * base_a * (1.0 - over_a)  # (H,W,3)

    comp_rgba = np.zeros_like(rgba_base, dtype=np.uint8)
    comp_rgba[..., :3] = np.clip(comp_rgb * 255.0, 0, 255).astype(np.uint8)
    comp_rgba[...,  3] = np.clip(comp_a[..., 0] * 255.0, 0, 255).astype(np.uint8)
    return comp_rgba


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    start: int = None,
    end:   int = None
):
    """
    После подключения клиент передаёт параметры start и end
    через параметры запроса (?start=...&end=...). 
    Обрабатываем кадры в диапазоне [start..end].
    """
    await websocket.accept()
    print(f"[СЕРВЕР] WebSocket открыт (кадры {start}–{end})")

    # Если переданы неверные параметры, закрываем соединение
    if start is None or end is None or start < 0 or end < start:
        await websocket.send_text(json.dumps({"error": "Неверный диапазон кадров"}))
        await websocket.close()
        return

    try:
        # 1) Один раз инициализируем ZED и параметры
        zed = init_camera(SVO_PATH)
        fx, fy, *_ = get_intrinsics_from_svo(SVO_PATH)
        vis_cfg = VisualizationConfig(show_plots=False, plot_figsize=(4, 3))

        # 2) Проходим по каждому кадру в диапазоне [start..end]
        for frame_idx in range(start, end + 1):
            print(f"[СЕРВЕР] Обработка кадра #{frame_idx}")

            # 2.1) Захват кадра
            image, depth, depth_shading, point_cloud = grab_frame(zed, frame_idx)
            if image is None or depth is None:
                print(f"[СЕРВЕР] WARNING: кадр {frame_idx} не получен, пропускаем")
                continue

            # 2.2) Получаем цветной BGR (убираем альфу, если есть)
            color_np = image.get_data()
            if color_np.ndim == 3 and color_np.shape[2] == 4:
                color_np = color_np[:, :, :3]

            # 2.3) Глубина
            depth_np = depth.get_data().astype(np.float32)

            # 2.4) Обрезка (если требуется)
            if CROP:
                color_np, _ = apply_crop(color_np, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)
                depth_np, _ = apply_crop(depth_np, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)

            # 3) Сегментация depth:
            #    analyze_depth_frame → (out_bgr_d, rgba_d, widths_d_px, heights_d_px, diagonals_d_mm, xs_d)
            out_bgr_d, rgba_d, widths_d_px, heights_d_px, diagonals_d_mm, xs_d = analyze_depth_frame(
                depth_np, depth_configs[0][1], vis_cfg,
                label=depth_configs[0][0],
                output_dir=None,
                save_plots=False,
                fx=fx, fy=fy
            )

            # 4) Сегментация color:
            #    analyze_color_frame → (out_bgr_c, rgba_c, widths_c_px, heights_c_px, diagonals_c_mm, xs_c)
            out_bgr_c, rgba_c, widths_c_px, heights_c_px, diagonals_c_mm, xs_c = analyze_color_frame(
                color_np, color_configs[0][1], vis_cfg,
                label=color_configs[0][0],
                output_dir=None,
                save_plots=False,
                depth_img=depth_np,
                fx=fx, fy=fy
            )

            # 5) Формируем базовый RGBA-фон из color_np
            h, w = color_np.shape[:2]
            rgba_base = np.zeros((h, w, 4), dtype=np.uint8)
            rgba_base[..., :3] = color_np
            rgba_base[...,  3] = 255  # полностью непрозрачный фон

            # 6) Альфа-композитинг: сначала контуры глубины, потом контуры цвета
            comp1 = alpha_composite(rgba_base, rgba_d)
            comp2 = alpha_composite(comp1, rgba_c)

            # 7) Кодируем comp2 (RGBA) → PNG → base64
            success, buffer = cv2.imencode('.png', comp2)
            if not success:
                print(f"[СЕРВЕР] Ошибка кодирования RGBA-изображения кадра {frame_idx}")
                continue
            overlay_b64 = base64.b64encode(buffer).decode('utf-8')

            # 8) Отправляем JSON-пэйлоад с новыми полями:
            payload = {
                "frame": frame_idx,
                "overlay_b64": overlay_b64,
                # передаём параметры частиц по глубине:
                "widths_depth_px":    widths_d_px,
                "heights_depth_px":   heights_d_px,
                "diagonals_depth_mm": diagonals_d_mm,
                "xs_depth":           xs_d,
                # параметры частиц по цвету:
                "widths_color_px":    widths_c_px,
                "heights_color_px":   heights_c_px,
                "diagonals_color_mm": diagonals_c_mm,
                "xs_color":           xs_c,
            }
            await websocket.send_text(json.dumps(payload))

        # 10) Закрываем камеру и WebSocket
        zed.close()
        print("[СЕРВЕР] Обработка диапазона завершена, камера закрыта.")
        await websocket.close()

    except WebSocketDisconnect:
        print("[СЕРВЕР] Клиент отключился.")
    except Exception as e:
        print(f"[СЕРВЕР] Ошибка: {e}")
        try:
            await websocket.close()
        except:
            pass
