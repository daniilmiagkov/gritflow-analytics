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
# analyze_* возвращают 6 полей: (out_bgr, rgba_contours, widths_px, heights_px, diagonals_mm, xs)
from processing.analyzer import analyze_color_frame, analyze_depth_frame

from config import (
    SVO_PATH,
    FRAME_NUMBER,  # стартовый кадр (игнорируется, если клиент задаёт диапазон)
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
# Диапазон кадров теперь берётся из параметров запроса (start..end)
# Оставим NUM_FRAMES лишь как некий потолок, если захочется ограничить длину
NUM_FRAMES = 1000

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
        bbox_color=(255, 0, 0),  # красные контуры — глубина
        invert=True,
        equalize_hist=False,
        use_clahe=True,
    )),
    # при желании сюда можно добавить ещё depth-конфиги
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
        bbox_color=(0, 255, 0),  # зелёные контуры для “малых”
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
    Клиент передаёт параметры start и end через query-параметры (?start=..&end=..).
    Обрабатываем диапазон кадров [start..end], по каждому конфигу из depth_configs
    и color_configs.
    """
    await websocket.accept()
    print(f"[СЕРВЕР] WebSocket открыт (кадры {start}–{end})")

    # Если неверные параметры — закрываем
    if start is None or end is None or start < 0 or end < start:
        await websocket.send_text(json.dumps({"error": "Неверный диапазон кадров"}))
        await websocket.close()
        return

    try:
        # 1) Инициализируем ZED-камеру и получаем параметры
        zed = init_camera(SVO_PATH)
        fx, fy, *_ = get_intrinsics_from_svo(SVO_PATH)
        vis_cfg = VisualizationConfig(show_plots=False, plot_figsize=(4, 3))

        # 2) Цикл по каждому кадру
        for frame_idx in range(start, end + 1):
            # Если вдруг превысили заданный потолок, выходим
            if frame_idx - start >= NUM_FRAMES:
                break

            print(f"[СЕРВЕР] Обработка кадра #{frame_idx}")

            # 2.1) Захват текущего кадра
            image, depth, depth_shading, point_cloud = grab_frame(zed, frame_idx)
            if image is None or depth is None:
                print(f"[СЕРВЕР] WARNING: кадр {frame_idx} не получен, пропускаем")
                continue

            # 2.2) Цвет BGR (убираем альфу, если есть)
            color_np = image.get_data()
            if color_np.ndim == 3 and color_np.shape[2] == 4:
                color_np = color_np[:, :, :3]

            # 2.3) Глубина
            depth_np = depth.get_data().astype(np.float32)

            # 2.4) Обрезка (если включено)
            if CROP:
                color_np, _ = apply_crop(color_np, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)
                depth_np, _ = apply_crop(depth_np, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)

            # 3) Сегментация по глубине — проходим по всем depth_configs
            depth_overlays = []
            all_diags_depth = {}
            all_xs_depth   = {}
            for label, cfg in depth_configs:
                out_bgr_d, rgba_d, widths_d_px, heights_d_px, diagonals_d_mm, xs_d = analyze_depth_frame(
                    depth_np, cfg, vis_cfg,
                    label=label,
                    output_dir=None,
                    save_plots=False,
                    fx=fx, fy=fy
                )
                # Соберём overlay-слои
                depth_overlays.append(rgba_d)
                # Запомним результаты по ключу label (например, "Depth-Big")
                safe_label = label.lower().replace(" ", "_")  # "depth_big"
                all_diags_depth[safe_label] = diagonals_d_mm
                all_xs_depth[safe_label]   = xs_d

            # 4) Сегментация по цвету — проходим по всем color_configs
            color_overlays = []
            all_diags_color = {}
            all_xs_color   = {}

            for label, cfg in color_configs:
                out_bgr_c, rgba_c, widths_c_px, heights_c_px, diagonals_c_mm, xs_c = analyze_color_frame(
                    color_np, cfg, vis_cfg,
                    label=label,
                    output_dir=None,
                    save_plots=False,
                    depth_img=depth_np,
                    fx=fx, fy=fy
                )
                color_overlays.append(rgba_c)
                safe_label = label.lower().replace(" ", "_")  # e.g. "color_small", "color_large"
                all_diags_color[safe_label] = diagonals_c_mm
                all_xs_color[safe_label]   = xs_c

            # 5) Готовим базовый RGBA-фон (исходное цветное изображение)
            h, w = color_np.shape[:2]
            rgba_base = np.zeros((h, w, 4), dtype=np.uint8)
            rgba_base[..., :3] = color_np
            rgba_base[...,  3] = 255  # полностью непрозрачный фон

            # 6) Альфа-композитинг: сначала все depth_overlays, затем все color_overlays
            comp = rgba_base
            for rgba_d in depth_overlays:
                comp = alpha_composite(comp, rgba_d)
            for rgba_c in color_overlays:
                comp = alpha_composite(comp, rgba_c)

            # 7) Кодируем итоговый overlay (RGBA) → PNG → base64
            success, buffer = cv2.imencode('.png', comp)
            if not success:
                print(f"[СЕРВЕР] Ошибка кодирования RGBA-изображения кадра {frame_idx}")
                continue
            overlay_b64 = base64.b64encode(buffer).decode('utf-8')

            # 8) Собираем payload
            payload = {
                "frame": frame_idx,
                "overlay_b64": overlay_b64,
                # Для каждого depth-конфига пропишем заново:
            }
            # Добавляем поля depth для каждого label
            for label, _ in depth_configs:
                safe_label = label.lower().replace(" ", "_")
                payload[f"diagonals_depth_{safe_label}_mm"] = all_diags_depth[safe_label]
                payload[f"xs_depth_{safe_label}"]          = all_xs_depth[safe_label]

            # Добавляем поля color для каждого label
            for label, _ in color_configs:
                safe_label = label.lower().replace(" ", "_")
                payload[f"diagonals_color_{safe_label}_mm"] = all_diags_color[safe_label]
                payload[f"xs_color_{safe_label}"]           = all_xs_color[safe_label]

            # Отправляем в JSON
            await websocket.send_text(json.dumps(payload))

        # 9) Завершаем и закрываем
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
