# server.py
import os
import cv2
import tifffile
import base64
import json
import asyncio

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from zed.zed_loader import get_intrinsics_from_svo, init_camera, grab_frame
from processing.analyzer import analyze_color_frame, analyze_depth_frame

from config import (
    SVO_PATH,
    FRAME_NUMBER,  # стартовый кадр
    CROP, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT,
)
from processing.config import Config
from processing.visual_config import VisualizationConfig

# -----------------------------------------
# Настройки
NUM_FRAMES = 20  # Сколько кадров подряд обрабатываем
# -----------------------------------------

# Конфиги для анализа (по глубине и по цвету)
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

def apply_crop(image: np.ndarray, x, y, width, height):
    """
    Обрезает изображение по прямоугольнику (x, y, width, height).
    Возвращает (обрезанное_изображение, (x1, y1, new_width, new_height)).
    """
    h, w = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x1 + width)
    y2 = min(h, y1 + height)
    return image[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)

app = FastAPI()

# -----------------------------------------
# Паспорт HTML: теперь с кнопкой «Start» и отсроченным созданием WebSocket.
# -----------------------------------------
html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Live Segmentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #container { display: flex; flex-direction: column; align-items: center; }
        #charts { display: flex; gap: 40px; margin-top: 20px; }
        canvas { background: #f4f4f4; border: 1px solid #ccc; }
        #status { margin-top: 10px; font-weight: bold; }
        #startBtn { padding: 10px 20px; font-size: 16px; cursor: pointer; }
    </style>
</head>
<body>
    <div id="container">
        <h2>Live Segmentation из SVO</h2>
        <button id="startBtn">Запустить трансляцию</button>
        <div id="status">Статус: не подключено</div>
        <img id="segmentedImage" src="" alt="Segmentation" width="640" height="360" style="margin-top:20px;" />
        <div id="charts">
            <div>
                <h4>Распределение диаметров (mm)</h4>
                <canvas id="diamChart" width="400" height="300"></canvas>
            </div>
            <div>
                <h4>Распределение по X</h4>
                <canvas id="xChart" width="400" height="300"></canvas>
            </div>
        </div>
    </div>

    
    <!-- Подключаем Chart.js (CDN) -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>

    let ws = null;
        const startBtn = document.getElementById("startBtn");
        const statusDiv = document.getElementById("status");
        const imgElem = document.getElementById("segmentedImage");

        // Инициализируем две пустые гистограммы через Chart.js
        const diamCtx = document.getElementById("diamChart").getContext("2d");
        const xCtx = document.getElementById("xChart").getContext("2d");

        const diamChart = new Chart(diamCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Частота',
                    data: [],
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: false,
                scales: {
                    x: { title: { display: true, text: 'Диаметр (мм)' } },
                    y: { title: { display: true, text: 'Частота' } }
                }
            }
        });

        const xChart = new Chart(xCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Частота',
                    data: [],
                    backgroundColor: 'rgba(255, 159, 64, 0.6)',
                    borderColor: 'rgba(255, 159, 64, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: { title: { display: true, text: 'X-координата (px)' } },
                    y: { title: { display: true, text: 'Частота' } }
                }
            }
        });
    /**
     * Обновлённая функция для биннинга и отрисовки гистограммы:
     * - Если length(values) <= numBins → создаёт по одной корзине на каждое уникальное значение.
     * - Иначе → классическое равномерное разбитие диапазона [minVal, maxVal] на numBins.
     *
     * @param {Chart} chartObj – экземпляр Chart.js
     * @param {Array<number>} values – массив чисел (например, diams_mm или xs)
     * @param {number} numBins – желаемое число корзин (bins)
     */
    function updateHistogram(chartObj, values, numBins) {
        // 1) Если массив пуст или не определён → очищаем график
        if (!values || values.length === 0) {
            chartObj.data.labels = [];
            chartObj.data.datasets[0].data = [];
            chartObj.update();
            return;
        }

        // 2) Если значений меньше либо равно numBins → создаём корзину для каждого уникального значения
        if (values.length <= numBins) {
            // Сначала найдём уникальные значения (в порядке появления)
            const uniqueMap = new Map();
            values.forEach(v => {
                const key = v.toFixed(1); // округлим до 1-го знака, чтобы избежать "мелких" плавающих отклонений
                uniqueMap.set(key, (uniqueMap.get(key) || 0) + 1);
            });

            const labels = Array.from(uniqueMap.keys());
            const counts = Array.from(uniqueMap.values());

            chartObj.data.labels = labels;
            chartObj.data.datasets[0].data = counts;
            chartObj.update();
            return;
        }

        // 3) Иначе: когда значений больше numBins → обычное равномерное биннинг-деление
        const minVal = Math.min(...values);
        const maxVal = Math.max(...values);

        // Если весь диапазон сводится в одну точку (теоретически) – одна корзина
        if (minVal === maxVal) {
            chartObj.data.labels = [minVal.toFixed(1)];
            chartObj.data.datasets[0].data = [values.length];
            chartObj.update();
            return;
        }

        // 4) Обычный случай: разбиваем диапазон [minVal, maxVal] на numBins равных частей
        const bins = numBins;
        const range = maxVal - minVal;
        const binSize = range / bins;
        const labels = [];
        const counts = new Array(bins).fill(0);

        for (let i = 0; i < bins; i++) {
            const left = minVal + i * binSize;
            const right = left + binSize;
            labels.push(left.toFixed(1) + "–" + right.toFixed(1));
        }

        // 5) Распределяем каждое значение по корзине
        values.forEach(v => {
            let idx = Math.floor((v - minVal) / binSize);
            if (idx < 0) idx = 0;
            if (idx >= bins) idx = bins - 1; // если v == maxVal
            counts[idx]++;
        });

        chartObj.data.labels = labels;
        chartObj.data.datasets[0].data = counts;
        chartObj.update();
    }

    // Далее ваш код для WebSocket:
    function startWebSocket() {
        if (ws !== null) {
            return;
        }
        ws = new WebSocket("ws://" + window.location.host + "/ws");

        ws.onopen = function() {
            statusDiv.innerText = "Статус: подключено";
            startBtn.disabled = true;
        };

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);

            // 1) Обновляем изображение
            imgElem.src = "data:image/png;base64," + data.image_b64;

            console.log("Получены diams_mm:", data.diams_mm);
console.log("Получены xs:", data.xs);

            // 2) Обновляем гистограмму диаметров (20 корзин)
            updateHistogram(diamChart, data.diams_mm, 20);

            // 3) Обновляем гистограмму по X (20 корзин)
            updateHistogram(xChart, data.xs, 20);
        };

        ws.onclose = function() {
            statusDiv.innerText = "Статус: отключено";
            ws = null;
            startBtn.disabled = false;
        };

        ws.onerror = function(err) {
            console.error("Ошибка WebSocket:", err);
            statusDiv.innerText = "Статус: ошибка";
        };
    }

    startBtn.addEventListener("click", () => {
        statusDiv.innerText = "Статус: подключение...";
        startWebSocket();
    });
</script>

</body>
</html>
"""

@app.get("/")
async def get_index():
    """
    Возвращает HTML-страницу с кнопкой и пустыми холстами Chart.js.
    """
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket-эндпоинт. После подключения строго по запросу клиента
    (при клике кнопки) начинает отправлять обработанные кадры в JSON.
    """
    await websocket.accept()
    print("Клиент подключился по WebSocket!")

    try:
        # 1) Инициализируем камеру и параметры
        zed = init_camera(SVO_PATH)
        fx, fy, *_ = get_intrinsics_from_svo(SVO_PATH)
        vis_cfg = VisualizationConfig(show_plots=False, plot_figsize=(4, 3))

        # 2) Цикл по кадрам
        for i in range(NUM_FRAMES):
            current_frame = FRAME_NUMBER + i
            print(f"[СЕРВЕР] Обработка кадра #{current_frame}")

            image, depth, depth_shading, point_cloud = grab_frame(zed, current_frame)
            if image is None or depth is None:
                print(f"[СЕРВЕР] WARNING: не удалось захватить кадр {current_frame}")
                continue

            color_np = image.get_data()
            depth_np = depth.get_data().astype(np.float32)

            # Обрезка, если нужно
            if CROP:
                color_np, _ = apply_crop(color_np, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)
                depth_np, _ = apply_crop(depth_np, CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT)

            # --- Анализ Depth (первый конфиг) ---
            label, cfg = depth_configs[0]
            seg, diams_px, diams_mm, xs = analyze_depth_frame(
                depth_np, cfg, vis_cfg,
                label=label,
                output_dir=None,   # не сохранять локально
                save_plots=False,  # не рисовать серверные графики
                fx=fx, fy=fy
            )

            # Кодируем сегментированное изображение в base64
            success, buffer = cv2.imencode('.png', seg)
            if not success:
                print(f"[СЕРВЕР] Ошибка кодирования сегментации кадра {current_frame}")
                continue
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            # Собираем JSON-пэйлоад
            payload = {
                "frame": current_frame,
                "image_b64": img_b64,
                "diams_mm": diams_mm,
                "xs": xs,
            }
            await websocket.send_text(json.dumps(payload))

            # Небольшая пауза, чтобы не забивать канал
            await asyncio.sleep(0.1)

        # После всех кадров закрываем камеру и соединение
        zed.close()
        print("[СЕРВЕР] Обработка завершена, камера закрыта.")
        await websocket.close()

    except WebSocketDisconnect:
        print("Клиент отключился.")
    except Exception as e:
        print(f"[СЕРВЕР] Ошибка: {e}")
        try:
            await websocket.close()
        except:
            pass
