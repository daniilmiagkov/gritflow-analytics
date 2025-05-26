import os
import cv2
import tifffile
import numpy as np
import itertools
import json
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count, get_context
from collections import defaultdict
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List

# Путь к корню проекта
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from processing.analyzer import analyze_frame
from processing.depth_config import DepthConfig
from processing.color_config import ColorConfig
from processing.visual_config import VisualizationConfig
from config import OUTPUT_DIR, FRAME_NUMBER


import matplotlib.pyplot as plt
import seaborn as sns

# --- Глобальные переменные для multiprocessing (инициализируются один раз в каждом процессе) ---
global_color: Optional[np.ndarray] = None
global_depth: Optional[np.ndarray] = None


def init_worker_shared_data(frame_number: int):
    """Инициализирует общие данные один раз в каждом процессе."""
    global global_color, global_depth

    color_path = os.path.join(OUTPUT_DIR, f"frame_{frame_number}_color.png")
    depth_path = os.path.join(OUTPUT_DIR, f"frame_{frame_number}_depth.tiff")

    color = cv2.imread(color_path, cv2.IMREAD_COLOR)
    depth = tifffile.imread(depth_path)

    if color is None:
        raise FileNotFoundError(f"❌ Не найден файл изображения: {color_path}")
    if depth is None or not isinstance(depth, np.ndarray):
        raise FileNotFoundError(f"❌ Не найден файл глубины или он некорректен: {depth_path}")

    global_color = color
    global_depth = depth


def worker(args):
    override, base_cfg_dict, frame_number = args
    global global_color

    try:
        cfg = ColorConfig(**base_cfg_dict)
        for k, v in override.items():
            setattr(cfg, k, v)

        vis_cfg = VisualizationConfig(show_plots=False)

        c_res, c_diams, _ = analyze_frame(global_color, cfg, vis_cfg)

        MIN_COUNT = 150
        BIG_THRESHOLD = 40.0
        total_count = len(c_diams)
        big_count = sum(1 for d in c_diams if d > BIG_THRESHOLD)

        if total_count < MIN_COUNT or not (2 <= big_count <= 3):
            return None

        parts = [f"{k[:2]}-{str(v).replace(' ', '')}" for k, v in override.items()]
        name = f"color_{'_'.join(parts)}_frame-{frame_number}.png"
        save_path = os.path.join(OUTPUT_DIR, name)
        cv2.imwrite(save_path, c_res)

        return {
            "path": save_path,
            "params": override,
            "total_count": total_count,
            "big_count": big_count,
        }

    except Exception as e:
        print(f"[PID {os.getpid()}] ❌ Ошибка в конфигурации {override}: {e}")
        return None
    
def analyze_results(results: List[Dict[str, Any]], output_csv_path: str = "param_analysis.csv"):
    print("\n=== 📊 Анализ результатов ===")
    param_stats = defaultdict(lambda: defaultdict(int))

    for r in results:
        for k, v in r["params"].items():
            param_stats[k][v] += 1

    # --- Вывод статистики по параметрам ---
    print("\nЧастоты параметров (в успешных конфигурациях):")
    for param, val_counts in param_stats.items():
        print(f"\n[{param}]")
        for val, count in sorted(val_counts.items(), key=lambda x: -x[1]):
            print(f"  {val}: {count} раз")

    # --- Построение графиков ---
    print("\n📈 Генерация графиков по параметрам...")
    num_params = len(param_stats)
    fig, axes = plt.subplots(nrows=num_params, ncols=1, figsize=(10, 5 * num_params))

    if num_params == 1:
        axes = [axes]  # чтобы итерация работала

    for ax, (param, val_counts) in zip(axes, param_stats.items()):
        items = sorted(val_counts.items(), key=lambda x: x[0])
        labels = [str(k) for k, _ in items]
        values = [v for _, v in items]

        sns.barplot(x=labels, y=values, ax=ax)
        ax.set_title(f"Распределение параметра: {param}")
        ax.set_ylabel("Количество успешных конфигураций")
        ax.set_xlabel(param)
        ax.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "param_distribution.png")
    plt.savefig(plot_path)
    print(f"📊 Плот сохранён: {plot_path}")

    # # --- CSV сохранение ---
    # df = pd.DataFrame([
    #     {**r["params"], "total_count": r["total_count"], "big_count": r["big_count"]}
    #     for r in results
    # ])
    # csv_path = os.path.join(OUTPUT_DIR, output_csv_path)
    # df.to_csv(csv_path, index=False)
    # print(f"\n📁 CSV сохранён: {csv_path}")

    # # --- Вывод удачных конфигураций в стиле ColorConfig ---
    # print("\n=== ✅ Успешные конфигурации ===")
    # for r in results:
    #     cfg = ColorConfig(**{**vars(ColorConfig()), **r["params"]})  # безопасно обновляем дефолт
    #     print(f"\n{cfg}")
    #     print(f"  🔸 total_count = {r['total_count']}, big_count = {r['big_count']}")



def main():
    base_cfg = ColorConfig()
    base_cfg_dict = vars(base_cfg)

    # param_tests = {
    #     "median_blur_size":           [1, 3],
    #     "adaptive_thresh":            [True],
    #     "adaptive_block_size": [5, 7, 9],
    #     "adaptive_C": [-3, -2, -1, 0, 1, 2, 3, 4, 5],
    #     "use_otsu":                   [True],
    #     "binary_thresh":              [0],
    #     "morph_kernel_shape":         ["rect", "ellipse", "cross"],
    #     "morph_kernel_size":          [1, 2, 3, 4, 5, 7, 11],
    #     "morph_iterations":           [0, 1, 2, 3, 4],
    #     "min_particle_size":          [10],
    #     "max_particle_size":          [200],
    #     "distance_transform_mask":    [0, 3, 5],
    #     "foreground_threshold_ratio": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
    #     "dilate_iterations":          [0, 1, 2, 3],
    # }

    param_tests = {
        # === Предобработка ===
        "median_blur_size":           [3, 5],  # добавим 5 — проверим, поможет ли большее сглаживание

        # === Пороговая сегментация ===
        "adaptive_thresh":            [True],  # только адаптивный порог
        "adaptive_block_size":        [7, 9, 11, 13, 15],  # расширяем диапазон вверх для крупных объектов
        "adaptive_C":                 [-2, 0, 2, 3],  # фокус на нейтральных и положительных значениях
        "use_otsu":                   [False],  # отключаем, так как с adaptive не нужен
        "binary_thresh":              [0],  # используется только если adaptive_thresh=False (а тут — не используется)

        # === Морфологическая обработка ===
        "morph_kernel_shape":         ["rect", "ellipse"],  # оставим только наиболее полезные формы
        "morph_kernel_size":          [2, 3, 5],  # убираем слишком большие — искажают мелкие объекты
        "morph_iterations":           [1, 2, 3],  # даёт выбор для усиления чистки и разрыва кластеров

        # === Фильтрация и анализ контуров ===
        "min_particle_size":          [5],  # пропускаем и мелкие, и крупные
        "max_particle_size":          [150],  # расширим на случай очень крупных объектов

        # === Доп. сегментация (distance transform) ===
        "distance_transform_mask":    [3, 5],  # оставим все варианты
        "foreground_threshold_ratio": [0.1, 0.16, 0.2],  # добавим 0.16 из твоей текущей настройки
        "dilate_iterations":          [1],  # уменьшили, чтобы не объединялись лишние объекты
    }



    # param_tests = {
    #     "median_blur_size":           [1, 3, 5],
    #     "adaptive_thresh":            [True],
    #     "binary_thresh":              [0],
    #     "morph_kernel_size":          [1, 2, ],
    #     "morph_iterations":           [1, 2],
    #     "min_particle_size":          [2, 3],
    #     "max_particle_size":          [200],
    #     "distance_transform_mask":    [1, 2, ],
    #     "foreground_threshold_ratio": [0.01, ],
    #     "dilate_iterations":          [1, 2, 3],
    # }


    keys = list(param_tests.keys())
    value_lists = [param_tests[k] for k in keys]
    tasks = [
        (dict(zip(keys, combo)), base_cfg_dict, FRAME_NUMBER)
        for combo in itertools.product(*value_lists)
    ]

    print(f"\n=== 🚀 Запуск {len(tasks)} задач на {cpu_count()} ядрах ===\n")

    ctx = get_context("spawn")  # Особенно важно для Windows
    with ctx.Pool(cpu_count(), initializer=init_worker_shared_data, initargs=(FRAME_NUMBER,)) as pool:
        results = pool.map(worker, tasks)

    # Отбрасываем None
    filtered = [r for r in results if r]
    print(f"\n=== ✅ Успешных конфигураций: {len(filtered)} ===")
    for r in filtered:
        print(f"✔ {r['path']} — всего: {r['total_count']}, больших: {r['big_count']}")
        cfg = ColorConfig(**{**vars(ColorConfig()), **r["params"]})
        print(f"{cfg}\n")


    analyze_results(filtered)


if __name__ == "__main__":
    main()
