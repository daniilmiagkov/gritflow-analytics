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

# Путь к корню проекта
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from processing.analyzer import analyze_frame
from processing.depth_config import DepthConfig
from processing.color_config import ColorConfig
from processing.visual_config import VisualizationConfig
from config import OUTPUT_DIR, FRAME_NUMBER

# --- Глобальные переменные (будут инициализированы через init_worker) ---
global_color = None
global_depth = None

def init_worker_shared_data(frame_number: int):
    """Инициализирует общие данные один раз в каждом процессе"""
    global global_color, global_depth
    color_path = os.path.join(OUTPUT_DIR, f"frame_{frame_number}_color.png")
    depth_path = os.path.join(OUTPUT_DIR, f"frame_{frame_number}_depth.tiff")
    global_color = cv2.imread(color_path, cv2.IMREAD_COLOR)
    global_depth = tifffile.imread(depth_path)
    if global_color is None or global_depth is None:
        raise FileNotFoundError(f"Не найден файл: {color_path} или {depth_path}")

def worker(args):
    """Один воркер обрабатывает конкретную комбинацию параметров"""
    override, base_cfg_dict, frame_number = args
    global global_color, global_depth

    try:
        cfg = ColorConfig(**base_cfg_dict)
        for k, v in override.items():
            setattr(cfg, k, v)

        depth_cfg = DepthConfig()
        vis_cfg = VisualizationConfig(show_plots=False)

        (c_res, c_diams, _), _ = analyze_frame(
            global_color, global_depth,
            depth_cfg, cfg, vis_cfg,
            use_depth=False
        )

        MIN_COUNT = 30
        BIG_THRESHOLD = 30.0
        total_count = len(c_diams)
        big_count = sum(1 for d in c_diams if d > BIG_THRESHOLD)

        if total_count < MIN_COUNT or not (1 <= big_count <= 4):
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

def analyze_results(results, output_csv_path="param_analysis.csv"):
    print("\n=== 📊 Анализ результатов ===")
    param_stats = defaultdict(lambda: defaultdict(int))

    for r in results:
        for k, v in r["params"].items():
            param_stats[k][v] += 1

    print("\nЧастоты параметров (в успешных конфигурациях):")
    for param, val_counts in param_stats.items():
        print(f"\n[{param}]")
        for val, count in sorted(val_counts.items(), key=lambda x: -x[1]):
            print(f"  {val}: {count} раз")

    # Сохраняем все результаты в CSV
    df = pd.DataFrame([
        {**r["params"], "total_count": r["total_count"], "big_count": r["big_count"]}
        for r in results
    ])
    csv_path = os.path.join(OUTPUT_DIR, output_csv_path)
    df.to_csv(csv_path, index=False)
    print(f"\n📁 CSV сохранён: {csv_path}")

def main():
    base_cfg = ColorConfig()
    base_cfg_dict = vars(base_cfg)

    param_tests = {
        "median_blur_size":           [1, 3, 5],
        "adaptive_thresh":            [True],
        "binary_thresh":              [0],
        "morph_kernel_size":          [1, 2, 3, 4, 5, 7, 11, 15],
        "morph_iterations":           [1, 2, 3, 4, 5],
        "min_particle_size":          [2, 3],
        "max_particle_size":          [200],
        "distance_transform_mask":    [1, 2, 3, 4, 5, 7, 9, 11],
        "foreground_threshold_ratio": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
        "dilate_iterations":          [1, 2, 3],
    }

    keys = list(param_tests.keys())
    value_lists = [param_tests[k] for k in keys]
    tasks = [(dict(zip(keys, combo)), base_cfg_dict, FRAME_NUMBER)
             for combo in itertools.product(*value_lists)]

    print(f"\n=== Запуск {len(tasks)} задач на {cpu_count()} ядрах ===\n")

    ctx = get_context("spawn")  # Для совместимости с Windows
    with ctx.Pool(cpu_count(), initializer=init_worker_shared_data, initargs=(FRAME_NUMBER,)) as pool:
        results = pool.map(worker, tasks)

    # Убираем None
    filtered = [r for r in results if r]
    print(f"\n=== ✅ Успешных конфигураций: {len(filtered)} ===")
    for r in filtered:
        print(r["path"])

    analyze_results(filtered)

if __name__ == "__main__":
    main()
