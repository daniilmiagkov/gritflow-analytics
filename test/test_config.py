import os
import cv2
import tifffile
import numpy as np
import itertools
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count, get_context
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List

# Путь к корню проекта
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from processing.analyzer import analyze_frame
from processing.color_config import ColorConfig
from processing.visual_config import VisualizationConfig
from config import OUTPUT_DIR, FRAME_NUMBER


# --- Глобальные переменные для multiprocessing ---
global_color: Optional[np.ndarray] = None
global_depth: Optional[np.ndarray] = None


def init_worker_shared_data(frame_number: int):
    """Инициализация общих данных в каждом процессе."""
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

        # Анализ изображения
        c_res, c_diams, _ = analyze_frame(global_color, cfg, vis_cfg)

        # Метрики
        MIN_COUNT = 150
        BIG_THRESHOLD = 40.0
        SMALL_THRESHOLD = 15.0  # условный порог для маленьких

        total_count = len(c_diams)
        big_count = sum(1 for d in c_diams if d > BIG_THRESHOLD)
        small_count = sum(1 for d in c_diams if d < SMALL_THRESHOLD)
        mean_diameter = float(np.mean(c_diams)) if c_diams else 0.0
        median_diameter = float(np.median(c_diams)) if c_diams else 0.0

        # Фильтрация неудачных конфигураций
        if total_count < MIN_COUNT or not (2 <= big_count <= 3):
            return None

        # Сохраняем изображение
        parts = [f"{k[:2]}-{str(v).replace(' ', '')}" for k, v in override.items()]
        name = f"color_{'_'.join(parts)}_frame-{frame_number}.png"
        save_path = os.path.join(OUTPUT_DIR, name)
        cv2.imwrite(save_path, c_res)

        return {
            "path": save_path,
            "params": override,
            "total_count": total_count,
            "big_count": big_count,
            "small_count": small_count,
            "mean_diameter": mean_diameter,
            "median_diameter": median_diameter,
        }

    except Exception as e:
        print(f"[PID {os.getpid()}] ❌ Ошибка в конфигурации {override}: {e}")
        return None



def analyze_results(results: List[Dict[str, Any]], output_dir: str):
    """Анализирует результаты после параллельного запуска и строит графики."""

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

    # Построение графиков
    print("\n📈 Генерация графиков по параметрам...")
    num_params = len(param_stats)
    fig, axes = plt.subplots(nrows=num_params, ncols=1, figsize=(10, 5 * num_params))

    if num_params == 1:
        axes = [axes]

    for ax, (param, val_counts) in zip(axes, param_stats.items()):
        items = sorted(val_counts.items(), key=lambda x: str(x[0]))
        labels = [str(k) for k, _ in items]
        values = [v for _, v in items]

        sns.barplot(x=labels, y=values, ax=ax)
        ax.set_title(f"Распределение параметра: {param}")
        ax.set_ylabel("Количество успешных конфигураций")
        ax.set_xlabel(param)
        ax.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "param_distribution.png")
    plt.savefig(plot_path)
    print(f"📊 Плот сохранён: {plot_path}")


def main():
    base_cfg = ColorConfig()
    base_cfg_dict = vars(base_cfg)

    param_tests = {
        "median_blur_size":           [3, 5, 7],
        "adaptive_thresh":            [True],
        "adaptive_block_size":        [5, 7, 9, 11, 13, 15],
        "adaptive_C":                 [-2, 0, 2, 3],
        "use_otsu":                   [False],
        "binary_thresh":              [0],
        "morph_kernel_shape":         ["ellipse"],
        "morph_kernel_size":          [2, 3, 5],
        "morph_iterations":           [1, 2, 3],
        "min_particle_size":          [10],
        "max_particle_size":          [150],
        "distance_transform_mask":    [3, 5],
        "foreground_threshold_ratio": [0.1, 0.16, 0.2],
        "dilate_iterations":          [1],
    }

    keys = list(param_tests.keys())
    value_lists = [param_tests[k] for k in keys]

    tasks = [
        (dict(zip(keys, combo)), base_cfg_dict, FRAME_NUMBER)
        for combo in itertools.product(*value_lists)
    ]

    print(f"\n=== 🚀 Запуск {len(tasks)} задач на {cpu_count()} ядрах ===\n")

    ctx = get_context("spawn")
    with ctx.Pool(cpu_count(), initializer=init_worker_shared_data, initargs=(FRAME_NUMBER,)) as pool:
        results = pool.map(worker, tasks)

    filtered = [r for r in results if r]

    print(f"\n=== ✅ Успешных конфигураций: {len(filtered)} ===")
    for r in filtered:
        print(f"✔ {r['path']} — всего: {r['total_count']}, больших: {r['big_count']}")
        # Можно показать конфиг, если надо:
        cfg = ColorConfig(**{**vars(ColorConfig()), **r["params"]})
        print(cfg)
        print()

    # Анализируем и строим графики по результатам
    analyze_results(filtered, OUTPUT_DIR)


if __name__ == "__main__":
    main()
