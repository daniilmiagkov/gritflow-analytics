from dataclasses import dataclass
from typing import Tuple

@dataclass
class DepthConfig:
    # --- Новое поле для медианной фильтрации ---
    median_blur_size: int = 5          # Размер ядра медианного фильтра (должен быть нечётным) :contentReference[oaicite:0]{index=0}

    depth_min: float = 300
    depth_max: float = 1500
    slope_correction: bool = False

    adaptive_thresh: bool = True
    binary_thresh: int = 0

    morph_kernel_size: int = 5
    morph_iterations: int = 1

    distance_transform_mask: int = 5
    foreground_threshold_ratio: float = 0.04
    dilate_iterations: int = 2

    min_particle_size: float = 20
    max_particle_size: float = 100

    bbox_color: Tuple[int, int, int] = (0, 0, 255)
    bbox_thickness: int = 2
    font_scale: float = 0.5
    font_thickness: int = 1
