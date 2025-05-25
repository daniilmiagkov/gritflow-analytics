from dataclasses import dataclass
from typing import Tuple

@dataclass
class ColorConfig:
    # --- Новое поле для медианной фильтрации ---
    median_blur_size: int = 3       
    adaptive_thresh: bool = True
    binary_thresh: int = 100

    morph_kernel_size: int = 2
    morph_iterations: int = 3

    distance_transform_mask: int = 1
    foreground_threshold_ratio: float = 0.5
    dilate_iterations: int = 3

    min_particle_size: float = 3
    max_particle_size: float = 100

    bbox_color: Tuple[int, int, int] = (0, 255, 0)
    bbox_thickness: int = 1
    font_scale: float = 0.5
    font_thickness: int = 0
