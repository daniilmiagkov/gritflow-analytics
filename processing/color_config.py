from dataclasses import dataclass
from typing import Tuple

@dataclass
class ColorConfig:
    # --- Новое поле для медианной фильтрации ---
    median_blur_size: int = 3          # Нечётный размер ядра для cv2.medianBlur :contentReference[oaicite:1]{index=1}

    adaptive_thresh: bool = True
    binary_thresh: int = 10

    morph_kernel_size: int = 2
    morph_iterations: int = 1

    distance_transform_mask: int = 3
    foreground_threshold_ratio: float = 0.04
    dilate_iterations: int = 2

    min_particle_size: float = 4
    max_particle_size: float = 65

    bbox_color: Tuple[int, int, int] = (0, 255, 0)
    bbox_thickness: int = 2
    font_scale: float = 0.5
    font_thickness: int = 1
