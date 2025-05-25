from dataclasses import dataclass
from typing import Tuple

@dataclass
class ColorConfig:
    # === Предобработка ===
    median_blur_size: int = 3

    # === Бинаризация ===
    adaptive_thresh: bool = True
    adaptive_block_size: int = 25
    adaptive_C: int = 5
    binary_thresh: int = 0  # если 0 — применится Otsu
    use_otsu: bool = True

    # === Морфология ===
    morph_kernel_size: int = 3
    morph_iterations: int = 1
    morph_kernel_shape: str = "rect"  # "rect", "ellipse", "cross"

    # === Фильтрация по размерам частиц ===
    min_particle_size: float = 5.0
    max_particle_size: float = 1000.0
    min_contour_area: float = 20.0

    # === Визуализация ===
    bbox_color: Tuple[int, int, int] = (0, 255, 0)
    bbox_thickness: int = 2
    font_scale: float = 0.5
    font_thickness: int = 1