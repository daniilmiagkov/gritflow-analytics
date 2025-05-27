from dataclasses import dataclass
from typing import Tuple

@dataclass
class ColorConfig:
    median_blur_size: int = 3
    adaptive_thresh: bool = True
    adaptive_block_size: int = 3
    adaptive_C: int = -2
    binary_thresh: int = 0
    use_otsu: bool = False
    morph_kernel_size: int = 2
    morph_iterations: int = 1
    morph_kernel_shape: str = 'ellipse'
    min_contour_area: int = 15
    min_particle_size: int = 5
    max_particle_size: int = 150
    bbox_color: Tuple[int, int, int] = (0, 255, 0)
    bbox_thickness: int = 1
    font_scale: float = 0.5
    font_thickness: int = 1
    distance_transform_mask: int = 3
    foreground_threshold_ratio: float = 0.2
    dilate_iterations: int = 1
    invert: bool = False
    equalize_hist: bool = False 
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)


# from dataclasses import dataclass
# from typing import Tuple

# @dataclass
# class ColorConfig:
#     # === Предобработка ===
#     median_blur_size: int = 3  # достаточно слабый не размоет мелкие детали

#     # === Пороговая сегментация ===
#     adaptive_thresh: bool = True
#     adaptive_block_size: int = 13  # побольше блок чтобы лучше различать границы у больших объектов
#     adaptive_C: int = 2  # чуть ниже — снижает шанс "перепороговать"
#     binary_thresh: int = 0
#     use_otsu: bool = False  # Отсу не нужен если используем adaptive

#     # === Морфологическая обработка ===
#     morph_kernel_size: int = 3  # небольшой — не съест мелкие детали
#     morph_iterations: int = 2  # чуть больше чтобы разорвать слипшиеся кластеры мелких
#     morph_kernel_shape: str = "ellipse"  # более мягкое воздействие чем "rect"

#     # === Фильтрация и анализ контуров ===
#     min_contour_area: float = 15  # чуть выше чтобы отфильтровывать шум и пыль
#     min_particle_size: float = 5  # всё ещё допускаем мелкие
#     max_particle_size: float = 200  # увеличиваем чтобы не отсекать крупные камни

#     # === Визуализация ===
#     bbox_color: Tuple[int int int] = (0 255 0)
#     bbox_thickness: int = 1
#     font_scale: float = 0.5
#     font_thickness: int = 1

#     # === Доп. сегментация (distance transform) ===
#     distance_transform_mask: int = 1
#     foreground_threshold_ratio: float = 0.16  # немного выше — поможет разбить слитые камни
#     dilate_iterations: int = 1  # меньше дилатации чтобы не соединялись случайно в один объект
