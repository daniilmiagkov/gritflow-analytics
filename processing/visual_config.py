from dataclasses import dataclass
from typing import Tuple

@dataclass
class VisualizationConfig:
    show_plots: bool = True
    plot_figsize: Tuple[int, int] = (15, 10)
