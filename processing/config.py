# Параметры обработки глубины
DEPTH_MIN = 300        # Минимальная глубина в мм
DEPTH_MAX = 2500       # Максимальная глубина в мм

# Параметры фильтрации
MEDIAN_BLUR_SIZE = 5   # Размер ядра медианного фильтра
BINARY_THRESH = 30     # Порог бинаризации (0-255)

# Параметры морфологических операций
MORPH_KERNEL_SIZE = 3  # Размер ядра морфологических операций
MORPH_ITERATIONS = 2   # Количество итераций

# Параметры watershed
DISTANCE_TRANSFORM_MASK = 5       # Размер маски для distance transform
FOREGROUND_THRESHOLD_RATIO = 0.5  # Порог для выделения переднего плана
DILATE_ITERATIONS = 3             # Итерации расширения для фона

# Параметры визуализации
SHOW_PLOTS = True                 # Показывать промежуточные графики
PLOT_FIGSIZE = (15, 10)           # Размер фигуры для графиков
BORDER_COLOR = [0, 0, 255]        # Цвет границ (BGR)
BBOX_COLOR = (0, 255, 0)          # Цвет bounding box (BGR)
BBOX_THICKNESS = 1                # Толщина линий bounding box
FONT_SCALE = 0.3                  # Масштаб текста
FONT_THICKNESS = 1                # Толщина текста
HIST_BINS = 20                    # Количество бинов гистограммы