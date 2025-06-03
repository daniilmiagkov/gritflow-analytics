import os

# Путь к SVO
SVO_PATH = r"C:\Downloads\Telegram Desktop\HD720_SN34708318_11-30-24.svo"

# Директория вывода
OUTPUT_DIR = r"C:\files\study\suai\diploma\data\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Кадр
FRAME_NUMBER = 1874


# config.py

# Обрезка изображения
CROP = True
CROP_X = 100  # x-координата верхнего левого угла
CROP_Y = 200  # y-координата верхнего левого угла
CROP_WIDTH = 800
CROP_HEIGHT = 350
# Диапазон допустимых глубин в мм
DEPTH_MIN = 990
DEPTH_MAX = 2500
