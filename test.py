import os
import sys
import pyzed.sl as sl
import numpy as np
import cv2

def main():
    # 1. Путь к SVO-файлу — используем raw string
    svo_path = r"C:\Downloads\Telegram Desktop\HD720_SN34708318_11-30-24.svo"

    # 2. Папка для вывода — должна существовать и быть доступной
    output_dir = r"C:\files\study\suai\diploma"
    os.makedirs(output_dir, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        print(f"Ошибка: нет прав на запись в {output_dir}")
        sys.exit(1)

    # 3. Инициализация ZED для чтения SVO
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL   # рекомендуемый режим
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Не удалось открыть SVO-файл")
        sys.exit(1)

    # 4. Захват и получение данных
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        img = sl.Mat()
        depth = sl.Mat()
        zed.retrieve_image(img, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        # Конвертация в NumPy
        image_np = img.get_data()
        depth_np = depth.get_data()

        # 5. Сохранение в доступную директорию
        img_path   = os.path.join(output_dir, "image.png")
        depth_path = os.path.join(output_dir, "depth.png")

        # Сохраняем цветное изображение
        if not cv2.imwrite(img_path, image_np):
            print(f"Не удалось сохранить {img_path}")
        # Нормализуем и сохраняем карту глубины
        depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX)
        if not cv2.imwrite(depth_path, depth_norm.astype(np.uint8)):
            print(f"Не удалось сохранить {depth_path}")

        print("Успешно сохранено:", img_path, depth_path)
    else:
        print("Не удалось захватить кадр")

    zed.close()

if __name__ == "__main__":
    main()
