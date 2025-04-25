import os
import sys
import pyzed.sl as sl
import numpy as np
import cv2

def main():
    svo_path = r"C:\Downloads\Telegram Desktop\HD720_SN34708318_11-30-24.svo"
    output_dir = r"C:\files\study\suai\diploma"
    target_frame_number = 1874  # Укажите здесь нужный номер кадра
    
    os.makedirs(output_dir, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        print(f"Ошибка: нет прав на запись в {output_dir}")
        sys.exit(1)

    zed = sl.Camera()
    init_params = sl.InitParameters()
    
    # Настройки из изображения
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init_params.depth_minimum_distance = 300
    init_params.depth_maximum_distance = 2500
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.depth_stabilization = True
    
    runtime_params = sl.RuntimeParameters()
    runtime_params.remove_saturated_areas = True
    runtime_params.confidence_threshold = 100
    runtime_params.texture_confidence_threshold = 100

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Не удалось открыть SVO-файл")
        sys.exit(1)

    # Установка позиции на нужный кадр
    zed.set_svo_position(target_frame_number)

    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        img = sl.Mat()
        depth = sl.Mat()
        zed.retrieve_image(img, sl.VIEW.LEFT)
        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)

        image_np = img.get_data()
        depth_np = depth.get_data()

        # Добавляем номер кадра в имя файла
        img_path = os.path.join(output_dir, f"image_frame_{target_frame_number}.png")
        depth_path = os.path.join(output_dir, f"depth_frame_{target_frame_number}.png")

        # Сохраняем цветное изображение
        if not cv2.imwrite(img_path, image_np):
            print(f"Не удалось сохранить {img_path}")
        
        # Обработка карты глубины
        try:
            # Заменяем NaN/Inf значения на 0 перед нормализацией
            depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Нормализация только если есть валидные значения
            if np.any(depth_np > 0):
                depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                if not cv2.imwrite(depth_path, depth_norm):
                    print(f"Не удалось сохранить {depth_path}")
            else:
                print("Предупреждение: карта глубины не содержит валидных данных")
                cv2.imwrite(depth_path, np.zeros_like(depth_np, dtype=np.uint8))
        except Exception as e:
            print(f"Ошибка при обработке глубины: {str(e)}")

        print(f"Успешно сохранен кадр {target_frame_number}: {img_path}, {depth_path}")
    else:
        print(f"Не удалось захватить кадр {target_frame_number}")

    zed.close()

if __name__ == "__main__":
    main()