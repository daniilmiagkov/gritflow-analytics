import pyzed.sl as sl
import os

def init_camera(svo_path):
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init_params.depth_minimum_distance = 300
    init_params.depth_maximum_distance = 2500
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.depth_stabilization = True

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Не удалось открыть SVO-файл")
    
    return zed

def grab_frame(zed, frame_number):
    # Настройки параметров обработки из скриншота
    runtime_params = sl.RuntimeParameters()
    runtime_params.remove_saturated_areas = True  # Удаление насыщенных областей
    runtime_params.confidence_threshold = 95      # Порог доверия 95%
    runtime_params.texture_confidence_threshold = 100  # Текстурное доверие 100%

    # Установка позиции кадра
    zed.set_svo_position(frame_number)

    if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
        return None, None, None, None

    # Инициализация объектов данных
    left_image = sl.Mat()
    depth_measure = sl.Mat()
    depth_shading = sl.Mat()  # Для Depth Shading
    point_cloud = sl.Mat()

    # Получение данных
    zed.retrieve_image(left_image, sl.VIEW.LEFT)        # Левый RGB кадр
    zed.retrieve_measure(depth_measure, sl.MEASURE.DEPTH)  # Сырые данные глубины
    zed.retrieve_image(depth_shading, sl.VIEW.DEPTH)    # Depth Shading (цветная визуализация)
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)  # Облако точек

    return left_image, depth_measure, depth_shading, point_cloud