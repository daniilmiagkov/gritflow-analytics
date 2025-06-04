import pyzed.sl as sl
import os
import config

def init_camera(svo_path):
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
    init_params.depth_minimum_distance = float(config.DEPTH_MIN)  # минимум из конфига
    init_params.depth_maximum_distance = float(config.DEPTH_MAX)  # максимум из конфига
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.depth_stabilization = False

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError("Не удалось открыть SVO-файл")
    
    return zed

def grab_frame(zed, frame_number):
    runtime_params = sl.RuntimeParameters()
    runtime_params.remove_saturated_areas = True
    runtime_params.confidence_threshold = 95
    runtime_params.texture_confidence_threshold = 100

    zed.set_svo_position(frame_number)

    if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
        return None, None, None, None

    left_image = sl.Mat()
    depth_measure = sl.Mat()
    depth_shading = sl.Mat()
    point_cloud = sl.Mat()

    zed.retrieve_image(left_image, sl.VIEW.LEFT)
    zed.retrieve_measure(depth_measure, sl.MEASURE.DEPTH)
    zed.retrieve_image(depth_shading, sl.VIEW.DEPTH)
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

    return left_image, depth_measure, depth_shading, point_cloud

def get_intrinsics_from_svo(svo_path: str):
    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.svo_real_time_mode = False

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Ошибка при открытии SVO: {status}")
        return

    calibration = zed.get_camera_information().camera_configuration.calibration_parameters

    left_cam = calibration.left_cam
    fx = left_cam.fx
    fy = left_cam.fy
    cx = left_cam.cx
    cy = left_cam.cy

    print(f"fx: {fx}, fy: {fy}")
    print(f"cx: {cx}, cy: {cy}")
    print(f"Resolution: {left_cam.image_size.width}x{left_cam.image_size.height}")

    zed.close()
    return fx, fy, cx, cy