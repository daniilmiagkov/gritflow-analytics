def pixel_to_mm(d_px: float, depth_value: float, fx: float, fy: float) -> float:
    """
    Преобразует размер в пикселях (d_px) в миллиметры, используя глубину (depth_value)
    и фокусные расстояния камеры fx, fy в пикселях.
    """
    # Примерная оценка мм/пиксель по горизонтали и вертикали
    # z - глубина в миллиметрах
    z = depth_value
    if z <= 0:
        return 0.0  # некорректная глубина

    # мм/пиксель в x и y
    mm_per_px_x = z / fx
    mm_per_px_y = z / fy

    # Средний масштаб
    mm_per_px = (mm_per_px_x + mm_per_px_y) / 2.0
    return d_px * mm_per_px
