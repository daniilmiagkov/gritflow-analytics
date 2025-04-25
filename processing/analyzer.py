import cv2
import numpy as np

def analyze_frame(image_np, depth_np):
    # 1. Очистка глубины (убираем нули и шум)
    depth_clean = np.where((depth_np > 300) & (depth_np < 2500), depth_np, 0).astype(np.float32)
    depth_norm = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_blur = cv2.medianBlur(depth_norm, 5)

    # 2. Простая бинаризация по глубине
    _, depth_bin = cv2.threshold(depth_blur, 30, 255, cv2.THRESH_BINARY)

    # 3. Морфология для очистки маски
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(depth_bin, cv2.MORPH_OPEN, kernel, iterations=2)

    # 4. Поиск маркеров для watershed
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    fg = fg.astype(np.uint8)
    bg = cv2.dilate(opening, kernel, iterations=3)
    unknown = cv2.subtract(bg, fg)
    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 5. Watershed
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(img_bgr, markers)
    img_bgr[markers == -1] = [0, 0, 255]

    # 6. Измерение по компонентам
    segmented_mask = (markers > 1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmented_mask)

    # 7. Диаметры по площади (без фона)
    diameters = []
    for stat in stats[1:]:  # пропускаем фон
        area = stat[cv2.CC_STAT_AREA]
        diameter = 2 * np.sqrt(area / np.pi)
        diameters.append(diameter)

    return img_bgr, diameters
