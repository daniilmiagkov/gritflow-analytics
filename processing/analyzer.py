import cv2
import numpy as np
import matplotlib.pyplot as plt
from processing.config import *

def analyze_frame(image_np, depth_np, show_plots=SHOW_PLOTS):
    plt.figure(figsize=PLOT_FIGSIZE)
    
    # 1. Очистка глубины
    depth_clean = np.where((depth_np > DEPTH_MIN) & (depth_np < DEPTH_MAX), depth_np, 0).astype(np.float32)
    depth_norm = cv2.normalize(depth_clean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_blur = cv2.medianBlur(depth_norm, MEDIAN_BLUR_SIZE)
    
    if show_plots:
        plt.subplot(3, 4, 1)
        plt.imshow(depth_clean, cmap='jet')
        plt.title('1. Очищенная глубина')
        plt.colorbar()
        
        plt.subplot(3, 4, 2)
        plt.imshow(depth_blur, cmap='gray')
        plt.title('2. После медианного фильтра')

    # 2. Бинаризация
    _, depth_bin = cv2.threshold(depth_blur, BINARY_THRESH, 255, cv2.THRESH_BINARY)
    
    if show_plots:
        plt.subplot(3, 4, 3)
        plt.imshow(depth_bin, cmap='gray')
        plt.title('3. Бинаризация')

    # 3. Морфология
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    opening = cv2.morphologyEx(depth_bin, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS)
    
    if show_plots:
        plt.subplot(3, 4, 4)
        plt.imshow(opening, cmap='gray')
        plt.title('4. После морфологии')

    # 4. Поиск маркеров
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, DISTANCE_TRANSFORM_MASK)
    _, fg = cv2.threshold(dist, FOREGROUND_THRESHOLD_RATIO * dist.max(), 255, 0)
    fg = fg.astype(np.uint8)
    bg = cv2.dilate(opening, kernel, iterations=DILATE_ITERATIONS)
    unknown = cv2.subtract(bg, fg)
    
    if show_plots:
        plt.subplot(3, 4, 5)
        plt.imshow(dist, cmap='jet')
        plt.title('5. Distance Transform')
        
        plt.subplot(3, 4, 6)
        plt.imshow(fg, cmap='gray')
        plt.title('6. Передний план (FG)')
        
        plt.subplot(3, 4, 7)
        plt.imshow(bg, cmap='gray')
        plt.title('7. Задний план (BG)')

    # 5. Watershed
    _, markers = cv2.connectedComponents(fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    markers = cv2.watershed(img_bgr, markers)
    img_bgr[markers == -1] = BORDER_COLOR  # Границы
    
    if show_plots:
        plt.subplot(3, 4, 8)
        plt.imshow(markers, cmap='nipy_spectral')
        plt.title('8. Маркеры для watershed')
        
        plt.subplot(3, 4, 9)
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.title('9. Результат watershed')

    # 6. Анализ компонент
    segmented_mask = (markers > 1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segmented_mask)
    
    # 7. Расчет диаметров
    diameters = []
    for stat in stats[1:]:
        area = stat[cv2.CC_STAT_AREA]
        diameter = 2 * np.sqrt(area / np.pi)
        diameters.append(diameter)
    
    # Визуализация результатов
    result_img = img_bgr.copy()
    for i, stat in enumerate(stats[1:]):
        x, y, w, h, area = stat[:5]
        cv2.rectangle(result_img, (x, y), (x+w, y+h), BBOX_COLOR, BBOX_THICKNESS)
        cv2.putText(result_img, f"{diameters[i]:.1f}", (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, BBOX_COLOR, FONT_THICKNESS)
    
    if show_plots:
        plt.subplot(3, 4, 10)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title('10. Результат с размерами')
        
        plt.subplot(3, 4, 11)
        plt.hist(diameters, bins=HIST_BINS)
        plt.title('11. Распределение диаметров')
        plt.xlabel('Диаметр (пиксели)')
        plt.ylabel('Количество')
        
        plt.tight_layout()
        plt.show()

    return result_img, diameters