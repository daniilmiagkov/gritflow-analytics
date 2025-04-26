import cv2
import numpy as np
import matplotlib.pyplot as plt
from processing.config import (
    DEPTH_MIN, DEPTH_MAX, SLOPE_CORRECTION, 
    USE_COLOR_SEGMENTATION, COLOR_WEIGHT, DEPTH_WEIGHT,
    ADAPTIVE_THRESH, BINARY_THRESH,
    MORPH_KERNEL_SIZE, MORPH_ITERATIONS,
    DISTANCE_TRANSFORM_MASK, FOREGROUND_THRESHOLD_RATIO, DILATE_ITERATIONS,
    MIN_PARTICLE_SIZE, MAX_PARTICLE_SIZE,
    BBOX_COLOR, BBOX_THICKNESS, FONT_SCALE, FONT_THICKNESS,
    SHOW_PLOTS, PLOT_FIGSIZE
)

def correct_slope(depth_map: np.ndarray) -> np.ndarray:
    """
    Fit a plane to the depth_map and subtract it to level the scene.
    Vectorized per-pixel correction via meshgrid and SVD on downsampled points.
    """
    h, w = depth_map.shape
    # Downscale factor based on memory footprint
    scale = 0.1 if depth_map.nbytes > 100e6 else 0.25

    small = cv2.resize(depth_map, (0, 0), fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    mask = small > DEPTH_MIN
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        return depth_map  # Not enough valid points

    zs = small[ys, xs]
    xs_full = xs / scale
    ys_full = ys / scale
    pts = np.stack([xs_full, ys_full, zs], axis=1)

    # Compute best-fit plane via SVD (normal = last row of Vt) :contentReference[oaicite:7]{index=7}
    centroid = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = Vt[2, :]

    # Create correction plane Z(x,y) = ...
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z_plane = ((normal[0]*(X - centroid[0]) +
                normal[1]*(Y - centroid[1])) / normal[2]) + centroid[2]

    # Subtract plane from depth, clip to valid range 
    corrected = np.where(depth_map > DEPTH_MIN,
                         depth_map - Z_plane,
                         depth_map)
    return np.clip(corrected, DEPTH_MIN, DEPTH_MAX).astype(np.float32)


def combined_segmentation(color_img: np.ndarray,
                          depth_map: np.ndarray) -> np.ndarray:
    """
    Combine grayscale histogram-equalized RGB and normalized depth
    into a single 8-bit image for thresholding.
    """
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    depth_norm = cv2.normalize(depth_map.astype(np.float32),
                               None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if USE_COLOR_SEGMENTATION:
        combined = cv2.addWeighted(gray.astype(np.float32), COLOR_WEIGHT,
                                   depth_norm.astype(np.float32), DEPTH_WEIGHT, 0)
        combined = combined.astype(np.uint8)
    else:
        combined = depth_norm

    # Adaptive or Otsu/given threshold :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
    if ADAPTIVE_THRESH:
        return cv2.adaptiveThreshold(combined, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    else:
        flag = cv2.THRESH_BINARY
        if BINARY_THRESH == 0:
            flag |= cv2.THRESH_OTSU
        _, binarized = cv2.threshold(combined, BINARY_THRESH, 255, flag)
        return binarized


def analyze_frame(color_img: np.ndarray,
                  depth_map: np.ndarray,
                  show_plots: bool = SHOW_PLOTS):
    """
    Full pipeline:
      1) Optional slope correction
      2) Combined RGB-D segmentation
      3) Morphology
      4) Marker computation
      5) Watershed
      6) Extract and annotate particles
    """
    # Ensure correct dtypes
    if color_img.dtype != np.uint8 and color_img.max() <= 1.0:
        color_img = (color_img * 255).astype(np.uint8)
    if depth_map.dtype != np.float32:
        depth_map = depth_map.astype(np.float32)

    # 1. Correct scene slope :contentReference[oaicite:10]{index=10}
    if SLOPE_CORRECTION:
        depth_map = correct_slope(depth_map)

    # 2. Segmentation mask
    mask = combined_segmentation(color_img, depth_map)

    # 3. Morphological opening + erosion
    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                              iterations=MORPH_ITERATIONS)
    opened = cv2.erode(opened, np.ones((2, 2), np.uint8), iterations=1)

    # 4. Compute markers via distance transform :contentReference[oaicite:11]{index=11}
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, DISTANCE_TRANSFORM_MASK)
    _, fg = cv2.threshold(dist,
                          FOREGROUND_THRESHOLD_RATIO * dist.max(),
                          255, 0)
    fg = fg.astype(np.uint8)
    bg = cv2.dilate(opened, kernel, iterations=DILATE_ITERATIONS)
    unknown = cv2.subtract(bg, fg)

    # 5. Watershed on grayscale version :contentReference[oaicite:12]{index=12}
    _, markers = cv2.connectedComponents(fg)
    markers = (markers + 1).astype(np.int32)
    markers[unknown == 255] = 0

    gray3 = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.equalizeHist(gray3)
    markers = cv2.watershed(cv2.merge([gray3] * 3), markers)

    # 6. Extract stats and annotate :contentReference[oaicite:13]{index=13}
    segmented = (markers > 1).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(segmented, connectivity=8)

    result = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    diameters = []
    for stat in stats[1:]:  # skip background
        area = stat[cv2.CC_STAT_AREA]
        diameter = 2 * np.sqrt(area / np.pi)
        if MIN_PARTICLE_SIZE <= diameter <= MAX_PARTICLE_SIZE:
            diameters.append(diameter)
            x, y, w, h = stat[:4]
            cv2.rectangle(result, (x, y), (x + w, y + h),
                          BBOX_COLOR, BBOX_THICKNESS)
            cv2.putText(result, f"{diameter:.1f}px", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                        BBOX_COLOR, FONT_THICKNESS)

    # Debug plots
    if show_plots:
        plt.figure(figsize=PLOT_FIGSIZE)
        plt.subplot(2, 3, 1); plt.title("Depth Corrected")
        plt.imshow(depth_map, cmap='inferno'); plt.axis('off')
        plt.subplot(2, 3, 2); plt.title("Combined Mask")
        plt.imshow(mask, cmap='gray'); plt.axis('off')
        plt.subplot(2, 3, 3); plt.title("Opened")
        plt.imshow(opened, cmap='gray'); plt.axis('off')
        plt.subplot(2, 3, 4); plt.title("Distance")
        plt.imshow(dist, cmap='jet'); plt.axis('off')
        plt.subplot(2, 3, 5); plt.title("Markers")
        plt.imshow(markers, cmap='tab20'); plt.axis('off')
        plt.subplot(2, 3, 6); plt.title("Result")
        plt.imshow(result); plt.axis('off')
        plt.tight_layout()
        plt.show()

    return result, diameters or "No particles detected"
