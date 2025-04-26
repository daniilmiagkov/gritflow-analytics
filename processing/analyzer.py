import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Union

from processing.depth_config import DepthConfig
from processing.color_config import ColorConfig
from processing.visual_config import VisualizationConfig

def correct_slope(depth_map: np.ndarray, cfg: DepthConfig) -> np.ndarray:
    h, w = depth_map.shape
    scale = 0.1 if depth_map.nbytes > 100e6 else 0.25
    small = cv2.resize(depth_map, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ys, xs = np.nonzero(small > cfg.depth_min)
    if len(xs) < 100:
        return depth_map
    zs = small[ys,xs]
    xs_f, ys_f = xs/scale, ys/scale
    pts = np.stack([xs_f, ys_f, zs], axis=1)
    centroid = pts.mean(axis=0)
    _,_,Vt = np.linalg.svd(pts-centroid, full_matrices=False)
    normal = Vt[2]
    X,Y = np.meshgrid(np.arange(w), np.arange(h))
    Zp = ((normal[0]*(X-centroid[0])+normal[1]*(Y-centroid[1]))/normal[2]) + centroid[2]
    corr = np.where(depth_map>cfg.depth_min, depth_map - Zp, depth_map)
    return np.clip(corr, cfg.depth_min, cfg.depth_max).astype(np.float32)

def segment_color(img: np.ndarray, cfg: ColorConfig) -> np.ndarray:
        # 0. Медианный фильтр для подавления шума :contentReference[oaicite:3]{index=3}
    if cfg.median_blur_size > 1:
        img = cv2.medianBlur(img, cfg.median_blur_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    if cfg.adaptive_thresh:
        return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,11,2)
    flag = cv2.THRESH_BINARY
    if cfg.binary_thresh==0: flag |= cv2.THRESH_OTSU
    _,b = cv2.threshold(gray, cfg.binary_thresh,255,flag)
    return b

def segment_depth(dm: np.ndarray, cfg: DepthConfig) -> np.ndarray:
        # 0. Медианный фильтр по глубине для сглаживания выбросов :contentReference[oaicite:4]{index=4}
    if cfg.median_blur_size > 1:
        dm = cv2.medianBlur(dm, cfg.median_blur_size)
    dn = cv2.normalize(dm,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    if cfg.adaptive_thresh:
        return cv2.adaptiveThreshold(dn,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY,11,2)
    flag = cv2.THRESH_BINARY
    if cfg.binary_thresh==0: flag |= cv2.THRESH_OTSU
    _,b = cv2.threshold(dn, cfg.binary_thresh,255,flag)
    return b

def process_mask(mask: np.ndarray, k:int, it:int) -> np.ndarray:
    ker = np.ones((k,k),np.uint8)
    op = cv2.morphologyEx(mask,cv2.MORPH_OPEN,ker,iterations=it)
    return cv2.erode(op, np.ones((2,2),np.uint8),iterations=1)

def analyze_mask(img: np.ndarray, mask: np.ndarray,
                 cfg: Union[ColorConfig,DepthConfig]) \
        -> Tuple[np.ndarray,List[float],List[float]]:
    dist = cv2.distanceTransform(mask,cv2.DIST_L2,cfg.distance_transform_mask)
    _,fg = cv2.threshold(dist, cfg.foreground_threshold_ratio*dist.max(),255,0)
    fg = fg.astype(np.uint8)
    bg = cv2.dilate(mask, np.ones((3,3),np.uint8),iterations=cfg.dilate_iterations)
    unk = cv2.subtract(bg,fg)
    _,m = cv2.connectedComponents(fg)
    m = (m+1).astype(np.int32); m[unk==255]=0
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    markers = cv2.watershed(cv2.merge([gray]*3), m)
    seg = (markers>1).astype(np.uint8)
    _,_,stats,_ = cv2.connectedComponentsWithStats(seg, connectivity=8)
    res = img.copy()
    diams,xs = [],[]
    for st in stats[1:]:
        area = st[cv2.CC_STAT_AREA]
        d = 2*np.sqrt(area/np.pi)
        if cfg.min_particle_size<=d<=cfg.max_particle_size:
            diams.append(d)
            x,y,w,h = st[:4]
            cx = x + w/2; xs.append(cx)
            cv2.rectangle(res,(x,y),(x+w,y+h),cfg.bbox_color,cfg.bbox_thickness)
            cv2.putText(res,f"{d:.1f}",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,
                        cfg.font_scale,cfg.bbox_color,cfg.font_thickness)
    return res,diams,xs

def analyze_frame(color_img: np.ndarray, depth_map: np.ndarray,
                  d_cfg: DepthConfig, c_cfg: ColorConfig,
                  v_cfg: VisualizationConfig):
    # types
    if color_img.dtype!=np.uint8 and color_img.max()<=1.0:
        color_img=(color_img*255).astype(np.uint8)
    if depth_map.dtype!=np.float32:
        depth_map=depth_map.astype(np.float32)
    # slope
    if d_cfg.slope_correction:
        depth_map = correct_slope(depth_map,d_cfg)
    # color pipeline
    cm = segment_color(color_img,c_cfg)
    cp = process_mask(cm,c_cfg.morph_kernel_size,c_cfg.morph_iterations)
    cres, cdi, cxs = analyze_mask(color_img,cp,c_cfg)
    # depth pipeline
    dm = segment_depth(depth_map,d_cfg)
    dp = process_mask(dm,d_cfg.morph_kernel_size,d_cfg.morph_iterations)
    dres, ddi, dxs = analyze_mask(color_img,dp,d_cfg)
    # show
    if v_cfg.show_plots:
        plt.figure(figsize=v_cfg.plot_figsize)
        titles=["Color Mask","Depth Mask","Color Proc","Depth Proc","Color Res","Depth Res"]
        imgs=[cm,dm,cp,dp,cres,dres]
        for i,(t,im) in enumerate(zip(titles,imgs),1):
            plt.subplot(2,3,i); plt.title(t)
            plt.imshow(im,cmap='gray' if im.ndim==2 else None); plt.axis('off')
        plt.tight_layout(); plt.show()
    return (cres,cdi,cxs),(dres,ddi,dxs)
