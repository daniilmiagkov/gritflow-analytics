import cv2
import os

def save_image(mat, filename):
    cv2.imwrite(filename, mat.get_data())

def save_point_cloud(point_cloud, output_path):
    point_cloud.write(output_path)
