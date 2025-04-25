import open3d as o3d
import numpy as np

def visualize_ply(ply_path):
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        print("Облако точек не содержит данных.")
        return

    pcd.transform([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="ZED Open3D Viewer", width=1280, height=720)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.2, 0.2, 0.2])
    opt.point_size = 1.5
    opt.show_coordinate_frame = True

    vis.run()
    vis.destroy_window()
