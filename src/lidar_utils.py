import open3d as o3d
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt

def visualize_point_cloud(points_1, points_2 = None, bounding_boxes = None):
    geometries = []
    pcd_1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_1))
    if points_2 is not None:
        pcd_2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_2))
        pcd_1.paint_uniform_color([0.8, 0.2, 0.2])      # red for first
        pcd_2.paint_uniform_color([0.2, 0.8, 0.2])      # green for second
        geometries.append(pcd_2)
    elif bounding_boxes:
        for box in bounding_boxes:
            geometries.append(box)
    geometries.append(pcd_1)
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
    o3d.visualization.draw_geometries(geometries)

def visualize_point_cloud_clusters(clusters_points):
    geometries = []
    cmap = plt.get_cmap("tab20")
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
    for i, (label, pts) in enumerate(clusters_points.items()):
        if label < 0:
            color = [0, 0, 0]   # black for noise
        else:
            color = cmap(label % 20)[:3]
        cluster_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(pts)))
        cluster_pcd.paint_uniform_color(color)
        geometries.append(cluster_pcd)
    o3d.visualization.draw_geometries(geometries)

def apply_roi_filter(points, roi):
    x_range, y_range, z_range = roi['x'], roi['y'], roi['z']
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
    )
    return points[mask]

def downsample(points, voxel_size):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return pcd.voxel_down_sample(voxel_size).points

def remove_ground_ransac(points, max_distance_threshold):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    plane_model, inliers = pcd.segment_plane(distance_threshold=max_distance_threshold, ransac_n=3, num_iterations=100)    # max distance, n=3 for a plane
    pcd_non_ground = pcd.select_by_index(inliers, invert=True)
    # remove points below the ground
    [a, b, c, d] = plane_model
    points = np.asarray(pcd_non_ground.points)
    distances = a*points[:,0] + b*points[:,1] + c*points[:,2] + d       # compute distances to plane
    return points[distances > 0.05]     # 0.05 meters tolerance

def dbscan_clustering(points, neighborhood_radius, min_cluster_size):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))   # estimate normals (optional, can help clustering in Open3D)
    labels = np.array(pcd.cluster_dbscan(eps=neighborhood_radius, min_points=min_cluster_size, print_progress=True))     # possibility for an adaptive approach
    cluster_points = defaultdict(list)
    for point, label in zip(np.asarray(pcd.points), labels):
        if label == -1:  # skip non-labeled
            continue
        cluster_points[label].append(point)
    return cluster_points

def clusters_bounding_boxes(clusters_points, oriented=False):
    bounding_boxes = []
    for cluster_id, pts in clusters_points.items():
        pts_np = np.array(pts)
        min_bound = pts_np.min(axis=0)
        max_bound = pts_np.max(axis=0)
        if oriented:
            if len(pts_np) >= 4:
                box = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts_np))   # tigher outline
            else:
                continue
        else:
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        box.color = (1, 0, 0)
        bounding_boxes.append((box, cluster_id))
    return bounding_boxes

def lidar_to_camera(points_lidar, tf):
    pts = (tf['R_lsensor'] @ points_lidar.T).T + tf['T_lsensor']    # lidar → ego (lidar)
    pts = (tf['R_lego'] @ pts.T).T + tf['T_lego']                   # ego (lidar) → global
    pts = (tf['R_cego'].T @ (pts - tf['T_cego']).T).T               # global → ego (camera)
    pts = (tf['R_csensor'].T @ (pts - tf['T_csensor']).T).T         # ego (camera) → camera
    return pts

def project_to_camera(points_cam, tf):
    mask = points_cam[:,2] > 0                  # discard points behind camera (z <= 0)
    pts = points_cam[mask]
    pts_h = (tf['K'] @ pts.T).T   # (M,3)
    pts_2d = pts_h[:, :2] / pts_h[:, 2:3]       # pinhole camera model (divice X and Y by Z) to get the prespective projection
    return pts_2d

def lidar_bounding_boxes_to_camera(clusters_bbs, transformations):
    bbs_in_image = []
    for bb, cluster_id in clusters_bbs:
        corners = np.asarray(bb.get_box_points())   # (8,3)
        cam_pts = lidar_to_camera(corners, transformations)
        img_pts = project_to_camera(cam_pts, transformations)
        if img_pts.shape[0] >= 4:       # at least 4 corners are visible
            x_min, y_min = img_pts.min(axis=0).astype(int)
            x_max, y_max = img_pts.max(axis=0).astype(int)
            bbs_in_image.append(([x_min, y_min, x_max, y_max], cluster_id))
    return bbs_in_image
