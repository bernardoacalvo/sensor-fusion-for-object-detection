from nuscenes_loader import *
from lidar_utils import *
from image_utils import *
from image_object_detection import *
from fusion import *
from ttc import *
from tqdm import tqdm
import os

NUSCENES_VERSION = "v1.0-mini"      # name of dataset
NUSCENES_PATH = "/Users/bernardocalvo/Programming/Projects/obstacle-detection/datasets/nuscenes/v1.0-mini"

# ROI in meters. Driving scenario -> forward-looking
LIDAR_ROI = {
    "x": (-6, 6),     # sides
    "y": (0, 50),     # forward
    "z": (-2, 2),     # road surface, typical vehicles heights, excluding buildings, trees
}

DOWNSAMPLING_VOXEL_SIZE = 0.2
MAXIMUM_DISTANCE_RANSCAN = 0.1
DBSCAN_EPS = 0.4
DBSCAN_MIN_SIZE = 10
FUSION_IOU_THRESHOLD = 0.2

def compute_detections(i, data_loader, od_model):
    sample = data_loader[i]
    pcd_multisweep_points = data_loader.get_multisweep_points(sample, 10)
    #visualize_point_cloud(pcd_multisweep_points)

    pcd_roi_points = apply_roi_filter(pcd_multisweep_points, LIDAR_ROI)
    #visualize_point_cloud(pcd_roi_points, pcd_multisweep_points)

    pcd_downsample_points = downsample(pcd_roi_points, DOWNSAMPLING_VOXEL_SIZE)
    print(f"Downsampled from {len(pcd_roi_points)} points to {len(pcd_downsample_points)}.")
    #visualize_point_cloud(pcd_downsample_points)

    pcd_non_ground_points = remove_ground_ransac(pcd_downsample_points, MAXIMUM_DISTANCE_RANSCAN)
    #visualize_point_cloud(pcd_non_ground_points)

    clusters_points = dbscan_clustering(pcd_non_ground_points, DBSCAN_EPS, DBSCAN_MIN_SIZE)
    print(f"Detected {len(clusters_points)} clusters.")
    #visualize_point_cloud_clusters(clusters_points)

    filtered_clusters_points = data_loader.filter_clusters_on_road(sample, clusters_points)
    #visualize_point_cloud_clusters(filtered_clusters_points)

    clusters_bbs = clusters_bounding_boxes(filtered_clusters_points, False)
    #visualize_point_cloud(points_1=pcd_non_ground_points, bounding_boxes=[box for box, _ in clusters_bbs])

    transformations = data_loader.get_camera_to_lidar_transformations(sample)
    lidar_bbs_in_image = lidar_bounding_boxes_to_camera(clusters_bbs, transformations)
    img_path = data_loader.get_image_path(sample)
    #visualize_detections(img_path, [{'bounding_box': bb} for bb, _ in lidar_bbs_in_image], 'LiDAR detections')

    od_detections = od_model.run(img_path)
    #visualize_detections(img_path, od_detections, 'YOLO detections')

    fusion_detections = fuse_detections(lidar_bbs_in_image, od_detections, FUSION_IOU_THRESHOLD)
    #visualize_detections(img_path, fusion_detections, 'Fusion detections')
    return fusion_detections, clusters_points


def run_detections_and_save(output_dir, data_loader, od_model):
    os.makedirs(output_dir, exist_ok=True)
    frame_ttcs = []
    centroids_prev = {}
    frame_paths = []

    for i in tqdm(range(len(data_loader))):
        # compute and get fusion detections
        fusion_detections, cluster_points = compute_detections(i, data_loader, od_model)
        matched_ids = {det['cluster_id'] for det in fusion_detections if 'cluster_id' in det}
        cluster_points_curr = {cid: cluster_points[cid] for cid in matched_ids if cid in cluster_points}

        if i > 0:
            centroids_curr = compute_cluster_centroids(cluster_points_curr)
            associations = associate_clusters(centroids_prev, centroids_curr)
            ttc_dict = compute_ttc(centroids_prev, centroids_curr, associations, data_loader.get_dt(i))

            centroids_prev = centroids_curr
            filtered_ttcs = {cid: ttc for cid, ttc in ttc_dict.items()}
            min_ttc = min(filtered_ttcs.values()) if filtered_ttcs else float('inf')
        else:
            centroids_prev = compute_cluster_centroids(cluster_points_curr)
            min_ttc = float('inf')
            
        frame_ttcs.append(min_ttc)
        sample_img_path = data_loader.get_image_path(data_loader[i])
        out_img_path = save_frame_with_detections(i, sample_img_path, output_dir, fusion_detections, min_ttc)
        frame_paths.append(out_img_path)

    generate_gif(output_dir, frame_paths)
    print(f"Results saved in: {output_dir}")


def main():
    data_loader = NuScenesLoader(NUSCENES_VERSION, NUSCENES_PATH)
    od_model = ObjectDetectionModel()
    #run_detections_and_save("output_results", data_loader, od_model)
    
    # one sample example
    #compute_detections(20, data_loader, od_model)

if __name__ == "__main__":
    main()
