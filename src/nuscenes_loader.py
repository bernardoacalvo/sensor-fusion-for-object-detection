from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import numpy as np
from shapely.geometry import Point, Polygon

class NuScenesLoader:
    def __init__(self, version, path):
        self.data_path = path
        self.nusc = NuScenes(version=version, dataroot=path, verbose=True)
        self.samples = self.nusc.sample

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        return self.nusc.get('sample', self.samples[i]['token'])
    
    def get_dt(self, i):
        if i == 0:
            return None
        curr_time = self.samples[i]['timestamp']
        prev_time = self.samples[i - 1]['timestamp']
        return (curr_time - prev_time) / 1e6        # in seconds

    def get_multisweep_points(self, sample, nsweeps=10):
        #Merged point cloud from 10 past sweeps
        pcd_sweeps, _ = LidarPointCloud.from_file_multisweep(self.nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)
        return pcd_sweeps.points[:3, :].T          # [D, N], D = (X, Y, Z, Intensity, ... ), N points, Transpose to make it (N,3)

    def get_image_path(self, sample):
        cam_token = sample['data']['CAM_FRONT']
        return self.nusc.get_sample_data_path(cam_token)
    
    def get_sensor_and_ego(self, sample, sensor_name):
        sensor_token = sample['data'][sensor_name]
        sensor_sample = self.nusc.get('sample_data', sensor_token)
        csensor = self.nusc.get('calibrated_sensor', sensor_sample['calibrated_sensor_token'])      # where the sensor is on the car
        pose = self.nusc.get('ego_pose', sensor_sample['ego_pose_token'])                           # where the car is in global space
        return csensor, pose

    def get_camera_to_lidar_transformations(self, sample):
        cam_csensor, cam_pose = self.get_sensor_and_ego(sample, 'CAM_FRONT')
        lidar_csensor, lidar_pose = self.get_sensor_and_ego(sample, 'LIDAR_TOP')
        K = np.array(cam_csensor['camera_intrinsic'])    # camera intrinsic matrix (3×3)
        tr = {
            'R_lsensor': Quaternion(lidar_csensor['rotation']).rotation_matrix,      # lidar points → ego pose (lidar timestamp)
            'T_lsensor': np.array(lidar_csensor['translation']),
            'R_lego': Quaternion(lidar_pose['rotation']).rotation_matrix,            # ego pose (lidar timestamp) → global
            'T_lego': np.array(lidar_pose['translation']),
            'R_cego': Quaternion(cam_pose['rotation']).rotation_matrix,              # global → ego pose (camera timestamp)
            'T_cego': np.array(cam_pose['translation']),
            'R_csensor': Quaternion(cam_csensor['rotation']).rotation_matrix,        # ego pose (camera timestamp) → camera points
            'T_csensor': np.array(cam_csensor['translation']),
            'K': K
        }
        return tr

    def filter_clusters_on_road(self, sample, clusters_points):
        log = [log for log in self.nusc.log if log['token'] == self.nusc.get('scene', sample['scene_token'])['log_token']][0]
        nusc_map = NuScenesMap(dataroot=self.data_path, map_name=log['location'])
        drivable_polys = []
        for record in nusc_map.drivable_area:
            for poly_token in record["polygon_tokens"]:
                coords = nusc_map.extract_polygon(poly_token)
                drivable_polys.append(Polygon(coords))

        lidar_csensor, lidar_pose = self.get_sensor_and_ego(sample, 'LIDAR_TOP')
        # LiDAR sensor → ego frame
        R_sensor = Quaternion(lidar_csensor['rotation']).rotation_matrix
        T_sensor = np.array(lidar_csensor['translation'])
        # ego → global
        R_ego = Quaternion(lidar_pose['rotation']).rotation_matrix
        T_ego = np.array(lidar_pose['translation'])

        filtered = {}
        for cid, pts in clusters_points.items():
            centroid = np.mean(pts, axis=0)
            ego_pt = R_sensor @ centroid + T_sensor
            global_pt = R_ego @ ego_pt + T_ego
            gx, gy = global_pt[:2]
            if any(poly.contains(Point(gx, gy)) for poly in drivable_polys):
                filtered[cid] = pts
        return filtered

