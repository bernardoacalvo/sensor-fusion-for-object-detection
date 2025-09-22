## LiDAR and Camera data fusion for object detection
The goal of this project is to implement a solution to fuse lidar and camera data from an autonomous car to perform object detection (mainly detect moving objects like cars, bicycles, people), predict TTC (Time To Colision), and learn along the way.

![Results](./output_results/detection_results.gif)

## Setup

Create a conda environment (pyenv37) using 3.7 Python version (supported version for nuScenes). Here it was used the 3.7.16.

```bash
conda create -n pyenv37 python=3.7
conda activate pyenv37
```

Install requirements.

```bash
pip install -r requirements.txt
```

Register in the [nuScenes website](https://www.nuscenes.org/nuscenes) by Motional and download v1.0-mini dataset as well as the map expansion data. Place it inside the same data folder. The structure shold be like:
```css
root/
├── src/
│   ├── main.py
│   └── ...
├── datasets/
│   └── nuscenes/
│       └── v1.0-mini/
│           ├── maps/
│           ├── samples/
│           ├── sweeps/
│           └── ...
└── README.md
```

## Blog

Here is a blog where I explain the whole pipeline and some important factors and topics I have learned during the development of this project.

Before working on a dataset have a look into the available data its structure and how to access it correctly: [nuScenes](https://www.nuscenes.org/nuscenes).

Delving into the solution itself, these are the pipeline steps and the reasoning behind it:

#### 1. Multisweep

Starting off with the LiDAR data, a single LiDAR sweep in nuScenes captures a 360º scan in ~20 Hz (basically 1 full scan in 0.05s). To increase point density on each sample, we can get a multisweep for each sample. It merges several past sweeps, in this case the last 10 sweeps, into the current frame by transforming them into the same coordinate space (based on extrinsic matrices transformations that are explained later). This provides a richer point cloud that improves object coverage and makes faraway objects more visible.

#### 2. Region of Interest (ROI) filtering

The LiDAR sensor covers all around the car, however, we are only interested in the front of it. Therefore, we crop the point cloud to a specific region, here is a forward-looking region, which is similar to the driving scenario. By filtering we remove uncessary points behind or far from the vehicle, decreasing processing time.

#### 3. Downsampling

To speed up processing even more, we can apply downsampling that reduces the number of points and noise while perserving the original strutucture. It creates voxels (3d cubes) based on a grid, where each voxel generates one point (centroid) by averaging all points inside. 

#### 4. Ground removal (RANSAC)

As we are only interested in movable objects, this step removes ground points, otherwise points close to the ground that represent cars or people are clustered together as one in the following step. For this, is used the RANSAC (RANdom SAmple Consensus) that is an iterative algorithm to find the best model that fits the data. It picks 3 random points (least number of points to form a plane), generates a candidate plane model and counts the inliers points according to the defined distance to that plane. After all iterations, the plane that got the largest number of inliers is chosen as the best fit. Additionally, the points below the ground are removed.

#### 5. Clustering (DBSCAN)

Use DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to find groups of points that are within a radius of each other (dense regions). Discards isolated points as noise. This identifies distinct objects like cars, people and bikes.

#### 6. Road filtering using maps

For simplicity, it is also used the nuScenes HD map API to check if each cluster centroid (detected previously) lies on a drivable area. Therefore, off-road clusters like walls, trees and others are discarded. It also involves extrinsic transformations that are explained later.

#### 7. Cluster bounding boxes

For each cluster, compute a 3D bounding box that covers all points using the minimum and maximum values for all 3 coordinates.

#### 8. Project LiDAR to camera

The following step is to project the LiDAR's detected 3D bounding boxes into a 2D image. For this, we need to transform points across multiple coordinate systems using calibration data.

To better understand both intrinsics and extrinsics matrices watch this [video](https://www.youtube.com/watch?v=ND2fa08vxkY). Here, the extrinsic matrices are used to transform points between coordinate frames, while the intrinsic projects 3D camera-frame points onto the 2D image.
To accurately transform points between sensors, it is required to transform from the ego frame (vehicle's position frame) to a global coordinate system that aligns all ego poses, which takes into account their correct pose at each sensor's timestamp because different sensors collect data at different times.

The projections order are:
1. LiDAR → Ego (LiDAR): apply the sensor's *extrinsic* matrix
2. Ego (LiDAR) → Global: ego pose at LiDAR timestamp
3. Global → Ego (Camera): ego pose at camera timestamp
4. Ego (Camera) → Camera Frame: apply camera's *extrinsic* matrix
5. Camera Frame → Image Plane: apply camera's *intrinsic* matrix

#### 9. Image object detection

For object detection in images, the YOLO (You Only Look Once) allows to detect cars, pedestrians, bicycles and others. Watch this [video](https://youtu.be/svn9-xV7wjk) for details.
Note: YOLO struggles to detect small objects. The SAHI (Sliced Aided Hyper Inference) approach slices the image into smaller patches, allowing an effective detection when running YOLO on each patch.

### 10. Sensor Fusion (LiDAR + Image)

Finally, the 2D bounding boxes detections from LiDAR data and images are fused together to improve reliability on the predictions. This is easily done by defining a threshold and using IoU (Intersection over Union) that measures how much two bounding boxes overlap.

### Time-To-Collision (TTC) estimation

Now that the objects are detected, it is possible to estimate how soon these might collied with the ego vehicle.
Because we require distance to compute it, first for each detected object associated with a cluster of points, we compute its centroid. Where consecutive frames, associate current centroids with closest previous ones. Then, compute the distance from the ego vehicle to each centroid and estimate relative velocity using the change in centroid position. So, if the distance is decreasing the object is approaching, otherwise, it is moving away. In case relative speed is positive, compute TTC (current distance/relative speed).