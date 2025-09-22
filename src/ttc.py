import numpy as np

def compute_cluster_centroids(cluster_points):
    centroids = {}
    for cluster_id, points in cluster_points.items():
        points = np.array(points)
        if len(points) < 3:
            continue        # skip small clusters
        centroid = np.mean(points, axis=0)
        centroids[cluster_id] = centroid
    return centroids

def associate_clusters(centroids_prev, centroids_curr, max_dist=1.0):
    associations = {}
    for curr_id, curr_c in centroids_curr.items():
        best_id = None
        best_dist = float('inf')
        for prev_id, prev_c in centroids_prev.items():
            d = np.linalg.norm(curr_c - prev_c)
            if d < best_dist and d < max_dist:
                best_dist = d
                best_id = prev_id
        if best_id is not None:
            associations[curr_id] = best_id
    return associations

def compute_ttc(centroids_prev, centroids_curr, associations, dt):
    ttcs = {}
    for curr_id, prev_id in associations.items():
        prev_distance = centroids_prev[prev_id][1]  # depth before
        curr_distance = centroids_curr[curr_id][1]  # depth now
        distance_change = prev_distance - curr_distance     # positive = approaching
        relative_speed = distance_change / dt
        if relative_speed > 0:
            ttc = curr_distance / relative_speed
            ttcs[curr_id] = ttc
        else:
            ttcs[curr_id] = float('inf')  # not approaching
        #print(f"prev_d: {prev_distance}, curr_d: {curr_distance}, change: {distance_change}, speed: {relative_speed}")
    return ttcs

