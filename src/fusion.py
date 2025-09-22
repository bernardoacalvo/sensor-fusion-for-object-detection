import torch
from torchvision.ops import box_iou

def fuse_detections(lidar_bbs, yolo_detections, iou_threshold = 0.2):
    res = []
    for lidar_bb, cluster_id in lidar_bbs:
        best_iou = 0
        best_idx = -1
        for idx, yolo_detection in enumerate(yolo_detections):
            iou = box_iou(torch.tensor([lidar_bb]), torch.tensor([yolo_detection['bounding_box']]))[0][0].item()
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou >= iou_threshold:
            yolo_det = yolo_detections[best_idx]
            res.append({
                'source': 'fusion',
                'bounding_box': yolo_det['bounding_box'],
                'label': yolo_det['label'],
                'score': yolo_det['score'],
                'cluster_id': cluster_id
            })
        '''
        else:   # lidar-only
            w = lidar_bb[2] - lidar_bb[0]
            h = lidar_bb[3] - lidar_bb[1]
            if w * h > 20000:       # large box
                res.append({
                    'source': 'lidar',
                    'bounding_box': lidar_bb,
                    'label': None,
                    'score': None,
                    'cluster_id': cluster_id
                })
        '''
    return res