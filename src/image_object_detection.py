from ultralytics import YOLO
import cv2

class ObjectDetectionModel:
    def __init__(self):
        self.model = YOLO('yolov8s.pt')

    def run(self, image_path):
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        detections = self.model(img_rgb)[0]
        res = []
        for det in detections.boxes:
            x_min, y_min, x_max, y_max = map(int, det.xyxy[0].tolist())
            conf = float(det.conf[0])
            cls = int(det.cls[0])
            detection = {
                'bounding_box': [x_min, y_min, x_max, y_max],
                'label': self.model.names[cls],
                'score': conf
            }
            res.append(detection)
        return res