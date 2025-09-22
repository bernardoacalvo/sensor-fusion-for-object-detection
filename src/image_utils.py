import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

def visualize_detections(image_path, detections_data, title='Detections'):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}. Please check the path.")
        return

    for detection in detections_data:
        x1, y1, x2, y2 = detection['bounding_box']
        label = detection.get('label', None)
        score = detection.get('score', None)
        source = detection.get('source', None)

        # color based on source or red for default (bgr)
        color = (0, 0, 255)
        if source is not None:
            if source == 'fusion':
                color = (255, 0, 0)  # blue
            elif source == 'lidar':
                color = (0, 0, 255)  # red
            elif source == 'yolo':
                color = (0, 255, 0)  # green
            else:
                color = (128, 128, 128)     # gray for unknown

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = ":".join([str(label) if label else "?", f"{score:.2f}" if score else "?"])
        text_y_pos = y1 - 7 if y1 - 7 > 10 else y1 + 15     # adjust text position
        cv2.putText(img, text, (x1, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_frame_with_detections(i, image_path, output_dir, bbs_detections, tcc):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 40)
    for det in bbs_detections:
        box = det.get('bounding_box')
        label = det.get('label', '')
        if box:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline='red', width=7)
            draw.text((x1, y1 - 40), label, fill='red', font=font)

    ttc_text = f"Min TTC: {tcc:.2f}s" if tcc != float('inf') else "Min TTC: inf"
    draw.text((20, 20), ttc_text, fill='yellow', font=font)
    out_img_path = os.path.join(output_dir, f"frame_{i:04d}.png")
    image.save(out_img_path)
    return out_img_path

def generate_gif(output_dir, frame_paths):
    gif_path = os.path.join(output_dir, "detection_results.gif")
    with imageio.get_writer(gif_path, mode='I', duration=1) as writer:
        for path in frame_paths:
            frame = imageio.imread(path)
            writer.append_data(frame)