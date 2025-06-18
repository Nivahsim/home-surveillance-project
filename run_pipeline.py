import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import torch

from ultralytics import YOLO, SAM

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

ROOT = Path(__file__).parent.resolve()

# Check CUDA availability
cuda_device = "0" if torch.cuda.is_available() else "cpu"

# Initialize models ONCE

# cat_model = YOLO(str(ROOT / 'cat-classifier' / 'weights' / 'cat_classifier_best.pt'))
cat_model = YOLO(str(ROOT / 'cat-classifier' / 'weights' / 'cat_classifier_best_fine-tuned.pt'))
fire_model = YOLO(str(ROOT / 'fire-detector' / 'weights' / 'fire_detector_best.pt'))
sam_model = SAM("sam2_t.pt")


def cat_classifier_run_detector(image):
    # Initialize YOLO model
    results = cat_model(image, conf=0.68, device=cuda_device)
    detected_boxes = []
    boxes = results[0].boxes  # Access the Boxes object
    if boxes is not None:
        # Use .xyxy to get coordinates and check if conf and cls are available as separate attributes
        for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
            x1, y1, x2, y2 = box[:4]  # Extract coordinates
            detected_boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)))

    return detected_boxes



def run_fire_detector(image):
    # Initialize YOLO model
    results = fire_model(image, conf=0.80, device=cuda_device)
    detected_boxes = []
    boxes = results[0].boxes  # Access the Boxes object
    if boxes is not None:
        # Use .xyxy to get coordinates and check if conf and cls are available as separate attributes
        for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
            x1, y1, x2, y2 = box[:4]  # Extract coordinates
            detected_boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)))

    return detected_boxes


def run_segmentation(image, box):
    # Unpack the first four values: x1, y1, x2, y2
    x1, y1, x2, y2, *_ = box  # Use *_ to ignore the extra values (conf and cls)
    
    # Load the model
    results = sam_model(source=image, bboxes=[x1, y1, x2, y2], device=cuda_device)  # generator of Results objects
    
    if results:
        # Extract mask from the results
        mask = None
        for result in results:
            if hasattr(result, 'masks') and result.masks:
                mask = result.masks.data[0].cpu().numpy()  # Get first mask as numpy array
    
    return mask


def process_box(image, box, box_color):
    x1, y1, x2, y2, conf, cls = box

    # Нарисуем YOLO box
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

    # Запускаем SAM для сегментации
    mask = run_segmentation(image, box)

    if mask is not None:
        # Convert mask to uint8
        mask_vis = (mask * 255).astype(np.uint8)

        # Create a color mask
        mask_color_img = np.zeros_like(image)
        mask_color_img[:, :, 1] = mask_vis  # Green channel

        # Blend with the original image
        alpha = 0.5
        image = cv2.addWeighted(image, 1.0, mask_color_img, alpha, 0)
    
    return image


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Cat classifier YOLO Detection
            boxes_cat = cat_classifier_run_detector(image)
            boxes_fire = run_fire_detector(image)
            
            # Draw SAM2 segmentation
            for box in boxes_cat:
                color = (0, 255, 0) if box[5] == 0 else (255, 0, 0)
                image = process_box(image, box, color)

            # Draw SAM2 segmentation
            for box in boxes_fire:
                image = process_box(image, box, (255, 0, 255))


            # Pose detection with MediaPipe
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            image.flags.writeable = True    
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Display the image
            cv2.imshow('YOLO + SAM + MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
