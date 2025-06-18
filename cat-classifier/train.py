from ultralytics import YOLO
import subprocess
import torch

# Model selection
model = 'yolo11n.pt'

# Check CUDA availability
cuda_device = "0" if torch.cuda.is_available() else "cpu"

# YOLOv8/YOLOv11 training command with augmentations
# yolo_command = (
#     f'yolo task=detect mode=train model={model} '
#     'data="data.yaml" '
#     'epochs=300 imgsz=1280 batch=32 workers=4 '
#     f'device={cuda_device} '
#     'project="cat-classifier" name="exp_augmented" '
#     'mosaic=1 mixup=0.2 '
#     'hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 '
#     'degrees=10 translate=0.1 scale=0.5 shear=2 '
#     'flipud=0.5 fliplr=0.5 '
#     'save_period=10 plots=True '
# )
yolo_command = (
    f'yolo task=detect mode=train model=/home/nivahsim/cv-project/home-surveillance-project/cat-classifier/cat-classifier/exp_augmented2/weights/best.pt '
    'data="data.yaml" '
    'epochs=100 imgsz=1280 batch=32 workers=4 '
    f'device={cuda_device} '
    'project="cat-classifier" name="fine-tune_ep-100_no-mosaic_lr-00001" '
    'mosaic=0 mixup=0.2 '
    'hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 '
    'degrees=10 translate=0.1 scale=0.5 shear=2 '
    'flipud=0.5 fliplr=0.5 lr0=0.0001 '
    'save_period=10 plots=True '
)

# Run training
subprocess.run(yolo_command, shell=True, check=True)
