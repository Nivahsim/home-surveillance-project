from ultralytics import YOLO
import subprocess
import torch

model = "./weights/cat_classifier_best.pt"

# Check CUDA availability
cuda_device = "0" if torch.cuda.is_available() else "cpu"

yolo_command = (
    # f'yolo task=detect mode=predict model={model} source="./dataset/test/Ghera_1.jpg"'
    f'yolo task=detect mode=predict model={model} source="/mnt/c/Users/user/Desktop/IMG_4765.mp4"'
)

# command = f"{yolo_command} batch=8 workers=2 device={cuda_device}"
subprocess.run(yolo_command, shell=True, check=True)
