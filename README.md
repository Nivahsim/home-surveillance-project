# Home Surveillance Project

This is a **Home Surveillance** project written in Python 3.  
It is trained on a custom dataset featuring my own cats — *Ghera* and *Shami*.

The system combines several state-of-the-art computer vision models:

- **YOLOv11** — for object detection (cats and fire)
- **SAM2** — for object segmentation
- **MediaPipe Pose** — for human pose estimation

(![Demo photo](https://github.com/user-attachments/assets/7de643fa-cd9e-4054-a4e5-da5a440a6df6)


---

## Features

✅ Real-time object detection (cats & fire)  
✅ Real-time segmentation with SAM2  
✅ Real-time human pose tracking  
✅ Live camera feed visualization with all overlays

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your_username/home-surveillance-project.git
cd home-surveillance-project
```
2. Install dependencies:
```
pip install -r requirements.txt
```
## Usage

Run the main application:

``` bash
python3 run_pipeline.py
```
