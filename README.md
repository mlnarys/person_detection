# Person detection with YOLO

![Python](https://img.shields.io/badge/Python-3.11-blue)
[![YOLO](https://img.shields.io/badge/YOLO-v11-8A2BE2)](https://github.com/AlexeyAB/darknet)  
[![NumPy](https://img.shields.io/badge/NumPy-1.27-lightblue)](https://numpy.org/)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)


## Introduction

This project enables person detection in videos using YOLO11 (You Only Look Once, version 11), a state-of-the-art deep learning framework for computer vision. It allows detecting people in videos and drawing bounding boxes around them.

The project demonstrates practical applications of YOLO11 for analyzing crowds, monitoring public spaces, and other tasks where human detection is essential. 

The system supports a Command Line Interface (CLI), allowing flexible workflows. It can process pre-recorded videos or real-time streams from webcams, highlighting detected people with either boxes or masks.

**Core Features:**

- Detection of people in videos using bounding boxes

- Real-time processing through webcam input

- Usage modes: CLI 

- Easy integration with custom projects and datasets

## Installation & Configuration
### Environment Setup
1. Download the project
2. Install required packages
```
cd person_detection
pip install -r requirements.txt

```
### Project Structure

```
person_detection/
├─ src/
│  ├─ detect.py           # Основной скрипт для детекции и сегментации людей
│  └─ utils.py            # Вспомогательные функции             
├─ README.md
├─ REPORT.md
└─ requirements.txt

```

### Required Files
- Pre-trained Model (automatically downloaded on first use):
    - yolo11s.pt - YOLO11 small detection model
- Input Files:
    - Videos: .mp4, .avi, .mov formats
    - Camera: Webcam device (source=0)

## Usage / How to Run

**CLI:**

```
python src/detect.py --input crowd.mp4 --output crowd_out.mp4 --weights yolo11s.pt --conf 0.4
```
