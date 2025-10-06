
# Real-Time Human Detection in Low-Light UAV Imagery with GPU-Accelerated Contrast Enhancement

[English](README.md) | [Русский](README_RU.md)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv11-orange?logo=pytorch)](https://docs.ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org)
[![OpenGL](https://img.shields.io/badge/OpenGL-GLSL-purple?logo=opengl)](https://www.opengl.org)

# Overview
This project addresses a critical challenge in drone-based computer vision: reliable human detection in aerial imagery under poor lighting conditions (e.g., dawn, dusk, night, or strong backlight). Standard object detectors often fail in such scenarios due to low contrast and small object scale.

To overcome this, the system integrates:

1. A custom GPU-accelerated implementation of CLAHE (Contrast Limited Adaptive Histogram Equalization) using OpenGL compute shaders for real-time contrast enhancement.
2. A fine-tuned YOLOv11m model, optimized for drone-captured images and trained on a curated subset of the [VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset?ysclid=mgfbmexmsb933349844) dataset.

The result is a robust, fast, and accurate pipeline that outperforms existing drone-based detection systems in both speed and detection quality - especially in low-light environments.

# Key Features

-  **Real-time performance** (tested on NVIDIA RTX 3070):
    - Up to 35 FPS on 1360×765 resolution
    - 24 FPS on Full HD (1920×1080)
- **Enhanced low-light detection:**
CLAHE preprocessing improves recall and mAP on night-time images (+6% recall, +2.6% mAP@0.5).

- **Custom GPU-optimized CLAHE:**
Implemented via OpenGL compute shaders - 17% faster than OpenCV’s CUDA-accelerated version.

# Repository structure
```text
src/
├── detection/                # YOLOv11 inference, trained models (.pt), and detection logic
├── image_processing/         # CLAHE implementations: ShaderProcessor (GLSL) and OpenCVProcessor
├── test/                     # Benchmarking tools and dataset evaluation scripts
│   └── testing_results/      # Tables and plots
└── main.py                   # Entry point for full pipeline
```
# Quick start

**Run the full pipeline**
```Python
python src/main.py -i <input_path> -p <processed_save_path> -d <detection_result_path>
```
