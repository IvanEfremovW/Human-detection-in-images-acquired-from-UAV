# Детекция людей на изображениях, полученных с БПЛА с использованием улучшения контрастности с помощью графического процессора

[English](README.md) | [Русский](README_RU.md)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv11-orange?logo=pytorch)](https://docs.ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)](https://opencv.org)
[![OpenGL](https://img.shields.io/badge/OpenGL-GLSL-purple?logo=opengl)](https://www.opengl.org)

## Обзор
Этот проект решает тважную задачу в области компьютерного зрения, используемого в дронах: обнаружение человека на аэрофотоснимках в условиях плохой освещенности (например, на рассвете, в сумерках, ночью или при ярком контровом свете). 
Стандартные детекторы объектов часто обладают низкими показателями качества в таких ситуациях из-за низкой контрастности изображений и малого размера объекта.

Система состоит из двух ключевых этапов:
1. **Предварительное улучшение контраста** с помощью **собственной GPU-ускоренной реализации CLAHE** (на основе вычислительных шейдеров OpenGL).
2. **Детекция объектов** с использованием **дообученной модели YOLOv11m**, обученной на отфильтрованной и модифицированной версии датасета [VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset?ysclid=mgfbmexmsb933349844).

## Ключевые особенности

- **Обработка в реальном времени**:  
  - **35 FPS** при разрешении 1360×765  
  - **24 FPS** при Full HD (1920×1080)  
  *(тестирование на NVIDIA RTX 3070)*

- **Улучшение детекции в условиях низкого освещения**:  
  +6% recall, +2.6% mAP@0.5 на ночных изображениях после предобработки.
  
- **Собственная реализация CLAHE на GPU**:  
  Реализована на вычислительных шейдерах с **OpenGL**, **на 17% быстрее**, чем версия из OpenCV с CUDA.

## 📂 Структура репозитория
```text
src/
├── detection/                # Логика детекции, обученные модели (.pt)
├── image_processing/          Реализации CLAHE: ShaderProcessor (GLSL) и OpenCVProcessor
├── test/                     # Инструменты для тестирования на датасетах
│   └── testing_results/      # Таблицы и графики
└── main.py                   # Точка входа для запуска полного pipeline
```

# Быстрый старт

**Запуск полного pipeline**
```Python
python src/main.py -i <input_path> -p <processed_save_path> -d <detection_result_path>
```
