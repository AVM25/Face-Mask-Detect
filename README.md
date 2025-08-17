# Face Mask Detection System

This project implements a real-time face mask detection application using deep learning and computer vision techniques.

## Overview
- **Real-time Inference**: Uses OpenCV's DNN face detection and TensorFlow/Keras MobileNetV2 model for mask classification.
- **High Accuracy**: Achieved over **97% validation accuracy** with transfer learning and data augmentation.
- **Optimized Performance**: Processes live video at **30+ FPS** for seamless real-time monitoring on standard hardware.
- **Compact Model**: Training done over **20 epochs** with model size reduced by **50%** leveraging MobileNetV2’s efficient architecture.

## Files and Structure

├── detect_mask.py # Real-time face and mask detection using webcam
├── train_mask_detector.py # Training script using MobileNetV2 and augmentation
├── mask_detector.model # Saved trained model
├── requirements.txt # Python dependencies
├── plot.png # Training loss and accuracy plot
└── face_detector/ # Face detector Caffe model files
├── deploy.prototxt
└── res10_300x300_ssd_iter_140000.caffemodel


## Setup and Usage
1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Train the model (optional):
    ```
    python train_mask_detector.py --dataset <dataset_path> --epochs 20 --batch-size 32
    ```
3. Run real-time detection:
    ```
    python detect_mask.py
    ```

## Key Metrics
- Validation Accuracy: **97%+**
- Frame Rate: **30+ FPS**
- Training Epochs: **20**

