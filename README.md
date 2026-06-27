# 🛡️ InnoGrid: Explainable Multi-Model Deepfake Detection

## Overview

InnoGrid is an AI-powered explainable deepfake detection system developed for **AI Nexus 2026**. The project combines multiple deep learning models with explainable AI techniques to detect manipulated facial images while providing visual evidence supporting the prediction.

Unlike conventional detectors that rely on a single model, InnoGrid employs a **multi-model fusion architecture** consisting of two EfficientNet-B4 classifiers and a CLIP semantic verification model. The final prediction is generated through a weighted fusion engine, while Grad-CAM heatmaps provide explainability by highlighting manipulated image regions.

---

## Key Features

* Multi-model Deepfake Detection
* EfficientNet-B4 (GAN Detector)
* EfficientNet-B4 (Diffusion Detector)
* CLIP Semantic Verification
* Weighted Decision Fusion
* Grad-CAM Explainability
* Fake Generation Type Estimation
* Interactive Streamlit Web Application

---

## System Architecture

```
                    Input Image
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
 Branch A           Branch B          CLIP Model
(EfficientNet)   (EfficientNet)   Semantic Verification
        │                │                │
        └────────────────┼────────────────┘
                         │
                  Weighted Fusion
                         │
              Fake / Real Classification
                         │
             Generation Type Prediction
                         │
                 Grad-CAM Heatmaps
```

---

## Technology Stack

### Programming Language

* Python

### Deep Learning

* PyTorch
* TorchVision
* timm

### Explainable AI

* Grad-CAM

### Vision-Language Model

* OpenAI CLIP

### Web Framework

* Streamlit

### Image Processing

* OpenCV
* Pillow
* NumPy

---

## Project Structure

```
InnoGrid/
│
├── app.py
├── predict.py
├── clip_module.py
├── fusion.py
├── gradcam.py
├── utils.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/nehaaraobtech24-cyber/InnoGrid-Explainable-Deepfake-Detection.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the trained model weights inside:

```
models/
```

Run the application:

```bash
streamlit run app.py
```

---

## Model Pipeline

### Branch A

* EfficientNet-B4
* Specialized for GAN-generated images

### Branch B

* EfficientNet-B4
* Specialized for Diffusion-generated images

### Branch C

* OpenAI CLIP
* Semantic verification using image-text similarity

---

## Explainability

Grad-CAM is used to visualize image regions that contribute most strongly to each EfficientNet model's prediction, improving interpretability and user trust.

---

## Fusion Strategy

Final predictions are generated using a weighted fusion engine that combines:

* Branch A confidence
* Branch B confidence
* CLIP semantic confidence

The system also estimates the most likely image generation technique (GAN or Diffusion) based on branch confidence scores.

---

## Future Enhancements

* Video deepfake detection
* Audio deepfake detection
* Additional diffusion model support
* Real-time webcam inference
* Cloud deployment
* Confidence calibration

---

## Developed For

**AI Nexus 2026**

SRM Connects

Theme:

> *Code the Future. Build the Impossible.*

---

## License

This project is released for academic and educational purposes.
