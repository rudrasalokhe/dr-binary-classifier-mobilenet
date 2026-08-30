# RetinaScope — Retinal Disease Detection

AI-powered retinal disease screening tool that classifies fundus photographs into **Normal**, **Diabetic Retinopathy**, **Glaucoma**, and **AMD** using a CNN classifier with a clinical preprocessing pipeline.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey?logo=flask&logoColor=white)

## Features

- **Preprocessing Pipeline** — Green channel extraction → CLAHE contrast enhancement → Gaussian denoising → Canny edge segmentation → 3-channel stacking → Resize & normalize
- **CNN Classifier** — Sequential model with 3 Conv2D blocks, GlobalAveragePooling, and softmax output
- **Web Interface** — Drag-and-drop upload, real-time pipeline visualization, confidence breakdown chart
- **Explainability** — Grad-CAM support for model interpretability (XAI)

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask (Python) |
| Frontend | Vanilla HTML/CSS/JS |
| Model | TensorFlow/Keras CNN |
| Preprocessing | OpenCV |

## Project Structure

```
├── app.py                     # Flask server + preprocessing pipeline
├── generate_test_images.py    # Synthetic fundus image generator
├── retinal_cnn_model.keras    # Trained model (not in repo — see Setup)
├── templates/
│   └── index.html             # Main page
└── static/
    ├── style.css              # UI styles
    └── script.js              # Client logic
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR_USERNAME/retinal-disease-detection.git
   cd retinal-disease-detection
   ```

2. **Install dependencies**
   ```bash
   pip install flask tensorflow opencv-python-headless numpy
   ```

3. **Add your trained model**
   Place your `retinal_cnn_model.keras` file in the project root. (Model file is excluded from git due to size.)

4. **Generate test images** (optional)
   ```bash
   python generate_test_images.py
   ```

5. **Run the app**
   ```bash
   python app.py
   ```
   Open [http://localhost:5000](http://localhost:5000)

## Preprocessing Pipeline

```
Input Image (BGR)
    │
    ├── Green Channel Extraction
    ├── CLAHE (clipLimit=2.5)
    ├── Gaussian Blur (5×5)
    ├── Canny Edge Detection (40, 120)
    └── Grayscale Conversion
         │
         ▼
    Stack [denoised, edges, grayscale] → Resize 128×128 → Normalize [0,1]
         │
         ▼
    CNN → Softmax → [Normal, DR, Glaucoma, AMD]
```

## Disclaimer

> ⚠️ This is a **student research project** — not a medical diagnostic tool. Do not use for clinical decision-making.

## License

MIT
