# RetinaScope AI — 5-Class Retinal Disease Detection

AI-powered ophthalmic screening suite that classifies retinal fundus photographs into **5 key conditions**:
- **Normal** (Healthy retinal architecture)
- **Diabetic Retinopathy** (Microvascular alterations, hemorrhages, exudates)
- **Glaucoma** (Optic nerve cup-to-disc enlargement, neuroretinal rim thinning)
- **AMD** (Age-Related Macular Degeneration — drusen deposits, pigmentary changes)
- **Cataract** (Lens opacification pattern)

Powered by a **MobileNetV2** deep neural network with a comprehensive clinical computer vision pipeline (Circular FOV masking, Green channel isolation, CLAHE illumination correction, and Top-Hat morphological vessel extraction).

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x%20%2F%20Keras%203-orange?logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey?logo=flask&logoColor=white)

## Features

- **Clinical Preprocessing Pipeline** — Circular FOV masking → Green channel isolation → CLAHE contrast enhancement → Top-Hat morphological filtering → 224×224 Lanczos4 scaling & normalization.
- **5-Class MobileNetV2 Architecture** — Lightweight, high-accuracy transfer learning network trained for multi-disease classification.
- **Dual Presentation & Diagnostic HUD** —
  - **Showcase Landing** (`/`) — Presentation-ready interface with interactive 3D retinal visualization and telemetry.
  - **Clinical Studio** (`/app`) — Interactive benchmark library (10 curated cases), 5-layer optical vision stack explorer, confidence gauges, and exportable clinical JSON reports.
- **Explainability & Biomarkers** — Computes vessel density %, estimated cup-to-disc ratio (CDR), field-of-view (FOV) coverage %, and triage urgency recommendations.
- **Multi-Class Training Pipeline** — Full data augmentation and transfer learning script (`train_multiclass.py`) for custom dataset training.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Flask (Python) |
| Frontend | Vanilla HTML5 / Modern CSS / JavaScript (Design System) |
| Deep Learning Model | MobileNetV2 (`retinal_cnn_model_5class (1).keras`, TensorFlow / Keras) |
| Image Processing | OpenCV (CLAHE, Morphology, FOV) |

## Project Structure

```
├── app.py                           # Flask server + clinical preprocessing pipeline + /predict API
├── train_multiclass.py              # Multi-class MobileNetV2 training pipeline
├── generate_test_images.py          # Synthetic fundus test image generator
├── classes.json                     # Active class definitions (AMD, Cataract, DR, Glaucoma, Normal)
├── requirements.txt                 # Project dependencies
├── test_images/                     # Benchmark fundus test images
├── templates/
│   ├── landing.html                 # SoloPilot showcase & triage portal (/)
│   └── index.html                   # Diagnostic studio interface (/app)
└── static/
    ├── landing.css & landing.js     # Showcase styles & controller
    ├── style.css & script.js        # Studio styles & controller
```

## Setup & Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/rudrasalokhe/dr-binary-classifier-mobilenet.git
   cd dr-binary-classifier-mobilenet
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate synthetic test images (Optional)**
   ```bash
   python generate_test_images.py
   ```

4. **Run the application**
   ```bash
   python app.py
   ```
   Open [http://localhost:5000](http://localhost:5000) or [http://localhost:5000/app](http://localhost:5000/app) in your browser.

## Preprocessing & Inference Flow

```
Input Fundus Scan (BGR)
    │
    ├── Green Channel Extraction (vascular contrast)
    ├── Circular Field-of-View (FOV) Masking
    ├── CLAHE Contrast Equalization
    ├── Top-Hat Morphological Microvascular Filtering
    └── 224 × 224 Lanczos4 Resampling & Normalization
         │
         ▼
    MobileNetV2 Deep CNN → Softmax (5 Classes)
         │
         ▼
    [AMD, Cataract, Diabetic Retinopathy, Glaucoma, Normal]
```

## Disclaimer

> ⚠️ This is a **student research project** — not a medical diagnostic tool. Do not use for clinical decision-making.

## License

MIT
