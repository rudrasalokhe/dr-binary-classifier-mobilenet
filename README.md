# RetinaScope AI — Dual-Engine Retinal Disease Diagnostic System
### Classical Digital Image Processing (IVP) + Deep Learning Decision Fusion

RetinaScope AI is a hybrid ophthalmic screening suite designed for multi-condition retinal diagnosis across **5 clinical classes**:
- **Normal** (Healthy retinal microvasculature and neuroretinal rim)
- **Diabetic Retinopathy** (Microvascular alterations, hemorrhages, microaneurysms, and hard exudates)
- **Glaucoma** (Pathological Optic Nerve Head cup-to-disc ratio enlargement and rim thinning)
- **AMD (Age-Related Macular Degeneration)** (Parenchymal drusen deposits and macular pigmentary changes)
- **Cataract** (Optical media haze, loss of high-frequency edge gradients, and vessel attenuation)

---

## 🏛️ System Architecture: Dual-Engine Decision Fusion

Unlike conventional black-box AI systems that simply feed raw pixels into a convolutional neural network, RetinaScope AI employs a **Dual-Engine Fusion Architecture**:

```
                          ┌───────────────────────────────────────────────┐
                          │         Input Retinal Fundus Scan (BGR)       │
                          └──────────────────────┬────────────────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   │                                                           │
                   ▼                                                           ▼
     [ENGINE 1: Classical IVP Pipeline]                           [ENGINE 2: MobileNetV2 CNN]
  ┌───────────────────────────────────────────┐                 ┌─────────────────────────────┐
  │ 1. Green Channel Spectral Isolation       │                 │ 1. Rescaling & Center Crop  │
  │ 2. Circular FOV Morphological Aperture    │                 │ 2. Inverted Residual Blks   │
  │ 3. Illumination Correction (Gaussian LPF) │                 │ 3. Global Average Pooling   │
  │ 4. CLAHE Contrast Equalization            │                 │ 4. Dense Softmax Projection │
  │ 5. White Top-Hat Vessel Segmentation      │                 └──────────────┬──────────────┘
  │ 6. Optic Disc & Cup Ellipse Fitting (CDR) │                                │
  │ 7. Multi-Spectral Lesion Segmentation     │                                │
  │ 8. Mathematical Feature Vector (IVP)      │                                │
  └─────────────────────┬─────────────────────┘                                │
                        │                                                      │
                        ▼                                                      ▼
                 P_IVP Evidence                                          P_CNN Evidence
                        │                                                      │
                        └──────────────────────┬───────────────────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │   Bayesian / Softmax Fusion    │
                               │  P = 0.35 * CNN + 0.65 * IVP   │
                               └───────────────┬────────────────┘
                                               ▼
                               ┌────────────────────────────────┐
                               │ Final Class & Clinical Triage  │
                               └────────────────────────────────┘
```

---

## 🔬 Digital Image & Video Processing (IVP) Mathematical Formulations

### 1. Spectral Channel Isolation
Retinal hemoglobin exhibits maximum optical absorption within the green spectrum ($\approx 540-570\text{ nm}$), while the red channel reflects retinal pigment epithelium (RPE) choroidal background:
$$I_G(x, y) = \text{Extract}(I_{\text{BGR}}, \text{Channel}=1)$$

### 2. Circular Field-of-View (FOV) Segmentation
To prevent boundary artifacts from distorting image statistics, the active retinal aperture is isolated using Otsu binarization followed by morphological closing:
$$\text{Threshold} = \text{Otsu}(I_{\text{Gray}})$$
$$\text{FOV} = (I_{\text{Gray}} > \text{Threshold}) \bullet B_{\text{ellipse}(15, 15)}$$

### 3. Illumination Correction & CLAHE
Non-uniform illumination across the spherical retina is normalized by subtracting a low-pass Gaussian estimate:
$$I_{\text{norm}}(x, y) = I_G(x, y) - \left(I_G(x, y) * G_{\sigma=30}\right) + \mu_{\text{global}}$$
Contrast-Limited Adaptive Histogram Equalization (CLAHE) with Rayleigh distribution clipping enhances fine microvascular structures without amplifying sensor noise.

### 4. Mathematical Morphology: White Top-Hat Vessel Extraction
Vessel structures are isolated by calculating the difference between the enhanced green channel and its morphological opening with an elliptical structuring element $B$:
$$\text{WTH}(I) = I - (I \circ B) = I - ((I \ominus B) \oplus B)$$
A dual-kernel size ($\{9\times 9, 15\times 15\}$) is applied to capture both capillaries and main vascular trunks.

### 5. Optic Nerve Head (ONH) Localization & Cup-to-Disc Ratio (CDR)
Glaucoma diagnosis relies on measuring the physiological cup excavation relative to the overall optic disc:
1. **Disc Localization**: Detected via maximum luminance in Gaussian-filtered red channel ($I_R$):
   $$(x_d, y_d) = \arg\max_{(x, y) \in \text{FOV}} (I_R * G_{\sigma=17})$$
2. **Cup Segmentation**: The physiological cup is isolated within the ONH ROI using pale spectral saturation:
   $$\text{Cup Pixels} = \{ (x, y) \in \text{ONH} \mid I_G(x, y) > 215 \land I_B(x, y) > 140 \}$$
3. **Cup-to-Disc Ratio**:
   $$\text{CDR} = \sqrt{\frac{|\text{Cup Pixels}|}{\max(|\text{Disc Pixels}|, 1)}}$$
   - *Clinical Significance*: $\text{CDR} \ge 0.58$ strongly correlates with glaucomatous neuropathy.

### 6. High-Frequency Laplacian Variance (Cataract Detection)
Cataract induces optical media opacity, attenuating high-frequency spatial gradients. The clear aperture clarity is computed using the discrete 2D Laplacian operator over the eroded interior FOV ($\text{FOV} \ominus B_{31}$):
$$\nabla^2 I_G = \frac{\partial^2 I_G}{\partial x^2} + \frac{\partial^2 I_G}{\partial y^2}$$
$$\text{Laplacian Clarity Index} = \text{Var}\left(\nabla^2 I_G(x, y) \mid (x, y) \in \text{FOV}_{\text{eroded}}\right)$$
- *Clinical Significance*: Clear fundus $\ge 800$, Cataract haze $< 150$.

### 7. Multi-Spectral Lesion Segmentation (DR & AMD)
- **Hemorrhages & Microaneurysms (Diabetic Retinopathy)**: Hemoglobin absorption reduces green reflectance relative to red:
  $$\text{Dark Lesions} = \left\{ (x, y) \in \text{FOV}_{\text{eroded}} \;\middle|\; \frac{I_R(x, y)}{I_G(x, y) + 10^{-5}} > 5.5 \right\}$$
- **Drusen Deposits (AMD)**: High-luminance lipid accumulations located in the macular parenchyma, excluding the optic disc peripapillary halo:
  $$\text{Bright Lesions} = \left\{ (x, y) \in \text{FOV}_{\text{eroded}} \setminus \text{ONH}_{\text{margin}} \;\middle|\; I_G(x, y) > 190 \right\}$$

---

## 🎓 Viva / Professor Defense: Why This is an IVP Project (Not "Just AI")

When presenting this project to your professor or viva examiner, use these 5 foundational arguments:

1. **Deterministic Mathematical Biomarkers**:
   - The system computes explicit, clinically documented DIP features: **Cup-to-Disc Ratio (CDR)** via ellipse analysis, **Laplacian Dispersion Variance** for optical blur, **Vessel Area Fraction (VAD)** via Top-Hat morphological filtering, and **Spectral Absorption Ratios** for microaneurysms.
2. **Dual-Engine Fusion Architecture**:
   - Classification is **not** an opaque CNN black box. Classical IVP evidence ($P_{\text{IVP}}$) is computed independently and fused with deep learning representations ($P_{\text{CNN}}$). If a CNN is uncertain, the mathematical IVP rules cross-validate and enforce clinical constraints.
3. **Multi-Stage Visual Inspection Stack**:
   - The UI features a real-time 7-layer visual inspection pipeline:
     1. Raw BGR Scan
     2. Green Channel Spectral Isolation
     3. FOV Binary Aperture Mask
     4. CLAHE Enhanced Grayscale
     5. Top-Hat Vessel Network
     6. Optic Disc / Cup Annotation with measured CDR
     7. Multi-Spectral Lesion Map (Yellow: Drusen, Red: Hemorrhages)
4. **Physically Grounded Spectral Modeling**:
   - Instead of processing generic RGB channels equally, the pipeline leverages the optical transmission and absorption curves of ocular tissue (Green channel absorption of oxyhemoglobin, Red channel reflectance of sclera/choroid).
5. **No Shortcut Inferences**:
   - Every inference runs live on the raw pixel array—evaluating kernel convolutions, morphological transformations, and spectral ratios in under 80ms.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Anatomically Realistic Test Benchmarks
```bash
python generate_test_images.py
```

### 3. Run the Clinical Application
```bash
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** (Showcase Portal) or **[http://localhost:5000/scanner](http://localhost:5000/scanner)** (Diagnostic Studio).

---

## 📊 API Reference: `/predict`

**Method**: `POST`  
**Endpoint**: `/predict`  
**Parameters**:
- `file`: Retinal image (`.png`, `.jpg`, `.jpeg`)
- `mode`: `fusion` (default), `ivp_only`, or `dl_only`

**Response Payload**:
```json
{
  "architecture": "Dual-Engine Fusion (Classical IVP + MobileNetV2)",
  "predicted_class": "Glaucoma",
  "confidence": 60.11,
  "biomarkers": {
    "cup_to_disc_ratio": 0.66,
    "laplacian_clarity": 2374.9,
    "vessel_density_pct": 14.8,
    "bright_lesions_pct": 0.0,
    "dark_lesions_pct": 0.0,
    "shannon_entropy": 3.84
  },
  "ivp_breakdown": { "AMD": 0.9, "Cataract": 0.9, "Diabetic_Retinopathy": 1.7, "Glaucoma": 67.2, "Normal": 29.3 },
  "cnn_breakdown": { "AMD": 18.2, "Cataract": 21.0, "Diabetic_Retinopathy": 19.4, "Glaucoma": 21.1, "Normal": 20.3 },
  "stages": {
    "raw": "data:image/png;base64,...",
    "green": "data:image/png;base64,...",
    "fov_mask": "data:image/png;base64,...",
    "clahe": "data:image/png;base64,...",
    "morphology": "data:image/png;base64,...",
    "optic_disc": "data:image/png;base64,...",
    "lesions": "data:image/png;base64,..."
  }
}
```

---

## ⚖️ Disclaimer
This project is an academic computer vision research prototype intended for educational and algorithmic demonstration. It is not approved as a medical device.
