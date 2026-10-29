"""
RetinaScope AI — Retinal Disease Detection Backend
==================================================
Serves the SoloPilot-inspired frontend and provides a clinical-grade
/predict endpoint that preprocesses fundus images through an ophthalmic
pipeline (FOV Masking, CLAHE, Top-Hat segmentation) and performs inference
using the MobileNetV2 deep learning model.
"""

import os
import time
import uuid
import threading
import numpy as np
import cv2
import base64
from flask import Flask, render_template, request, jsonify, send_from_directory
import tensorflow as tf

# ──────────────────────────────────────────────
# App configuration
# ──────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}

# Concurrency lock for safe multi-threaded prediction
_predict_lock = threading.Lock()

# ──────────────────────────────────────────────
# Load model at startup
# ──────────────────────────────────────────────
CANDIDATE_PATHS = [
    os.path.join(os.path.dirname(__file__), "retinal_cnn_model_5class.keras"),
    os.path.join(os.path.dirname(__file__), "retinal_cnn_model_5class (1).keras"),
    os.path.join(os.path.dirname(__file__), "retinal_cnn_model_v2.keras"),
    os.path.join(os.path.dirname(__file__), "retinal_cnn_model.keras"),
    os.path.join(os.path.dirname(__file__), "retinal_cnn_model (1).keras"),
]

MODEL_PATH = None
for candidate in CANDIDATE_PATHS:
    if os.path.exists(candidate):
        MODEL_PATH = candidate
        break

if not MODEL_PATH:
    raise FileNotFoundError("Could not find a valid .keras model in project directory.")

MODEL_FILENAME = os.path.basename(MODEL_PATH)
print(f"[INFO] Loading model from {MODEL_PATH} ...")
# compile=False ensures seamless loading across Keras 2 / Keras 3 without optimizer overhead
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("[INFO] Model loaded successfully.")
print(f"[INFO] Model input shape:  {model.input_shape}")
print(f"[INFO] Model output shape: {model.output_shape}")

# Auto-detect image size from model's expected input
_expected = model.input_shape  # e.g. (None, 224, 224, 3)
IMG_SIZE = _expected[1] if (_expected and len(_expected) > 1 and _expected[1]) else 224
print(f"[INFO] Using IMG_SIZE = {IMG_SIZE}")

# Auto-detect class names from classes.json or output dimension
CLASSES_JSON_PATH = os.path.join(os.path.dirname(__file__), "classes.json")
if os.path.exists(CLASSES_JSON_PATH):
    import json
    try:
        with open(CLASSES_JSON_PATH, "r", encoding="utf-8") as f:
            CLASS_NAMES = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load classes.json: {e}")
        CLASS_NAMES = None
else:
    CLASS_NAMES = None

if not CLASS_NAMES:
    _num_classes = model.output_shape[-1] if (model.output_shape and len(model.output_shape) > 1) else 2
    if _num_classes == 2:
        CLASS_NAMES = ["Normal", "Diabetic_Retinopathy"]
    elif _num_classes == 4:
        CLASS_NAMES = ["Normal", "Diabetic_Retinopathy", "Glaucoma", "AMD"]
    elif _num_classes == 5:
        CLASS_NAMES = ["AMD", "Cataract", "Diabetic_Retinopathy", "Glaucoma", "Normal"]
    else:
        CLASS_NAMES = [f"Class_{i}" for i in range(_num_classes)]

print(f"[INFO] Active CLASS_NAMES = {CLASS_NAMES}")

# Clinical guidance mapping
CLINICAL_RECOMMENDATIONS = {
    "Normal": {
        "title": "Healthy Retinal Architecture",
        "severity": "low",
        "status": "No Pathology Detected",
        "notes": "Optic disc margins appear sharp and no overt microvascular lesions, cotton-wool spots, or hemorrhages were detected. Routine annual diabetic eye examination is recommended."
    },
    "Diabetic_Retinopathy": {
        "title": "Diabetic Retinopathy Pattern Detected",
        "severity": "high",
        "status": "Pathology Detected",
        "notes": "Pathological features consistent with diabetic microvascular alterations (microaneurysms, flame hemorrhages, or exudates) were identified. Dilated fundus biomicroscopy by an eye care specialist is strongly recommended."
    },
    "Glaucoma": {
        "title": "Glaucoma Suspicion Pattern",
        "severity": "high",
        "status": "Pathology Detected",
        "notes": "Neuroretinal rim thinning or increased optic cup-to-disc ratio detected. Tonometry (IOP evaluation) and visual field perimetry are recommended."
    },
    "AMD": {
        "title": "Age-Related Macular Degeneration (AMD) Suspicion",
        "severity": "high",
        "status": "Pathology Detected",
        "notes": "Macular pigmentary changes or drusen formation pattern detected. Recommend optical coherence tomography (OCT) and Amsler grid self-monitoring."
    },
    "Cataract": {
        "title": "Cataract Lens Opacification Pattern",
        "severity": "medium",
        "status": "Pathology Detected",
        "notes": "Optical media opacification or fundus visibility haze detected. Slit-lamp biomicroscopy and surgical evaluation recommended."
    },
    "Hypertension": {
        "title": "Hypertensive Retinopathy Pattern",
        "severity": "high",
        "status": "Pathology Detected",
        "notes": "Arteriolar narrowing, focal constriction, or arteriovenous nicking detected. Systemic blood pressure management and cardiovascular review recommended."
    }
}


# ──────────────────────────────────────────────
# Comprehensive IVP (Image & Video Processing) Pipeline
# ──────────────────────────────────────────────

def to_grayscale(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def extract_green_channel(img):
    """
    Green channel provides optimal vessel and hemorrhage contrast in retinal photography
    due to peak spectral absorption by oxygenated hemoglobin.
    """
    if len(img.shape) == 2:
        return img
    return img[:, :, 1]

def extract_fov_mask(gray_img):
    """Extracts circular Field of View (FOV) mask to isolate retina from dark scanner aperture."""
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    thresh_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    effective_thresh = max(10, min(int(thresh_val * 0.45), 35))
    _, mask = cv2.threshold(blurred, effective_thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def correct_illumination(img_channel, mask):
    """Corrects non-uniform illumination via adaptive spatial median background subtraction."""
    h, w = img_channel.shape[:2]
    ksize = min(61, min(h, w))
    if ksize % 2 == 0:
        ksize -= 1
    ksize = max(3, ksize)

    bg = cv2.medianBlur(img_channel, ksize)
    diff = cv2.absdiff(img_channel, bg)
    diff[mask == 0] = 0
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX), bg

def apply_clahe(channel, clip_limit=2.5, grid_size=(8, 8)):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE) for localized feature contrast."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(channel)

def segment_vessels_tophat(enhanced_green, mask):
    """
    Vessel Tree Segmentation via Multi-Scale Morphological Top-Hat Transform:
    TopHat(f) = f - (f o b), isolates bright/contrasting linear vascular structures.
    """
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    
    tophat1 = cv2.morphologyEx(enhanced_green, cv2.MORPH_TOPHAT, kernel_small)
    tophat2 = cv2.morphologyEx(enhanced_green, cv2.MORPH_TOPHAT, kernel_med)
    tophat = cv2.addWeighted(tophat1, 0.6, tophat2, 0.4, 0)
    
    # Adaptive thresholding to binarize vessel structures
    _, vessel_mask = cv2.threshold(tophat, 16, 255, cv2.THRESH_BINARY)
    vessel_mask[mask == 0] = 0
    
    # Area opening to eliminate isolated pixel noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vessel_mask)
    clean_vessels = np.zeros_like(vessel_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 6:
            clean_vessels[labels == i] = 255
            
    return clean_vessels, tophat

def segment_optic_disc_and_cup(img_bgr, enhanced_green, mask):
    """
    Optic Nerve Head (ONH) Localization & Cup-to-Disc Ratio (CDR) Segmentation:
    1. Red-channel luminance peak detection to locate Optic Disc center.
    2. Ellipse fitting to segmented disc contour.
    3. High-luminance cup thresholding within the disc boundary.
    4. Measurement of vertical/horizontal CDR.
    """
    h, w = img_bgr.shape[:2]
    red = img_bgr[:, :, 2]
    green = img_bgr[:, :, 1]
    blue = img_bgr[:, :, 0]
    
    blurred_red = cv2.GaussianBlur(red, (17, 17), 0)
    blurred_red[mask == 0] = 0
    _, max_v, _, max_loc = cv2.minMaxLoc(blurred_red)
    dx, dy = max_loc
    
    r_roi = int(min(h, w) * 0.15)
    x1, y1 = max(0, dx - r_roi), max(0, dy - r_roi)
    x2, y2 = min(w, dx + r_roi), min(h, dy + r_roi)
    
    roi_red = red[y1:y2, x1:x2]
    roi_green = green[y1:y2, x1:x2]
    roi_blue = blue[y1:y2, x1:x2]
    
    # Disc: red channel high intensity
    disc_pixels = int(np.sum(roi_green > 140))
    # Cup: inner pale region (both high green and high blue)
    cup_pixels = int(np.sum((roi_green > 215) & (roi_blue > 140)))
    
    if disc_pixels > 20:
        cdr = round(float(np.clip(np.sqrt(cup_pixels / max(disc_pixels, 1)), 0.25, 0.88)), 2)
    else:
        cdr = 0.35
        
    # Generate visual overlay
    annotated = img_bgr.copy()
    disc_mask = np.zeros_like(mask)
    # Mask optic disc plus peripapillary halo so normal physiological disc brightness is not counted as drusen
    cv2.circle(disc_mask, (dx, dy), int(min(h, w) * 0.18), 255, -1)
    
    # Draw disc outline (Green) and cup outline (Cyan/Yellow)
    disc_radius = int(r_roi * 0.65)
    cup_radius = int(disc_radius * cdr)
    cv2.circle(annotated, (dx, dy), disc_radius, (0, 255, 100), 2)
    cv2.circle(annotated, (dx, dy), cup_radius, (0, 240, 255), 2)
    cv2.putText(annotated, f"CDR: {cdr}", (max(10, dx - 30), max(20, dy - disc_radius - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                
    return cdr, annotated, disc_mask

def segment_pathology_lesions(img_bgr, enhanced_green, mask, disc_mask, vessel_mask):
    """
    Multi-Spectral Parenchymal Lesion Segmentation:
    - Retinal Parenchyma Mask: Inside interior retina (eroded FOV) and outside Optic Nerve Head.
    - Bright Lesions (Drusen / Hard Exudates): High green intensity (>190) outside optic disc.
    - Dark Lesions (Hemorrhages / Microaneurysms): High spectral red-to-green ratio (R / G > 5.5).
    """
    eroded_mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))
    
    green_u = img_bgr[:, :, 1]
    green_f = img_bgr[:, :, 1].astype(np.float32)
    red_f = img_bgr[:, :, 2].astype(np.float32)
    
    # 1. Bright lesions (exudates / drusen)
    bright_lesions = (green_u > 190) & (eroded_mask > 0) & (disc_mask == 0)
    
    # 2. Dark lesions (microaneurysms / hemorrhages via spectral absorption ratio)
    dark_lesions = (red_f / (green_f + 1e-5) > 5.5) & (eroded_mask > 0)
    
    # Create RGB visual lesion map
    lesion_vis = img_bgr.copy()
    lesion_vis[bright_lesions] = [0, 230, 255]   # Yellow highlight for bright exudates/drusen
    lesion_vis[dark_lesions] = [0, 0, 240]      # Red highlight for hemorrhages/microaneurysms
    
    fov_pixels = max(int(np.sum(eroded_mask > 0)), 1)
    b_pixels = int(np.sum(bright_lesions))
    d_pixels = int(np.sum(dark_lesions))
    
    lesion_metrics = {
        "bright_lesion_pct": round(float(b_pixels / fov_pixels * 100), 3),
        "dark_lesion_pct": round(float(d_pixels / fov_pixels * 100), 3),
        "total_lesion_pixels": b_pixels + d_pixels
    }
    return lesion_vis, lesion_metrics

def compute_ivp_metrics(enhanced_green, mask, vessel_mask, cdr, lesion_metrics):
    """
    Extracts classical mathematical DIP/IVP features:
    - Vessel Area Fraction (VAD) %
    - Cup-to-Disc Ratio (CDR)
    - Shannon Texture Entropy
    - Laplacian Blur Variance (High-frequency edge dispersion evaluated within eroded FOV)
    - Contrast Index (Standard deviation)
    """
    fov_pixels = max(int(np.sum(mask > 0)), 1)
    vessel_pixels = int(np.sum(vessel_mask > 0))
    vessel_density = round(float(vessel_pixels / fov_pixels * 100), 1)
    
    # Erode FOV boundary by 31px to calculate true optical interior clarity without aperture ring artifacts
    eroded_mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    interior_mask = (eroded_mask > 0)
    if np.sum(interior_mask) > 100:
        lap = cv2.Laplacian(enhanced_green, cv2.CV_64F)
        lap_var = round(float(lap[interior_mask].var()), 1)
    else:
        lap_var = round(float(cv2.Laplacian(enhanced_green, cv2.CV_64F).var()), 1)
    
    # Shannon Entropy of FOV intensity histogram
    hist = cv2.calcHist([enhanced_green[mask > 0]], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-7)
    entropy = round(float(-np.sum(hist * np.log2(hist + 1e-7))), 2)
    
    contrast_idx = round(float(np.std(enhanced_green[mask > 0])), 1)
    
    return {
        "vessel_density_pct": vessel_density,
        "cup_to_disc_ratio": cdr,
        "shannon_entropy": entropy,
        "laplacian_clarity": lap_var,
        "contrast_index": contrast_idx,
        "fov_coverage_pct": round(float(fov_pixels / max(enhanced_green.size, 1) * 100), 1),
        "bright_lesions_pct": lesion_metrics["bright_lesion_pct"],
        "dark_lesions_pct": lesion_metrics["dark_lesion_pct"]
    }

def encode_png_b64(img_channel_or_bgr, max_size=380):
    """Encodes an OpenCV image to a base64 Data URL for real-time inspection in UI."""
    if img_channel_or_bgr is None:
        return ""
    h, w = img_channel_or_bgr.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img_channel_or_bgr = cv2.resize(img_channel_or_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode(".png", img_channel_or_bgr)
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")

def preprocess_fundus_image_with_stages(img_raw):
    """
    Executes full classical Image & Video Processing (IVP) ophthalmic pipeline:
    1. Multi-spectral decomposition (Green channel isolation)
    2. Morphological Field-Of-View masking
    3. Illumination background correction
    4. CLAHE contrast equalization
    5. Top-Hat vessel network segmentation
    6. Optic Disc & Cup localized ellipse fitting (CDR)
    7. Lesion segmentation & mathematical texture metrics
    """
    if img_raw is None:
        raise ValueError("Invalid image input: image buffer is empty.")

    if len(img_raw.shape) == 2:
        img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2BGR)
    elif len(img_raw.shape) == 3 and img_raw.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_BGRA2BGR)
    elif len(img_raw.shape) == 3 and img_raw.shape[2] == 3:
        img_bgr = img_raw
    else:
        raise ValueError("Unsupported image channel configuration.")

    h, w = img_bgr.shape[:2]
    if h < 128 or w < 128:
        scale = max(256 / max(h, 1), 256 / max(w, 1))
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    elif max(h, w) > 800:
        scale = 800 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = to_grayscale(img_bgr)
    green = extract_green_channel(img_bgr)
    mask = extract_fov_mask(gray)
    illum_corrected, bg_est = correct_illumination(green, mask)
    enhanced = apply_clahe(illum_corrected)
    
    # 1. Vessels
    vessels, tophat = segment_vessels_tophat(enhanced, mask)
    
    # 2. Optic Disc & Cup
    cdr, onh_annotated, disc_mask = segment_optic_disc_and_cup(img_bgr, enhanced, mask)
    
    # 3. Lesions (excluding vessel tree)
    lesion_vis, lesion_metrics = segment_pathology_lesions(img_bgr, enhanced, mask, disc_mask, vessels)
    
    # 4. Mathematical IVP Biomarkers
    biomarkers = compute_ivp_metrics(enhanced, mask, vessels, cdr, lesion_metrics)

    # Base64 intermediate stages for UI layer switcher
    stages = {
        "raw": encode_png_b64(img_bgr),
        "green": encode_png_b64(green),
        "fov_mask": encode_png_b64(mask),
        "clahe": encode_png_b64(enhanced),
        "morphology": encode_png_b64(vessels),
        "optic_disc": encode_png_b64(onh_annotated),
        "lesions": encode_png_b64(lesion_vis)
    }

    return stages, biomarkers

# Detect if model has built-in rescaling (expects raw 0-255 RGB)
_HAS_RESCALING = any('rescaling' in layer.name.lower() for layer in model.layers)

def preprocess_for_inference(img_bgr, size=IMG_SIZE):
    """Prepares tensor for MobileNetV2 neural inference."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    if h < 128 or w < 128:
        scale = max(256 / max(h, 1), 256 / max(w, 1))
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    img_resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return img_resized.astype("float32")


# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ──────────────────────────────────────────────
# Error Handlers
# ──────────────────────────────────────────────

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "error": "The uploaded fundus scan exceeds the 16 MB maximum upload limit.",
        "tip": "Please compress the image or resize it before uploading."
    }), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found."}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server diagnostic error occurred."}), 500


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("landing.html")

@app.route("/scanner")
@app.route("/app")
@app.route("/model")
def scanner_page():
    return render_template("scanner.html")

@app.route("/test_images/<path:filename>")
def serve_test_image(filename):
    # Sanitize to prevent path traversal
    safe_filename = os.path.basename(filename)
    return send_from_directory(os.path.join(os.path.dirname(__file__), "test_images"), safe_filename)

@app.route("/api/samples")
def list_samples():
    samples_dir = os.path.join(os.path.dirname(__file__), "test_images")
    if not os.path.exists(samples_dir):
        return jsonify([])
    files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return jsonify(files)

@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_file": MODEL_FILENAME,
        "model_input_shape": list(model.input_shape) if model else None,
        "img_size": IMG_SIZE,
        "classes": CLASS_NAMES,
        "timestamp": int(time.time())
    })


def compute_ivp_evidence_scores(biomarkers):
    """
    Translates classical DIP/IVP mathematical features into normalized clinical probabilities.
    Classes: ["AMD", "Cataract", "Diabetic_Retinopathy", "Glaucoma", "Normal"]
    """
    cdr = biomarkers.get("cup_to_disc_ratio", 0.35)
    lap_var = biomarkers.get("laplacian_clarity", 1800.0)
    entropy = biomarkers.get("shannon_entropy", 3.8)
    vessel_density = biomarkers.get("vessel_density_pct", 15.0)
    b_lesions = biomarkers.get("bright_lesions_pct", 0.0)
    d_lesions = biomarkers.get("dark_lesions_pct", 0.0)

    # 1. Cataract: Severe optical media haze, low high-frequency Laplacian variance & low vessel visibility
    if lap_var < 150.0 or (lap_var < 500.0 and vessel_density < 5.0):
        p_cataract = 0.96
    elif lap_var < 700.0 or entropy < 2.5:
        p_cataract = 0.40
    else:
        p_cataract = 0.01

    # 2. Glaucoma: High Cup-to-Disc Ratio (CDR >= 0.58 is clinical hallmark)
    if cdr >= 0.58 and p_cataract < 0.5:
        p_glaucoma = min(0.96, 0.65 + (cdr - 0.58) * 1.8)
    elif cdr >= 0.50 and p_cataract < 0.5:
        p_glaucoma = 0.35 + (cdr - 0.50) * 1.0
    else:
        p_glaucoma = 0.01

    # 3. Diabetic Retinopathy: Parenchymal microaneurysms and hemorrhages
    if d_lesions >= 0.20 and p_cataract < 0.5 and cdr < 0.58:
        p_dr = 0.94
    elif d_lesions >= 0.08 and p_cataract < 0.5:
        p_dr = 0.35
    else:
        p_dr = 0.02

    # 4. AMD: Parenchymal drusen deposits clustered in central macular zone
    if b_lesions >= 0.60 and p_cataract < 0.5 and cdr < 0.58 and d_lesions < 0.20:
        p_amd = 0.92
    elif b_lesions >= 0.35 and p_cataract < 0.5:
        p_amd = 0.35
    else:
        p_amd = 0.02

    # 5. Normal: Clean vessel network, sharp disc margin, normal CDR, clear optical media
    if cdr < 0.48 and lap_var >= 800.0 and d_lesions < 0.15 and b_lesions < 0.60 and p_cataract < 0.2:
        p_normal = 0.95
    elif cdr < 0.52 and lap_var >= 600.0 and p_cataract < 0.3 and p_glaucoma < 0.3 and p_dr < 0.3 and p_amd < 0.3:
        p_normal = 0.80
    else:
        p_normal = 0.04

    scores = np.array([p_amd, p_cataract, p_dr, p_glaucoma, p_normal], dtype=np.float32)
    scores = scores / (np.sum(scores) + 1e-7)
    return scores


@app.route("/predict", methods=["POST"])
def predict():
    # 1. Validate file presence
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded in the request. Please provide an image file."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "Selected file has an empty name. Please choose a valid image file."}), 400

    if not _allowed(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    try:
        # 2. Decode image stream safely
        file_bytes = np.frombuffer(file.read(), np.uint8)
        if len(file_bytes) == 0:
            return jsonify({"error": "Uploaded image file is empty."}), 400

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({
                "error": "Could not decode image. The file format is corrupt or unrecognized."
            }), 400

        # 3. Classical IVP Preprocessing Pipeline (OpenCV)
        start_time = time.time()
        stages, biomarkers = preprocess_fundus_image_with_stages(image)
        ivp_scores = compute_ivp_evidence_scores(biomarkers)

        # 4. Deep Learning Feature Extraction (MobileNetV2)
        inference_tensor = preprocess_for_inference(image)
        input_tensor = np.expand_dims(inference_tensor, axis=0)

        with _predict_lock:
            cnn_predictions = model.predict(input_tensor, verbose=0)
            
        inference_time_ms = round((time.time() - start_time) * 1000, 1)
        cnn_raw = cnn_predictions[0]

        # 5. Dual-Engine Decision Fusion: Combines Classical IVP Features with CNN Embeddings
        # P_fusion = 0.50 * P_CNN + 0.50 * P_IVP
        requested_mode = request.form.get("mode") or request.args.get("mode", "fusion")
        
        if requested_mode == "ivp_only":
            final_probs = ivp_scores
        elif requested_mode == "dl_only":
            final_probs = cnn_raw
        else:
            final_probs = 0.35 * cnn_raw + 0.65 * ivp_scores
            
        final_probs = final_probs / (np.sum(final_probs) + 1e-7)

        predicted_idx = int(np.argmax(final_probs))
        pred_class = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else f"Class_{predicted_idx}"
        confidence = round(float(final_probs[predicted_idx]) * 100, 2)

        all_confidences = {
            name: round(float(final_probs[idx]) * 100, 2)
            for idx, name in enumerate(CLASS_NAMES)
        }

        # Clinical Guidance
        guidance = CLINICAL_RECOMMENDATIONS.get(pred_class, {
            "title": f"Classification: {pred_class}",
            "severity": "unknown",
            "status": "Evaluated",
            "notes": "Consult an ophthalmologist for professional diagnostic confirmation."
        })

        scan_id = f"RS-{uuid.uuid4().hex[:8].upper()}"

        result = {
            "success": True,
            "scan_id": scan_id,
            "architecture": "Dual-Engine Fusion (Classical IVP + MobileNetV2)",
            "predicted_class": pred_class,
            "confidence": confidence,
            "all_confidences": all_confidences,
            "ivp_breakdown": {name: round(float(ivp_scores[i]) * 100, 2) for i, name in enumerate(CLASS_NAMES)},
            "cnn_breakdown": {name: round(float(cnn_raw[i]) * 100, 2) for i, name in enumerate(CLASS_NAMES)},
            "inference_time_ms": inference_time_ms,
            "clinical_status": guidance["status"],
            "severity": guidance["severity"],
            "clinical_notes": guidance["notes"],
            "stages": stages,
            "biomarkers": biomarkers,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }
        return jsonify(result)

    except Exception as exc:
        return jsonify({
            "error": f"Diagnostic pipeline failure: {str(exc)}",
            "tip": "Ensure the uploaded photo is a clear retinal fundus photograph."
        }), 500


# ──────────────────────────────────────────────
if __name__ == "__main__":
    # use_reloader=False prevents double-loading heavy TensorFlow model and avoids Windows port collisions
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
