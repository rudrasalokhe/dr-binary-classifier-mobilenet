"""
Retinal Disease Detection — Flask Backend
==========================================
Serves the frontend and provides a /predict endpoint that accepts
a fundus image, preprocesses it through the clinical pipeline, and
returns class-wise confidence scores from the Keras CNN model.
"""

import os
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

# ──────────────────────────────────────────────
# App configuration
# ──────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}
CLASS_NAMES = ["Normal", "Diabetic_Retinopathy", "Glaucoma", "AMD"]

# ──────────────────────────────────────────────
# Load model once at startup
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "retinal_cnn_model (1).keras")
print(f"[INFO] Loading model from {MODEL_PATH} …")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully.")
print(f"[INFO] Model input shape:  {model.input_shape}")
print(f"[INFO] Model output shape: {model.output_shape}")

# Auto-detect image size from model's expected input
_expected = model.input_shape  # e.g. (None, 128, 128, 3)
IMG_SIZE = _expected[1] if _expected[1] else 128
print(f"[INFO] Using IMG_SIZE = {IMG_SIZE}")


# ──────────────────────────────────────────────
# Preprocessing pipeline — MUST match training
# ──────────────────────────────────────────────

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def extract_green_channel(img):
    # Green channel has the best vessel/lesion contrast in fundus images
    return img[:, :, 1]

def extract_fov_mask(gray_img):
    """Extracts the circular Field of View (FOV) mask to ignore black borders."""
    _, mask = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def correct_illumination(img_channel, mask):
    """Corrects uneven illumination by subtracting a heavily blurred background estimate."""
    bg = cv2.medianBlur(img_channel, 61)
    diff = cv2.absdiff(img_channel, bg)
    diff[mask == 0] = 0
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

def apply_clahe(channel, clip_limit=2.5, grid_size=(8, 8)):
    """Applies Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(channel)

def morphological_feature_extraction(enhanced_gray, mask):
    """
    Uses Top-Hat morphological operations to highlight retinal structures 
    like blood vessels, microaneurysms, and exudates.
    """
    inv = cv2.bitwise_not(enhanced_gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    tophat = cv2.morphologyEx(inv, cv2.MORPH_TOPHAT, kernel)
    
    # Enhance features and enforce mask
    features = cv2.addWeighted(inv, 1.0, tophat, 1.5, 0)
    features[mask == 0] = 0
    return features

def resize_and_normalize(img, size=IMG_SIZE):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return img.astype("float32") / 255.0

def preprocess_fundus_image(img_bgr):
    """
    Advanced Clinical Preprocessing Pipeline:
    1. Extract Green Channel (highest contrast for retinal structures)
    2. FOV Masking (ignore dark background)
    3. Illumination Equalization
    4. CLAHE (Contrast Enhancement)
    5. Morphological Feature Extraction (Top-Hat Transform)
    """
    gray = to_grayscale(img_bgr)
    green = extract_green_channel(img_bgr)
    
    # Pipeline execution
    mask = extract_fov_mask(gray)
    illum_corrected = correct_illumination(green, mask)
    enhanced = apply_clahe(illum_corrected)
    seg = morphological_feature_extraction(enhanced, mask)
    
    # Combine 3 processed views into a 3-channel tensor for CNN inference
    stacked = np.stack([enhanced, seg, gray], axis=-1)
    return resize_and_normalize(stacked)



# ──────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("landing.html")

@app.route("/app")
def app_page():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # --- validate upload ---
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not _allowed(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    try:
        # --- read image bytes into OpenCV ---
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Could not decode the image. The file may be corrupted."}), 400

        # --- preprocess ---
        processed = preprocess_fundus_image(image)
        input_tensor = np.expand_dims(processed, axis=0)  # (1, 224, 224, 3)

        # --- predict ---
        predictions = model.predict(input_tensor, verbose=0)
        probabilities = predictions[0]  # shape: (4,)

        # Build response
        predicted_idx = int(np.argmax(probabilities))
        result = {
            "predicted_class": CLASS_NAMES[predicted_idx],
            "confidence": round(float(probabilities[predicted_idx]) * 100, 2),
            "all_confidences": {
                name: round(float(prob) * 100, 2)
                for name, prob in zip(CLASS_NAMES, probabilities)
            },
        }
        return jsonify(result)

    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500


# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
