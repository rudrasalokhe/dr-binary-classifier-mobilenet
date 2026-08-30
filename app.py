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

def apply_clahe(gray_img):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray_img)

def gaussian_denoise(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def simple_vessel_segmentation(gray_img):
    # Simple rule-based segmentation: edge map to highlight vessel/lesion
    # boundaries, as noted in your "simple rule-based prediction" idea.
    seg = cv2.Canny(gray_img, 40, 120)
    seg = cv2.GaussianBlur(seg, (5, 5), 0)
    return seg

def resize_and_normalize(img, size=IMG_SIZE):
    img = cv2.resize(img, (size, size))
    return img.astype("float32") / 255.0

def preprocess_fundus_image(img_bgr):
    """
    Full pipeline matching training:
    green channel -> CLAHE (contrast enhancement) -> Gaussian filter (noise reduction)
    -> segmentation map -> stack [denoised, segmentation, grayscale] as 3 channels
    -> resize + normalize
    """
    green = extract_green_channel(img_bgr)
    enhanced = apply_clahe(green)
    denoised = gaussian_denoise(enhanced)
    seg = simple_vessel_segmentation(denoised)
    gray = to_grayscale(img_bgr)

    # Combine 3 processed views into a 3-channel "image" the CNN can eat
    stacked = np.stack([denoised, seg, gray], axis=-1)
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
