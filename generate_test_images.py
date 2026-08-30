"""Generate synthetic fundus test images matching the training data distribution."""
import os
import cv2
import numpy as np

CLASSES = ["Normal", "Diabetic_Retinopathy", "Glaucoma", "AMD"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "test_images")
os.makedirs(OUT_DIR, exist_ok=True)

def make_synthetic_fundus(label_idx, size=300):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2.circle(img, center, size // 2 - 5, (30, 60, 120), -1)

    rng = np.random.default_rng()
    n_blobs, blob_size, blob_color = {
        0: (2, (3, 6), (40, 70, 130)),           # Normal: few, small, subtle
        1: (14, (3, 8), (0, 0, 230)),            # DR: many small red dot hemorrhages
        2: (1, (40, 60), (210, 210, 210)),       # Glaucoma: one large pale cupped disc
        3: (5, (10, 20), (0, 210, 230)),         # AMD: few larger yellow deposits
    }[label_idx]

    for _ in range(n_blobs):
        x = rng.integers(size // 4, 3 * size // 4)
        yv = rng.integers(size // 4, 3 * size // 4)
        r = rng.integers(blob_size[0], blob_size[1])
        cv2.circle(img, (x, yv), r, blob_color, -1)

    noise = rng.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

# Generate 3 images per class
for idx, cls in enumerate(CLASSES):
    for i in range(3):
        img = make_synthetic_fundus(idx)
        path = os.path.join(OUT_DIR, f"{cls}_{i+1}.png")
        cv2.imwrite(path, img)
        print(f"Saved: {path}")

print(f"\nDone! {len(CLASSES) * 3} test images saved to {OUT_DIR}")
