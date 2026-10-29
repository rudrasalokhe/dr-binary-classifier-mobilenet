"""
RetinaScope AI — Realistic Fundus Benchmark Generator
=====================================================
Generates anatomically consistent retinal fundus images with:
- Red-orange fundus gradient and circular FOV aperture
- Optic Disc and Optic Cup (with variable Cup-to-Disc Ratio)
- Arborescent blood vessel tree branching from the optic nerve head
- Foveal Avascular Zone (Macula)
- Pathology hallmarks (DR microaneurysms/exudates, Glaucoma cupping, AMD drusen, Cataract haze)
"""

import os
import cv2
import numpy as np

CLASSES = ["Normal", "Diabetic_Retinopathy", "Glaucoma", "AMD", "Cataract"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "test_images")
os.makedirs(OUT_DIR, exist_ok=True)

def generate_fundus_image(cls_idx, seed=42, size=512):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.46)
    
    # 1. Base retinal background with radial gradient
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = dist_from_center <= radius
    
    norm_dist = np.clip(dist_from_center / radius, 0, 1)
    # BGR fundus palette: rich orange-red center, deeper burgundy edge
    b = np.clip(28 - 18 * norm_dist, 5, 45).astype(np.uint8)
    g = np.clip(78 - 38 * norm_dist, 18, 95).astype(np.uint8)
    r = np.clip(192 - 60 * norm_dist, 75, 215).astype(np.uint8)
    
    img[mask, 0] = b[mask]
    img[mask, 1] = g[mask]
    img[mask, 2] = r[mask]
    
    rng = np.random.default_rng(seed)
    
    # Add subtle choroidal texture
    noise = rng.normal(0, 4, (size, size, 3)).astype(np.int16)
    img_textured = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img[mask] = img_textured[mask]
    
    # Optic disc position (nasal side, ~33% from left, 50% height)
    disc_x, disc_y = int(size * 0.33), int(size * 0.50)
    disc_rx, disc_ry = int(size * 0.075), int(size * 0.088)
    
    # Draw Optic Disc (yellowish-orange oval)
    cv2.ellipse(img, (disc_x, disc_y), (disc_rx, disc_ry), 0, 0, 360, (115, 198, 248), -1)
    
    # Optic cup (inner pale white-yellow region)
    # Glaucoma has pathologically enlarged cup (CDR > 0.65)
    cup_scale = 0.72 if cls_idx == 2 else 0.36
    cup_rx, cup_ry = int(disc_rx * cup_scale), int(disc_ry * cup_scale)
    cv2.ellipse(img, (disc_x, disc_y), (cup_rx, cup_ry), 0, 0, 360, (175, 235, 255), -1)
    
    # Macula / Foveal avascular zone (darker spot temporal to optic disc)
    macula_x, macula_y = int(size * 0.62), int(size * 0.50)
    cv2.circle(img, (macula_x, macula_y), int(size * 0.055), (12, 42, 115), -1)
    
    # Branching blood vessel tree radiating from the optic disc
    angles = [25, 55, 75, -25, -55, -75, 115, 145, -115, -145]
    for angle_deg in angles:
        rad = np.radians(angle_deg)
        curr_x, curr_y = disc_x, disc_y
        vessel_bgr = (15, 32, 110)
        thickness = 4
        
        # Draw branch with natural organic jitter
        for step in range(16):
            step_len = rng.integers(14, 20)
            jitter_x = rng.integers(-3, 4)
            jitter_y = rng.integers(-3, 4)
            next_x = int(curr_x + np.cos(rad) * step_len + jitter_x)
            next_y = int(curr_y + np.sin(rad) * step_len + jitter_y)
            
            d_sq = np.sqrt((next_x - center[0])**2 + (next_y - center[1])**2)
            if d_sq < radius - 10:
                cv2.line(img, (curr_x, curr_y), (next_x, next_y), vessel_bgr, max(1, thickness))
                
                # Small bifurcations
                if step in (4, 8) and rng.random() > 0.3:
                    b_rad = rad + np.radians(rng.choice([-28, 28]))
                    bx, by = next_x, next_y
                    for _ in range(5):
                        nbx = int(bx + np.cos(b_rad) * 10 + rng.integers(-2, 3))
                        nby = int(by + np.sin(b_rad) * 10 + rng.integers(-2, 3))
                        if np.sqrt((nbx - center[0])**2 + (nby - center[1])**2) < radius - 10:
                            cv2.line(img, (bx, by), (nbx, nby), vessel_bgr, 1)
                        bx, by = nbx, nby
                        
            curr_x, curr_y = next_x, next_y
            if step > 3:
                thickness = 3
            if step > 7:
                thickness = 2
            if step > 11:
                thickness = 1
                
    # Class-specific pathological features
    if cls_idx == 1:
        # Diabetic Retinopathy: Microaneurysms, dot/blot hemorrhages, and hard exudates
        for _ in range(24):
            hx = rng.integers(int(size * 0.22), int(size * 0.78))
            hy = rng.integers(int(size * 0.22), int(size * 0.78))
            if np.sqrt((hx - center[0])**2 + (hy - center[1])**2) < radius - 15:
                cv2.circle(img, (hx, hy), rng.integers(2, 6), (10, 15, 160), -1)
        for _ in range(16):
            ex = rng.integers(int(size * 0.35), int(size * 0.72))
            ey = rng.integers(int(size * 0.30), int(size * 0.70))
            if np.sqrt((ex - center[0])**2 + (ey - center[1])**2) < radius - 15:
                cv2.circle(img, (ex, ey), rng.integers(2, 5), (130, 225, 245), -1)
                
    elif cls_idx == 3:
        # AMD: Macular drusen deposits
        for _ in range(22):
            dx = int(macula_x + rng.normal(0, size * 0.055))
            dy = int(macula_y + rng.normal(0, size * 0.055))
            if np.sqrt((dx - center[0])**2 + (dy - center[1])**2) < radius - 15:
                cv2.circle(img, (dx, dy), rng.integers(3, 8), (110, 210, 240), -1)
                
    elif cls_idx == 4:
        # Cataract: Optical media haze & loss of high-frequency vessel sharpness
        haze = np.full_like(img, (145, 165, 180), dtype=np.uint8)
        img = cv2.addWeighted(img, 0.42, haze, 0.58, 0)
        img = cv2.GaussianBlur(img, (19, 19), 0)
        img[~mask] = 0

    return img

if __name__ == "__main__":
    for idx, cls in enumerate(CLASSES):
        for i in range(3):
            seed = (idx + 1) * 100 + i
            fundus = generate_fundus_image(idx, seed=seed)
            out_file = os.path.join(OUT_DIR, f"{cls}_{i+1}.png")
            cv2.imwrite(out_file, fundus)
            print(f"Generated: {out_file}")
    print(f"\nAll benchmark fundus images updated successfully in {OUT_DIR}")
