"""Render mascot.png from mascot.svg with clean transparent edges.

Removes white fringe by:
1. Render SVG at 2x resolution
2. Iteratively remove white-ish edge pixels (brightness>180, saturation<0.2)
3. Median-blend remaining edges with 9x9 grid of opaque neighbors
4. Downscale to final resolution with LANCZOS

Usage:
    pip install cairosvg pillow numpy scipy
    python docs/render_mascot.py
"""

import cairosvg
from PIL import Image
import numpy as np
from scipy.ndimage import minimum_filter
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
SVG_PATH = SCRIPT_DIR / "mascot.svg"
PNG_PATH = SCRIPT_DIR / "mascot.png"

FINAL_SIZE = 1000
RENDER_SIZE = FINAL_SIZE * 2  # 2x for quality
BRIGHTNESS_THRESHOLD = 180
SATURATION_THRESHOLD = 0.2
MEDIAN_GRID = 9  # 9x9
MAX_EROSION_STEPS = 15


def render():
    print(f"Rendering {SVG_PATH} at {RENDER_SIZE}px...")
    cairosvg.svg2png(url=str(SVG_PATH), write_to=str(PNG_PATH), output_width=RENDER_SIZE)

    img = Image.open(PNG_PATH).convert("RGBA")
    data = np.array(img)
    h, w = data.shape[:2]

    # Step 1: iteratively remove white-ish edge pixels
    for step in range(MAX_EROSION_STEPS):
        alpha = data[:, :, 3]
        is_visible = alpha > 0
        has_transparent_neighbor = ~minimum_filter(is_visible, size=3)
        edge_mask = is_visible & has_transparent_neighbor

        removed = 0
        for y in range(h):
            for x in range(w):
                if not edge_mask[y, x]:
                    continue
                r, g, b = int(data[y, x, 0]), int(data[y, x, 1]), int(data[y, x, 2])
                brightness = (r + g + b) / 3.0
                max_c = max(r, g, b)
                min_c = min(r, g, b)
                saturation = (max_c - min_c) / (max_c + 1)
                if brightness > BRIGHTNESS_THRESHOLD and saturation < SATURATION_THRESHOLD:
                    data[y, x, 3] = 0
                    removed += 1

        print(f"  Step {step + 1}: removed {removed} white edge pixels")
        if removed == 0:
            break

    # Step 2: median blend remaining edges
    alpha = data[:, :, 3]
    is_visible = alpha > 0
    has_transparent_neighbor = ~minimum_filter(is_visible, size=3)
    edge_mask = is_visible & has_transparent_neighbor

    pad = MEDIAN_GRID // 2
    result = data.copy()
    count = 0
    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            if not edge_mask[y, x]:
                continue
            patch = data[y - pad:y + pad + 1, x - pad:x + pad + 1]
            opaque_patch = patch[:, :, 3] >= 200
            if opaque_patch.sum() > 0:
                for c in range(3):
                    result[y, x, c] = int(np.median(patch[:, :, c][opaque_patch]))
                count += 1

    print(f"  Edge blend: {count} pixels")

    # Step 3: downscale
    final = Image.fromarray(result).resize((FINAL_SIZE, FINAL_SIZE), Image.LANCZOS)
    final.save(PNG_PATH)
    print(f"Saved {PNG_PATH} ({FINAL_SIZE}x{FINAL_SIZE})")


if __name__ == "__main__":
    render()
