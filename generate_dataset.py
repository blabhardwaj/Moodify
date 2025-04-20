import numpy as np
import pandas as pd
import random
import os
from PIL import Image, ImageDraw

# Define color themes for moods (RGB)
MOOD_COLORS = {
    "happy": [(255, 223, 0), (255, 165, 0), (255, 105, 180)],        # Yellow, Orange, Pink
    "sad": [(70, 130, 180), (100, 149, 237), (25, 25, 112)],         # Blues
    "angry": [(255, 0, 0), (139, 0, 0), (255, 69, 0)],               # Red tones
    "calm": [(176, 224, 230), (240, 255, 255), (152, 251, 152)],     # Light greens & aquas
    "excited": [(255, 20, 147), (255, 140, 0), (255, 0, 255)]        # Bright purples & pinks
}

OUTPUT_CSV = "mood_color_dataset.csv"
SAMPLES_PER_MOOD = 200

def generate_color_image(color, size=(64, 64), noise=False):
    """Create an image from a solid color or with noise."""
    img = Image.new("RGB", size, color)
    if noise:
        draw = ImageDraw.Draw(img)
        for _ in range(500):
            rand_color = tuple(np.clip(np.array(color) + np.random.randint(-50, 50, 3), 0, 255))
            x, y = np.random.randint(0, size[0]), np.random.randint(0, size[1])
            draw.point((x, y), fill=rand_color)
    return img

def build_dataset():
    data = []
    for mood, colors in MOOD_COLORS.items():
        for _ in range(SAMPLES_PER_MOOD):
            base_color = random.choice(colors)
            image = generate_color_image(base_color, noise=True)
            arr = np.array(image)
            features = arr.reshape(-1, 3).mean(axis=0)
            data.append({
                "r": features[0],
                "g": features[1],
                "b": features[2],
                "label": mood
            })
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Generated dataset saved to `{OUTPUT_CSV}` with {len(df)} samples.")

if __name__ == "__main__":
    build_dataset()
