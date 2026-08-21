import pickle
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

def decode_image(pixels_str: str) -> np.ndarray:
    pixels = np.fromstring(pixels_str, dtype=np.uint8, sep=" ")
    return pixels.reshape(48, 48)

print("[INFO] Loading PKL and CSV...")
# 1. Đọc file PKL
with open("outputs/fer2013_landmarks.pkl", "rb") as f:
    landmarks_data = pickle.load(f)

# 2. Tìm các index bị Failed (Value is None)
failed_keys = [k for k, v in landmarks_data.items() if v is None]
failed_indices = [int(k.split("/")[1]) for k in failed_keys]

print(f"[INFO] Found {len(failed_indices)} failed images.")

# 3. Đọc CSV và trích xuất đúng các dòng Failed
df = pd.read_csv("data/fer2013.csv")
failed_df = df.iloc[failed_indices].copy()

# 4. Chọn ngẫu nhiên 40 ảnh để vẽ
sample_df = failed_df.sample(n=min(40, len(failed_df)), random_state=42).reset_index()

# 5. Plot
n_cols, n_rows = 8, 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
fig.suptitle(f"Sanity Check: FER2013 Failed Landmark Detections (Sample of 40/{len(failed_indices)})", fontsize=16, fontweight="bold")
axes_flat = axes.flatten()

emotions = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

for i, (_, row) in enumerate(sample_df.iterrows()):
    ax = axes_flat[i]
    img = decode_image(row["pixels"])
    emo_str = emotions.get(int(row["emotion"]), "Unknown")
    
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    ax.set_title(f"{emo_str} | ID: {row['index']}", fontsize=9, color="red")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = Path("outputs/failed_landmarks_sanity.png")
plt.savefig(out_path, dpi=150)
print(f"[RESULT] Saved visualization to {out_path.resolve()}")