#!/usr/bin/env python3
import os
from PIL import Image

# ◀── EDIT THESE TWO PATHS ──▶
SRC_DIR = "/home/kaicyang/Data/Code/stable-diffusion-webui-master/dataset"
DST_DIR = "/home/kaicyang/Data/Code/textual_inversion/imgs"

os.makedirs(DST_DIR, exist_ok=True)

for fname in os.listdir(SRC_DIR):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")):
        continue

    src_path = os.path.join(SRC_DIR, fname)
    dst_path = os.path.join(DST_DIR, fname)

    with Image.open(src_path) as img:
        # convert to RGB (drops alpha) if you don’t need transparency
        img = img.convert("RGB")
        img = img.resize((512, 512), Image.LANCZOS)
        img.save(dst_path)

    print(f"Resized and saved → {dst_path}")
