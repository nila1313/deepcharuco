import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


SRC_DIR = Path("../my_dataset/opencv_pseudolabels_raw_320")
SRC_IMG_DIR = SRC_DIR / "images"
SRC_KPT_DIR = SRC_DIR / "keypoints"

OUT_DIR = Path("../my_dataset/opencv_pseudolabels_raw_320_min6_aug")
OUT_IMG_DIR = OUT_DIR / "images"
OUT_KPT_DIR = OUT_DIR / "keypoints"
OUT_VIS_DIR = OUT_DIR / "vis"

OUT_W, OUT_H = 320, 240

# Number of augmented versions per original image.
AUGS_PER_IMAGE = 6

# Minimum transformed keypoints to keep.
MIN_KEYPOINTS_TO_KEEP = 4

random.seed(42)
np.random.seed(42)


def reset_output_dir():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_KPT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_VIS_DIR.mkdir(parents=True, exist_ok=True)


def read_keypoints(csv_path):
    keypoints = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            cid = int(row["corner_id"])
            x = float(row["x"])
            y = float(row["y"])
            keypoints.append([cid, x, y])

    return keypoints


def write_keypoints(csv_path, keypoints):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["corner_id", "x", "y"])

        for cid, x, y in keypoints:
            writer.writerow([int(cid), float(x), float(y)])


def draw_vis(img, keypoints):
    vis = img.copy()

    for cid, x, y in keypoints:
        x_i = int(round(x))
        y_i = int(round(y))

        cv2.circle(vis, (x_i, y_i), 3, (0, 255, 0), -1)
        cv2.putText(
            vis,
            str(int(cid)),
            (x_i + 3, y_i - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return vis


def photometric_aug(img):
    out = img.astype(np.float32)

    # Brightness/contrast
    alpha = random.uniform(0.85, 1.20)
    beta = random.uniform(-18, 18)
    out = out * alpha + beta
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Optional blur
    if random.random() < 0.35:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    # Optional noise
    if random.random() < 0.35:
        noise = np.random.normal(0, random.uniform(2, 6), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out


def affine_aug(img, keypoints):
    """
    Mild geometry augmentation:
    - tiny rotation
    - tiny scale
    - small translation

    Keypoints are transformed with the same affine matrix.
    """
    angle = random.uniform(-4.0, 4.0)
    scale = random.uniform(0.96, 1.04)
    tx = random.uniform(-8, 8)
    ty = random.uniform(-6, 6)

    center = (OUT_W / 2.0, OUT_H / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    aug_img = cv2.warpAffine(
        img,
        M,
        (OUT_W, OUT_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    aug_keypoints = []

    for cid, x, y in keypoints:
        pt = np.array([x, y, 1.0], dtype=np.float32)
        x_new, y_new = M @ pt

        if 0 <= x_new < OUT_W and 0 <= y_new < OUT_H:
            aug_keypoints.append([cid, float(x_new), float(y_new)])

    return aug_img, aug_keypoints


def save_sample(name, img, keypoints):
    img_path = OUT_IMG_DIR / f"{name}.png"
    csv_path = OUT_KPT_DIR / f"{name}.csv"
    vis_path = OUT_VIS_DIR / f"{name}.png"

    cv2.imwrite(str(img_path), img)
    write_keypoints(csv_path, keypoints)

    vis = draw_vis(img, keypoints)
    cv2.imwrite(str(vis_path), vis)


def main():
    reset_output_dir()

    image_paths = sorted(SRC_IMG_DIR.glob("frame_*.png"))

    print("Source image folder:", SRC_IMG_DIR)
    print("Source keypoint folder:", SRC_KPT_DIR)
    print("Source images:", len(image_paths))
    print("Output folder:", OUT_DIR)
    print("Augmentations per image:", AUGS_PER_IMAGE)

    total_saved = 0
    skipped = 0

    for img_path in image_paths:
        csv_path = SRC_KPT_DIR / img_path.name.replace(".png", ".csv")

        if not csv_path.exists():
            skipped += 1
            continue

        img = cv2.imread(str(img_path))

        if img is None:
            skipped += 1
            continue

        img = cv2.resize(img, (OUT_W, OUT_H))
        keypoints = read_keypoints(csv_path)

        if len(keypoints) < MIN_KEYPOINTS_TO_KEEP:
            skipped += 1
            continue

        stem = img_path.stem

        # Save original too.
        save_sample(f"{stem}_orig", img, keypoints)
        total_saved += 1

        for aug_idx in range(AUGS_PER_IMAGE):
            aug_img, aug_kpts = affine_aug(img, keypoints)

            if len(aug_kpts) < MIN_KEYPOINTS_TO_KEEP:
                skipped += 1
                continue

            aug_img = photometric_aug(aug_img)

            out_name = f"{stem}_aug{aug_idx:02d}"
            save_sample(out_name, aug_img, aug_kpts)
            total_saved += 1

    print("\nDone.")
    print("Saved samples:", total_saved)
    print("Skipped:", skipped)
    print("Images saved to:", OUT_IMG_DIR)
    print("Keypoints saved to:", OUT_KPT_DIR)
    print("Visualizations saved to:", OUT_VIS_DIR)


if __name__ == "__main__":
    main()