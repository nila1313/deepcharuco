import os
import glob
import cv2
import torch
import numpy as np

from inference import load_models, infer_image


# -----------------------------
# Paths
# -----------------------------
image_dir = "../my_dataset/raw_frames"
output_img_dir = "../my_dataset/inference_output_best_aug_min6_epoch5_raw/images"
output_kpt_dir = "../my_dataset/inference_output_best_aug_min6_epoch5_raw/keypoints"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_kpt_dir, exist_ok=True)


## -----------------------------
# Your trained checkpoints
# -----------------------------
deepc_path = "./tb_logs/ckpts_deepcharuco_masked_pseudo_min6_aug/best_aug_min6_epoch5.ckpt"
refinenet_path = "./tb_logs/ckpts_refinenet/epoch=193-step=9312.ckpt"
# -----------------------------
# Your board configuration
# 7 x 7 board -> 6 x 6 inner corners = 36
# -----------------------------
n_ids = 36


# -----------------------------
# Device
# CPU is safest for first test
# -----------------------------
device = "cpu"

print("Using device:", device)
print("Loading models...")

deepc, refinenet = load_models(
    deepc_path,
    refinenet_path,
    n_ids=n_ids,
    device=device
)

print("Models loaded successfully.")


# -----------------------------
# Select images
# First test only 20 images
# -----------------------------
image_paths = sorted(glob.glob(os.path.join(image_dir, "frame_*.png")))

print("Total images found:", len(image_paths))

#image_paths = image_paths[:100]

all_results = {}

for img_path in image_paths:
    filename = os.path.basename(img_path)
    print("Processing:", filename)

    img = cv2.imread(img_path)

    if img is None:
        print("Could not read image:", img_path)
        continue

    img = cv2.resize(img, (320, 240), cv2.INTER_LINEAR)

    if img is None:
        print("Could not read image:", img_path)
        continue

    keypoints, out_img = infer_image(
        img,
        n_ids,
        deepc,
        refinenet,
        draw_pred=True,
        device=device
    )

    keypoints = np.array(keypoints)

    print("Detected keypoints:", len(keypoints))

    # Save visualization image
    out_img_path = os.path.join(output_img_dir, filename)
    cv2.imwrite(out_img_path, out_img)

    # Save keypoints as CSV
    # format: x, y, corner_id
    out_csv_path = os.path.join(
        output_kpt_dir,
        filename.replace(".png", ".csv")
    )

    if len(keypoints) > 0:
        np.savetxt(
            out_csv_path,
            keypoints,
            delimiter=",",
            header="x,y,corner_id",
            comments=""
        )
    else:
        with open(out_csv_path, "w") as f:
            f.write("x,y,corner_id\n")

    all_results[filename] = keypoints

print("Inference finished.")
print("Output images saved to:", output_img_dir)
print("Keypoints saved to:", output_kpt_dir)