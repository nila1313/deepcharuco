import os
import glob
import shutil
import random
import cv2
import json
from PIL import Image

raw_dir = "../my_dataset/raw_frames"
train_dir = "../my_dataset/clean_train"
val_dir = "../my_dataset/clean_val"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for folder in [train_dir, val_dir]:
    for f in os.listdir(folder):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(folder, f))

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
detector_params = cv2.aruco.DetectorParameters()

image_paths = sorted(glob.glob(os.path.join(raw_dir, "frame_*.png")))

clean_frames = []

for path in image_paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=detector_params
    )

    marker_count = 0 if ids is None else len(ids)

    if marker_count == 0:
        clean_frames.append(path)

print("Total raw frames:", len(image_paths))
print("Clean frames with 0 detected ArUco markers:", len(clean_frames))

random.seed(42)
random.shuffle(clean_frames)

val_count = max(1, int(len(clean_frames) * 0.2))

val_frames = clean_frames[:val_count]
train_frames = clean_frames[val_count:]

for path in train_frames:
    shutil.copy(path, os.path.join(train_dir, os.path.basename(path)))

for path in val_frames:
    shutil.copy(path, os.path.join(val_dir, os.path.basename(path)))

print("Clean train frames:", len(train_frames))
print("Clean val frames:", len(val_frames))


def make_json(image_dir, output_json):
    images = []
    annotations = []

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    for idx, filename in enumerate(image_files):
        path = os.path.join(image_dir, filename)

        with Image.open(path) as img:
            width, height = img.size

        images.append({
            "id": idx,
            "file_name": filename,
            "width": width,
            "height": height
        })

        annotations.append({
            "id": idx,
            "image_id": idx,
            "caption": "clean custom background image"
        })

    data = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": []
    }

    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)

    print("Saved:", output_json)
    print("Number of images:", len(images))


make_json(train_dir, "../my_dataset/clean_train_custom.json")
make_json(val_dir, "../my_dataset/clean_val_custom.json")