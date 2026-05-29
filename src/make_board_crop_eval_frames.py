import cv2
import numpy as np
from pathlib import Path

src_dir = Path("../my_dataset/raw_frames")
out_dir = Path("../my_dataset/eval_frames_board_crop")
out_dir.mkdir(parents=True, exist_ok=True)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

try:
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    def detect_markers(img):
        corners, ids, rejected = detector.detectMarkers(img)
        return corners, ids

except AttributeError:
    params = cv2.aruco.DetectorParameters_create()

    def detect_markers(img):
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img, aruco_dict, parameters=params
        )
        return corners, ids

OUT_W, OUT_H = 320, 240

def crop_with_margin(img, corners, margin_ratio=0.35):
    pts = np.concatenate([c.reshape(-1, 2) for c in corners], axis=0)

    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)

    w = x2 - x1
    h = y2 - y1
    margin = margin_ratio * max(w, h)

    x1 = int(max(0, x1 - margin))
    y1 = int(max(0, y1 - margin))
    x2 = int(min(img.shape[1], x2 + margin))
    y2 = int(min(img.shape[0], y2 + margin))

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return cv2.resize(img, (OUT_W, OUT_H))

    return cv2.resize(crop, (OUT_W, OUT_H))

image_paths = sorted(list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpg")))

print("Input folder:", src_dir)
print("Found images:", len(image_paths))

for img_path in image_paths:
    img = cv2.imread(str(img_path))

    if img is None:
        print("Could not read:", img_path)
        continue

    corners, ids = detect_markers(img)

    if ids is not None and len(corners) > 0:
        out = crop_with_margin(img, corners)
    else:
        out = cv2.resize(img, (OUT_W, OUT_H))

    cv2.imwrite(str(out_dir / img_path.name), out)

print("Saved cropped eval frames to:", out_dir)