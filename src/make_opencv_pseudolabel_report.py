import cv2
import csv
import numpy as np
from pathlib import Path

SRC_DIR = Path("../my_dataset/raw_frames")
OUT_DIR = Path("../my_dataset/opencv_pseudolabels_raw_320")
IMG_OUT = OUT_DIR / "images"
CSV_OUT = OUT_DIR / "keypoints"
VIS_OUT = OUT_DIR / "vis"

IMG_OUT.mkdir(parents=True, exist_ok=True)
CSV_OUT.mkdir(parents=True, exist_ok=True)
VIS_OUT.mkdir(parents=True, exist_ok=True)

OUT_W, OUT_H = 320, 240
MIN_CORNERS_TO_KEEP = 6

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

try:
    board = cv2.aruco.CharucoBoard(
        (7, 7),
        0.14285714285714285,
        0.10,
        aruco_dict
    )
except Exception:
    board = cv2.aruco.CharucoBoard_create(
        7,
        7,
        0.14285714285714285,
        0.10,
        aruco_dict
    )

try:
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    def detect_markers(gray):
        corners, ids, rejected = detector.detectMarkers(gray)
        return corners, ids

except AttributeError:
    params = cv2.aruco.DetectorParameters_create()

    def detect_markers(gray):
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=params
        )
        return corners, ids


def interpolate_charuco(corners, ids, gray):
    if ids is None or len(corners) == 0:
        return None, None

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners,
        ids,
        gray,
        board
    )

    if charuco_ids is None or charuco_corners is None:
        return None, None

    return charuco_corners, charuco_ids


summary_rows = []
image_paths = sorted(SRC_DIR.glob("frame_*.png"))

print("Input folder:", SRC_DIR)
print("Total raw frames:", len(image_paths))

for img_path in image_paths:
    img = cv2.imread(str(img_path))

    if img is None:
        continue

    img_320 = cv2.resize(img, (OUT_W, OUT_H))
    gray = cv2.cvtColor(img_320, cv2.COLOR_BGR2GRAY)

    marker_corners, marker_ids = detect_markers(gray)
    charuco_corners, charuco_ids = interpolate_charuco(marker_corners, marker_ids, gray)

    n_corners = 0
    kept = False

    if charuco_corners is not None and charuco_ids is not None:
        n_corners = len(charuco_ids)

        if n_corners >= MIN_CORNERS_TO_KEEP:
            kept = True

            out_img_path = IMG_OUT / img_path.name
            cv2.imwrite(str(out_img_path), img_320)

            out_csv_path = CSV_OUT / img_path.name.replace(".png", ".csv")
            with open(out_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["corner_id", "x", "y"])

                for corner, cid in zip(charuco_corners, charuco_ids):
                    x, y = corner.reshape(2)
                    writer.writerow([int(cid[0]), float(x), float(y)])

            vis = img_320.copy()
            for corner, cid in zip(charuco_corners, charuco_ids):
                x, y = corner.reshape(2)
                x, y = int(round(x)), int(round(y))
                cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
                cv2.putText(
                    vis,
                    str(int(cid[0])),
                    (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

            cv2.imwrite(str(VIS_OUT / img_path.name), vis)

    summary_rows.append({
        "frame": img_path.name,
        "opencv_charuco_corners": n_corners,
        "kept_for_pseudolabel": kept
    })

summary_path = OUT_DIR / "opencv_pseudolabel_summary.csv"

with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["frame", "opencv_charuco_corners", "kept_for_pseudolabel"]
    )
    writer.writeheader()
    writer.writerows(summary_rows)

counts = [r["opencv_charuco_corners"] for r in summary_rows]
kept_count = sum(r["kept_for_pseudolabel"] for r in summary_rows)

print("Saved pseudo-label folder:", OUT_DIR)
print("Saved summary:", summary_path)
print("Frames kept with >=", MIN_CORNERS_TO_KEEP, "corners:", kept_count)
print("Max OpenCV ChArUco corners:", max(counts) if counts else 0)
print("Mean OpenCV ChArUco corners:", sum(counts) / len(counts) if counts else 0)
print("Frames with >= 4 corners:", sum(c >= 4 for c in counts))
print("Frames with >= 6 corners:", sum(c >= 6 for c in counts))
print("Frames with >= 8 corners:", sum(c >= 8 for c in counts))
print("Frames with >= 12 corners:", sum(c >= 12 for c in counts))
