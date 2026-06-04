import os
import glob
import csv
from pathlib import Path

import cv2
import numpy as np


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
calib_path = Path("../my_dataset/initial_calibration_from_pseudolabels/initial_fisheye_calibration.npz")
raw_image_dir = Path("../my_dataset/raw_frames")

out_dir = Path("../my_dataset/raw_frames_320_undistorted_fisheye")
out_img_dir = out_dir / "images"
out_vis_dir = out_dir / "comparison_vis"
out_img_dir.mkdir(parents=True, exist_ok=True)
out_vis_dir.mkdir(parents=True, exist_ok=True)

summary_csv = out_dir / "opencv_raw_vs_undistorted_summary.csv"


# ------------------------------------------------------------
# Board setup
# ------------------------------------------------------------
squares_x = 7
squares_y = 7
square_length = 0.14285714285714285
marker_length = 0.10

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

try:
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        aruco_dict
    )
except Exception:
    board = cv2.aruco.CharucoBoard_create(
        squares_x,
        squares_y,
        square_length,
        marker_length,
        aruco_dict
    )


def detect_charuco(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    parameters = cv2.aruco.DetectorParameters()

    try:
        if hasattr(cv2.aruco, "CharucoDetector"):
            charuco_params = cv2.aruco.CharucoParameters()
            detector = cv2.aruco.CharucoDetector(board, charuco_params, parameters)
            charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        else:
            if hasattr(cv2.aruco, "ArucoDetector"):
                detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                marker_corners, marker_ids, _ = detector.detectMarkers(gray)
            else:
                marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                    gray,
                    aruco_dict,
                    parameters=parameters
                )

            if marker_ids is None or len(marker_ids) == 0:
                return []

            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                board
            )

        if charuco_ids is None or charuco_corners is None:
            return []

        pts = []
        for corner, cid in zip(charuco_corners, charuco_ids.flatten()):
            x, y = corner.ravel()
            pts.append((int(cid), float(x), float(y)))

        return pts

    except Exception:
        return []


def draw_points(image, pts, color, prefix):
    vis = image.copy()

    for cid, x, y in pts:
        cv2.circle(vis, (int(round(x)), int(round(y))), 4, color, -1)
        cv2.putText(
            vis,
            f"{prefix}{cid}",
            (int(round(x)) + 4, int(round(y)) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA
        )

    return vis


# ------------------------------------------------------------
# Load calibration
# ------------------------------------------------------------
data = np.load(calib_path, allow_pickle=True)
K = data["K"].astype(np.float64)
D = data["D"].astype(np.float64)
image_size = tuple(data["image_size"].astype(int).tolist())

print("Loaded fisheye calibration:")
print("K:")
print(K)
print("D:", D.ravel())
print("image_size:", image_size)


# ------------------------------------------------------------
# Build undistortion map
# ------------------------------------------------------------
w, h = image_size

# balance=0 keeps fewer black borders and stronger crop.
# balance=1 keeps more FOV but can create more black/warped border.
balance = 0.0

new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K,
    D,
    (w, h),
    np.eye(3),
    balance=balance,
    new_size=(w, h),
    fov_scale=1.0
)

map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K,
    D,
    np.eye(3),
    new_K,
    (w, h),
    cv2.CV_16SC2
)

print("new_K:")
print(new_K)
print("balance:", balance)


# ------------------------------------------------------------
# Process frames
# ------------------------------------------------------------
image_paths = sorted(raw_image_dir.glob("frame_*.png"))

rows = []

for img_path in image_paths:
    frame = img_path.name

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    # Important: calibration is 320 x 240, so resize first.
    raw_320 = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    undist = cv2.remap(
        raw_320,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    out_img_path = out_img_dir / frame
    cv2.imwrite(str(out_img_path), undist)

    raw_pts = detect_charuco(raw_320)
    undist_pts = detect_charuco(undist)

    raw_vis = draw_points(raw_320, raw_pts, (0, 255, 0), "R")
    undist_vis = draw_points(undist, undist_pts, (0, 165, 255), "U")

    side = np.hstack([raw_vis, undist_vis])
    cv2.imwrite(str(out_vis_dir / frame.replace(".png", "_raw_vs_undist.png")), side)

    rows.append({
        "frame": frame,
        "raw_corners": len(raw_pts),
        "undistorted_corners": len(undist_pts),
        "delta": len(undist_pts) - len(raw_pts),
    })

with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["frame", "raw_corners", "undistorted_corners", "delta"]
    )
    writer.writeheader()
    writer.writerows(rows)

print()
print("Finished.")
print("Undistorted images:", out_img_dir)
print("Visual comparisons:", out_vis_dir)
print("Summary:", summary_csv)

raw_frames = sum(1 for r in rows if r["raw_corners"] > 0)
undist_frames = sum(1 for r in rows if r["undistorted_corners"] > 0)

print()
print("SUMMARY")
print("Total frames:", len(rows))
print("Raw OpenCV frames:", raw_frames)
print("Undistorted OpenCV frames:", undist_frames)
print("Raw total corners:", sum(r["raw_corners"] for r in rows))
print("Undistorted total corners:", sum(r["undistorted_corners"] for r in rows))
print("Average raw corners/frame:", round(sum(r["raw_corners"] for r in rows) / len(rows), 3))
print("Average undistorted corners/frame:", round(sum(r["undistorted_corners"] for r in rows) / len(rows), 3))
