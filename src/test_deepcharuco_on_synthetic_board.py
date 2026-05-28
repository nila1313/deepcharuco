import os
import csv
import cv2
import torch
import numpy as np

from inference import load_models, infer_image


# ============================================================
# Output folder
# ============================================================

output_dir = "../my_dataset/synthetic_board_test"
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# Model checkpoints
# ============================================================

deepc_path = "./tb_logs/ckpts_deepcharuco/epoch=98-step=2376.ckpt"
refinenet_path = "./tb_logs/ckpts_refinenet/epoch=194-step=9360.ckpt"


# ============================================================
# Board configuration
# 7 x 7 squares -> 6 x 6 inner ChArUco corners = 36
# ============================================================

squares_x = 7
squares_y = 7
square_length = 0.14285714285714285
marker_length = 0.10
n_ids = 36

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)


# ============================================================
# Create board, compatible with different OpenCV versions
# ============================================================

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


# ============================================================
# Generate synthetic board image
# ============================================================

img_w, img_h = 320, 240

try:
    board_gray = board.generateImage((img_w, img_h), marginSize=10)
except Exception:
    board_gray = board.draw((img_w, img_h), marginSize=10)

if len(board_gray.shape) == 2:
    board_bgr = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)
else:
    board_bgr = board_gray.copy()

raw_board_path = os.path.join(output_dir, "synthetic_board_raw.png")
cv2.imwrite(raw_board_path, board_bgr)

print("Saved raw synthetic board to:", raw_board_path)


# ============================================================
# OpenCV ChArUco detection
# ============================================================

gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)

parameters = cv2.aruco.DetectorParameters()

try:
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
except Exception:
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters
    )

opencv_points = {}

opencv_vis = board_bgr.copy()

if marker_ids is not None and len(marker_ids) > 0:
    cv2.aruco.drawDetectedMarkers(opencv_vis, marker_corners, marker_ids)

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board
    )

    if charuco_ids is not None and charuco_corners is not None:
        for corner, cid in zip(charuco_corners, charuco_ids.flatten()):
            x, y = corner.ravel()
            cid = int(cid)
            opencv_points[cid] = (float(x), float(y))

            cv2.circle(
                opencv_vis,
                (int(round(x)), int(round(y))),
                4,
                (0, 255, 0),
                -1
            )
            cv2.putText(
                opencv_vis,
                f"O{cid}",
                (int(round(x)) + 4, int(round(y)) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

opencv_vis_path = os.path.join(output_dir, "opencv_ids_on_synthetic_board.png")
cv2.imwrite(opencv_vis_path, opencv_vis)

print("OpenCV detected ChArUco corners:", len(opencv_points))
print("OpenCV IDs:", sorted(opencv_points.keys()))
print("Saved OpenCV visualization to:", opencv_vis_path)


# ============================================================
# Load DeepChArUco model
# ============================================================

device = "cpu"

print()
print("Using device:", device)
print("Loading DeepChArUco models...")

deepc, refinenet = load_models(
    deepc_path,
    refinenet_path,
    n_ids=n_ids,
    device=device
)

print("Models loaded successfully.")


# ============================================================
# Run DeepChArUco inference on synthetic board
# ============================================================

keypoints, deep_vis = infer_image(
    board_bgr,
    n_ids,
    deepc,
    refinenet,
    draw_pred=True,
    device=device
)

keypoints = np.array(keypoints)

deep_points = {}

for kp in keypoints:
    x, y, cid = kp
    cid = int(cid)

    if cid not in deep_points:
        deep_points[cid] = (float(x), float(y))

deep_vis_path = os.path.join(output_dir, "deepcharuco_on_synthetic_board.png")
cv2.imwrite(deep_vis_path, deep_vis)

deep_csv_path = os.path.join(output_dir, "deepcharuco_synthetic_keypoints.csv")

with open(deep_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "corner_id"])
    for kp in keypoints:
        writer.writerow([float(kp[0]), float(kp[1]), int(kp[2])])

print()
print("DeepChArUco detected keypoints:", len(keypoints))
print("DeepChArUco unique IDs:", sorted(deep_points.keys()))
print("Saved DeepChArUco visualization to:", deep_vis_path)
print("Saved DeepChArUco keypoints to:", deep_csv_path)


# ============================================================
# Compare OpenCV and DeepChArUco IDs
# ============================================================

matched_ids = sorted(set(opencv_points.keys()) & set(deep_points.keys()))

errors = []

comparison_vis = board_bgr.copy()

# Draw OpenCV points in green
for cid, (x, y) in opencv_points.items():
    cv2.circle(
        comparison_vis,
        (int(round(x)), int(round(y))),
        4,
        (0, 255, 0),
        -1
    )
    cv2.putText(
        comparison_vis,
        f"O{cid}",
        (int(round(x)) + 4, int(round(y)) - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

# Draw DeepChArUco points in red
for cid, (x, y) in deep_points.items():
    cv2.drawMarker(
        comparison_vis,
        (int(round(x)), int(round(y))),
        (0, 0, 255),
        markerType=cv2.MARKER_TILTED_CROSS,
        markerSize=8,
        thickness=2
    )
    cv2.putText(
        comparison_vis,
        f"D{cid}",
        (int(round(x)) + 4, int(round(y)) + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 0, 255),
        1,
        cv2.LINE_AA
    )

# Draw blue lines for matched IDs
for cid in matched_ids:
    ox, oy = opencv_points[cid]
    dx, dy = deep_points[cid]

    error = np.sqrt((dx - ox) ** 2 + (dy - oy) ** 2)
    errors.append(error)

    cv2.line(
        comparison_vis,
        (int(round(ox)), int(round(oy))),
        (int(round(dx)), int(round(dy))),
        (255, 0, 0),
        1
    )

comparison_path = os.path.join(output_dir, "opencv_vs_deepcharuco_synthetic.png")
cv2.imwrite(comparison_path, comparison_vis)

print()
print("Matched IDs:", matched_ids)
print("Number of matched IDs:", len(matched_ids))

if len(errors) > 0:
    print("Mean pixel error:", float(np.mean(errors)))
    print("Median pixel error:", float(np.median(errors)))
    print("Max pixel error:", float(np.max(errors)))
else:
    print("No matched IDs, so no pixel error computed.")

print("Saved comparison image to:", comparison_path)
print()
print("Synthetic board test finished.")