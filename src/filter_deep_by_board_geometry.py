import os
import glob
import csv
import cv2
import numpy as np

# Input: current no-RefineNet prediction CSVs
input_dir = "../my_dataset/inference_output_best_aug_min6_epoch5_iddust_no_refinenet/keypoints"

# Output: geometry-filtered prediction CSVs
output_dir = "../my_dataset/inference_output_best_aug_min6_epoch5_iddust_no_refinenet_geom_filtered/keypoints"
os.makedirs(output_dir, exist_ok=True)

# Your board: 7x7 squares -> 6x6 inner ChArUco corners
inner_cols = 6
inner_rows = 6
n_ids = 36

# RANSAC reprojection threshold in pixels.
# 20 px is loose enough for fisheye distortion, but should remove very wrong IDs.
ransac_thresh_px = 20.0


def read_points(csv_path):
    points = []

    if not os.path.exists(csv_path):
        return points

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row["x"])
                y = float(row["y"])
                cid = int(float(row["corner_id"]))
            except Exception:
                continue

            if 0 <= cid < n_ids:
                points.append((x, y, cid))

    return points


def board_xy_from_id(cid):
    """
    ChArUco inner-corner ID to ideal board-grid coordinate.
    For a 6x6 inner-corner grid:
      id 0 -> (0,0)
      id 1 -> (1,0)
      ...
      id 6 -> (0,1)
    """
    col = cid % inner_cols
    row = cid // inner_cols
    return float(col), float(row)


def write_points(csv_path, points):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "corner_id"])
        for x, y, cid in points:
            writer.writerow([x, y, cid])


csv_paths = sorted(glob.glob(os.path.join(input_dir, "frame_*.csv")))

total_before = 0
total_after = 0
frames_with_filter = 0

for csv_path in csv_paths:
    name = os.path.basename(csv_path)
    points = read_points(csv_path)
    total_before += len(points)

    out_path = os.path.join(output_dir, name)

    # Need at least 4 points for homography/RANSAC.
    # If fewer than 4, keep them unchanged because we cannot verify geometry.
    if len(points) < 4:
        filtered = points
        write_points(out_path, filtered)
        total_after += len(filtered)
        continue

    board_pts = []
    image_pts = []

    for x, y, cid in points:
        bx, by = board_xy_from_id(cid)
        board_pts.append([bx, by])
        image_pts.append([x, y])

    board_pts = np.asarray(board_pts, dtype=np.float32)
    image_pts = np.asarray(image_pts, dtype=np.float32)

    H, inlier_mask = cv2.findHomography(
        board_pts,
        image_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh_px
    )

    if H is None or inlier_mask is None:
        filtered = []
    else:
        inlier_mask = inlier_mask.ravel().astype(bool)
        filtered = [pt for pt, keep in zip(points, inlier_mask) if keep]

    frames_with_filter += 1
    total_after += len(filtered)
    write_points(out_path, filtered)

print("Geometry filtering finished.")
print("Input dir:", input_dir)
print("Output dir:", output_dir)
print("Frames processed:", len(csv_paths))
print("Frames with >=4 points:", frames_with_filter)
print("Total points before:", total_before)
print("Total points after:", total_after)
print("RANSAC threshold px:", ransac_thresh_px)
