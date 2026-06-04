import os
import glob
import csv
import cv2
import numpy as np

# Existing sparse OpenCV pseudo-labels
input_csv_dir = "../my_dataset/opencv_pseudolabels_raw_320_min4/keypoints"

# Raw 320x240 frames
image_dir = "../my_dataset/opencv_pseudolabels_raw_320_min4/images"

# New dense pseudo-label output
output_csv_dir = "../my_dataset/opencv_pseudolabels_raw_320_min4_dense_homography_conservative/keypoints"
output_vis_dir = "../my_dataset/opencv_pseudolabels_raw_320_min4_dense_homography_conservative/visual_check"

os.makedirs(output_csv_dir, exist_ok=True)
os.makedirs(output_vis_dir, exist_ok=True)

# 7x7 squares -> 6x6 inner ChArUco corners
inner_cols = 6
inner_rows = 6
n_ids = 36

# Conservative settings
min_points_for_homography = 8
max_median_reproj_error = 2.0
max_mean_reproj_error = 3.0

# Expand visible board-coordinate range slightly
board_margin = 0.05


def id_to_board_xy(cid):
    col = cid % inner_cols
    row = cid // inner_cols
    return float(col), float(row)


def read_sparse_csv(csv_path):
    pts = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                cid = int(float(row["corner_id"]))
                x = float(row["x"])
                y = float(row["y"])
            except Exception:
                continue

            if 0 <= cid < n_ids:
                pts.append((cid, x, y))

    return pts


def write_csv(csv_path, pts):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["corner_id", "x", "y"])
        for cid, x, y in pts:
            writer.writerow([cid, x, y])


def project_points(H, board_points):
    pts = np.asarray(board_points, dtype=np.float32).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return proj


csv_paths = sorted(glob.glob(os.path.join(input_csv_dir, "frame_*.csv")))

report_rows = []

for csv_path in csv_paths:
    name = os.path.basename(csv_path)
    stem = os.path.splitext(name)[0]
    img_path = os.path.join(image_dir, stem + ".png")

    sparse = read_sparse_csv(csv_path)

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    # If not enough sparse points, just copy original sparse labels.
    if len(sparse) < min_points_for_homography:
        dense = sparse
        status = "copied_too_few_points"
        median_err = np.nan
        mean_err = np.nan
    else:
        board_pts = []
        image_pts = []

        for cid, x, y in sparse:
            bx, by = id_to_board_xy(cid)
            board_pts.append([bx, by])
            image_pts.append([x, y])

        board_pts = np.asarray(board_pts, dtype=np.float32)
        image_pts = np.asarray(image_pts, dtype=np.float32)

        H, mask = cv2.findHomography(
            board_pts,
            image_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=6.0
        )

        if H is None:
            dense = sparse
            status = "copied_homography_failed"
            median_err = np.nan
            mean_err = np.nan
        else:
            reproj = project_points(H, board_pts)
            errors = np.linalg.norm(reproj - image_pts, axis=1)
            median_err = float(np.median(errors))
            mean_err = float(np.mean(errors))

            if median_err > max_median_reproj_error or mean_err > max_mean_reproj_error:
                dense = sparse
                status = "copied_bad_homography"
            else:
                existing_ids = {cid for cid, _, _ in sparse}

                sparse_cols = [cid % inner_cols for cid, _, _ in sparse]
                sparse_rows = [cid // inner_cols for cid, _, _ in sparse]

                min_col = min(sparse_cols) - board_margin
                max_col = max(sparse_cols) + board_margin
                min_row = min(sparse_rows) - board_margin
                max_row = max(sparse_rows) + board_margin

                dense_dict = {cid: (x, y) for cid, x, y in sparse}

                candidate_board = []
                candidate_ids = []

                for cid in range(n_ids):
                    if cid in existing_ids:
                        continue

                    col = cid % inner_cols
                    row = cid // inner_cols

                    # Only fill missing corners inside/near the sparse visible region.
                    if not (min_col <= col <= max_col and min_row <= row <= max_row):
                        continue

                    candidate_ids.append(cid)
                    candidate_board.append([float(col), float(row)])

                if len(candidate_board) > 0:
                    candidate_proj = project_points(H, candidate_board)

                    for cid, (x, y) in zip(candidate_ids, candidate_proj):
                        # Keep only points inside the 320x240 image.
                        if 0 <= x < w and 0 <= y < h:
                            dense_dict[cid] = (float(x), float(y))

                dense = [(cid, xy[0], xy[1]) for cid, xy in sorted(dense_dict.items())]
                status = "densified"

    out_csv = os.path.join(output_csv_dir, name)
    write_csv(out_csv, dense)

    # Visualization
    vis = img.copy()

    sparse_ids = {cid for cid, _, _ in sparse}

    for cid, x, y in dense:
        if cid in sparse_ids:
            color = (0, 255, 0)      # green = original OpenCV pseudo-label
            prefix = "O"
        else:
            color = (0, 165, 255)    # orange = homography-filled label
            prefix = "H"

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

    out_vis = os.path.join(output_vis_dir, stem + "_dense_check.png")
    cv2.imwrite(out_vis, vis)

    report_rows.append({
        "frame": stem + ".png",
        "sparse_points": len(sparse),
        "dense_points": len(dense),
        "added_points": len(dense) - len(sparse),
        "status": status,
        "median_reproj_error": median_err,
        "mean_reproj_error": mean_err,
    })

# Save report
report_path = "../my_dataset/opencv_pseudolabels_raw_320_min4_dense_homography_conservative/dense_pseudo_report.csv"
os.makedirs(os.path.dirname(report_path), exist_ok=True)

with open(report_path, "w", newline="") as f:
    fieldnames = [
        "frame",
        "sparse_points",
        "dense_points",
        "added_points",
        "status",
        "median_reproj_error",
        "mean_reproj_error",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(report_rows)

print("Dense pseudo-label generation finished.")
print("Input sparse CSV dir:", input_csv_dir)
print("Output dense CSV dir:", output_csv_dir)
print("Output visual dir:", output_vis_dir)
print("Report:", report_path)

# Quick summary
total_sparse = sum(r["sparse_points"] for r in report_rows)
total_dense = sum(r["dense_points"] for r in report_rows)
densified_frames = sum(1 for r in report_rows if r["status"] == "densified")

print()
print("SUMMARY")
print("Frames processed:", len(report_rows))
print("Frames densified:", densified_frames)
print("Total sparse points:", total_sparse)
print("Total dense points:", total_dense)
print("Total added points:", total_dense - total_sparse)
