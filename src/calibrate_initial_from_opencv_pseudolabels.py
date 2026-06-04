import csv
from pathlib import Path
import cv2
import numpy as np


# ------------------------------------------------------------
# Input pseudo-labels
# ------------------------------------------------------------
keypoint_dir = Path("../my_dataset/opencv_pseudolabels_raw_320_min4/keypoints")
image_dir = Path("../my_dataset/opencv_pseudolabels_raw_320_min4/images")

# Output
out_dir = Path("../my_dataset/initial_calibration_from_pseudolabels")
out_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Board setup
# 7 x 7 squares -> 6 x 6 inner ChArUco corners = 36
# ------------------------------------------------------------
inner_cols = 6
inner_rows = 6
n_ids = 36

square_len = 0.14285714285714285

min_corners_per_frame = 6


def id_to_object_point(cid):
    """
    OpenCV ChArUco inner-corner ID -> 3D board coordinate.
    ID layout is row-major on the 6 x 6 inner-corner grid.
    """
    col = cid % inner_cols
    row = cid // inner_cols

    x = col * square_len
    y = row * square_len
    z = 0.0

    return [x, y, z]


def read_csv_points(csv_path):
    obj_pts = []
    img_pts = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            cid = int(float(row["corner_id"]))
            x = float(row["x"])
            y = float(row["y"])

            if 0 <= cid < n_ids:
                obj_pts.append(id_to_object_point(cid))
                img_pts.append([x, y])

    return np.asarray(obj_pts, dtype=np.float32), np.asarray(img_pts, dtype=np.float32)


# ------------------------------------------------------------
# Collect calibration frames
# ------------------------------------------------------------
objpoints = []
imgpoints = []
used_frames = []

image_size = None

csv_paths = sorted(keypoint_dir.glob("frame_*.csv"))

for csv_path in csv_paths:
    frame_name = csv_path.stem + ".png"
    img_path = image_dir / frame_name

    if not img_path.exists():
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]
    image_size = (w, h)

    obj, imgp = read_csv_points(csv_path)

    if len(obj) < min_corners_per_frame:
        continue

    objpoints.append(obj.reshape(-1, 1, 3))
    imgpoints.append(imgp.reshape(-1, 1, 2))
    used_frames.append(frame_name)


print("Initial calibration from OpenCV pseudo-labels")
print("---------------------------------------------")
print("CSV folder:", keypoint_dir)
print("Image folder:", image_dir)
print("Image size:", image_size)
print("Total CSV files:", len(csv_paths))
print("Used frames:", len(used_frames))
print("Minimum corners/frame:", min_corners_per_frame)

if image_size is None or len(objpoints) < 5:
    raise RuntimeError("Not enough usable frames for calibration.")


# ------------------------------------------------------------
# Try fisheye calibration first
# ------------------------------------------------------------
w, h = image_size

K_init = np.array([
    [w, 0, w / 2.0],
    [0, w, h / 2.0],
    [0, 0, 1.0],
], dtype=np.float64)

D_init = np.zeros((4, 1), dtype=np.float64)

flags_fisheye = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    + cv2.fisheye.CALIB_FIX_SKEW
)

print()
print("Trying cv2.fisheye.calibrate...")

try:
    rms_fisheye, K_fisheye, D_fisheye, rvecs_fisheye, tvecs_fisheye = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        image_size,
        K_init.copy(),
        D_init.copy(),
        flags=flags_fisheye,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6),
    )

    print("Fisheye calibration SUCCESS")
    print("RMS:", rms_fisheye)
    print("K:")
    print(K_fisheye)
    print("D:")
    print(D_fisheye.ravel())

    np.savez(
        out_dir / "initial_fisheye_calibration.npz",
        model="fisheye",
        image_size=np.asarray(image_size),
        K=K_fisheye,
        D=D_fisheye,
        rms=rms_fisheye,
        used_frames=np.asarray(used_frames),
    )

except Exception as e:
    print("Fisheye calibration FAILED")
    print("Reason:", repr(e))


# ------------------------------------------------------------
# Also run normal pinhole calibration as fallback/reference
# ------------------------------------------------------------
print()
print("Trying cv2.calibrateCamera pinhole fallback...")

objpoints_pin = [o.reshape(-1, 3) for o in objpoints]
imgpoints_pin = [p.reshape(-1, 2) for p in imgpoints]

try:
    rms_pin, K_pin, D_pin, rvecs_pin, tvecs_pin = cv2.calibrateCamera(
        objpoints_pin,
        imgpoints_pin,
        image_size,
        None,
        None,
        flags=cv2.CALIB_RATIONAL_MODEL,
    )

    print("Pinhole calibration SUCCESS")
    print("RMS:", rms_pin)
    print("K:")
    print(K_pin)
    print("D:")
    print(D_pin.ravel())

    np.savez(
        out_dir / "initial_pinhole_calibration.npz",
        model="pinhole",
        image_size=np.asarray(image_size),
        K=K_pin,
        D=D_pin,
        rms=rms_pin,
        used_frames=np.asarray(used_frames),
    )

except Exception as e:
    print("Pinhole calibration FAILED")
    print("Reason:", repr(e))


# ------------------------------------------------------------
# Save used frame list
# ------------------------------------------------------------
with open(out_dir / "used_frames.txt", "w") as f:
    for name in used_frames:
        f.write(name + "\n")

print()
print("Saved outputs to:", out_dir)
print("Used frame list:", out_dir / "used_frames.txt")
