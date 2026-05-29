import cv2
import numpy as np

from configs import load_configuration
from aruco_utils import get_board, board_image, draw_inner_corners

config = load_configuration("config.yaml")

board = get_board(config)

# Generate synthetic board exactly like training
img, synthetic_corners = board_image(
    board,
    (480, 480),
    config.row_count,
    config.col_count
)

synthetic_ids = np.arange(config.n_ids)

# Draw synthetic labels in red
img_synthetic = draw_inner_corners(
    img,
    synthetic_corners,
    synthetic_ids,
    draw_ids=True,
    radius=4,
    color=(0, 0, 255)
)

# Now let OpenCV detect the same generated board
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
marker_corners, marker_ids, _ = detector.detectMarkers(gray)

retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
    marker_corners,
    marker_ids,
    gray,
    board
)

img_opencv = img.copy()

if charuco_ids is not None:
    for corner, cid in zip(charuco_corners, charuco_ids.flatten()):
        x, y = corner.ravel()
        cv2.circle(img_opencv, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.putText(
            img_opencv,
            str(int(cid)),
            (int(x) + 5, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

cv2.imwrite("synthetic_training_ids.png", img_synthetic)
cv2.imwrite("opencv_detected_ids.png", img_opencv)

print("Saved synthetic_training_ids.png")
print("Saved opencv_detected_ids.png")
print("OpenCV detected IDs:", None if charuco_ids is None else charuco_ids.flatten())
