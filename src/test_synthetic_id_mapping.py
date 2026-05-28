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

# Draw training IDs in red
img_synthetic = draw_inner_corners(
    img,
    synthetic_corners,
    synthetic_ids,
    draw_ids=True,
    radius=4,
    color=(0, 0, 255)
)

cv2.imwrite("synthetic_training_ids.png", img_synthetic)
print("Saved synthetic_training_ids.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
parameters = cv2.aruco.DetectorParameters()

charuco_corners = None
charuco_ids = None

# Try newer OpenCV API first
if hasattr(cv2.aruco, "CharucoDetector"):
    try:
        charuco_params = cv2.aruco.CharucoParameters()
        detector = cv2.aruco.CharucoDetector(board, charuco_params, parameters)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        print("Used CharucoDetector API")
    except Exception as e:
        print("CharucoDetector failed:", e)

# Try older OpenCV API if needed
if charuco_ids is None:
    try:
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            marker_corners, marker_ids, _ = detector.detectMarkers(gray)
        else:
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(
                gray,
                aruco_dict,
                parameters=parameters
            )

        if marker_ids is not None:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners,
                marker_ids,
                gray,
                board
            )
            print("Used interpolateCornersCharuco API")
    except Exception as e:
        print("interpolateCornersCharuco failed:", e)

img_opencv = img.copy()

if charuco_ids is not None and charuco_corners is not None:
    for corner, cid in zip(charuco_corners, charuco_ids.flatten()):
        x, y = corner.ravel()
        cv2.circle(img_opencv, (int(round(x)), int(round(y))), 4, (0, 255, 0), -1)
        cv2.putText(
            img_opencv,
            str(int(cid)),
            (int(round(x)) + 5, int(round(y)) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    print("OpenCV detected IDs:", charuco_ids.flatten().tolist())
else:
    print("OpenCV did not detect ChArUco corners on synthetic board")

cv2.imwrite("opencv_detected_ids.png", img_opencv)
print("Saved opencv_detected_ids.png")
