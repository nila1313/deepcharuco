import os
import glob
import cv2

image_dir = "../my_dataset/raw_frames"

CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 7
SQUARE_LENGTH = 0.14285714285714285
MARKER_LENGTH = 0.10

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)

board = cv2.aruco.CharucoBoard(
    (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dict
)

detector_params = cv2.aruco.DetectorParameters()

image_paths = sorted(glob.glob(os.path.join(image_dir, "frame_*.png")))[:20]

for path in image_paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = cv2.aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=detector_params
    )

    print(os.path.basename(path))

    if ids is None:
        print("  ArUco markers detected: 0")
        print("  ChArUco corners detected: 0")
        print("----------------------")
        continue

    print("  ArUco markers detected:", len(ids))

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners,
        ids,
        gray,
        board
    )

    if charuco_ids is None:
        print("  ChArUco corners detected: 0")
    else:
        print("  ChArUco corners detected:", len(charuco_ids))

    print("----------------------")