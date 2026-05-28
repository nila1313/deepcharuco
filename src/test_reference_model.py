import os
import cv2
from inference import load_models, infer_image

deepc_path = "./reference/longrun-epoch=99-step=369700.ckpt"
refinenet_path = "./reference/second-refinenet-epoch-100-step=373k.ckpt"

n_ids = 16
device = "cpu"

print("Loading reference models...")
deepc, refinenet = load_models(
    deepc_path,
    refinenet_path,
    n_ids=n_ids,
    device=device
)
print("Reference models loaded.")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

board = cv2.aruco.CharucoBoard(
    (5, 5),
    0.01,
    0.0075,
    aruco_dict
)

try:
    board_gray = board.generateImage((800, 800))
except Exception:
    board_gray = cv2.aruco.drawPlanarBoard(board, (800, 800), 20, 1)

board_bgr = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)

os.makedirs("../my_dataset/reference_test", exist_ok=True)
cv2.imwrite("../my_dataset/reference_test/default_clean_board.png", board_bgr)

keypoints, out_img = infer_image(
    board_bgr,
    n_ids,
    deepc,
    refinenet,
    draw_pred=True,
    device=device
)

print("Detected keypoints:", len(keypoints))
print(keypoints)

cv2.imwrite("../my_dataset/reference_test/default_clean_board_prediction.png", out_img)
print("Saved to ../my_dataset/reference_test")