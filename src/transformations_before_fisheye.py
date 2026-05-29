import imgaug
import albumentations as A
import numpy as np
import cv2
import random
from aruco_utils import board_image, get_board
from custom_aug.custom_aug import PasteBoard


# Monkey patching Albumentations 1.3.0 CoarseDropout bug :)
# https://github.com/albumentations-team/albumentations/pull/1330
def apply_to_keypoints(self, keypoints, holes, **params):
    result = set(keypoints)
    for hole in holes:
        for kp in keypoints:
            if self._keypoint_in_hole(kp, hole):
                result.discard(kp)
    return list(result)

A.CoarseDropout.apply_to_keypoints = apply_to_keypoints  # noqa: E305


def board_transformations(refinenet, input_size):
    """
    More realistic augmentation for our real omnidirectional ChArUco video.

    Main model:
    - board is large
    - partially cropped
    - moderate rotation
    - moderate shear
    - fewer random occlusions

    RefineNet:
    - smaller transformations
    - focuses on local refinement
    """

    if refinenet:
        trans = (-0.05, 0.05)
        scale = (0.85, 1.15)
        rotate = (-10, 10)
        shear = (-5, 5)
        cd_p = 0.0
    else:
        trans = (-0.20, 0.20)
        scale = (0.70, 1.35)
        rotate = (-35, 35)
        shear = (-15, 15)
        cd_p = 0.15

    max_holes = 6
    min_holes = 1
    maxs = 64
    mins = 16

    transf = [
        A.PadIfNeeded(
            min_height=input_size[1],
            min_width=input_size[0],
            always_apply=True,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
        ),

        A.OneOf(
    [
        A.Affine(
            scale=scale,
            rotate=(-45, 45),
            shear=shear,
            translate_percent=trans,
            keep_ratio=True,
            fit_output=False,
            always_apply=True,
        ),
        A.Affine(
            scale=scale,
            rotate=(135, 225),
            shear=shear,
            translate_percent=trans,
            keep_ratio=True,
            fit_output=False,
            always_apply=True,
        ),
    ],
    p=1.0,
),

        A.Resize(
            height=input_size[1],
            width=input_size[0],
            always_apply=True,
        ),

        A.OneOf(
            [
                A.CoarseDropout(
                    max_holes=max_holes,
                    max_height=maxs,
                    max_width=maxs,
                    min_holes=min_holes,
                    min_height=mins,
                    min_width=mins,
                    mask_fill_value=0,
                ),

                *[
                    A.CoarseDropout(
                        max_holes=max_holes,
                        max_height=maxs,
                        max_width=maxs,
                        min_holes=min_holes,
                        min_height=mins,
                        min_width=mins,
                        fill_value=f,
                        mask_fill_value=255,
                    )
                    for f in [0, 128, 255]
                ],
            ],
            p=cd_p,
        ),
    ]

    return A.Compose(
        transf,
        keypoint_params=A.KeypointParams(
            format="xy",
            label_fields=["ids"],
            remove_invisible=True,
        ),
    )


class Transformation:
    """
    Class to apply augmentation on COCO dataset to train deepcharuco.
    Steps:
    0) Choose if is a negative sample, in that case just return an augmented coco
    1) Augment board image (+ mask + corners)
    2) ~ Histogram matching of board image given coco image
    3) Paste image on coco image
    4) Augment coco + board image
    5) Profit!
    """
    def __init__(self, configs, negative_p=0.05, refinenet=False, seed=None):
        self.seed = seed
        self.negative_p = negative_p
        if seed is not None:
            random.seed(seed)
            imgaug.random.seed(seed)

        self.refinenet = refinenet

        min_r = min(configs.input_size)
        board = get_board(configs)
        board_img, corners = board_image(board, (min_r, min_r),
                                         configs.row_count, configs.col_count)

        self.board_img = board_img
        self.corners = corners
        self.ids = np.arange(self.corners.shape[0])
        self.board_mask = np.full((board_img.shape[0], board_img.shape[1]),
                                  dtype=np.uint8, fill_value=255)

        # 1) Create transformation for board image
        self._transf_board = board_transformations(self.refinenet, configs.input_size)

        # 1bis) COCO transformation
        self._transf_coco = A.Compose([
            A.Flip(p=0.5),
            A.Rotate(limit=(-180, 180), crop_border=True, p=0.5),
            A.PadIfNeeded(min_height=configs.input_size[1],
                          min_width=configs.input_size[0], always_apply=True,
                          border_mode=cv2.BORDER_CONSTANT, value=0,
                          mask_value=0),
            A.RandomCrop(height=configs.input_size[1],
                         width=configs.input_size[0], always_apply=True),
        ])

        # 2 + 3) Apply histogram matching then Paste transformation
        self._transf_joint = A.Compose([
            PasteBoard(always_apply=True),

            A.ColorJitter(brightness=0, p=0.5),
            A.RGBShift(p=0.5),

            # Augmentations as from paper
            A.GaussNoise(p=0.5),
            A.MotionBlur(blur_limit=5, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.25),
            A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.8, 0.35),
                                       contrast_limit=0, p=0.5),

        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['ids'],
                                            remove_invisible=True)
        )

    def _transform_board(self):
        t_res = self._transf_board(image=self.board_img,
                                   mask=self.board_mask,
                                   keypoints=self.corners,
                                   ids=self.ids)
        return t_res

    def __call__(self, coco_img):
        return self.transform(coco_img)

    def transform(self, coco_img):
        res = self._transform_board()  # Generate board image

        # Adapt coco image to input_size
        coco_img = self._transf_coco(image=coco_img)['image']

        # We also generate negative instances without board (if not refinenet)
        isnegative = False if self.refinenet else (random.random() < self.negative_p)

        # Apply joint pipeline
        res = self._transf_joint(**res, target=coco_img, isnegative=isnegative)
        return {'image': res['image'], 'keypoints': res['keypoints'],
                'ids': res['ids'], 'isnegative': isnegative}
