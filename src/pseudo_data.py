import csv
import os
from pathlib import Path

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset

from data import create_label
from models.model_utils import pre_bgr_image


class PseudoCharucoDataset(Dataset):
    """
    Dataset for real frames with OpenCV pseudo-labels.

    Expects:
    images_folder/
        frame_000000.png
        frame_000014.png
        ...

    keypoints_folder/
        frame_000000.csv
        frame_000014.csv
        ...

    CSV format:
        corner_id,x,y
    """

    def __init__(self, configs, images_folder, keypoints_folder):
        super().__init__()

        self.configs = configs
        self.images_folder = Path(images_folder)
        self.keypoints_folder = Path(keypoints_folder)

        self.image_paths = sorted(self.images_folder.glob("frame_*.png"))

        self.samples = []
        for img_path in self.image_paths:
            csv_path = self.keypoints_folder / img_path.name.replace(".png", ".csv")
            if csv_path.exists():
                self.samples.append((img_path, csv_path))

        print("PseudoCharucoDataset")
        print("Images folder:", self.images_folder)
        print("Keypoints folder:", self.keypoints_folder)
        print("Usable pseudo-labeled samples:", len(self.samples))

        if len(self.samples) == 0:
            raise RuntimeError("No pseudo-labeled samples found. Check image/keypoint folders.")

    def __len__(self):
        return len(self.samples)

    def _read_keypoints_csv(self, csv_path):
        keypoints = []
        kpts_ids = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                cid = int(row["corner_id"])
                x = float(row["x"])
                y = float(row["y"])

                keypoints.append([x, y])
                kpts_ids.append(cid)

        keypoints = np.asarray(keypoints, dtype=np.float32)
        kpts_ids = np.asarray(kpts_ids, dtype=np.int64)

        return keypoints, kpts_ids

    def __getitem__(self, idx):
        img_path, csv_path = self.samples[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        # Your pseudo-label images are already 320 x 240,
        # but keep this safe resize in case one file is different.
        image = cv2.resize(image, tuple(self.configs.input_size))

        keypoints, kpts_ids = self._read_keypoints_csv(csv_path)

        isnegative = False
        dust_bin_ids = self.configs.n_ids

        loc, ids = create_label(
            image=image,
            keypoints=keypoints,
            kpts_ids=kpts_ids,
            isnegative=isnegative,
            dust_bin_ids=dust_bin_ids
        )

        image = pre_bgr_image(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        sample = {
            "image": image,
            "label": (loc, ids),
        }

        return sample