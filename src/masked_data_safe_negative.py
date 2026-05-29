import numpy as np

from data import CharucoDataset
from pseudo_data import PseudoCharucoDataset


class SyntheticSafeNegativeDataset(CharucoDataset):
    """
    Synthetic samples are fully labeled.

    For synthetic data:
        - positive ChArUco cells are trusted
        - dustbin/background cells are trusted

    So mask = 1 everywhere.
    """

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        loc, ids = sample["label"]

        mask = np.ones_like(ids, dtype=np.float32)

        sample["mask"] = mask

        return sample


class PseudoSafeNegativeDataset(PseudoCharucoDataset):
    """
    Pseudo-real samples are partially labeled.

    For pseudo-real data:
        1. OpenCV-labeled ChArUco cells are positive.
        2. Unlabeled cells near the labeled board region are ignored.
        3. Far-away cells outside the expanded board box are treated as safe background.

    This is safer than:
        - full pseudo loss: wrongly treats missed board corners as background
        - positive-only mask: gives no background pressure and causes false positives
    """

    def __init__(
        self,
        configs,
        images_folder,
        keypoints_folder,
        ignore_margin_cells=8,
    ):
        super().__init__(configs, images_folder, keypoints_folder)
        self.ignore_margin_cells = ignore_margin_cells

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        loc, ids = sample["label"]

        h, w = ids.shape

        # Positive cells: OpenCV-labeled ChArUco corners.
        pos_mask = ids < self.configs.n_ids

        # Start by ignoring everything.
        mask = np.zeros_like(ids, dtype=np.float32)

        # Always train positives.
        mask[pos_mask] = 1.0

        pos_y, pos_x = np.where(pos_mask)

        # If no pseudo labels somehow exist, ignore the whole sample.
        if len(pos_y) == 0:
            sample["mask"] = mask
            return sample

        # Bounding box around detected pseudo labels.
        y1 = max(0, int(pos_y.min()) - self.ignore_margin_cells)
        y2 = min(h - 1, int(pos_y.max()) + self.ignore_margin_cells)
        x1 = max(0, int(pos_x.min()) - self.ignore_margin_cells)
        x2 = min(w - 1, int(pos_x.max()) + self.ignore_margin_cells)

        # Ignore the expanded board-like region.
        ignore_region = np.zeros_like(ids, dtype=bool)
        ignore_region[y1:y2 + 1, x1:x2 + 1] = True

        # Safe negatives are outside the expanded board-like region.
        safe_negative = ~ignore_region

        # For safe negatives, ids/loc are already dustbin from create_label().
        # So we can train them as background.
        mask[safe_negative] = 1.0

        # Positives must remain active.
        mask[pos_mask] = 1.0

        sample["mask"] = mask

        return sample