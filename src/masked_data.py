import numpy as np

from data import CharucoDataset
from pseudo_data import PseudoCharucoDataset


class SyntheticMaskedDataset(CharucoDataset):
    """
    Synthetic samples are fully labeled.

    For synthetic training, every grid cell is trusted:
        - real ChArUco cells
        - dustbin/background cells

    So the mask is 1 everywhere.
    """

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        loc, ids = sample["label"]

        # All cells are valid for synthetic data.
        mask = np.ones_like(ids, dtype=np.float32)

        sample["mask"] = mask

        return sample


class PseudoMaskedDataset(PseudoCharucoDataset):
    """
    Pseudo-real samples are only partially labeled.

    For OpenCV pseudo labels, only detected ChArUco cells are trusted.
    Unlabeled cells should NOT be treated as true background, because OpenCV
    may have missed visible board corners.

    So the mask is:
        1 for cells where ids < n_ids
        0 for unlabeled/dustbin cells
    """

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        loc, ids = sample["label"]

        # Valid pseudo-labeled cells are the OpenCV-labeled ChArUco IDs.
        # Dustbin id == config.n_ids.
        mask = (ids < self.configs.n_ids).astype(np.float32)

        sample["mask"] = mask

        return sample