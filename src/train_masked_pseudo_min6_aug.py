from pathlib import Path
import torch

from torch.utils.data import DataLoader, ConcatDataset, random_split

from configs import load_configuration
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import configs

from masked_data import SyntheticMaskedDataset, PseudoMaskedDataset
from models.net import lModel, dcModel


def latest_checkpoint(ckpt_dir):
    ckpts = sorted(
        Path(ckpt_dir).glob("*.ckpt"),
        key=lambda p: p.stat().st_mtime
    )

    if len(ckpts) == 0:
        raise RuntimeError(f"No checkpoint found in {ckpt_dir}")

    return str(ckpts[-1])


class SetLearningRate(pl.Callback):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def on_train_start(self, trainer, pl_module):
        print(f"Setting masked fine-tuning learning rate to {self.lr}")

        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = self.lr


def make_loader(dataset, batch_size, shuffle, num_workers):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    if num_workers > 0:
        kwargs["prefetch_factor"] = 10

    return DataLoader(**kwargs)


if __name__ == "__main__":
    config = load_configuration(configs.CONFIG_PATH)

    synthetic_train = SyntheticMaskedDataset(
        config,
        config.train_labels,
        config.train_images,
        visualize=False,
        validation=False,
    )

    synthetic_val = SyntheticMaskedDataset(
        config,
        config.val_labels,
        config.val_images,
        visualize=False,
        validation=True,
    )

    pseudo_all = PseudoMaskedDataset(
        config,
        "../my_dataset/opencv_pseudolabels_raw_320_min6_aug/images",
        "../my_dataset/opencv_pseudolabels_raw_320_min6_aug/keypoints",
    )

    n_total = len(pseudo_all)
    n_val = max(10, int(0.2 * n_total))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(42)

    pseudo_train, pseudo_val = random_split(
        pseudo_all,
        [n_train, n_val],
        generator=generator,
    )

    print("Synthetic train samples:", len(synthetic_train))
    print("Synthetic val samples:", len(synthetic_val))
    print("Pseudo min6_aug total samples:", len(pseudo_all))
    print("Pseudo min6_aug train samples:", len(pseudo_train))
    print("Pseudo min6_aug val samples:", len(pseudo_val))

    # For masked pseudo training, pseudo samples are repeated more strongly
    # because their loss only applies on a few labeled cells.
    mixed_train = ConcatDataset([
        synthetic_train,
        pseudo_train,
        pseudo_train,
        pseudo_train,
    ])

    mixed_val = ConcatDataset([
        synthetic_val,
        pseudo_val,
    ])

    print("Mixed train samples:", len(mixed_train))
    print("Mixed val samples:", len(mixed_val))

    train_loader = make_loader(
        mixed_train,
        batch_size=config.bs_train,
        shuffle=True,
        num_workers=config.num_workers,
    )

    val_loader = make_loader(
        mixed_val,
        batch_size=config.bs_val,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = dcModel(n_ids=config.n_ids)
    train_model = lModel(model)

    source_ckpt = latest_checkpoint("tb_logs/ckpts_deepcharuco/")
    print("Loading source checkpoint:", source_ckpt)

    ckpt = torch.load(source_ckpt, map_location="cpu")
    missing, unexpected = train_model.load_state_dict(
        ckpt["state_dict"],
        strict=False
    )

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    logger = TensorBoardLogger(
        "tb_logs",
        name="deepcharuco_masked_pseudo_min6_aug"
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="tb_logs/ckpts_deepcharuco_masked_pseudo_min6_aug/",
        save_top_k=10,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=8,
        logger=logger,
        accelerator="auto",
        callbacks=[
            checkpoint_callback,
            SetLearningRate(lr=1e-5),
        ],
    )

    trainer.fit(train_model, train_loader, val_loader)
    