from pathlib import Path
import torch

from torch.utils.data import DataLoader, ConcatDataset, random_split

from configs import load_configuration
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import configs

from data import CharucoDataset
from pseudo_data import PseudoCharucoDataset
from models.net import lModel, dcModel


def latest_checkpoint(ckpt_dir):
    ckpts = sorted(Path(ckpt_dir).glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if len(ckpts) == 0:
        raise RuntimeError(f"No checkpoint found in {ckpt_dir}")
    return str(ckpts[-1])


class SetLearningRate(pl.Callback):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def on_train_start(self, trainer, pl_module):
        print(f"Setting fine-tuning learning rate to {self.lr}")
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

    # 1. Original synthetic/fisheye dataset
    synthetic_train = CharucoDataset(
        config,
        config.train_labels,
        config.train_images,
        visualize=False,
        validation=False,
    )

    synthetic_val = CharucoDataset(
        config,
        config.val_labels,
        config.val_images,
        visualize=False,
        validation=True,
    )

    # 2. Real OpenCV pseudo-labeled dataset
    pseudo_all = PseudoCharucoDataset(
        config,
        "../my_dataset/opencv_pseudolabels_raw_320/images",
        "../my_dataset/opencv_pseudolabels_raw_320/keypoints",
    )

    # 27 pseudo samples -> about 22 train, 5 val
    n_total = len(pseudo_all)
    n_val = max(5, int(0.2 * n_total))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(42)
    pseudo_train, pseudo_val = random_split(
        pseudo_all,
        [n_train, n_val],
        generator=generator,
    )

    print("Synthetic train samples:", len(synthetic_train))
    print("Synthetic val samples:", len(synthetic_val))
    print("Pseudo train samples:", len(pseudo_train))
    print("Pseudo val samples:", len(pseudo_val))

    # Mix synthetic and pseudo-real data.
    # Pseudo is repeated once to give real frames stronger influence,
    # but not enough to completely overfit 27 images.
    mixed_train = ConcatDataset([
        synthetic_train,
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

    # 3. Build model
    model = dcModel(n_ids=config.n_ids)
    train_model = lModel(model)

    # 4. Load latest Option A / normal DeepChArUco checkpoint
    source_ckpt = latest_checkpoint("tb_logs/ckpts_deepcharuco/")
    print("Loading source checkpoint:", source_ckpt)

    ckpt = torch.load(source_ckpt, map_location="cpu")
    missing, unexpected = train_model.load_state_dict(ckpt["state_dict"], strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # 5. Save pseudo fine-tuned checkpoints separately
    logger = TensorBoardLogger("tb_logs", name="deepcharuco_pseudo_finetune")

    checkpoint_callback = ModelCheckpoint(
        dirpath="tb_logs/ckpts_deepcharuco_pseudo/",
        save_top_k=5,
        monitor="val_loss",
        mode="min",
    )

    lr_callback = SetLearningRate(lr=1e-5)

    trainer = pl.Trainer(
        max_epochs=25,
        logger=logger,
        accelerator="auto",
        callbacks=[checkpoint_callback, lr_callback],
    )

    trainer.fit(train_model, train_loader, val_loader)
    