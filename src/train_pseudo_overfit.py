from pathlib import Path
import torch

from torch.utils.data import DataLoader

from configs import load_configuration
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import configs

from pseudo_data import PseudoCharucoDataset
from models.net import lModel, dcModel


class SetLearningRate(pl.Callback):
    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def on_train_start(self, trainer, pl_module):
        print(f"Setting overfit learning rate to {self.lr}")
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


def choose_source_checkpoint():
    candidates = [
        Path("./tb_logs/ckpts_deepcharuco_pseudo/best_optionB_epoch6.ckpt"),
        Path("./tb_logs/ckpts_deepcharuco_pseudo/epoch=6-step=322.ckpt"),
    ]

    for c in candidates:
        if c.exists():
            return str(c)

    ckpts = sorted(
        Path("./tb_logs/ckpts_deepcharuco_pseudo").glob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
    )

    if len(ckpts) == 0:
        raise RuntimeError("No pseudo checkpoint found.")

    return str(ckpts[0])


if __name__ == "__main__":
    config = load_configuration(configs.CONFIG_PATH)

    pseudo_dataset = PseudoCharucoDataset(
        config,
        "../my_dataset/opencv_pseudolabels_raw_320/images",
        "../my_dataset/opencv_pseudolabels_raw_320/keypoints",
    )

    print("Pseudo-only overfit samples:", len(pseudo_dataset))

    train_loader = make_loader(
        pseudo_dataset,
        batch_size=min(4, config.bs_train),
        shuffle=True,
        num_workers=config.num_workers,
    )

    # Use the same pseudo dataset as validation.
    # This is intentional: this is a memorization sanity test, not a real validation.
    val_loader = make_loader(
        pseudo_dataset,
        batch_size=min(4, config.bs_val),
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = dcModel(n_ids=config.n_ids)
    train_model = lModel(model)

    source_ckpt = choose_source_checkpoint()
    print("Loading source checkpoint:", source_ckpt)

    ckpt = torch.load(source_ckpt, map_location="cpu")
    missing, unexpected = train_model.load_state_dict(ckpt["state_dict"], strict=False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    logger = TensorBoardLogger("tb_logs", name="deepcharuco_pseudo_overfit")

    checkpoint_callback = ModelCheckpoint(
        dirpath="tb_logs/ckpts_deepcharuco_pseudo_overfit/",
        save_top_k=5,
        monitor="val_loss",
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=200,
        logger=logger,
        accelerator="auto",
        callbacks=[
            checkpoint_callback,
            SetLearningRate(lr=1e-4),
        ],
    )

    trainer.fit(train_model, train_loader, val_loader)