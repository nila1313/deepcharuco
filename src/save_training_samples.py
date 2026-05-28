import os
import cv2
import yaml
from configs import Config
from data import CharucoDataset

with open("config.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

config = Config(**config_dict)

output_dir = "../my_dataset/debug_training_samples"
os.makedirs(output_dir, exist_ok=True)

dataset = CharucoDataset(
    config,
    config.train_labels,
    config.train_images,
    validation=False,
    visualize=False
)

print("Dataset size:", len(dataset))

for i in range(20):
    sample = dataset[i]

    image = sample["image"]

    # image is probably normalized grayscale tensor/array.
    # Convert it safely to viewable image.
    if hasattr(image, "numpy"):
        image = image.numpy()

    if image.ndim == 3:
        image = image.squeeze()

    img = image.copy()

    # Normalize to 0-255 for saving
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img = (img * 255).astype("uint8")

    save_path = os.path.join(output_dir, f"sample_{i:03d}.png")
    cv2.imwrite(save_path, img)

    print("Saved:", save_path)