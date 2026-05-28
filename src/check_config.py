import yaml
from configs import Config

with open("config.yaml", "r") as f:
    config_dict = yaml.safe_load(f)

config = Config(**config_dict)

print("Train labels:", config.train_labels)
print("Val labels:", config.val_labels)
print("Train images:", config.train_images)
print("Val images:", config.val_images)

print("Board:", config.board_name)
print("Rows:", config.row_count)
print("Cols:", config.col_count)
print("Square length:", config.square_len)
print("Marker length:", config.marker_len)
print("n_ids:", config.n_ids)