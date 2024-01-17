from dataloader.spliter import split_dataset
from os.path import exists

if not exists("dataset/victim/Caltech-256-Splitted"):
    split_dataset(dataset_name="Caltech-256", split_ratio=(0.6, 0.2, 0.2))