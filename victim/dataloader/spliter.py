import splitfolders
import os
root = "dataset/victim/"
def split_dataset(dataset_name:str = "Caltech-256", split_ratio:tuple = (0.6,0.2,0.2)):
    path = root + dataset_name
    splitfolders.ratio(path, output = path + "-Splitted", ratio=split_ratio, seed=42)