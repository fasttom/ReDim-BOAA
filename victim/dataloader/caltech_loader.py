import torch
import logging
import warnings

from timm.data import create_dataset
from timm.data.loader import create_loader

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings("ignore")


def load_data(dataset_type: str = "timm", dataset_name: str = "Caltech-256-Splitted", input_size: tuple = (3, 224, 224), 
    train_batch = 32, test_batch = 32):
    dataset_path = "./dataset/victim/" + dataset_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_type == "timm":
        train_dataset = create_dataset(
            name="train",
            root=dataset_path,
            split="train",
            seed=42
        )

        train_loader = create_loader(         
            dataset=train_dataset,
            input_size=input_size,
            batch_size=train_batch,
            is_training=True,
            use_prefetcher=False,
            no_aug=True
        )

        test_dataset = create_dataset(
            name="val",
            root=dataset_path,
            split="val",
            seed=42
        )

        test_loader = create_loader(
            dataset=test_dataset,
            input_size=input_size,
            batch_size=test_batch,
            is_training=False,
            use_prefetcher=False,
        )

        return train_loader, test_loader