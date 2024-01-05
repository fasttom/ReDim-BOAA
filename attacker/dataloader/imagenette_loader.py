import torch
import logging
import warnings

from timm.data import create_dataset
from timm.data.loader import create_loader

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings("ignore")

def load_data(dataset_type: str = "timm", dataset_name: str = "imagenette2-320", input_size: tuple = (3, 224, 224), train_batch: int = 32, test_batch: int = 1):
    dataset_path = "./dataset/" + dataset_name
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

        attack_loader = create_loader(
            dataset=test_dataset,
            input_size=input_size,
            batch_size=1,
            is_training=False,
            use_prefetcher=False,
        )

        return train_loader, test_loader, attack_loader

