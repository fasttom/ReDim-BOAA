import torch
import logging
import warnings

from timm.data import create_dataset
from timm.data.loader import create_loader

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings("ignore")


def load_vicim_data(dataset_type: str = "timm", dataset_name: str = "Caltech-256-Splitted", input_size:tuple = (3, 224, 224), victim_batch = 32):
    dataset_path = "./../victim/dataset/victim/" + dataset_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_type == "timm":
        attack_dataset = create_dataset(
            name="test",
            root=dataset_path,
            split="test",
            seed=42
        )

        attack_loader = create_loader(
            dataset=attack_dataset,
            input_size=input_size,
            batch_size=victim_batch,
            is_training=False,
            use_prefetcher=False,
        )

        return attack_loader
    

def load_vicim_train_data(dataset_type: str = "timm", dataset_name: str = "Caltech-256-Splitted", input_size:tuple = (3, 224, 224), train_batch = 32):
    dataset_path = "./../victim/dataset/victim/" + dataset_name
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
            is_training=False,
            use_prefetcher=False,
        )

        return train_loader
    

def load_vicim_val_data(dataset_type: str = "timm", dataset_name: str = "Caltech-256-Splitted", input_size:tuple = (3, 224, 224), val_batch = 32):
    dataset_path = "./../victim/dataset/victim/" + dataset_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset_type == "timm":
        val_dataset = create_dataset(
            name="val",
            root=dataset_path,
            split="val",
            seed=42
        )

        val_loader = create_loader(
            dataset=val_dataset,
            input_size=input_size,
            batch_size=val_batch,
            is_training=False,
            use_prefetcher=False,
        )

        return val_loader


def load_AE_data(dataset_type: str = "timm", dataset_name: str = "imagenette2-320", input_size: tuple = (3, 224, 224), train_batch: int = 32, test_batch: int = 32):
    dataset_path = "./dataset/autoencoder/" + dataset_name
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