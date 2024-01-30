from dataloader.spliter import split_dataset
from dataloader.caltech_loader import load_data

from models.train_model import train_model

from os.path import exists

if not exists("dataset/victim/Caltech-256-Splitted"):
    split_dataset(dataset_name="Caltech-256", split_ratio=(0.6, 0.2, 0.2))

train_loader, test_loader = load_data(dataset_type="timm", dataset_name="Caltech-256-Splitted", input_size=(3, 224, 224),
    train_batch=256, test_batch=32)

vgg_model = train_model(train_set=train_loader, model_name="vgg16", epochs=50)
resnet_model = train_model(train_set=train_loader, model_name="resnet50", epochs=50)
