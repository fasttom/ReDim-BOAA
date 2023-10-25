import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torchvision
from autoencoder.models.vgg import VGGAutoEncoder
from autoencoder.models.resnet import ResNetAutoEncoder


def get_autoencoder(encoder_path:str, arch:str):
    arch = arch.lower()
    assert arch in ["vgg11", "vgg13", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    if arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        model = VGGAutoEncoder()
    else:
        model = ResNetAutoEncoder()
    true_path = (os.path.join(os.curdir,"../autoencoder/"+encoder_path))
    checkpoint = torch.load(true_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    encoder = model.encoder
    decoder = model.decoder
    return encoder, decoder