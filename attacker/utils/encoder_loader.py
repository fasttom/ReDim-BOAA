import os
import sys
import torch
import torchvision

from attacker.autoencoder.models.vgg import VGGAutoEncoder
from attacker.autoencoder.models.resnet import ResNetAutoEncoder
from attacker.autoencoder.models.vgg import get_configs as vgg_config
from attacker.autoencoder.models.resnet import get_configs as resnet_config

def get_autoencoder(encoder_name:str, arch:str):
    arch = arch.lower()
    assert arch in ["vgg11", "vgg13", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    if arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        model = VGGAutoEncoder(vgg_config(arch))
    else:
        model = ResNetAutoEncoder(resnet_config(arch))
    true_path = os.path.join(os.curdir, "attacker/autoencoder/results/"+encoder_name)
    checkpoint = torch.load(true_path)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict, False)
    encoder = model.encoder
    decoder = model.decoder
    return encoder, decoder