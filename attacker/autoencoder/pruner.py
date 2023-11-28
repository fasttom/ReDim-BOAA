import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn.utils import prune
from models.resnet import ResNetAutoEncoder
from models.vgg import VGGAutoEncoder
def prune_model(model_path:str):
    model = VGGAutoEncoder([2, 2, 4, 4, 4])
    model.load_state_dict(torch.load(model_path)["state_dict"], strict=False)
    parameters_to_prune = []
    for block in model.encoder.modules():
        for layer in block.modules():
            if isinstance(layer, nn.Sequential):
                for sublayer in layer.modules():
                    if isinstance(sublayer, nn.Conv2d):
                        parameters_to_prune.append((sublayer, "weight"))
    for block in model.decoder.modules():
        for layer in block.modules():
            if isinstance(layer, nn.Sequential):
                for sublayer in layer.modules():
                    if isinstance(sublayer, nn.Conv2d):
                        parameters_to_prune.append((sublayer, "weight"))
    prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
    )
    return model


# temp... todo: move to main.py
pruned_model = prune_model("results/caltech256-vgg19.pth")
#print(pruned_model)
both_state_dict = pruned_model.state_dict()
pruned_state_dict = OrderedDict()
for key in both_state_dict.keys():
    if "_orig" not in key and "_mask" not in key:
        pruned_state_dict[key] = both_state_dict[key]
print(pruned_state_dict.keys())
torch.save({"state_dict": pruned_state_dict}, "results/caltech256-vgg19-pruned.pth")