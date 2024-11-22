# Autoencoder training part

import torch
from dataloader.imagenette_loader import load_AE_data, load_vicim_data
from autoencoder.classes.resnet_autoencoder import AE
from autoencoder.train_autoencoder import train_autoencoder
from utils.eval_AE import evaluate_AE
from utils.test_victim import test_victim
from utils.save_summary import save_summary
from optimizer.run_attack import run_attack
from timm.models import create_model


model = create_model("mobilenetv3_large_100", pretrained=True)
model.load_state_dict(torch.load(f"./../victim/results/mobilenetv3_large_100_best.pth"))
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)