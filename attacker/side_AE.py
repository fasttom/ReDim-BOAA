# Autoencoder training side experiment

import torch
from dataloader.imagenette_loader import load_AE_data, load_vicim_data
from autoencoder.classes.resnet_autoencoder import AE
from autoencoder.train_autoencoder import train_autoencoder
from utils.eval_AE import evaluate_AE, single_plot_AE, check_differece_AE
from utils.test_victim import test_victim
from utils.save_summary import save_summary
from optimizer.run_attack import run_attack


epochs = 100
train_AE = False

train_loader, val_loader = load_AE_data(dataset_type="timm", dataset_name="imagenette2-320", input_size=(3, 224, 224), train_batch=256, test_batch=32)
attack_loader = load_vicim_data(dataset_type="timm", dataset_name="Caltech-256-Splitted", input_size=(3, 224, 224), victim_batch=32)

if train_AE:
    model = train_autoencoder(train_loader, val_loader, num_layers=34, epochs=epochs)
else:
    model = AE(num_layers=34)
    model.load_state_dict(torch.load(f"./autoencoder/results/Res_AE_34_best.pth"))

# evaluating accuracy of victim model
# mobilenet_accuracy, mobilenet_attack_loader = test_victim(attack_loader, model_name="mobilenetv3_large_100")
# resnet_accuracy, resnet_attack_loader = test_victim(attack_loader, model_name="resnet50")
vgg_accuracy, vgg_attack_loader = test_victim(attack_loader, model_name="vgg19")
# vit_small_accuracy, vit_small_attack_loader = test_victim(attack_loader, model_name="vit_small_patch32_224")

# evaluating autoencoder
# evaluate_AE(val_loader, model_type="Res_AE", num_layers=34, dataset_name="imagenette2-320")

# AE quality test on open dataset
# single_plot_AE(val_loader, model_type="Res_AE", num_layers=34, dataset_name = "imagenette2-320", length = 30)

# AE quality test on target dataset
# single_plot_AE(vgg_attack_loader, model_type="Res_AE", num_layers=34, dataset_name= "Caltech-256-Splitted", length = 30)

# MAPE test
open_rmse = check_differece_AE(val_loader, model_type="Res_AE", num_layers=34, dataset_name= "imagenette2-320")
target_rmse = check_differece_AE(vgg_attack_loader, model_type="Res_AE", num_layers=34, dataset_name="Caltech-256-Splitted")

print("MAPE of ResNet Autoencoder over open dataset: ", open_rmse)
print("MAPE of ResNet Autoencoder over target dataset: ", target_rmse)