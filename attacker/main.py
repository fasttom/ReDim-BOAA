# Autoencoder training part

import torch
from dataloader.imagenette_loader import load_AE_data, load_vicim_data
from autoencoder.classes.resnet_autoencoder import AE
from autoencoder.train_autoencoder import train_autoencoder
from utils.eval_AE import evaluate_AE
from utils.test_victim import test_victim
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
mobilenet_accuracy, mobilenet_attack_loader = test_victim(attack_loader, model_name="mobilenetv3_large_100")
resnet_accuracy, resnet_attack_loader = test_victim(attack_loader, model_name="resnet50")
vgg_accuracy, vgg_attack_loader = test_victim(attack_loader, model_name="vgg19")
vit_small_accuracy, vit_small_attack_loader = test_victim(attack_loader, model_name="vit_small_patch32_224")

# evaluating autoencoder
evaluate_AE(val_loader, model_type="Res_AE", num_layers=34, dataset_name="imagenette2-320")

# evaluating autoencoder on vicitm dataset where victim model classified correctly
evaluate_AE(mobilenet_attack_loader, model_type="Res_AE", num_layers=34, dataset_name="Caltech-256-Splitted", victim_model="mobilenetv3_large_100")
evaluate_AE(resnet_attack_loader, model_type="Res_AE", num_layers=34, dataset_name="Caltech-256-Splitted", victim_model="resnet50")
evaluate_AE(vgg_attack_loader, model_type="Res_AE", num_layers=34, dataset_name="Caltech-256-Splitted", victim_model="vgg19")
evaluate_AE(vit_small_attack_loader, model_type="Res_AE", num_layers=34, dataset_name="Caltech-256-Splitted", victim_model="vit_small_patch32_224")

# run attacks
mobilenet_originals, mobilenet_advs, mobilenet_used_epochs, mobilenet_asr = run_attack("mobilenetv3_large_100", attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=200)
resnet_originals, resnet_advs, resnet_used_epochs, resnet_asr = run_attack("resnet50", attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=200)
vgg_originals, vgg_advs, vgg_used_epochs, vgg_asr = run_attack("vgg19", attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=200)
vit_small_originals, vit_small_advs, vit_small_used_epochs, vit_small_asr = run_attack("vit_small_patch32_224", attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=200)