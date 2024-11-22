# Autoencoder training part

import torch
from dataloader.imagenette_loader import load_AE_data, load_vicim_data
from autoencoder.classes.resnet_autoencoder import AE
from autoencoder.train_autoencoder import train_autoencoder
from utils.eval_AE import evaluate_AE
from utils.test_victim import test_victim
from utils.save_summary import save_summary
from optimizer.run_attack import run_attack_short, get_adversarial_classes, make_adv_class_dict, run_noise_plot


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

# mobilenet_originals, mobilenet_advs, mobilenet_idx = run_attack_short("mobilenetv3_large_100", mobilenet_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100)
# mobilenet_original_lables, mobilenet_adv_labels = get_adversarial_classes("mobilenetv3_large_100", mobilenet_originals, mobilenet_advs)
# mobilenet_dict = make_adv_class_dict(mobilenet_idx, mobilenet_original_lables, mobilenet_adv_labels)
# save_summary(mobilenet_dict, "mobilenetv3_large_100_class_compare")

# resnet_originals, resnet_advs, resnet_idx = run_attack_short("resnet50", resnet_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100)
# resnet_original_lables, resnet_adv_labels = get_adversarial_classes("resnet50", resnet_originals, resnet_advs)
# resnet_dict = make_adv_class_dict(resnet_idx, resnet_original_lables, resnet_adv_labels)
# save_summary(resnet_dict, "resnet50_class_compare")

# vgg_originals, vgg_advs, vgg_idx = run_attack_short("vgg19", vgg_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100)
# vgg_original_lables, vgg_adv_labels = get_adversarial_classes("vgg19", vgg_originals, vgg_advs)
# vgg_dict = make_adv_class_dict(vgg_idx, vgg_original_lables, vgg_adv_labels)
# save_summary(vgg_dict, "vgg19_class_compare")

# vit_small_originals, vit_small_advs, vit_small_idx = run_attack_short("vit_small_patch32_224", vit_small_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100)
# vit_small_original_lables, vit_small_adv_labels = get_adversarial_classes("vit_small_patch32_224", vit_small_originals, vit_small_advs)
# vit_small_dict = make_adv_class_dict(vit_small_idx, vit_small_original_lables, vit_small_adv_labels)
# save_summary(vit_small_dict, "vit_small_patch32_224_class_compare")

_, __, ___ = run_noise_plot("mobilenetv3_large_100", mobilenet_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100)
_, __, ___ = run_noise_plot("mobilenetv3_large_100", mobilenet_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100, is_semi=True)
_, __, ___ = run_noise_plot("mobilenetv3_large_100", mobilenet_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100, is_baseline=True)