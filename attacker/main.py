# Autoencoder training part

import torch
from dataloader.imagenette_loader import load_AE_data, load_vicim_data
from autoencoder.classes.resnet_autoencoder import AE
from autoencoder.train_autoencoder import train_autoencoder
from utils.eval_AE import evaluate_AE
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


"""
resnet_originals, resnet_advs, resnet_used_epochs, resnet_asr = run_attack("resnet50", resnet_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100)
resnet_summary = {"epochs": resnet_used_epochs, "asr": resnet_asr}
save_summary(resnet_summary, "resnet50")
"""

resnet_originals_base, resnet_advs_base, resnet_used_epochs_base, resnet_asr_base = run_attack("resnet50", resnet_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100,
                                                                                            is_baseline=True)
resnet_summary_base = {"epochs": resnet_used_epochs_base, "asr": resnet_asr_base}
save_summary(resnet_summary_base, "resnet50_base")

resnet_originals_semi, resnet_advs_semi, resnet_used_epochs_semi, resnet_asr_semi = run_attack("resnet50", resnet_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100,
                                                                                               is_semi=True)
resnet_summary_semi = {"epochs": resnet_used_epochs_semi, "asr": resnet_asr_semi}
save_summary(resnet_summary_semi, "resnet50_semi")


"""
vgg_originals, vgg_advs, vgg_used_epochs, vgg_asr = run_attack("vgg19", vgg_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100)
vgg_summary = {"epochs": vgg_used_epochs, "asr": vgg_asr}
save_summary(vgg_summary, "vgg19")
"""

"""
vgg_originals_base, vgg_advs_base, vgg_used_epochs_base, vgg_asr_base = run_attack("vgg19", vgg_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100, 
                                                                                                           is_baseline=True)
vgg_summary_base = {"epochs": vgg_used_epochs_base, "asr": vgg_asr_base}
save_summary(vgg_summary_base, "vgg19_base")
"""

"""
vgg_originals_semi, vgg_advs_semi, vgg_used_epochs_semi, vgg_asr_semi = run_attack("vgg19", vgg_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100, 
                                                                                                           is_semi=True)
vgg_summary_semi = {"epochs": vgg_used_epochs_semi, "asr": vgg_asr_semi}
save_summary(vgg_summary_semi, "vgg19_semi")
"""

"""
vit_small_originals, vit_small_advs, vit_small_used_epochs, vit_small_asr = run_attack("vit_small_patch32_224", vit_small_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100)
vit_small_summary = {"epochs": vit_small_used_epochs, "asr": vit_small_asr}
save_summary(vit_small_summary, "vit_small_patch32_224")
"""

"""
vit_small_originals_base, vit_small_advs_base, vit_small_used_epochs_base, vit_small_asr_base = run_attack("vit_small_patch32_224", vit_small_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100,
                                                                                                           is_baseline=True)
vit_small_summary_base = {"epochs": vit_small_used_epochs_base, "asr": vit_small_asr_base}
save_summary(vit_small_summary_base, "vit_small_patch32_224_base")
"""

"""
vit_small_originals_semi, vit_small_advs_semi, vit_small_used_epochs_semi, vit_small_asr_semi = run_attack("vit_small_patch32_224", vit_small_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100,
                                                                                                           is_semi=True)
vit_small_summary_semi = {"epochs": vit_small_used_epochs_semi, "asr": vit_small_asr_semi}
save_summary(vit_small_summary_semi, "vit_small_patch32_224_semi")
"""

"""
mobilenet_originals, mobilenet_advs, mobilenet_used_epochs, mobilenet_asr = run_attack("mobilenetv3_large_100", mobilenet_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100)
mobilenet_summary = {"epochs": mobilenet_used_epochs, "asr": mobilenet_asr}
save_summary(mobilenet_summary, "mobilenetv3_large_100")
"""

mobilenet_originals_base, mobilenet_advs_base, mobilenet_used_epochs_base, mobilenet_asr_base = run_attack("mobilenetv3_large_100", mobilenet_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100, 
                                                                                                           is_baseline=True)
mobilenet_summary_base = {"epochs": mobilenet_used_epochs_base, "asr": mobilenet_asr_base}
save_summary(mobilenet_summary_base, "mobilenetv3_large_100_base")

mobilenet_originals_semi, mobilenet_advs_semi, mobilenet_used_epochs_semi, mobilenet_asr_semi = run_attack("mobilenetv3_large_100", mobilenet_attack_loader, model, acquisition="EI", feature_len=7, num_channels=512, epoch_lim=100, 
                                                                                                           is_semi=True)
mobilenet_summary_semi = {"epochs": mobilenet_used_epochs_semi, "asr": mobilenet_asr_semi}
save_summary(mobilenet_summary_semi, "mobilenetv3_large_100_semi")
