# Autoencoder training part

from dataloader.imagenette_loader import load_AE_data, load_vicim_data
from autoencoder.classes.resnet_autoencoder import AE
from autoencoder.train_autoencoder import train_autoencoder
from utils.eval_AE import evaluate_AE
from utils.test_victim import test_victim

epochs = 100
train_AE = False

train_loader, val_loader = load_AE_data(dataset_type="timm", dataset_name="imagenette2-320", input_size=(3, 224, 224), train_batch=256, test_batch=32)
attack_loader = load_vicim_data(dataset_type="timm", dataset_name="Caltech-256-Splitted", input_size=(3, 224, 224), victim_batch=32)

if train_AE:
    model = train_autoencoder(train_loader, val_loader, num_layers=34, epochs=epochs)

# evaluating autoencoder
evaluate_AE(val_loader, model_type="Res_AE", num_layers=34, dataset_name="imagenette2-320")

# evaluating autoencoder with attack dataset
evaluate_AE(attack_loader, model_type="Res_AE", num_layers=34, dataset_name="Caltech-256-Splitted")

# evaluating accuracy of victim model
mobilenet_accuracy = test_victim(attack_loader, model_name="mobilenetv3_large_100")
resnet_accuracy = test_victim(attack_loader, model_name="resnet50")
vgg_accuracy = test_victim(attack_loader, model_name="vgg19")
vit_tiny_accuracy = test_victim(attack_loader, model_name="vit_small_patch32_224")

