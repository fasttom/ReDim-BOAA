# Autoencoder training part

from dataloader.imagenette_loader import load_AE_data, load_vicim_data
from autoencoder.classes.resnet_autoencoder import AE
from autoencoder.train_autoencoder import train_autoencoder
from utils.image_plotter import check_plot
from timm.models import create_model
import torch
import torch.nn as nn
import torch.nn.functional as F

epochs = 100
train_AE = False

train_loader, val_loader = load_AE_data(dataset_type="timm", dataset_name="imagenette2-320", input_size=(3, 224, 224), train_batch=256, test_batch=32)

if train_AE:
    model = train_autoencoder(train_loader, val_loader, num_layers=34, epochs=epochs)

# evaluating autoencoder
model = AE(network='default', num_layers=34).to("cuda")
model.load_state_dict(torch.load("./autoencoder/results/Res_AE_34_best.pth"))
encoder = model.encoder
decoder = model.decoder

encoder.eval()
decoder.eval()

check_plot(model, val_loader)


# evaluating autoencoder with attack dataset
model = AE(network='default', num_layers=34).to("cuda")
model.load_state_dict(torch.load("./autoencoder/results/Res_AE_34_best.pth"))
encoder = model.encoder
decoder = model.decoder

encoder.eval()
decoder.eval()

attack_loader = load_vicim_data(dataset_type="timm", dataset_name="Caltech-256-Splitted", input_size=(3, 224, 224), victim_batch=32)
check_plot(model, attack_loader)

# evaluating accuracy of victim model
victim_model = create_model("mobilenetv3_large_100", pretrained=True)
victim_model.load_state_dict(torch.load("./../victim/results/mobilenetv3_large_100_best.pth"))
victim_model.eval()
victim_model.to("cuda")

correct = 0
total = 0

with torch.no_grad():
    for data in attack_loader:
        images, labels = data
        images = images.to("cuda")
        labels = labels.to("cuda")
        outputs = victim_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the attack images: %d %%' % (100 * correct / total))

