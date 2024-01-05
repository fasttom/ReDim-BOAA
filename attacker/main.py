# Autoencoder training part

from dataloader.imagenette_loader import load_data
from autoencoder.classes.resnet_autoencoder import AE
from utils.image_plotter import check_plot
import torch
import torch.nn as nn
import torch.nn.functional as F

epochs = 1000
train_AE = False

train_loader, test_loader, attack_loader = load_data(dataset_type="timm", dataset_name="imagenette2-320", input_size=(3, 224, 224), train_batch=256, test_batch=32)

if train_AE:
    model = AE(network='default', num_layers=34).to("cuda")
    encoder = model.encoder
    decoder = model.decoder

    encoder.train()
    decoder.train()


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = 1000000
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to("cuda")
            target = target.to("cuda")
            optimizer.zero_grad()
            encoded = encoder(data)
            decoded = decoder(encoded)
            loss = F.mse_loss(decoded, data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch_idx, loss.item()))
        epoch_loss /= len(train_loader)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "./autoencoder/models/Res_AE_34_best.pth")
            print("Best Loss so far at epoch {} is {}".format(epoch, best_loss))
            print("Saved model")
    encoder.eval()
    decoder.eval()
            
if not train_AE:
    model = AE(network='default', num_layers=34).to("cuda")
    model.load_state_dict(torch.load("./autoencoder/models/Res_AE_34_best.pth"))
    encoder = model.encoder
    decoder = model.decoder

    encoder.eval()
    decoder.eval()
    check_plot(model, test_loader)