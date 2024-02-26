import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder.classes.resnet_autoencoder import AE

def train_autoencoder(train_loader, val_loader, num_layers, epochs = 100):
    # training autoencoder
    model = AE(network='default', num_layers=num_layers).to("cuda")
    encoder = model.encoder
    decoder = model.decoder

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = 1000000

    for epoch in range(epochs):
        # Training phase
        print("training phase")
        encoder.train()
        decoder.train()
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

        # Validation phase
        print("validation phase")
        encoder.eval()
        decoder.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.to("cuda")
            target = target.to("cuda")
            encoded = encoder(data)
            decoded = decoder(encoded)
            loss = F.mse_loss(decoded, data)
            val_loss += loss.item()
            if batch_idx % 10 == 0:
                print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch_idx, loss.item()))
        epoch_loss /= len(train_loader)
        val_loss /= len(val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "./autoencoder/results/Res_AE_{}_best.pth".format(num_layers))
            print("Best Loss so far at epoch {} is {}".format(epoch, best_loss))
            print("Saved model")
    encoder.eval()
    decoder.eval()
    return model
