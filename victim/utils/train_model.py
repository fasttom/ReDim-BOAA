import torch
import torch.nn.functional as F

from timm.models import create_model

def train_model(train_set, val_set, model_name, epochs = 100):
    model = create_model(model_name, pretrained=True)
    model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = 1000000
    for epoch in range(epochs):
        # training phase
        print("training phase")
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_set):
            data = data.to("cuda")
            target = target.to("cuda")
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch_idx, loss.item()))
        
        # validation phase
        print("validation phase")
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_set):
            data = data.to("cuda")
            target = target.to("cuda")
            output = model(data)
            loss = F.cross_entropy(output, target)
            val_loss += loss.item()
            if batch_idx % 10 == 0:
                print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, batch_idx, loss.item()))

        epoch_loss /= len(train_set)
        val_loss /= len(val_set)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "./results/{}_best.pth".format(model_name))
            print("Best Loss so far at epoch {} is {}".format(epoch, best_loss))
            print("Saved model")
    return model