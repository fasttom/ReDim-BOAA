import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
# import models
import torch.nn.functional as F
from FNS_BEO import FDE
import math

from tqdm import tqdm
import logging
import warnings


from timm.data import create_dataset
from timm.data.loader import create_loader
from timm.optim import create_optimizer_v2
from timm.models import create_model
from timm.loss import cross_entropy
from timm.scheduler.cosine_lr import CosineLRScheduler


logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings("ignore")


train_batch = 32
test_batch = 1
dataset_name="imagenette2-320"
dataset_path="./dataset/"+dataset_name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vic_epo = 10


train_dataset = create_dataset(
    name="train",
    root=dataset_path,
    split="train",
    seed=42
)


train_loader = create_loader(
    dataset=train_dataset,
    input_size=(3, 224, 224),
    batch_size=train_batch,
    is_training=True,
    use_prefetcher=False,
    no_aug=True
)


test_dataset = create_dataset(
    name="val",
    root=dataset_path,
    split="val",
    seed=42
)


test_loader = create_loader(
    dataset=test_dataset,
    input_size=(3, 224, 224),
    batch_size=test_batch,
    is_training=False,
    use_prefetcher=False,
)


class_names = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 
               'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1) 

    
inputs, classes = next(iter(train_loader))[:8]
out = torchvision.utils.make_grid(inputs, nrow=4)
imshow(out, title=[class_names[x.item()] for x in classes])


model = create_model(
    model_name="vit_small_patch32_224",
    pretrained=True
)


optimizer = create_optimizer_v2(
    model_or_params=model,
    opt="adam",
    lr=1e-4,
    weight_decay=0,
    momentum=0.9
)


model = model.to(device)
model.train()

class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(loader, model, loss_fn = nn.CrossEntropyLoss(), **optim_kwargs):
    logging.info(f"\ncreated model: {model.__class__.__name__}")
    logging.info(f"created optimizer: {optimizer.__class__.__name__}")
    
    losses = []
    loss_avg = AverageMeter()
    model = model.cuda()
    tk0 = tqdm(enumerate(loader), total=len(loader))
    for i, (inputs, targets) in tk0:
        inputs = inputs.to(device)
        targets = targets.to(device)
        preds = model(inputs)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_avg.update(loss.item(), loader.batch_size)
        losses.append(loss_avg.avg)
        tk0.set_postfix(loss=loss.item())
    return losses


def train_victim(loader, model, loss_fn = nn.CrossEntropyLoss(), **optim_kwargs):
    losses_list=[]
    for epo in range(vic_epo):
        epo_loss = train_one_epoch(loader=loader, model=model, loss_fn=loss_fn)
        losses_list.append(losses_list)
    return model


count = 0
total_count = 0
net_correct = 0

model = train_victim(train_loader, model, nn.CrossEntropyLoss())

model.eval()

def select_other_images(first_label):
    sum_images = []
    meet = 0
    first_label = first_label.cpu().detach().numpy()
    for images, labels in test_loader:
        if labels != first_label:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.cpu().detach().numpy()
            if np.argmax(outputs) == labels:
                target_images = torch.tensor(images)
                target_images = np.array(target_images.cpu())
                sum_images.append(target_images)
                meet += 1
                if meet == 20:
                    break
    sum_images = np.array(sum_images)
    num_images = np.size(sum_images, 0)
    if num_images > 1:
        sum_images = sum_images.squeeze()
    else:
        sum_images = sum_images.squeeze()
        sum_images = torch.tensor(sum_images)
        sum_images = sum_images.unsqueeze(0)
        sum_images = sum_images.detach().numpy()
    return sum_images


for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    output = model(images)
    _, pre = torch.max(output.data, 1)
    total_count += 1
    if pre == labels:
        net_correct += 1
        clean_soft = F.softmax(output, dim=1)[0]
        clean_info_entrophy = 0
        for i in range(10):
            clean_info_entrophy += clean_soft[i] * math.log(clean_soft[i])
        clean_info_entrophy = -clean_info_entrophy
        if net_correct <= 100:
            output = output.cpu().detach().numpy()
            min_value = np.min(output)
            output.itemset(labels, min_value)
            second_label = np.argmax(output)
            sum_images = select_other_images(labels)
            images, eva_num = FDE(model, images, sum_images, labels, clean_info_entrophy)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, pre = torch.max(outputs.data, 1)
            if pre == labels:
                count += 1
    
    attack_count = net_correct - count
    print('total count:', total_count, 'net correct:', net_correct, 'atatck fail:', count, 'attack success:', attack_count)
    if net_correct >0:
        print('Success ratio of attack: %f %%' % (100 * float(attack_count) / net_correct))
    if net_correct == 100:
        break