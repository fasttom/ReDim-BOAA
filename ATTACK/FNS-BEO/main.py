import numpy as np
import torch
import torch.utils.data as Data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
# import models
import torch.nn.functional as F
from FNS_BEO import FDE
import math
import warnings


warnings.filterwarnings("ignore")


batch_size = 1 # temporary
# batch_size = my_batch_size

# for CIFAR
train_dataset = dsets.CIFAR10(
    root="/workspaces/Attacking_Image_Models/pytorch-image-model-attack/ATTACK/dataset", #temporary absolute path
    # root ="../dataset" # to-do relative path
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ]),
    train=True
)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

test_dataset = dsets.CIFAR10(
    root="/workspaces/Attacking_Image_Models/pytorch-image-model-attack/ATTACK/dataset", #temporary absolute path
    # root ="../dataset" # to-do relative path
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ]),
    train=False
)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model_name().to(device)

# train_model......
# with optimizer.....

count = 0
total_count = 0
net_correct = 0
# model.eval() # enter to evalutaion mode

def select_other_images(first_label):
    sum_images = []
    meet = 0
    first_label = first_label.cpu().detach().numpy()
    for images, labels in test_loader:
        if labels != first_label:
            images = images.to(device)
            outputs = models(images)
            outputs = outputs.cpu().detach().numpy()
            if np.argmax(outputs) == labels:
                target_images = torch.tensor(images)
                target_images = np.array(target_images.cpu())
                sum_images.append(target_images)
                meet += 1
                if meet == 20:
                    break
    sum_images = np.array(sum_images)
    num_images = np.szie(sum_images, 0)
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
            images, eva_num = FDE(images, sum_images, labels, clean_info_entrophy)
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