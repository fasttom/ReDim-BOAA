import numpy as np
import torch
import torch.utils.data as Data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
# import models
import torch.nn.functional as F
# from FNS_BEO import FDE
import math
import warnings


from timm.data import create_dataset
from timm.data.loader import create_loader
from timm.optim import create_optimizer_v2
from timm.models import create_model


warnings.filterwarnings("ignore")


batch_size = 32
dataset_name="imagenette2-320"
dataset_path="./dataset/"+dataset_name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#mode_name = argparse로 불러온 model name
model = create_model(
    model_name="vit_small_patch32_224",
    pretrained=True
)

train_dataset = create_dataset(
    name="",
    root=dataset_path,
    split="train",
    is_training=True,
    batch_size=batch_size,
    seed=42,
)

train_loader = create_loader(
    dataset=train_dataset,
    input_size=(3, 224, 224),
    batch_size=batch_size,
    is_training=False,
    re_split=True
)

test_dataset = create_dataset(
    name="",
    root=dataset_path,
    split="validation",
    is_training=False,
    batch_size=batch_size,
    seed=42
)

test_loader = create_loader(
    dataset=test_dataset,
    input_size=(3, 224, 224),
    batch_size=batch_size,
    is_training=False,
    re_split=False
)




"""

# model = model_name().to(device)

# train_model......
# with optimizer.....
"""
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