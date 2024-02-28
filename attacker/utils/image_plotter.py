# Testing the quality of the autoencoder

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import os

def normalize_image(img):
    img = (img-img.min())/(img.max()-img.min())
    return img

def check_plot(model, loader, dataset_name:str, victim_model:str):
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):

            inputs = data.to("cuda")
            outputs = model.decoder(model.encoder(inputs))

            input_samples = inputs.permute(0,2,3,1).cpu().numpy() # BRG to RGB
            reconstructed_samples = outputs.permute(0,2,3,1).cpu().numpy() # BRG to RGB
            break

    input_samples = normalize_image(input_samples)
    reconstructed_samples = normalize_image(reconstructed_samples)

    columns = 8
    rows = 4

    fig = plt.figure(figsize=(columns, rows))

    for i in range(1, columns*rows + 1):
        img = input_samples[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(os.path.join("figs", f"{dataset_name}_{victim_model}_input_samples.png"))
    plt.show(block=False)
    

    fig = plt.figure(figsize=(columns, rows))

    for i in range(1, columns*rows + 1):
        img = reconstructed_samples[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(os.path.join("figs", f"{dataset_name}_{victim_model}_reconstructed_samples.png"))
    plt.show(block=False)
   