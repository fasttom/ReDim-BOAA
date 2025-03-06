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

def compare_plot(model, loader, dataset_name: str, length: int):
    with torch.no_grad():
        count = 0
        for batch_idx, (data, target) in enumerate(loader):
            sample_no = 32*count + 1
            inputs = data.to("cuda")
            outputs = model.decoder(model.encoder(inputs))

            input_samples = inputs.permute(0,2,3,1).cpu().numpy() # BRG to RGB
            reconstructed_samples = outputs.permute(0,2,3,1).cpu().numpy() # BRG to RGB

            input_samples = normalize_image(input_samples)
            reconstructed_samples = normalize_image(reconstructed_samples)

            fig, axs = plt.subplots(1,2)
            img = input_samples[0]
            recon_img = reconstructed_samples[0]

            axs[0].imshow(img)
            axs[0].set_title("original", fontsize = 16)
            axs[0].axis("off")

            axs[1].imshow(recon_img)
            axs[1].set_title("reconstructed", fontsize = 16)
            axs[1].axis("off")

            plt.savefig(os.path.join(f"figs/AE_compare/{dataset_name}", f"image_{sample_no}.png"))
            
            count = count+1
            if count == length:
                break

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

    return None

def adversarial_plot(image:torch.Tensor, adv_image:torch.Tensor, model_name:str, attack_idx:int):
    image = image.permute(1,2,0).cpu().detach().numpy()
    image = normalize_image(image)
    adv_image = adv_image.permute(1,2,0).cpu().detach().numpy()
    adv_image = normalize_image(adv_image)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Original")
    ax[0].axis('off')

    ax[1].imshow(adv_image)
    ax[1].set_title("Adversarial")
    ax[1].axis('off')

    # plt.savefig(os.path.join(f"figs/{model_name}", f"Caltech-256_attack_{attack_idx}.png"))
    plt.show(block=False)


def noise_plot(image:torch.Tensor, adv_image:torch.Tensor, model_name:str, attack_idx:int):
    noise = adv_image-image

    image = image.permute(1,2,0).cpu().detach().numpy()
    image = normalize_image(image)
    adv_image = adv_image.permute(1,2,0).cpu().detach().numpy()
    adv_image = normalize_image(adv_image)
    noise = noise.permute(1,2,0).cpu().detach().numpy()
    noise = normalize_image(noise)

    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    ax[0].imshow(image)
    ax[0].set_title("Original")
    ax[0].axis('off')

    ax[1].imshow(adv_image)
    ax[1].set_title("Adversarial")
    ax[1].axis('off')

    ax[2].imshow(noise)
    ax[2].set_title("Difference")
    ax[2].axis("off")
    # plt.savefig(os.path.join(f"figs/{model_name}", f"Caltech-256_attack_{attack_idx}.png"))
    plt.show(block=False)