#    Authors:    Taeyeong Kim
#    Hanyang University, Korea
#    Defense Innovation Institute, Chinese Academy of Military Science, China
#    EMAIL:      fasttom@hanyang.ac.kr
#    DATE:       #to-do
# ------------------------------------------------------------------------
# This code is part of the program that produces the results in the following paper:
#
# to-do
#
# You are free to use it for non-commercial purposes. However, we do not offer any forms of guanrantee or warranty associated with the code. We would appreciate your acknowledgement.
# ------------------------------------------------------------------------


import random
import torch
import numpy as np
# import VGG_16 model as vgg
# import our models to use
import scipy.stats
import torch.nn.functional as F
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
# from non_dominated_sorting import fast_non_dominated_sort # to implement later
# from latin import latin # to implement later
import math
import warnings


warnings.filterwarnings("ignore")
# from here
population_size = 50
generation = 100
MR = 0.5 # Mutation Rate
CR = 0.6 # Crossover Rate
# to here -- implement argument passing method
x_min, x_max = [-1, 1]
eps = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model_name().to(device) # the victim model
# model.train() # train the victim model (with model_optimizer)

def Kcalculate_fitness(target_image, sample_adv_images, population, first_labels, dim):
    target_image = target_image.cpu().detach().numpy()
    Kfitness = []
    function_value = np.zeros(100)
    attack_direction=np.zeros((100, 3, 32, 32))
    for i in range(100):
        for j in range(0,dim):
            attack_direction[i, :, :, :] = attack_direction[i, :, :, :] + population[i,j] * (sample_adv_images[j, :, :, :] - target_image[0, :, :, :])
        attack_direction[i, :, :, :] = np.sign(attack_direction[i, :, :, :])
    
    # model.eval()
    for b in range(100):
        attack_image = target_image + eps * attack_direction[b, :, :, :]
        attack_image = torch.from_numpy(attack_image).to(device)
        # outputs = model(attack_image.float())
        # outputs = outputs.cpu().detach().numpy()
        # d = outputs[0, first_labels]
        # c = np.min(outputs)
        # outputs.itemset(first_labels, c)
        # d = np.max(outputs)
        # function_value[b] = d-g
        # Kfitness.append(function_value[b])

    return Kfitness


def calculate_fitness(target_image, sample_adv_images, population, first_labels, dim, size):
    target_image = target_image.cpu().detach().numpy()
    adv_entropy = []
    fitness = []
    fucntion_value = np.zeros(size)
    attack_direction = np.zeros((size, 3, 32, 32))
    for i in range(size):
        for j in range(0, dim):
            attack_direction[i, :, :, :] = attack_direction[i, :, :, :] + population[i, j] * (sample_adv_images[j, :, :, :] - target_image[0, :, :, :])
        attack_direction[i, :, :, :] = np.sign(attack_direction[i, :, :, :])

    # model.eval()
    for b in range(size):
        attack_image = target_image + eps * attack_direction[b, :, :, :]
        attack_image = torch.from_numpy(attack_image).to(device)
        # outputs = model(attack_image.float())
        # adv_soft = F.softmax(outputs, dim=1)[0]
        info_entropy = 0
        # for i in range(10):
            # info_entropy += adv_soft[i] * math.log(adv_soft[i])
        info_entropy = -info_entropy
        info_entropy = info_entropy.cpu().detach().numpy()
        adv_entropy.append(info_entropy)
        # outputs = outputs.cpu().detach().numpy()
        # d = outputs[0, first_labels]
        # c = np.min(outputs)
        # outputs.itemset(first_labels, c)
        # g = np.max(outputs)
        # fucntion_value[b] = d-g
        # fitness.append(fucntion_value[b])

    return fitness, adv_entropy


def mutation(population, dim):
    Mpopulation = np.zeros((population_size, dim))
    for i in range(population_size):
        r1 = r2 = r3 = 0
        while r1 == i or r2 == i or r3 == i or r1 == r2 or r2 == r3 or r3 == r1:
            r1 = random.randint(0, population_size - 1)
            r2 = random.randint(0, population_size - 1)
            r3 = random.randint(0, population_size - 1)
        Mpopulation[i] = population[r1] + MR * (population[r2]-population[r3])
    
        for j in range(dim):
            if x_min <= Mpopulation[i,j] <= x_max:
                Mpopulation[i,j] = Mpopulation[i,j]
            else:
                Mpopulation[i,j] = x_min + random.random() * (x_max - x_min)
    return Mpopulation


