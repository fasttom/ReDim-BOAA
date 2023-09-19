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
generations = 100
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


def crossover(Mpopulation, population, dim):
    Cpopulation = np.zeros((population_size, dim))
    for i in range(population_size):
        for j in range(dim):
            rand_j = random.randint(0, dim - 1)
            rand_float = random.random()
            if rand_float <= CR or rand_j == j:
                Cpopulation[i,j] = Mpopulation[i, j]
            else:
                Cpopulation[i,j] = population[i, j]
    return Cpopulation


def selection(Cpopulation, population, gp, Best_solu):
    _, _, Cfitness, _, _ = surrogate_evalu(Cpopulation, gp, Best_solu)
    _, _, pfitness, _, _ = surrogate_evalu(population, gp, Best_solu)
    for i in range(population_size):
        if Cfitness[i] > pfitness[i]:
            population[i] = Cpopulation[i]
        else:
            population[i] = population[i]
    return population


def surrogate_evalu(x_set, gp, Best_solu):
    means, sigmas = gp.predict(x_set, return_std=True)
    PI = np.zeros(50)
    EI = np.zeros(50)
    LCB = np.zeros(50)
    for y in range(50):
        LCB[y] = means[y] - 2 * sigmas[y]
        z = (Best_solu - means[y]) / sigmas[y]
        PI[y] = scipy.stats.norm.cdf(z)
        EI[y] = (Best_solu - means[y]) * scipy.stats.norm.cdf(z) + sigmas[y] * scipy.stats.norm.pdf(z)
    return means, sigmas, PI, EI, LCB


def FDE(target_image, adversarial_images, first_labels, clean_entropy):
    clean_entropy = clean_entropy.cpu().detach().numpy()
    num = np.size(adversarial_images, 0)
    if num >= 10:
        dim = 10
        index = random.sample(range(0, num), dim)
        sample_adv_images = adversarial_images[index]
    else:
        dim = num
        sample_adv_images = adversarial_images

    # init the train data of the surrogate model
    # population = latin(100, dim, -1, 1)
    # computing the object value of the train data
    # K_fit = Kcalculate_fitness(target_image, sample_adv_images, population, first_labels, dim)
    eval_num = 100
    # Best_solu = min(K_fit)
    # Best_indi_index = np.argmin(K_fit)
    # Best_indi = population[Best_indi_index, :]
    # Surrogate Model
    kernel = RBF(1.0, (1e-5, 1e5))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    # xobs = population
    # yobs = np.array(K_fit)
    # gp.fit(xobs, yobs)

    # evolution population
    # population = population[0:50, :]
    pro = [1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1]
    r = [0, 0, 0, 0, 0, 0, 0]
    init_entropy = [clean_entropy, clean_entropy, clean_entropy, clean_entropy, clean_entropy, clean_entropy, clean_entropy]
    for step in range(generations):
        if Best_solu < 0:
            break
        # Mpopulation = mutation(population, dim)
        # Cpopulation = crossover(Mpopulation, population, dim)
        # population = selection(Cpopulation, population, gp, Best_solu)
        index = []
        rand_value = random.random()
        if rand_value < pro[0]:
            oper_id = 0
            # sorted_id = sorted(range(len(Ei)), key=lambda k: Ei[k], reverse=True)
            # index.append(sorted_id[0])
        if pro[0] <= rand_value < pro[1]:
            oper_id = 1
            # fronts = fast_non_dominated_sort(sfit, std)
            # fist_front = fronts[0]
            # rand_index1 = random.randint(0, len(fist_front)-1)
            # index.append(fist_front[rand_index1])
        if pro[1] <= rand_value < pro[2]:
            oper_id = 2
            # fronts = fast_non_dominated_sort(sfit, std)
            # fist_front = fronts[0]
            # front_fit = sfit[fist_front]
            # fit_index = np.argmin(front_fit)
            # index.append(fist_front[fit_index])
        if pro[2] <= rand_value < pro[3]:
            oper_id = 3
            # fronts = fast_non_dominated_sort(sfit, std)
            # fist_front = fronts[0]
            # front_std = std[fist_front]
            # std_index = np.argmin(front_std)
            # index.append(fist_front[std_index])
        if pro[3] <= rand_value < pro[4]:
            oper_id = 4
            # sorted_id = sorted(range(len(std)), key=lambda k: std[k], reverse=True)
            # index.append(sorted_id[0])
        if pro[4] <= rand_value < pro[5]:
            oper_id = 5
            # sorted_id = sorted(range(len(Pi)), key=lambda k: Pi[k], reverse=True)
            # index.append(sorted_id[0])
        if pro[5] <= rand_value < pro[6]:
            oper_id = 6
            # sorted_id = sorted(range(len(lcb)), key=lambda k: lcb[k], reverse=True)
            # index.append(sorted_id[-1])
        # add_xdata = population[index, :]
        size = len(index)
        Tfitness, adv_entro = calculate_fitness(target_image, sample_adv_images, add_xdata, first_labels, dim, size)
        init_entropy[oper_id] = max(adv_entro)
        max_entropy = max(init_entropy)
        min_entropy = min(init_entropy)
        for i in range(7):
            r[i] = (init_entropy[i]-max_entropy)/(max_entropy-min_entropy)
        sum_r = 0
        for i in range(7):
            sum_r += math.exp(r[i])
        pro[0] = math.exp(r[0]) / sum_r
        for i in range(1, 6):
            pro[i] = math.exp(r[i]) / sum_r + pro[i-1]
        eval_num += size
        add_ydata = Tfitness
        add_xdata = add_xdata.tolist()
        xobs = xobs.tolist()
        yobs = yobs.tolist()
        for c in range(size):
            xobs.append(add_xdata[c])
            yobs.append(add_ydata[c])
        xobs = np.array(xobs)
        yobs = np.array(yobs)
        gp.fit(xobs, yobs)
        sBest_solu = min(Tfitness)
        if Best_solu > sBest_solu:
            Best_solu = sBest_solu
            # if size == 1:
                # Best_indi = population[index[0], :]
            # else:
                # sBest_solu_index = np.argmin(Tfitness)
                # add_xdata = np.array(add_xdata)
                # Best_indi = add_xdata[sBest_solu_index, :]
        if eval_num >= 200:
            break
    
    Final_attack_sign = np.zeros((1, 3, 3 ,32))
    target_image = target_image.cpu().detach().numpy()
    # for j in range(0, dim):
        # Final_attack_sign[0, :, :, :] = Final_attack_sign[0, :, :, :] + Best_indi[j] * (sample_adv_images[j, :, :, :] - target_image[0, :, :, :])
    Final_direction = np.sign(Final_attack_sign)
    final_image = target_image + eps * Final_direction
    final_image = torch.from_numpy(final_image)
    final_image = final_image.float()
    final_image[0, :, :, :] = torch.clamp(final_image[0, :, :, :], -1, 1)
    return final_image, eval_num