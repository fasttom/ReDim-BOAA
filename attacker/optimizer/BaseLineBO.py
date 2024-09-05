import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from optimizer.utils import perturbate_baseline, baseline_loss_gain, random_noise, image_interpolarate
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf

def BaseLineBO(image:torch.Tensor,
            victim_model: torch.nn.Module, labels: list[int], true_label:int, 
            acquisition:str = "EI", epoch_lim: int = 200):
    image_shape = image.shape
    z = image.flatten()
    # first 20 perturbations are selected randomly.
    deltas = [random_noise(image_shape).flatten() for _ in range(20)] #initial 20 random perturbations
    perturbated_zs = [perturbate_baseline(image.flatten(), delta) for delta in deltas]
    loss_gains = [baseline_loss_gain(image, perturbated_z, labels, true_label, image_shape, torch.nn.CrossEntropyLoss(), victim_model) for perturbated_z in perturbated_zs]


    # random perturbation can be too far away from original image
    # so we will use the first successful perturbation as the initial point
    # even if initial perturbations are successful
    used_epoch = 20

    # if given perturbations are not successful, we will use bayesian optimization
    # until the epoch limit is reached
    for i in range(epoch_lim-20):
        # ititialize the model
        tensor_delta = torch.stack(deltas)
        tensor_delta = tensor_delta.unsqueeze(0)
        tensor_loss_gains = torch.Tensor(loss_gains)
        tensor_loss_gains = tensor_loss_gains.unsqueeze(-1)
        tensor_loss_gains = tensor_loss_gains.unsqueeze(0)
        gp = SingleTaskGP(tensor_delta, tensor_loss_gains)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        # fit the model
        fit_gpytorch_mll(mll)

        # initialize the acquisition function
        if acquisition == "EI":
            AQ = ExpectedImprovement(gp, best_f=0)
        elif acquisition == "UCB":
            AQ = UpperConfidenceBound(gp, beta=0.1)
        
        # initialize the bounds
        bounds = torch.stack([torch.ones(z.shape)*-1, torch.ones(z.shape)])

        # optimize the acquisition function
        candidate, acq_value = optimize_acqf(AQ, bounds=bounds, q=1, num_restarts=1, raw_samples=i+20)
        candidate = candidate.squeeze(0)

        # evaluate the candidiate
        perturbated_z = perturbate_baseline(z, candidate)
        loss_gain= baseline_loss_gain(image, perturbated_z, labels, true_label, image_shape, torch.nn.CrossEntropyLoss(), victim_model)
        used_epoch += 1
        if loss_gain > 0:
            adv_example = perturbated_z.unflatten(0, image_shape)
            adv_example = image_interpolarate(image, adv_example, 0.1)
            return adv_example, used_epoch, True
        else:
            deltas.append(candidate)
            perturbated_zs.append(perturbated_z)
            loss_gains.append(loss_gain)
    
    return None, np.nan, False