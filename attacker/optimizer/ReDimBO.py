import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from optimizer.utils import perturbate, relative_loss_gain, random_delta
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf

def ReDimBO(image:torch.Tensor,
            victim_model: torch.nn.Module, autoencoder: torch.nn.Module, labels: list[int], true_label:int, 
            acquisition:str = "EI", feature_len:int = 7, num_channels:int = 512, epoch_lim: int = 200):
    # first ten perturbations are selected randomly
    deltas = [random_delta() for _ in range(10)]
    z = autoencoder.encoder(image.unsqueeze(0)).squeeze(0)
    perturbated_zs = [perturbate(z, delta) for delta in deltas]
    _ = [relative_loss_gain(image, perturbated_z, labels, true_label, autoencoder, torch.nn.CrossEntropyLoss(), victim_model) for perturbated_z in perturbated_zs]
    loss_gains, real_loss_gains = zip(*_)
    loss_gains = list(loss_gains)
    real_loss_gains = list(real_loss_gains)

    for i in range(10): #if the attack is successful by random perturbation, 
        # we will not use the bayesian optimization
        if real_loss_gains[i]>0:
            real_z = perturbated_zs[i]
            adv_example = autoencoder.decoder(real_z.unsqueeze(0)).squeeze(0)
            return adv_example, 10, True
    used_epoch = 10

    # if given perturbations are not successful, we will use bayesian optimization
    # until the epoch limit is reached
    for i in range(epoch_lim-10):
        # ititialize the model
        gp = SingleTaskGP(deltas, loss_gains)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        # fit the model
        fit_gpytorch_mll(mll)

        # initialize the acquisition function
        if acquisition == "EI":
            AQ = ExpectedImprovement(gp, best_f=0)
        elif acquisition == "UCB":
            AQ = UpperConfidenceBound(gp, beta=0.1)
        
        # initialize the bounds
        bounds = torch.stack([torch.ones(feature_len*feature_len+num_channels)*-1, torch.ones(feature_len*feature_len+num_channels)])

        # optimize the acquisition function
        candidate, acq_value = optimize_acqf(AQ, bounds=bounds, q=1, num_restarts=5, raw_samples=int(5+0.5*i))

        # evaluate the candidiate
        perturbated_z = perturbate(image, candidate)
        loss_gain, real_loss_gain = relative_loss_gain(image, perturbated_z, labels, true_label, autoencoder, torch.nn.CrossEntropyLoss(), victim_model)
        used_epoch += 1
        if real_loss_gain > 0:
            adv_example = autoencoder.decoder(perturbated_z.unsqueeze(0)).squeeze(0)
            return adv_example, used_epoch, True
        else:
            deltas.append(candidate)
            perturbated_zs.append(perturbated_z)
            loss_gains.append(loss_gain)
            real_loss_gains.append(real_loss_gain)
    
    return None, np.nan, False

        




