import torch

def centerd_minmax_scaler(x: torch.Tensor):
    max = x.max()
    min = x.min()
    center = (max + min) / 2
    sclaed = 2*(x - center) / (max - min)
    return sclaed, (center, min, max)

def descaler(x: torch.Tensor, param: tuple):
    center, min, max = param
    return (x * (max - min) / 2) + center

def clip(x: torch.Tensor, min: float, max: float):
    return torch.max(torch.min(x, max), min)

def perturbate(x: torch.Tensor, delta: torch.Tensor, feat_len:int = 7, num_channels: int = 512, min: float = -1, max: float = 1):
    num_pixels = feat_len * feat_len
    delta = clip(delta, min, max)
    delta_pixel = delta[:num_pixels]
    delta_channel = delta[num_pixels:]
    delta_pixel = delta_pixel.expand_as(x)
    delta_channel = delta_channel.expand_as(x)
    abs_delta_pixel = torch.abs(delta_pixel)
    abs_delta_channel = torch.abs(delta_channel)
    scaled_x, sclae_param = centerd_minmax_scaler(x)
    scaled_x_perturbated = scaled_x + delta_pixel + delta_channel - abs_delta_pixel*scaled_x - abs_delta_channel*scaled_x
    scaled_x_perturbated = clip(scaled_x_perturbated, -1, 1)
    x_perturbated = descaler(scaled_x_perturbated, sclae_param)
    return x_perturbated

def relative_loss_gain(x: torch.Tensor, perturbated_x: torch.Tensor,label_list: list[int], true_label: int, 
                       loss_ft: torch.nn.Module, model: torch.nn.Module, alpha:float = 1.0):
    true_label_loss = loss_ft(model(perturbated_x), torch.tensor([true_label]))
    other_label_losses = torch.min([loss_ft(model(perturbated_x), torch.tensor([label])) for label in label_list if label != true_label])
    differnece = torch.mean(torch.abs(x, perturbated_x))
    real_gain = true_label_loss - other_label_losses # attack success when real_gain > 0
    regularized_gain = real_gain - alpha * differnece # bayesian optimize with this to minimize the differnece
    return regularized_gain, real_gain

