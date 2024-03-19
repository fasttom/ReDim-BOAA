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
    shaped_max = torch.ones_like(x) * max
    shaped_min = torch.ones_like(x) * min
    return torch.max(torch.min(x, shaped_max), shaped_min)

def perturbate(z: torch.Tensor, delta: torch.Tensor, feat_len:int = 7, num_channels: int = 512, min: float = -1, max: float = 1):
    num_pixels = feat_len * feat_len
    delta = clip(delta, min, max)
    delta_pixel = delta[:num_pixels].unflatten(0, (feat_len, feat_len)) # 7x7
    delta_pixel = delta_pixel.unsqueeze(0) # 1x7x7
    delta_pixel = delta_pixel.expand(num_channels, -1, -1) # 512x7x7
    delta_channel = delta[num_pixels:] # 512
    delta_channel = delta_channel.unsqueeze(-1) # 512x1
    delta_channel = delta_channel.unsqueeze(-1) # 512x1x1
    delta_channel = delta_channel.expand(-1, feat_len, feat_len) # 512x7x7
    abs_delta_pixel = torch.abs(delta_pixel)
    abs_delta_channel = torch.abs(delta_channel)
    scaled_z, sclae_param = centerd_minmax_scaler(z)
    scaled_z_perturbated = scaled_z + delta_pixel + delta_channel - abs_delta_pixel*scaled_z - abs_delta_channel*scaled_z
    scaled_z_perturbated = clip(scaled_z_perturbated, -1, 1)
    z_perturbated = descaler(scaled_z_perturbated, sclae_param)
    return z_perturbated

def relative_loss_gain(x: torch.Tensor, perturbated_z: torch.Tensor,label_list: list[int], true_label: int, 
                       autoencoder: torch.nn.Module,
                       loss_ft: torch.nn.Module, model: torch.nn.Module, alpha:float = 1.0):
    perturbated_x = autoencoder.decoder(perturbated_z.unsqueeze(0)).squeeze(0)
    true_label_loss = loss_ft(model(perturbated_x.unsqueeze(0)).squeeze(0), torch.tensor(true_label))
    other_label_losses = torch.Tensor([loss_ft(model(perturbated_x.unsqueeze(0)).squeeze(0), torch.tensor(label)) for label in label_list if label != true_label])
    other_label_loss = torch.min(other_label_losses)
    differnece = torch.mean(torch.abs(x - perturbated_x))
    real_gain = true_label_loss - other_label_loss # attack success when real_gain > 0
    regularized_gain = real_gain - alpha * differnece # bayesian optimize with this to minimize the differnece
    return regularized_gain, real_gain

def random_delta(feat_len:int = 7, num_channels: int = 512, min: float = -1, max: float = 1):
    num_pixels = feat_len * feat_len
    delta = torch.rand(size=(num_pixels + num_channels,)) * (max - min) + min
    return delta

