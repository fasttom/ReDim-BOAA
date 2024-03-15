from timm.
from optimizer.ReDimBO import ReDimBO
def run_attack(victim_model_name, dataloader, autoencoder,
               acquisition:str = "EI", feature_len:int = 7, num_channels:int = 512, epoch_lim: int = 200):
    victim