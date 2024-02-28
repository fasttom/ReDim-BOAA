import torch
from autoencoder.classes.resnet_autoencoder import AE
from utils.image_plotter import check_plot

def evaluate_AE(loader, model_type:str = "Res_AE", num_layers: int = 34, dataset_name:str = "imagenette2-320"):
    network_type = "default" if num_layers == 34 or 18 else "light"
    model = AE(network=network_type, num_layers=num_layers).to("cuda")
    model.load_state_dict(torch.load(f"./autoencoder/results/{model_type}_{num_layers}_best.pth"))
    
    encoder = model.encoder
    decoder = model.decoder
    
    encoder.eval()
    decoder.eval()

    check_plot(model, loader, dataset_name)

