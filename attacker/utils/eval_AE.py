import torch
from autoencoder.classes.resnet_autoencoder import AE
from utils.image_plotter import check_plot, compare_plot
from sklearn.metrics import root_mean_squared_error

def evaluate_AE(loader, model_type:str = "Res_AE", num_layers: int = 34, dataset_name:str = "imagenette2-320", victim_model:str = ""):
    network_type = "default" if num_layers == 34 or 18 else "light"
    model = AE(network=network_type, num_layers=num_layers).to("cuda")
    model.load_state_dict(torch.load(f"./autoencoder/results/{model_type}_{num_layers}_best.pth"))
    
    encoder = model.encoder
    decoder = model.decoder
    
    encoder.eval()
    decoder.eval()

    check_plot(model, loader, dataset_name, victim_model)

def single_plot_AE(loader, model_type: str = "Res_AE", num_layers: int = 34, dataset_name:str = "imagenette2-320", length: int = 30):
    network_type = "default" if num_layers == 34 or 18 else "light"
    model = AE(network=network_type, num_layers=num_layers).to("cuda")
    model.load_state_dict(torch.load(f"./autoencoder/results/{model_type}_{num_layers}_best.pth"))
    
    encoder = model.encoder
    decoder = model.decoder
    
    encoder.eval()
    decoder.eval()

    compare_plot(model, loader, dataset_name, length)

def check_differece_AE(loader, model_type: str = "Res_AE", num_layers: int = 34, dataset_name:str = "imagenette2-320"):
    network_type = "default" if num_layers == 34 or 18 else "light"
    model = AE(network=network_type, num_layers=num_layers).to("cuda")
    model.load_state_dict(torch.load(f"./autoencoder/results/{model_type}_{num_layers}_best.pth"))
    
    encoder = model.encoder
    decoder = model.decoder
    
    encoder.eval()
    decoder.eval()

    sample_count = 0
    rmse = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):

            inputs = data.to("cuda")
            outputs = model.decoder(model.encoder(inputs))

            input_samples = inputs.permute(0,2,3,1).cpu().numpy() # BRG to RGB
            reconstructed_samples = outputs.permute(0,2,3,1).cpu().numpy() # BRG to RGB

            for i in range(len(input_samples)):
                input = input_samples[i].ravel()/256
                reconstruction = reconstructed_samples[i].ravel()/256
                
                rmse += root_mean_squared_error(input, reconstruction)
                sample_count += 1

    rmse /= sample_count

    return rmse

