from timm.models import create_model
import torch
from utils.get_unique_labels import get_unique_labels
from optimizer.ReDimBO import ReDimBO
from utils.image_plotter import adversarial_plot
def run_attack(victim_model_name, dataloader, autoencoder,
               acquisition:str = "EI", feature_len:int = 7, num_channels:int = 512, epoch_lim: int = 200):
    victim_model = create_model(victim_model_name, pretrained=True)
    victim_model.load_state_dict(torch.load(f"./../victim/results/{victim_model_name}_best.pth"))
    labels = get_unique_labels(dataloader)
    victim_model.eval()
    autoencoder.eval()

    originals = []
    advs = []
    used_epochs = []
    num_success = 0
    num_fail = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        for i in range(len(data)):
            image = data[i]
            true_label = target[i]
            adv_example, used_epoch, success = ReDimBO(image, victim_model, autoencoder, labels, true_label, acquisition, feature_len, num_channels, epoch_lim)
            if success:
                print(f"Attack successful for image {num_success + num_fail} with {used_epoch} epochs.")
                originals.append(image)
                advs.append(adv_example)
                used_epochs.append(used_epoch)
                num_success += 1
                if num_success == 1: # at first success
                    adversarial_plot(image, adv_example)
            else:
                print(f"Attack failed for image {num_success + num_fail}")
                num_fail += 1
    asr = num_success / (num_success + num_fail)
    print(f"Attack success rate: {asr}")
    return originals, advs, used_epochs, asr