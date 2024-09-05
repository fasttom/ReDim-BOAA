from timm.models import create_model
import torch
from utils.get_unique_labels import get_unique_labels
from optimizer.ReDimBO import ReDimBO
from optimizer.BaseLineBO import BaseLineBO
from optimizer.SemiReDimBO import SemiReDimBO
from utils.image_plotter import adversarial_plot
def run_attack(victim_model_name, dataloader, autoencoder,
               acquisition:str = "EI", feature_len:int = 7, num_channels:int = 512, epoch_lim: int = 200,
               is_baseline: bool = False, is_semi: bool = False):
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
            if is_baseline:
                adv_example, used_epoch, success = BaseLineBO(image, victim_model, labels, true_label, acquisition, epoch_lim)
            elif is_semi:
                adv_example, used_epoch, success = SemiReDimBO(image, victim_model, autoencoder, labels, true_label, acquisition, feature_len, num_channels, epoch_lim)
            else:
                adv_example, used_epoch, success = ReDimBO(image, victim_model, autoencoder, labels, true_label, acquisition, feature_len, num_channels, epoch_lim)
            if success:
                print(f"Attack successful for image {num_success + num_fail} with {used_epoch} epochs.")
                originals.append(image)
                advs.append(adv_example)
                used_epochs.append(used_epoch)
                num_success += 1
                if is_baseline:
                    adversarial_plot(image, adv_example, victim_model_name+"_base", num_success + num_fail)
                elif is_semi:
                    adversarial_plot(image, adv_example, victim_model_name+"_semi", num_success + num_fail)
                else:
                    adversarial_plot(image, adv_example, victim_model_name, num_success + num_fail)
            else:
                print(f"Attack failed for image {num_success + num_fail}")
                num_fail += 1
    asr = num_success / (num_success + num_fail)
    print(f"Attack success rate: {asr}")
    return originals, advs, used_epochs, asr