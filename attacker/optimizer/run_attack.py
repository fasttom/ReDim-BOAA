from timm.models import create_model
import torch
from utils.get_unique_labels import get_unique_labels
from optimizer.ReDimBO import ReDimBO
from optimizer.BaseLineBO import BaseLineBO
from optimizer.SemiReDimBO import SemiReDimBO
from utils.image_plotter import adversarial_plot, noise_plot
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

def run_attack_short(victim_model_name, dataloader, autoencoder,
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
    succeed_idx = []
    num_success = 0
    num_fail = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        for i in [0]: # only examine first data in each batch
            image = data[i]
            true_label = target[i]
            if is_baseline:
                adv_example, used_epoch, success = BaseLineBO(image, victim_model, labels, true_label, acquisition, epoch_lim)
            elif is_semi:
                adv_example, used_epoch, success = SemiReDimBO(image, victim_model, autoencoder, labels, true_label, acquisition, feature_len, num_channels, epoch_lim)
            else:
                adv_example, used_epoch, success = ReDimBO(image, victim_model, autoencoder, labels, true_label, acquisition, feature_len, num_channels, epoch_lim)
            if success:
                print(f"Attack successful for image {32*(num_success + num_fail)} with {used_epoch} epochs.")
                originals.append(image)
                advs.append(adv_example)
                used_epochs.append(used_epoch)
                succeed_idx.append(32*(num_fail+num_success))
                num_success += 1
                if is_baseline:
                    adversarial_plot(image, adv_example, victim_model_name+"_base_class_compare", 32*(num_success + num_fail- 1))
                elif is_semi:
                    adversarial_plot(image, adv_example, victim_model_name+"_semi_class_compare", 32*(num_success + num_fail - 1))
                else:
                    adversarial_plot(image, adv_example, victim_model_name+"_class_compare", 32*(num_success + num_fail-1))
            else:
                print(f"Attack failed for image {32*(num_success + num_fail)}")
                num_fail += 1
        if num_success == 10: # we use only ten successful attacks
            break
    asr = num_success / (num_success + num_fail)
    print(f"Attack success rate: {asr}")
    return originals, advs, succeed_idx

def run_noise_plot(victim_model_name, dataloader, autoencoder,
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
    succeed_idx = []
    num_success = 0
    num_fail = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        for i in [0]: # only examine first data in each batch
            image = data[i]
            true_label = target[i]
            if is_baseline:
                adv_example, used_epoch, success = BaseLineBO(image, victim_model, labels, true_label, acquisition, epoch_lim)
            elif is_semi:
                adv_example, used_epoch, success = SemiReDimBO(image, victim_model, autoencoder, labels, true_label, acquisition, feature_len, num_channels, epoch_lim)
            else:
                adv_example, used_epoch, success = ReDimBO(image, victim_model, autoencoder, labels, true_label, acquisition, feature_len, num_channels, epoch_lim)
            if success:
                print(f"Attack successful for image {32*(num_success + num_fail)} with {used_epoch} epochs.")
                originals.append(image)
                advs.append(adv_example)
                used_epochs.append(used_epoch)
                succeed_idx.append(32*(num_fail+num_success))
                num_success += 1
                if is_baseline:
                    noise_plot(image, adv_example, victim_model_name+"_base_noise_plot", 32*(num_success + num_fail- 1))
                elif is_semi:
                    noise_plot(image, adv_example, victim_model_name+"_semi_noise_plot", 32*(num_success + num_fail - 1))
                else:
                    noise_plot(image, adv_example, victim_model_name+"_noise_plot", 32*(num_success + num_fail-1))
            else:
                print(f"Attack failed for image {32*(num_success + num_fail)}")
                num_fail += 1
        if num_success == 10: # we use only ten successful attacks
            break
    asr = num_success / (num_success + num_fail)
    print(f"Attack success rate: {asr}")
    return originals, advs, succeed_idx

def get_adversarial_classes(victim_model_name, originals, advs):
    victim_model = create_model(victim_model_name, pretrained=True)
    victim_model.load_state_dict(torch.load(f"./../victim/results/{victim_model_name}_best.pth"))
    victim_model.eval()
    true_labels = []
    adv_labels = []
    for i in range(len(originals)):
        original = torch.unsqueeze(originals[i], 0)
        adv = torch.unsqueeze(advs[i], 0)
        original_score = torch.nn.functional.softmax(victim_model(original), dim = 1)
        _, ori_class = torch.max(original_score, dim = 1)
        adv_score = torch.nn.functional.softmax(victim_model(adv), dim = 1)
        _, adv_class = torch.max(adv_score, dim=1)
        true_labels.append(int(ori_class.item()))
        adv_labels.append(int(adv_class.item()))
    return true_labels, adv_labels

def make_adv_class_dict(succeed_idx, true_labels, adv_labels):
    adv_class_dict = dict()
    adv_class_dict["idxs"] = succeed_idx
    adv_class_dict["true_labels"] = true_labels
    adv_class_dict["adv_labels"] = adv_labels
    return adv_class_dict