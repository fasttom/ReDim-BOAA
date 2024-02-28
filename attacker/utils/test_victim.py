import torch
from torch.utils.data import Subset
from timm.models import create_model
from timm.data.loader import create_loader

def test_victim(loader, model_name:str = "resnet50"):
    victim_model = create_model(model_name, pretrained=True)
    victim_model.load_state_dict(torch.load(f"./../victim/results/{model_name}_best.pth"))
    victim_model.eval()
    victim_model.to("cuda")

    correct = 0
    total = 0

    is_correct = []

    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to("cuda")
            labels = labels.to("cuda")
            outputs = victim_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            is_correct += (predicted == labels).tolist()
        accuracy = 100 * correct / total
        print(f'Accuracy of the {model_name} on the attack images: {accuracy:.2f} %')
        correct_idx = [i for i, x in enumerate(is_correct) if x]

    well_classified_dataset = Subset(loader.dataset, correct_idx)
    well_classified_loader = create_loader(
        well_classified_dataset,
        input_size=(3, 224, 224),
        batch_size=32,
        is_training=False,
        use_prefetcher=False,
    )
    return accuracy, well_classified_loader