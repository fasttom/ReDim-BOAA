import torch
from timm.models import create_model

def test_victim(loader, model_name:str = "resnet50"):
    victim_model = create_model(model_name, pretrained=True)
    victim_model.load_state_dict(torch.load(f"./../victim/results/{model_name}_best.pth"))
    victim_model.eval()
    victim_model.to("cuda")

    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to("cuda")
            labels = labels.to("cuda")
            outputs = victim_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy of the {model_name} on the attack images: {accuracy:.2f} %')
    return accuracy