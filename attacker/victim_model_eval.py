from dataloader.imagenette_loader import load_vicim_data, load_vicim_train_data, load_vicim_val_data
from utils.test_victim import test_victim

# load target dataset 
train_loader = load_vicim_train_data(dataset_type="timm", dataset_name="Caltech-256-Splitted", input_size=(3, 224, 224), train_batch=32)
val_loader = load_vicim_val_data(dataset_type="timm", dataset_name="Caltech-256-Splitted", input_size=(3, 224, 224), val_batch=32)
test_loader = load_vicim_data(dataset_type="timm", dataset_name="Caltech-256-Splitted", input_size=(3, 224, 224), victim_batch=32)

# Accuracy on train set
print("####################")
print("Accuracy on train set")
mobilenet_accuracy, _ = test_victim(train_loader, model_name="mobilenetv3_large_100")
resnet_accuracy, _ = test_victim(train_loader, model_name="resnet50")
vgg_accuracy, _ = test_victim(train_loader, model_name="vgg19")
vit_small_accuracy, _ = test_victim(train_loader, model_name="vit_small_patch32_224")
print("####################")
print("")

# Accuracy on val set
print("####################")
print("Accuracy on val set")
mobilenet_accuracy, _ = test_victim(val_loader, model_name="mobilenetv3_large_100")
resnet_accuracy, _ = test_victim(val_loader, model_name="resnet50")
vgg_accuracy, _ = test_victim(val_loader, model_name="vgg19")
vit_small_accuracy, _ = test_victim(val_loader, model_name="vit_small_patch32_224")
print("####################")
print("")

# Accuracy on test set
print("####################")
print("Accuracy on test set")
mobilenet_accuracy, _ = test_victim(test_loader, model_name="mobilenetv3_large_100")
resnet_accuracy, _ = test_victim(test_loader, model_name="resnet50")
vgg_accuracy, _ = test_victim(test_loader, model_name="vgg19")
vit_small_accuracy, _ = test_victim(test_loader, model_name="vit_small_patch32_224")
print("####################")
print("")