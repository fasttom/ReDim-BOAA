import os

def dict_reader(model_name:str, is_base:bool = False, is_semi:bool = False):
    d = {}
    if is_base:
        model_name = model_name+"_base"
    elif is_semi:
        model_name = model_name+"_semi"
    else:
        model_name = model_name
    with open(f"./results/summary_{model_name}.txt", 'r') as f:
        for line in f:
            line = line.replace(" ","")
            line = line.replace(":"," ")
            key, value = line.split(" ")
            value = eval(value)
            d[key] = value
    return d

    
def mean_epoch_calculator(summary_dict:dict):
    epochs = summary_dict["epochs"]
    mean_epoch = sum(epochs)/len(epochs)
    return mean_epoch

mobilenet_epoch = mean_epoch_calculator(dict_reader("mobilenetv3_large_100", False, False))
resnet_epoch = mean_epoch_calculator(dict_reader("resnet50", False, False))
vgg_epoch = mean_epoch_calculator(dict_reader("vgg19", False, False))
vit_epoch = mean_epoch_calculator(dict_reader("vit_small_patch32_224", False, False))

print("mobilenet mean: ", mobilenet_epoch)
print("resnet mean: ", resnet_epoch)
print("vgg_mean: ", vgg_epoch)
print("vit_mean: ", vit_epoch)

vgg_semi_epoch = mean_epoch_calculator(dict_reader("vgg19", False, True))
vgg_base_epoch = mean_epoch_calculator(dict_reader("vgg19", True, False))

print("vgg_semi_mean: ", vgg_semi_epoch)
print("vgg_base_mean: ", vgg_base_epoch)

vit_semi_epoch = mean_epoch_calculator(dict_reader("vit_small_patch32_224", False, True))
vit_base_epoch = mean_epoch_calculator(dict_reader("vit_small_patch32_224", True, False))

print("vit_semi_mean: ", vit_semi_epoch)
print("vit_base_mean: ", vit_base_epoch)

