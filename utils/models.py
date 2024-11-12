import torch
import segmentation_models_pytorch as smp

def _load_model_weights(model, path):
    if torch.cuda.is_available():
        try:
            loaded = torch.load(path)
        except FileNotFoundError:
            raise FileNotFoundError("{} doesn't exist.".format(path))
    else:
        try:
            loaded = torch.load(path, map_location='cpu')
        except FileNotFoundError:
            raise FileNotFoundError("{} doesn't exist.".format(path))
        
    model.load_state_dict(loaded['state_dict'])
    print(f"model {path} loading finished")
    return model

model_dict = {'deeplabv3plus':smp.DeepLabV3Plus}
