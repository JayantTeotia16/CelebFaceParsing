from torch import nn
import segmentation_models_pytorch as smp

torch_losses = {
    'crossentropyloss': nn.CrossEntropyLoss,
    'softcrossentropyloss': smp.losses.SoftCrossEntropyLoss,
    'focalloss': smp.losses.FocalLoss,
    'jaccardloss': smp.losses.JaccardLoss,
    'diceloss': smp.losses.DiceLoss
}

def get_loss(loss, loss_weights=None, custom_losses=None):
    if not isinstance(loss, dict):
        raise TypeError('The loss description is formatted improperly.'
                        ' See the docs for details.')
    if len(loss) > 1 :
        if loss_weights is None:
            weights = {k: 1 for k in loss.keys()}
        else:
            weights = loss_weights
        if list(loss.keys()).sort() != list(weights.keys()).sort():
            raise ValueError(
                'The losses and weights must have the same name keys.')
        return TorchCompositeLoss(loss, weights, custom_losses)
    else: 
        loss_name, loss_dict = list(loss.items())[0]
        return get_single_loss(loss_name, loss_dict, custom_losses)
    
def get_single_loss(loss_name, params_dict, custom_losses=None):
    if params_dict is None:
        return torch_losses.get(loss_name.lower())()
    else:
        return torch_losses.get(loss_name.lower())(**params_dict)

class TorchCompositeLoss(nn.Module):
    def __init__(self, loss_dict, weight_dict = None, custom_losses = None):
        super().__init__()
        self.weights = weight_dict
        self.losses = {loss_name: get_single_loss(loss_name,
                                                  loss_params,
                                                  custom_losses)
                       for loss_name, loss_params in loss_dict.items()}
        self.values = {}
    
    def forward(self, outputs, targets):
        loss = 0
        for func_name, weight in self.weights.items():
            self.values[func_name] = self.losses[func_name](outputs, targets)
            loss += weight*self.values[func_name]
        return loss