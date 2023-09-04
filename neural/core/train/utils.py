import torch



def get_parameters(model):
    return sum(i.numel() for i in model.parameters())