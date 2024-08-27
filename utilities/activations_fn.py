import torch 
from torchtyping import TensorType 


def sigmoid(x) -> TensorType[float]:
    return 1 / (1 + torch.exp(-1))

def bipolar_sigmoid(x) -> TensorType[float]:
    return (torch.exp(x) - 1) / (torch.exp(x) + 1)

def tanh(x) -> TensorType[float]:
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

 