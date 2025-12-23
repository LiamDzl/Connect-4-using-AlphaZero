import torch
from neural_network import policy

parameters = policy(structure=[42, 64, 64, 64, 8])
torch.save(parameters, "random_parameters.pt")
