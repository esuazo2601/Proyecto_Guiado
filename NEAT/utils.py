import numpy as np
import torch

def indice_max_probabilidad(probabilidades):
    # Mueve el tensor a la CPU antes de convertirlo a numpy
    if isinstance(probabilidades, torch.Tensor):
        probabilidades = probabilidades.cpu().detach().numpy()
    
    cdf = np.cumsum(probabilidades)
    if cdf[-1] == 0:
        return np.random.randint(0, len(probabilidades))
    else:
        s = np.sum(probabilidades)
        rand = np.random.rand() * s
        return np.searchsorted(cdf, rand)
