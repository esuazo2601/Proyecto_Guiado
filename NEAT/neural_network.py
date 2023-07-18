from .neat import NEAT
from .genome import Genome
from .connection_genes import ConnectionGenes, Connection
from .node_genes import NodeGenes, Node

import torch
import torch.nn as nn

from torch import autograd
# Clase encargada de ejecutar la red seleccionada por neat y reconstruirla
# a partir de los archivos .pickle generados por este
class NeuralNetwork:
    def __init__(self, genome: Genome):
        # TODO: todo lo que es construir un modelo de pytorch
        pass

    def train(self):        # Entrenar el modelo
        pass

    def test(self):         # Probar un modelo ya entrenado
        pass
