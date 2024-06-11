from .genome import Genome
from typing import Self
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch
import torch.nn.functional as F
import torch.nn as nn
import random

# Funciones de activación usando PyTorch
def relu(x):
    return F.relu(x)
def leaky_relu(x):
    return F.leaky_relu(x, negative_slope=0.01)
def tanh(x):
    return F.tanh(x)
def identity(x):
    return x
def softmax(x):
    return F.softmax(x, dim=0)
def sigmoid(x):
    return 1/(1 + np.exp(-x))


class Neuron(nn.Module):
    def __init__(self, type: str, device: torch.device) -> None:
        super(Neuron, self).__init__()
        self.input: list[Neuron] = []
        self.output: list[Neuron] = []
        self.value: float = 0  # Inicializa el tensor en el dispositivo
        self.weight: dict[Neuron, torch.Tensor] = {}
        self.type = type
        self.ready = False
        self.device = device

    def __str__(self):
        return f"""
        input = {self.input}
        output = {self.output}
        weights = {self.weight}
        type = {self.type}
        ready = {self.ready}
        """

class NeuralNetwork(nn.Module):
    def __init__(self, genome: Genome, inputSize, outputSize, device: torch.DeviceObjType):
        super(NeuralNetwork, self).__init__()
        self.neuron: dict[int, Neuron] = {}
        self.outputs = []
        self.inputs = []
        self.device = device

        self.output_size = inputSize
        self.input_size = outputSize

        # Inicializar neuronas
        for n in genome.nodes.genes:
            node = Neuron(n.type, device=device)
            self.neuron[n.id] = node
            if n.type == "OUTPUT":
                self.outputs.append(node)
            elif n.type == "INPUT":
                node.ready = True
                self.inputs.append(node)

        # Inicializar conexiones
        for n, conn in genome.connections.genes.items():
            if not conn.Enabled:
                continue

            _in = conn.Input
            _out = conn.Output
            
            if self.neuron[_in] is None:
                _in = conn.Input-1
            if self.neuron[_out] is None:
                _out = conn.Output-1
            
            weight = conn.Weight
            self.neuron[_out].input.append(self.neuron[_in])
            self.neuron[_in].output.append(self.neuron[_out])
            self.neuron[_in].weight[self.neuron[_out]] = weight

    def forward(self, _input: dict[int]):
        for i, value in _input.items():
            self.neuron[i].value = value

        queue = self.outputs.copy()
        while queue:
            n = queue.pop()
            n.ready = True
            for i in n.input:
                if i.ready:
                    n.value += i.value * i.weight[n]
                else:
                    n.ready = False
                    queue.insert(0, i)
            if n.ready and n.type != "INPUT":
                n.value = sigmoid(n.value)

        output_values = [x.value for x in self.outputs]
        output_values = torch.tensor(output_values, dtype=torch.float)
        return F.softmax(output_values, dim=0)
