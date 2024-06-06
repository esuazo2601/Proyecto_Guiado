from .genome import Genome
from typing import Self
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch
import torch.nn.functional as F
import torch.nn as nn

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

class Neuron(nn.Module):
    def __init__(self, type: str, device: torch.device) -> None:
        super(Neuron,self).__init__()
        self.input: list[Neuron] = []
        self.output: list[Neuron] = []
        self.value: torch.Tensor = torch.tensor(0.0, device=device)  # Inicializa el tensor en el dispositivo
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
    def __init__(self, genome: Genome, device: torch.device) -> torch.Tensor:
        super(NeuralNetwork,self).__init__()
        self.neuron: dict[int, Neuron] = {}
        self.outputs = []
        self.inputs = []
        self.device = device
        for n in genome.nodes.genes:
            node = Neuron(n.type, device)
            node.type = n.type
            self.neuron[n.id] = node
            if n.type == "OUTPUT":
                self.outputs.append(node)
            elif n.type == "INPUT":
                node.ready = True
                self.inputs.append(node)
        
        self.to(device=device)

        for n, conn in genome.connections.genes.items():
            if not conn.Enabled:
                continue
            _in = conn.Input
            _out = conn.Output
            _weight = conn.Weight
            self.neuron[_out].input.append(self.neuron[_in])
            self.neuron[_in].output.append(self.neuron[_out])
            self.neuron[_in].weight[self.neuron[_out]] = torch.tensor(_weight, device=device)  # Mueve el peso al dispositivo

    def forward(self, _input: torch.Tensor):
        # Set the values of the input nodes
        for i, node in enumerate(self.inputs):
            node.value = _input[i]

        queue = []
        for n in self.outputs:
            queue.append(n)

        while len(queue) != 0:
            n = queue.pop()
            n.ready = True
            for i in n.input:
                if i.ready:
                    n.value += i.value * i.weight[n]
                    #print(f"Neuron {n}: value {n.value}, weight {i.weight[n]}")  # Debugging line
                else:
                    n.ready = False
                    queue.insert(0, i)
            if n.ready:
                if n.type != "INPUT":
                    n.value = leaky_relu(n.value)
            else:
                queue.insert(0, n)

        # Get the values of the output nodes and apply the softmax function
        output_values = [x.value for x in self.outputs]
        output_values = torch.tensor(output_values,dtype=torch.float)
        return softmax(output_values)

