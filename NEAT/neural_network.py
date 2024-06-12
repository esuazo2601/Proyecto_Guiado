from .genome import Genome
from typing import List, Dict
import torch.nn as nn
import torch.nn.functional as F
import torch

class Neuron(nn.Module):
    def __init__(self, type: str, device: torch.device) -> None:
        super(Neuron, self).__init__()
        self.input: List[Neuron] = []
        self.output: List[Neuron] = []
        self.value: torch.Tensor = torch.tensor(0.0, device=device)
        self.weight: Dict[Neuron, torch.Tensor] = {}
        self.type = type
        self.device = device

    def __str__(self):
        return f"""
        input = {self.input}
        output = {self.output}
        weights = {self.weight}
        type = {self.type}
        """

class NeuralNetwork(nn.Module):
    def __init__(self, genome: Genome, inputSize: int, outputSize: int, device: torch.device):
        super(NeuralNetwork, self).__init__()
        self.neuron: Dict[int, Neuron] = {}
        self.outputs: List[Neuron] = []
        self.inputs: List[Neuron] = []
        self.device = device
        self.input_size = inputSize
        self.output_size = outputSize
        self.to(device=device)

        # Inicializar neuronas
        for n in genome.nodes.genes:
            node = Neuron(n.type, device=device)
            self.neuron[n.id] = node
            if n.type == "OUTPUT":
                self.outputs.append(node)
            elif n.type == "INPUT":
                node.value = torch.tensor(1.0, device=device)  # Inputs are ready with a value of 1
                self.inputs.append(node)

        # Inicializar conexiones
        for conn in genome.connections.genes.values():
            if not conn.Enabled:
                continue
            try:
                _in = conn.Input
                _out = conn.Output
                weight = torch.tensor(conn.Weight, device=device)
                self.neuron[_out].input.append(self.neuron[_in])
                self.neuron[_in].output.append(self.neuron[_out])
                self.neuron[_in].weight[self.neuron[_out]] = weight
            except KeyError as e:
                print(f"Error: {e}")
            #if _in in self.neuron and _out in self.neuron:
            #    weight = torch.tensor(conn.Weight, device=device)
            #    self.neuron[_out].input.append(self.neuron[_in])
            #    self.neuron[_in].output.append(self.neuron[_out])
            #    self.neuron[_in].weight[self.neuron[_out]] = weight
            #else:
            #    print(f"Warning: Invalid connection from {_in} to {_out} - Neuron not found.")

    def forward(self, _input: Dict[int, float]):
        for i, value in _input.items():
            if i in self.neuron:
                self.neuron[i].value = torch.tensor(value, device=self.device)

        queue = self.outputs.copy()
        while queue:
            n = queue.pop()
            for i in n.input:
                if i.value is not None:
                    n.value += i.value * i.weight[n]
                else:
                    queue.insert(0, i)
            if n.type != "INPUT":
                n.value = torch.sigmoid(n.value)

        output_values = [x.value for x in self.outputs]
        output_values = torch.stack(output_values)
        return F.softmax(output_values, dim=0)
