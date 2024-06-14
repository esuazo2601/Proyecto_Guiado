from .genome import Genome
from typing import List, Dict, Self
import numpy as np
import math
import random

# Funciones de activacion
def sig(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0.0, x)

def leaky_relu(x):
    if x > 0:
        return x
    else:
        return 0.01 * x

def tanh(x):
    return math.tanh(x)

def indentity(x):
    return x

def softmax_stable(x):
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

class Neuron:
    def __init__(self, type: str) -> None:
        self.input: List[Self] = []
        self.output: List[Self] = []
        self.value: float = 0
        self.weight: Dict[Self] = {}
        self.type = type
        self.ready = False

    def __str__(self):
        return f"""
        input = {self.input}
        output = {self.output}
        weights = {self.weight}
        type = {self.type}
        """

class NeuralNetwork:
    def __init__(self, genome: Genome, inputSize: int, outputSize: int):
        self.neuron: Dict[int, Neuron] = {}
        self.outputs = []
        self.inputs = []
        self.input_size = inputSize
        self.output_size = outputSize

        # Diccionarios de neuronas válidas por tipo
        self.valid_inputs = {}
        self.valid_outputs = {}

        # Inicializar neuronas
        for n in genome.nodes.genes:
            node = Neuron(n)
            node.type = n.type
            self.neuron[n.id] = node
            if n.type == "OUTPUT":
                self.outputs.append(node)
                self.valid_outputs[n.id] = node
            elif n.type == "INPUT":
                node.ready = True
                self.inputs.append(node)
                self.valid_inputs[n.id] = node

        # Inicializar conexiones
        for conn in genome.connections.genes.values():
            if not conn.Enabled:
                continue
            try:
                _in = conn.Input
                _out = conn.Output
                _weight = conn.Weight

                if _in not in self.neuron:
                    print(f"Input neuron {_in} not found. Selecting a valid input neuron.")
                    _in = random.choice(list(self.valid_inputs.keys()))
                if _out not in self.neuron:
                    print(f"Output neuron {_out} not found. Selecting a valid output neuron.")
                    _out = random.choice(list(self.valid_outputs.keys()))

                self.neuron[_out].input.append(self.neuron[_in])
                self.neuron[_in].output.append(self.neuron[_out])
                self.neuron[_in].weight[self.neuron[_out]] = _weight
            except KeyError as e:
                print(f"Error: {e}")

    def forward(self, _input: dict[int]):
        for i, value in _input.items():
            self.neuron[i].value = value

        queue = []
        for n in self.outputs:
            queue.append(n)
        while len(queue) != 0:
            n = queue.pop()
            n.ready = True
            for i in n.input:
                if i.ready:
                    n.value += i.value * i.weight[n]
                else:
                    n.ready = False
                    queue.insert(0, i)
            if n.ready:
                if n.type != "INPUT":
                    n.value = sig(n.value)
            else:
                queue.insert(0, n)
        output_values = [x.value for x in self.outputs]
        output_values = np.array(output_values)
        final_values = softmax_stable(output_values)
        return final_values

