from .genome import Genome
from typing import Self
import numpy as np
import math

# Activation functions
def sig(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0.0, x)

def leaky_relu(x):
    return x if x > 0 else 0.01 * x

def tanh(x):
    return math.tanh(x)

def indentity(x):
    return x

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def stable_softmax(x):
    x = np.array(x)
    if x.ndim == 1:
        z = x - np.max(x) 
    else:
        z = x - np.max(x, axis=1, keepdims=True)
    return np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)

class Neuron:
    def __init__(self, type: str) -> None:
        self.input: list[Self] = []
        self.output: list[Self] = []
        self.value: float = 0
        self.weight: dict[Self] = {}
        self.type = type
        self.ready = False
    
    def __str__(self):
        return f"""
input = {self.input}
output = {self.output}
weights = {self.weight}
type = {self.type}
ready = {self.ready}
        """

class NeuralNetwork:
    def __init__(self, genome: Genome):
        self.neuron: dict[int, Neuron] = {}
        self.outputs = []
        self.inputs = []

        # Initialize neurons
        for n in genome.nodes.genes:
            node = Neuron(n.type)
            self.neuron[n.id] = node
            if n.type == "OUTPUT":
                self.outputs.append(node)
            elif n.type == "INPUT":
                node.ready = True
                self.inputs.append(node)

        for conn in genome.connections.genes.values():
            if not conn.Enabled:
                continue
            _in = conn.Input
            _out = conn.Output
            _weight = conn.Weight

            # si no existen los nodos, se crean como HIDDEN
            if _in not in self.neuron:
                self.neuron[_in] = Neuron("HIDDEN")
            if _out not in self.neuron:
                self.neuron[_out] = Neuron("HIDDEN")

            self.neuron[_out].input.append(self.neuron[_in])
            self.neuron[_in].output.append(self.neuron[_out])
            self.neuron[_in].weight[self.neuron[_out]] = _weight

    def forward(self, _input: dict[int, float]):
        for node_id, value in _input.items():
            if node_id in self.neuron:
                self.neuron[node_id].value = value

        queue = []
        processed = set()

        for n in self.outputs:
            queue.append(n)

        while queue:
            n = queue.pop()
            if n in processed:
                continue
            processed.add(n)
            n.ready = True
            for i in n.input:
                if i.ready:
                    n.value += i.value * i.weight[n]
                else:
                    n.ready = False
                    queue.insert(0, n)
                    queue.insert(0, i)
                    break
            if n.ready and n.type != "INPUT":
                n.value = tanh(n.value)

        # Normalizar la salida con el softmax estable para los valores muy grandes generados con RELU por ejemplo
        output_values = [x.value for x in self.outputs]
        return stable_softmax(output_values)
