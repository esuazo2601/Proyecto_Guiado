from .genome import Genome
#from typing import Self
import numpy as np
import math

# Funciones de activacion
def sig(x):
 return 1/(1 + np.exp(-x))

def relu(x):
    return max(0.0, x)

def leaky_relu(x):
  if x>0 :
    return x
  else :
    return 0.01*x

def tanh(x):
    return math.tanh(x)

def indentity(x):
    return x

# Clase encargada de ejecutar la red seleccionada por neat y reconstruirla
# a partir de los archivos .pickle generados por este

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
weigths = {self.weight}
type = {self.type}
ready = {self.ready}
            """

class NeuralNetwork:
    def __init__(self, genome: Genome):
        self.neuron: dict[int] = {}
        self.outputs = []
        self.inputs = []
        for n in genome.nodes.genes:
            node = Neuron(n)
            node.type = n.type
            self.neuron[n.id] = node
            if n.type == "OUTPUT":
                self.outputs.append(node)
            elif n.type == "INPUT":
                node.ready = True
                self.inputs.append(node)
        
        for n, conn in genome.connections.genes.items():
            if not conn.Enabled:
                continue
            _in = conn.Input
            _out = conn.Output
            _weight = conn.Weight
            self.neuron[_out].input.append(self.neuron[_in])
            self.neuron[_in].output.append(self.neuron[_out])
            self.neuron[_in].weight[self.neuron[_out]] = _weight
        
        # DEBUG
        # for i, n in self.neuron.items():
        #     print(i, ":", n)


    # TODO: fix infinite loop
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
                    n.value += i.value*i.weight[n]
                else:
                    n.ready = False
                    queue.insert(0, i)
            if n.ready:
                if n.type != "INPUT":
                    n.value = sig(n.value)
            else:
                queue.insert(0, n)

        return [x.value for x in self.outputs]
