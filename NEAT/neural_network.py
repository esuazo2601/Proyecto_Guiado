from .neat import NEAT
from .genome import Genome
from .connection_genes import ConnectionGenes, Connection
from .node_genes import NodeGenes, Node
from typing import Self

# Clase encargada de ejecutar la red seleccionada por neat y reconstruirla
# a partir de los archivos .pickle generados por este

class Node:
    def __init__(self, type: str) -> None:
        self.input: list[Self] = []
        self.output = list[Self] = []
        self.value: float = 0
        self.weight: dict[Self]
        self.type = type
        self.ready = False

class NeuralNetwork:
    def __init__(self, genome: Genome, input: dict[int]):
        self.nodes: dict[int] = {}
        self.outputs = []
        self.inputs = []
        for n in genome.nodes.genes:
            node = Node(n)
            node.type = n.type
            self.nodes[n.id] = node
            if n.type == "OUTPUT":
                self.outputs.append(node)
            elif n.type == "INPUT":
                node.ready = True
                self.inputs.append(node)
        
        for i, value in input.items():
            self.nodes[i].value = value
        
        for n, conn in genome.connections.genes.items():
            if not conn.Enabled:
                continue
            _in = conn.Input
            _out = conn.Output
            _weight = conn.Weight
            self.nodes[_out].input.append(self.nodes[_in])
            self.nodes[_in].output.append(self.nodes[_out])
            self.nodes[_in].weight[self.nodes[_out]] = _weight

    def forward(self):
        # funcion iterativa que usa un stack
        stack = []
        for n in self.inputs:
            stack.append(n)
        
        while len(stack) != 0:
            n = stack.pop()
            if n.ready:
                for i in n.output:
                    i.value += n.value * n.weigth[i]
                    i.input.remove(n)
                    stack.append(i)
            else:
                if len(n.input) == 0:
                    n.ready = True
                stack.append(n)

    def train(self):        # Entrenar el modelo
        pass

    def test(self):         # Probar un modelo ya entrenado
        pass
