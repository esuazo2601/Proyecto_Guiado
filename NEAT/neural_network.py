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
        self.output: float
        self.weight: dict[float]
        self.type = type
        self.ready = False

class NeuralNetwork:
    def __init__(self, genome: Genome, input: dict[int]):
        self.nodes: dict[int] = {}
        self.outputs = []
        for n in genome.nodes.genes:
            node = Node(n)
            node.type = n.type
            self.nodes[n.id] = node
            if n.type == "OUTPUT":
                self.outputs.append(node)
            elif n.type == "INPUT":
                node.ready = True
        
        for i, value in input.items():
            self.nodes[i].output = value
        
        for n, conn in genome.connections.genes.items():
            if not conn.Enabled:
                continue
            _in = conn.Input
            _out = conn.Output
            _weight = conn.Weight
            self.nodes[_out].input.append(_in)
            self.weigth[_in] = _weight

    def forward(self):
        # funcion iterativa que usa un stack
        stack = []

    def train(self):        # Entrenar el modelo
        pass

    def test(self):         # Probar un modelo ya entrenado
        pass
