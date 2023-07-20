from .neat import NEAT
from .genome import Genome
from .connection_genes import ConnectionGenes, Connection
from .node_genes import NodeGenes, Genes
from typing import Self

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
    def __init__(self, genome: Genome, input: dict[int]):
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
        
        for i, value in input.items():
            self.neuron[i].value = value
        
        for n, conn in genome.connections.genes.items():
            if not conn.Enabled:
                continue
            _in = conn.Input
            _out = conn.Output
            _weight = conn.Weight
            self.neuron[_out].input.append(self.neuron[_in])
            self.neuron[_in].output.append(self.neuron[_out])
            self.neuron[_in].weight[self.neuron[_out]] = _weight
        
        for i, n in self.neuron.items():
            print(i, ":", n)


    # TODO: fix infinite loop
    def forward(self):
        # funcion iterativa que usa un stack
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
                    queue.insert(0, i)
        
        return [x.value for x in self.outputs]

    def train(self):        # Entrenar el modelo
        pass

    def test(self):         # Probar un modelo ya entrenado
        pass
