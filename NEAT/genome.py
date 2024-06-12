from .connection_genes import ConnectionGenes, Connection
from .node_genes import NodeGenes, Genes
import random
import copy

class Genome:
    def __init__(self, inputSize, outputSize):       
        # Cada Genome dispone de dos elementos:
        self.nodes: NodeGenes = NodeGenes(inputSize, outputSize)  # "Lista" de NodeGenes
        self.connections: ConnectionGenes = ConnectionGenes(self.nodes)  # "Lista" de ConnectionGenes
        self.fitness: float = 0  # Basado en el rendimiento
        self.age = 0
        self.inputSize = inputSize
        self.outputSize = outputSize                                         

    # Decide si mutar o no y qué mutar específicamente
    def mutate(self):
        choice: float = random.random()

        # Elige crear una Nueva Conexion (10%)
        if choice <= 0.1:
            rand_node1: Genes = random.choice(self.nodes.genes)
            rand_node2: Genes = random.choice(self.nodes.genes)
            
            while rand_node1.type == "OUTPUT":  # Nodo puede ser "INPUT" o "HIDDEN"
                rand_node1 = random.choice(self.nodes.genes)
            
            while rand_node1.id == rand_node2.id or rand_node2.type == "INPUT":  # Nodo puede ser "HIDDEN" o "OUTPUT" y distinto del otro
                rand_node2 = random.choice(self.nodes.genes)

            self.connections.mutate_add_connection(rand_node1.id, rand_node2.id)

        # Elige crear un Nuevo Nodo (10%)
        elif choice <= 0.2:
            new_connection = random.choice(list(self.connections.genes.items()))
            new_node = self.connections.mutate_add_node(new_connection[1])
            self.nodes.add_node(new_node)

        # Elige mutar todos los Pesos (80%)
        else:
            for _, conn in self.connections.genes.items():                
                uniform: float = random.random()

                if uniform <= 0.9:  # 90% chance of being uniformly perturbed
                    perturbation: float = random.uniform(-2.0, 2.0)
                    new_weight: float = conn.Weight * perturbation
                else:  # 10% chance of being assigned a new random value
                    new_weight: float = random.uniform(-1.0, 1.0)

                self.connections.mutate_weight(new_weight, conn)

    def copy(self):
        new_genome = Genome(self.inputSize, self.outputSize)
        new_genome.nodes = copy.deepcopy(self.nodes)
        new_genome.connections = copy.deepcopy(self.connections)
        new_genome.fitness = self.fitness
        return new_genome
