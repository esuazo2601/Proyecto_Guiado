from .neat import NEAT
from connection_genes import ConnectionGenes, Connection
from node_genes import NodeGenes, Genes
from innovation import Innovation
import random

#! MUY IMPORTANTE: El InnovationNumber lo conocen TODOS los Genomes de la Poblacion
class Genome():
    def __init__(self, inputSize, outputSize):                                  # Cada Genome dispone de dos elementos:
        self.nodes: NodeGenes = NodeGenes(inputSize, outputSize)                # "Lista" de NodeGenes
        self.connections: ConnectionGenes = ConnectionGenes(self.nodes)         # "Lista" de ConnectionGenes
        self.fitness: float = 0                                                 # Basado en el rendimiento

    # Decide si mutar o no y que mutar especificamente
    def mutate(self):
        choice = random.random()

        # Elige crear una Nueva Conexion (%10)
        if choice <= .10:
            rand_node1: Genes = random.choice(self.nodes.genes)
            rand_node2: Genes = random.choice(self.nodes.genes)
            
            while rand_node1 == rand_node2 and rand_node1.type == "OUTPUT" and rand_node2.type == "INPUT":
                rand_node1 = random.choice(self.nodes)
                rand_node2 = random.choice(self.nodes)
            
            fix = 0                                                                 # TODO: implementar innovation_number
            self.connections.mutate_add_connection(rand_node1, rand_node2, fix)

        # Elige crear un Nuevo Nodo (%10)
        elif choice <= .20:
            new_connection: Connection = random.choice(self.connections)
            new_node = self.connections.mutate_add_node(new_connection)
            self.nodes.add_node(new_node)

        # Elige mutar todos los Pesos (%10)
        else:
            for conn in self.connections.genes:                
                uniform = random.random()

                if uniform > .9: # in which case each weight had a 90% chance of being uniformly perturbed
                    for conn in self.connections:
                        perturbation = random.randrange(-2.0, 2.0, 0.01)
                        new_weight = conn.Weight * perturbation
                
                else:   # and a 10% chance of being assigned a new random value.
                    for conn in self.connections:
                        new_weight = random.random()

                self.connections.mutate_weight(new_weight, conn)