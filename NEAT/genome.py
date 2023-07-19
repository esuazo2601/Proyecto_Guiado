from .connection_genes import ConnectionGenes, Connection
from .node_genes import NodeGenes, Genes
import random

class Genome():
    def __init__(self, inputSize, outputSize):                                  # Cada Genome dispone de dos elementos:
        self.nodes: NodeGenes = NodeGenes(inputSize, outputSize)                # "Lista" de NodeGenes
        self.connections: ConnectionGenes = ConnectionGenes(self.nodes)         # "Lista" de ConnectionGenes
        self.fitness: float = 0                                                 # Basado en el rendimiento

    # Decide si mutar o no y que mutar especificamente
    def mutate(self):
        choice: float = random.randrange(0, 100) * 0.01

        # Elige crear una Nueva Conexion (%10)
        if choice <= .1:
            rand_node1: Genes = random.choice(self.nodes.genes)
            rand_node2: Genes = random.choice(self.nodes.genes)
            
            while rand_node1.type == "OUTPUT":                                  # Nodo puede ser "INPUT" o "HIDDEN"
                rand_node1 = random.choice(self.nodes.genes)
            
            while rand_node1.id is rand_node2.id or rand_node2.type == "INPUT": # Nodo puede ser "HIDDEN" o "OUTPUT" y distinto del otro
                rand_node2 = random.choice(self.nodes.genes)

            #print(rand_node1.type, rand_node2.type)

            self.connections.mutate_add_connection(rand_node1.id, rand_node2.id)

        # Elige crear un Nuevo Nodo (%10)
        elif choice <= .2:
            new_connection = random.choice(list(self.connections.genes.items()))

            #print(new_connection[0])

            new_node = self.connections.mutate_add_node(new_connection[1])
            self.nodes.add_node(new_node)

        # Elige mutar todos los Pesos (%80)
        else:
            for _, conn in self.connections.genes.items():                
                uniform: float = random.randrange(0, 100) * 0.01

                if uniform <= .9: # 90% chance of being uniformly perturbed
                    perturbation: float = random.randrange(-20, 20) * 0.1
                    while perturbation == .0:
                        perturbation = random.randrange(-20, 20) * 0.1
                    new_weight: float = conn.Weight * perturbation
                
                else:   # 10% chance of being assigned a new random value.
                    new_weight: float = random.randrange(-100, 100) * 0.01
                    while new_weight == .0:
                        new_weight = random.randrange(-100, 100) * 0.01 

                self.connections.mutate_weight(new_weight, conn)