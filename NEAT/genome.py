from .neat import NEAT
from connection_gene import ConnectionGene
from node_gene import NodeGene
from typing import Self
import random


class Genome:
    
    def __init__(self, neat: NEAT):
        self.neat = neat

        self.connections: list[ConnectionGene] = []
        self.nodes: list[NodeGene] = []
        self.innovation: int = 1

    def _distance(self, genome2: Self):
        genome1 = self

        highest_innovation_gene_1: int = 0
        if len(genome1.connections) != 1:
            genome1.connections.sort(key=lambda x: x.innovation)
            highest_innovation_gene_1 = genome1.connections[-1].innovation

        highest_innovation_gene_2: int = 0
        if len(genome1.connections) != 1:
            genome2.connections.sort(key=lambda x: x.innovation)
            highest_innovation_gene_2 = genome2.connections[-1].innovation

        if highest_innovation_gene_1 < highest_innovation_gene_2:
            g = genome1
            genome1 = genome2
            genome2 = g

        genome1_index: int = 0
        genome2_index: int = 0

        disjoint_genes: int = 0
        excess_genes: int = 0
        weight_difference: float = 0
        similar_genes: int = 0

        while genome1_index < len(genome1.connections) and genome2_index < len(genome2.connections):
            gene1 = genome1.connections[genome1_index]
            gene2 = genome2.connections[genome2_index]

            innovation1 = gene1.innovation
            innovation2 = gene2.innovation

            if innovation1 == innovation2:
                similar_genes += 1
                weight_difference += abs(gene1.weight - gene2.weight)
                genome1_index += 1
                genome2_index += 1

            elif innovation1 > innovation2:
                disjoint_genes += 1
                innovation2 += 1

            elif innovation2 > innovation1:
                disjoint_genes += 1
                innovation1 += 1

        weight_difference /= similar_genes
        excess_genes = len(genome1.connections) - genome1_index
        
        # N can be set to 1 if both genomes are small, i.e., consist of fewer than 20 gene
        N = max(len(genome1.connections), len(genome2.connections))
        N = 1 if N < 20 else N

        return self.neat.C1 * (disjoint_genes/N) + self.neat.C2 * (excess_genes/N) + self.neat.C3 * weight_difference

    def _cross_over(self, genome1, genome2):
        pass

    # There was a 75% chance that an inherited gene was disabled if it was disabled in either parent.
    # In each generation, 25% of offspring resulted from mutation without crossover
    def _mutate(self):
        pass

    # a single new connection gene with a random weight is added connecting two previously unconnected nodes
    def _mutate_link(self):
        # elegir dos nodos que no esten conectados
        in_node = random.choice(self.nodes)
        out_node = random.choice(self.nodes)
        conns = [(x.in_node, x.out_node) for x in self.connections]
        
        # si la conexion ya existe, elegir nuevos nodos al azar
        while (in_node, out_node) in conns or (out_node, in_node) in conns:
            in_node = random.choice(self.nodes)
            out_node = random.choice(self.nodes)
        
        # crear conexion y agregarla a la lista de conecciones
        new_connection = ConnectionGene(in_node, out_node, random.random(), self.innovation)
        self.innovation += 1
        self.connections.append(new_connection)

    # an existing connection is split and the new node placed where the old connection used to be
    # The old connection is disabled and two new connections are added to the genome
    # The new connection leading into the new node receives a weight of 1, and the new connection
    # leading out receives the same weight as the old connection
    def _mutate_node(self):
        conn = random.choice(self.connections)
        new_node = NodeGene(len(self.nodes), "HIDDEN")
        conn.enabled = False
        
        new_connection1 = ConnectionGene(conn.in_node, new_node, 1, self.innovation)
        self.innovation += 1
        new_connection2 = ConnectionGene(new_node, conn.out_node, conn.weight, self.innovation)
        self.innovation += 1
        
        self.connections.append(new_connection1)
        self.connections.append(new_connection2)
        self.nodes.append(new_node)
        

    # There was an 80% chance of a genome having its connection weights mutated,
    # in which case each weight had a 90% chance of being uniformly perturbed
    # and a 10% chance of being assigned a new random value.
    def _mutate_weight_shift(self):
        pass

    def _mutate_weight_random(self):
        pass

    # There was a 75% chance that an inherited gene was disabled if it was disabled
    # in either parent. In each generation, 25% of offspring resulted from mutation without crossover
    def _mutate_link_toggle(self):
        pass

    def _get_random_gene(self):
        pass
