from .neat import NEAT
from connection_gene import ConnectionGene
from node_gene import NodeGene
from typing import Self


class Genome:
    def __init__(self, neat: NEAT):
        self.neat = neat

        self.connections: list[ConnectionGene] = []
        self.nodes: list[NodeGene] = []

    def _distance(self, genome2: Self):
        genome1 = self

        highest_innovation_gene_1: int = 0
        if len(genome1.connections) != 1:
            genome1.connections.sort(key=lambda x: x.innovation_number)
            highest_innovation_gene_1 = genome1.connections[-1].innovation_number

        highest_innovation_gene_2: int = 0
        if len(genome1.connections) != 1:
            genome2.connections.sort(key=lambda x: x.innovation_number)
            highest_innovation_gene_2 = genome2.connections[-1].innovation_number

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
            gene1: ConnectionGene = genome1.connections[genome1_index]
            gene2: ConnectionGene = genome2.connections[genome2_index]

            innovation1: int = gene1.innovation_number
            innovation2: int = gene2.innovation_number

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

            N: float = max(len(genome1.connections), len(genome2.connections))
            N = 1 if N < 20 else N

            return self.neat.C1 * (disjoint_genes/N) + self.neat.C2 * (excess_genes/N) + self.neat.C3 * weight_difference

    def _cross_over(self, genome1, genome2):
        pass

    def _mutate(self):
        pass

    def _mutate_link(self):
        pass

    def _mutate_node(self):
        pass

    def _mutate_weight_shift(self):
        pass

    def _mutate_weight_random(self):
        pass

    def _mutate_link_toggle(self):
        pass

    def _get_random_connect(self):
        pass

    def _get_random_node(self):
        pass
