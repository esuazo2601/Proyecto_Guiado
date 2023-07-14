from genome import Genome
from node_genes import NodeGenes, Genes
from connection_genes import ConnectionGenes, Connection

import random

class Species():
    def __init__(self, n_species: int, genomes: list[Genome]):
        self.n_species: int = n_species
        self.genomes: list[Genome] = genomes
        self.offsprings: list[Genome] = []

    def fitness_function(self, ):
        pass

    def speciation(self, num_offsprings: int):          # TODO: implementar
                                                        # TODO: proceso de separar
        genome_species: list[Genome] = []

        for i in range(num_offsprings):

            genome1: Genome = random.choice(self.genomes)           #! TEMPORAL
            genome2: Genome = random.choice(self.genomes)           #! TEMPORAL

            genome_species.append(self.cross_over(genome1, genome2))
        
        for offspring in genome_species:                        # Le da la oportunidad a cada retoño de mutar
            offspring.mutate()

        return genome_species
    
    # Recibe dos Genomas, los cuales al cruzarse crearan una red 
    def cross_over(genome1: Genome, genome2: Genome):          # TODO: FIX
        # PENDIENTE: elegir realmente el padre mas apto
        # usar los genes del padre mas apto
        # asumir que padre 1 es el mas apto
        offspring: Genome = Genome()
        
        #! There was a 75% chance that an inherited gene was disabled if it was disabled in either parent.
        
        # TODO: arreglar el resto de la implementacion
        for node in genome1.nodes:
            offspring.nodes.append(node)

        # elegir al azar los matching genes
        # por cada nodo que esta en el padre 1 y padre 2, elegir al azar
        # innovs = [x.innovation for x in genome2.connections]
        innovs = {}
        for x in genome2.connections:
            innovs[x.innovation] = x

        for conn in genome1.connections:
            if conn.innovation in innovs.keys():
                # elegir al azar entre la conexion del padre1 y padre2
                selected = random.choice([conn, innovs[conn.innovation]])
                offspring.connections.append(selected.copy())
            # si no son matching genes, dejar los genes del padre mas apto
            else:
                offspring.connections.append(conn.copy())

        return offspring