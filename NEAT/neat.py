import os
from connection_genes import ConnectionGenes
from node_genes import NodeGenes
from genome import Genome
import pickle
import random

class NEAT:
    def __init__(self, inputSize, outputSize, populationSize, C1, C2, C3):
        self.input_size: int = inputSize            # Cantidad de nodos de entrada
        self.output_size: int = outputSize          # Cantidad de nodos de salida
        self.population_size: int = populationSize  # Cantidad maxima de genomas por generacion

        self.genomes: list[Genome] = []             # Lista de Genomas
        for i in range(populationSize):             # Itera hasta llenar la lista con la poblacion de genomas
            g = Genome(inputSize, outputSize)       # Crea un genoma nuevo
            self.genomes.append(g)                  # Agrega el genoma al listado

        self.C1: float = C1                         # Valor C1 para calcular fitness
        self.C2: float = C2                         # Valor C2 para calcular fitness
        self.C3: float = C3                         # Valor C3 para calcular fitness

    def speciation(self):
        pass

    def _distance(genome1: Genome, genome2: Genome):
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

        return genome1.neat.C1 * (disjoint_genes/N) + genome1.neat.C2 * (excess_genes/N) + genome1.neat.C3 * weight_difference

    def _cross_over(self):          # TODO: FIX
        # PENDIENTE: elegir realmente el padre mas apto
        # usar los genes del padre mas apto
        # asumir que padre 1 es el mas apto
        offspring: list[Genome] = []

        #! In each generation, 25% of offspring resulted from mutation without crossover
        population_no_crossover = int(self.population_size * .25)
        
        for i in range(population_no_crossover):
            pass
            rand_genome = random.choice(self.genomes)
            rand_genome.mutate()
            offspring.append(rand_genome)
        
        #! There was a 75% chance that an inherited gene was disabled if it was disabled in either parent.
        # TODO: arreglar el resto de la implementacion
        for i in range(self.population_size - population_no_crossover):
            for node in genome1.nodes:
                child.nodes.append(node)

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
                    child.connections.append(selected.copy())
                # si no son matching genes, dejar los genes del padre mas apto
                else:
                    child.connections.append(conn.copy())

            return child

    # Guardar red en un archivo .pkl
    def save_genomes(self, name: str):
        if os.path.isdir("./saved_model"):
            with open('./saved_model/{}.pkl', 'wb', name) as file:
                pickle.dumps(self, file)
        else:
            print("Error")

    # Cargar red desde un archivo .pkl
    def load_genomes(self, name: str):
        if os.path.isdir("./saved_model") and os.path.isfile("./saved_model/{}.pkl", name):
            with open('./saved_model/{}.pkl', 'rb', name) as file:
                model = pickle.dumps(self, file)

            self.input_size = model.inputSize              # Cantidad de nodos de entrada
            self.output_size = model.outputSize            # Cantidad de nodos de salida
            self.population_size = model.populationSize    # Cantidad maxima de genomas por generacion

            self.genomes = model.genomes                   # Lista de Genomas

            self.C1 = model.C1                             # Valor C1 para calcular fitness
            self.C2 = model.C2                             # Valor C2 para calcular fitness
            self.C3 = model.C3                             # Valor C3 para calcular fitness
        
        else:
            print("Error")

    def _get_connection(self, node1: NodeGenes, node2: NodeGenes):
        c = ConnectionGenes(node1, node2)
        if c in self.all_connections:
            c.innovation_number = self.all_connections[self.all_connections.index(c)].innovation_number
        else:
            c.innovation_number = len(self.all_connections) + 1     #! HAY QUE ACTUALIZAR
            self.all_connections.append(c)
        return c

    def _get_node(self, id=None):
        if id and id <= len(self.all_nodes):
            return self.all_nodes[id - 1]

        n = NodeGenes(len(self.all_nodes) + 1)
        self.all_nodes.append(n)
        return n
    
    def evaluate(self):
        pass
        for network in self.genomes:        # Evaluar cada genoma
            pass

    def evaluateGenome(self):
        pass