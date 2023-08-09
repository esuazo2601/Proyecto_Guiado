import os
from .connection_genes import ConnectionGenes, Connection
from .node_genes import NodeGenes
from .genome import Genome
from .species import Species
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

    # Separa los procesos para generar la siguiente generación de Genomas
    def next_generation(self, n_species: int):
        new_generation: list[Genome] = []
        #! In each generation, 25% of offspring resulted from mutation without crossover
        population_no_crossover = int(self.population_size * .25)
        
        for i in range(population_no_crossover):
            rand_genome = random.choice(self.genomes)
            rand_genome.mutate()
            new_generation.append(rand_genome)

        new_species: Species = Species(n_species)
        
        new_generation.extend(
            new_species.speciation(self.population_size - population_no_crossover), # Agrega los retoños que se generen de la especiacion
                                    self.genomes)                                   # Entrega los Genomas a ordenar

        self.genomes = new_generation                                               # Reemplaza los anteriores Genomas

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
