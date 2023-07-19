from .genome import Genome
from .node_genes import NodeGenes, Genes
from .connection_genes import ConnectionGenes, Connection
from .innovation import Innovation as inn_num

import random

class Species():
    def __init__(self, distance_t: float, genomes: list[Genome], C1: float, C2: float, C3: float):
        self.distance_t = distance_t                # TODO: definir distance_t en neat.py
        self.C1: float = C1                         # Valor C1 para calcular fitness
        self.C2: float = C2                         # Valor C2 para calcular fitness
        self.C3: float = C3                         # Valor C3 para calcular fitness
        self.genomes: list[Genome] = genomes
        self.offsprings: list[Genome] = []
        self.gen_dist: dict[(Genome, Genome)] = {}  # Memorization
    
    def adjusted_fitness(self, i: Genome):
        summation: int = 0
        for j in self.genomes:
            if i is not j:
                summation += self.sh(i, j)
        return i.fitness / summation

    # The sharing function sh is set to 0 when distance(i, j) is above the threshold distance_t; otherwise, sh(distance(i; j)) is set to 1
    def sh(self, i: Genome, j: Genome): # Sharing Function
        return 0 if self.distance(i, j) > self.distance_t else 1
    
    def distance(self, genome1: Genome, genome2: Genome):   # The distance from every other organism j in the population
        if self.gen_dist.get(genome1, genome2) is not None: # Se agrega un diccionario para ayudar en 
            return self.gen_dist[genome1, genome2]

        conn_parent1: dict[int] = {}
        for _, conn1 in genome1.connections.genes.items():
            conn_parent1[conn1.Innovation] = conn1
        conn_size1 = len(conn_parent1)
        nodes_size1 = len(genome1.nodes.genes)

        conn_parent2: dict[int] = {}
        for _, conn2 in genome2.connections.genes.items():
            conn_parent2[conn2.Innovation] = conn2
        conn_size2 = len(conn_parent2)
        nodes_size2 = len(genome2.nodes.genes)

        is_excess = True
        parent = 1 if conn_size1 > conn_size2 else 2
        weight_counter: int = 0

        E: int = 0                                                          # E: Number of Excess Genes
        D: int = 0                                                          # D: Number of Disjoint Genes
        N: int = nodes_size1 if nodes_size1 > nodes_size2 else nodes_size2  # N: The number of Genes in the larger Genome
        W: float = .0                                                       # W: The average Weight differences of matching Genes

        p1 = max(conn_parent1.keys())
        p2 = max(conn_parent2.keys())

        n = p1 if p1 >= p2 else p2

        for i in range(n, 0, -1):
            conn_in_1: Connection = conn_parent1.get(i) # Recibe una Conexion o None
            conn_in_2: Connection = conn_parent2.get(i) # Recibe una Conexion o None
            
            if ((conn_in_1 != None and conn_in_2 == None and is_excess and parent == 1) or      # Conexion existe y quedan Genes de Exceso
                (conn_in_1 == None and conn_in_2 != None and is_excess and parent == 2)):
                E += 1

            elif ((conn_in_1 == None and conn_in_2 != None and is_excess and parent == 1) or    # Conexion existe, pero ya no quedan Genes de Exceso
                  (conn_in_1 == None and conn_in_2 == None and is_excess and parent == 2)):
                is_excess = False
                D += 1
            
            elif ((conn_in_1 != None and conn_in_2 == None) or                          # Conexion existe y no quedan Genes de Exceso
                  (conn_in_1 == None and conn_in_2 != None)) and not is_excess:         # Solo quedan Genes Disjuntos
                D += 1

            else:                                                                           # Conexion existe en ambos Padres
                is_excess = False                                                           # En caso de que aun no se haya deshabilitado esta opcion
                weight_counter += 1
                W += abs(conn_in_1.Weight - conn_in_2.Weight)                               # Diferencia de Pesos

        W /= weight_counter

        value = self.C1*(E/N) + self.C2*(D/N) + self.C3*W
        self.gen_dist[genome1, genome2] = value
        self.gen_dist[genome2, genome1] = value

        return value
    
    # Recibe dos Genomas, los cuales al aparearse crearan un nuevo Genoma
    def cross_over(self, parent1: Genome, parent2: Genome):
        offspring: Genome = Genome(0, 0)
        
        conn_parent1: dict[int] = {}
        for _, conn1 in parent1.connections.genes.items():
            conn_parent1[conn1.Innovation] = conn1
        conn_size1 = len(conn_parent1)
        nodes_size1 = len(parent1.nodes.genes)

        conn_parent2: dict[int] = {}
        for _, conn2 in parent2.connections.genes.items():
            conn_parent2[conn2.Innovation] = conn2
        conn_size2 = len(conn_parent2)
        nodes_size2 = len(parent2.nodes.genes)


        new_genes: dict[(int, int)] = {}
        
        p1 = max(conn_parent1.keys())
        p2 = max(conn_parent2.keys())

        n = p1 if p1 >= p2 else p2

        for i in range(n, 0, -1):
            conn_in_1 = conn_parent1.get(i)                                 # Recibe una Conexion o None
            conn_in_2 = conn_parent2.get(i)                                 # Recibe una Conexion o None
            
            if conn_in_1 != None and conn_in_2 == None:                     # Conexion existe en el Padre 1
                new_genes[conn_in_1.Input, conn_in_1.Output] = conn_in_1

            elif conn_in_1 == None and conn_in_2 != None:                   # Conexion existe en el Padre 2
                new_genes[conn_in_2.Input, conn_in_2.Output] = conn_in_2

            else:                                                           # Conexion existe en ambos Padres
                new_genes[conn_in_1.Input, conn_in_1.Output] = conn_in_1
                new_genes[conn_in_1.Input, conn_in_1.Output].Weight = (conn_in_1.Weight + conn_in_2.Weight) / 2
                
                #! There was a 75% chance that an inherited gene was disabled if it was disabled in either parent.
                if (conn_in_1.Enabled and not conn_in_1.Enabled) or (not conn_in_1.Enabled and conn_in_2.Enabled):
                    new_genes[conn_in_1.Input, conn_in_1.Output].Enabled = False if random.randrange(0, 100) * 0.01 <= .75 else True
        
        #* Agrega las Conexiones al Retonyo
        offspring.connections.genes = new_genes

        #* Agrega los Nodos al Retonyo
        if nodes_size1 >= nodes_size2:
            offspring.nodes.genes = parent1.nodes.genes
            offspring.connections.node_count = nodes_size1
        else:
            offspring.nodes.genes = parent2.nodes.genes
            offspring.connections.node_count = nodes_size2

        #* Agrega el promedio de ambos Padres al Fitness del Retonyo
        offspring.fitness = (parent1.fitness + parent2.fitness) / 2
        
        #* Retorna el Genoma resultante
        return offspring
    
    def speciation(self, num_offsprings: int):          # TODO: implementar
                                                        # TODO: proceso de separar
        genome_species: list[Genome] = []

        for i in range(num_offsprings):

            genome1: Genome = random.choice(self.genomes)           #! TEMPORAL
            genome2: Genome = random.choice(self.genomes)           #! TEMPORAL

            genome_species.append(self.cross_over(genome1, genome2))
        
        for offspring in genome_species:                        # Le da la oportunidad a cada retonyo de mutar
            offspring.mutate()

        return genome_species