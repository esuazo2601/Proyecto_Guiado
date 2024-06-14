import os
from .connection_genes import ConnectionGenes
from .node_genes import NodeGenes
from .genome import Genome
from .species import Species
import numpy as np
import pickle
import random
from .neural_network import NeuralNetwork
import gymnasium as gym
import time
import numpy as np

class NEAT():
    def __init__(self, inputSize: int, outputSize: int, populationSize: int, C1: float, C2: float, C3: float):
        self.env = gym.make("SpaceInvaders-ramDeterministic-v4")
        #self.env = gym.wrappers.TimeLimit(self.env, max_episode_steps=200)
        self.input_size: int = inputSize
        self.output_size: int = outputSize
        self.population_size: int = populationSize

        self.genomes: list[Genome] = []
        for i in range(populationSize):
            g = Genome(inputSize, outputSize)
            self.genomes.append(g)
        
        self.C1: float = C1
        self.C2: float = C2
        self.C3: float = C3
        self.best_fit: float = 0
        self.best_genome: Genome = None

    def evaluate_genome(self, genome):
        network = NeuralNetwork(genome, self.input_size, self.output_size)
        state, info = self.env.reset()
        done = False
        truncated = False  # Inicializar la variable truncated
        score = 0
        
        while not done and not truncated:
            state = {i: state[i] for i in range(len(state))}
            actions = network.forward(state)
            final_action = np.argmax(actions)
            #print(final_action)
            n_state, reward, done, truncated, info = self.env.step(final_action)
            score += reward
            state = n_state
        
        genome.fitness = score
        return genome.fitness


    def train(self, epochs: int, goal: float, distance_t: float, output_file: str):
            with open(output_file, "w") as f:
                f.write("epoch;prom_fit;std_dev;max_fit\n")

            print(f"goal: {goal}, epochs: {epochs}, genomes: {len(self.genomes)}")


            for episode in range(1, epochs + 1):
                fits_epoch = []
                print(f"episode: {episode}\n")

                if self.best_fit >= goal:
                    self.save_genomes("results_" + str(epochs))
                    print(f"Epoch {episode}: Best Fitness: {self.best_fit}, Goal: {goal}")
                    break
                
                count_genomes = len(self.genomes)
                for i in range(count_genomes):
                    print(f"genome:{i}")
                    
                    start = time.time()
                    curr_fit = self.evaluate_genome(self.genomes[i])
                    end = time.time()
                    #print(curr_fit)
                    print(f"TIME: {(end-start)}")
                    
                    if curr_fit > self.best_fit:
                        self.best_fit = curr_fit
                        self.best_genome = self.genomes[i]
                    
                    fits_epoch.append(curr_fit)
                    #print(fits_epoch)

                prom = np.mean(fits_epoch)
                std_dev = np.std(fits_epoch)
                max_fit = np.max(fits_epoch)
                
                ep = str(episode)
                prom = str(prom)
                std_dev = "{:.3f}".format(std_dev)
                max_fit = str(max_fit)

                with open(output_file, 'a') as f:
                    f.write(ep + ";" + prom + ";" + std_dev + ";" + max_fit + "\n")

                if episode%50 == 0:
                    self.save_genomes(f"save_ep:{episode}")
                self.next_generation(distance_t)

    
    def test(self, _input: dict):
        if self.best_genome is not None:
            network = NeuralNetwork(self.best_genome,self.input_size,self.output_size)
            network.forward(_input)
        else:
            print("Error")

    #def next_generation(self, distance_t: float):
    #    new_generation: list[Genome] = []
    #    population_no_crossover = int(self.population_size * .05)
    #    
    #    for i in range(population_no_crossover):
    #        rand_genome = random.choice(self.genomes)
    #        rand_genome.mutate()
    #        new_generation.append(rand_genome)
    #
    #    new_species = Species(distance_t, self.genomes, self.C1, self.C2, self.C3)
    #    
    #    new_generation.extend(new_species.speciation(self.population_size - population_no_crossover))
    #    self.genomes = new_generation 

    def next_generation(self, distance_t: float):
        new_generation: list[Genome] = []
        
        # Se calcula la cantidad de elites
        elitism_part = int(self.population_size * 0.2)
        
        # Se ordenan de acuerdo al fitness y se extrae esa parte
        sorted_genomes = sorted(self.genomes, key=lambda g: g.fitness, reverse=True)
        elites = sorted_genomes[:elitism_part]
        #print(elites)
        
        # Se añaden las elites a la generación
        new_generation.extend(elites)
        
        # Generar nuevos genomas a través de mutación
        population_mutated = int(self.population_size * 0.05)
        for _ in range(population_mutated):
            rand_genome = random.choice(self.genomes)
            new_genome = rand_genome.copy()  # Copiar el genoma antes de mutar
            new_genome.mutate()
            new_generation.append(new_genome)
        
        # Crear nuevas especies y añadirlas a la nueva generación
        new_species = Species(distance_t, self.genomes, self.C1, self.C2, self.C3)
        new_generation.extend(new_species.speciation(self.population_size - len(new_generation)))
        
        # Asegurarse de que la nueva generación tiene el tamaño correcto
        while len(new_generation) < self.population_size:
            rand_genome = random.choice(self.genomes)
            new_genome = rand_genome.copy()  # Copiar el genoma antes de mutar
            new_genome.mutate()
            new_generation.append(new_genome)
        
        # Actualizar self.genomes con la nueva generación
        self.genomes = new_generation


    def save_genomes(self, name: str):
        if not os.path.isdir("./saved_model"):
            os.makedirs("./saved_model")

        with open(f'./saved_model/{name}.pkl', 'wb') as file:
            pickle.dump(self, file)

    def load_genomes(self, name: str):
        if os.path.isdir("./saved_model") and os.path.isfile(f"./saved_model/{name}.pkl"):
            with open(f'./saved_model/{name}.pkl', 'rb') as file:
                model = pickle.load(file)

            self.input_size = model.input_size
            self.output_size = model.output_size
            self.population_size = model.population_size
            self.genomes = model.genomes
            self.C1 = model.C1
            self.C2 = model.C2
            self.C3 = model.C3
            self.best_genome = model.best_genome
            print("Loaded genomes correctly")
        else:
            print("Error")

    def _get_connection(self, node1: NodeGenes, node2: NodeGenes):
        c = ConnectionGenes(node1, node2)
        if c in self.all_connections:
            c.innovation_number = self.all_connections[self.all_connections.index(c)].innovation_number
        else:
            c.innovation_number = len(self.all_connections) + 1
            self.all_connections.append(c)
        return c

    def _get_node(self, id=None):
        if id and id <= len(self.all_nodes):
            return self.all_nodes[id - 1]

        n = NodeGenes(len(self.all_nodes) + 1)
        self.all_nodes.append(n)
        return n