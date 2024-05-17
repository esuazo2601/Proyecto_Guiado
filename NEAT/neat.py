import os
from .connection_genes import ConnectionGenes, Connection
from .node_genes import NodeGenes
from .genome import Genome
from .species import Species
from .neural_network import NeuralNetwork
import numpy as np
from multiprocessing import Pool, cpu_count
from .utils import indice_max_probabilidad
import pickle
import random
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation


gym.logger.set_level(50)

class NEAT:
    def __init__(self, inputSize: int, outputSize: int, populationSize: int, C1: float, C2: float, C3: float):
        self.input_size = inputSize
        self.output_size = outputSize
        self.population_size = populationSize
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.best_genome = None
        self.batch_size = 10
        self.genomes = [Genome(inputSize, outputSize) for _ in range(populationSize)]

    def _make_env(self):
        return gym.make('SpaceInvaders-v4', render_mode='rgb_array')

    def _evaluate_genome(self, genome):
        env = self._make_env()
        state, _ = env.reset()
        score = 0
        done = False
        obs_ram = env.unwrapped.ale.getRAM()
        while not done:
            #dict_input = {j: state[j] for j in range(216)}
            dict_input = {i: int(valor) for i, valor in enumerate(obs_ram)} 
            
            action = np.argmax(genome.network.forward(dict_input))
            state, reward, terminated, truncated, _ = env.step(action)
            obs_ram = env.unwrapped.ale.getRAM()
            score += reward
            done = terminated or truncated

        env.close()
        return score

    def _evaluate_genomes_batch(self, genomes_batch):
        with Pool(cpu_count()) as pool:
            fitness_values = pool.map(self._evaluate_genome, genomes_batch)
        return fitness_values

    def _evaluate_genomes(self):
        fitness_values = np.zeros(self.population_size)
        for i in range(0, self.population_size, self.batch_size):
            end = min(i + self.batch_size, self.population_size)
            genomes_batch = self.genomes[i:end]
            print(f"Evaluating genomes {i} to {end-1}")
            batch_fitness = self._evaluate_genomes_batch(genomes_batch)
            fitness_values[i:end] = batch_fitness
        return fitness_values

    def train(self, epochs, goal, distance_t, output_file):
        with open(output_file, "w") as f:
            f.write("epoch;prom_fit;std_dev;best\n")

        best_fit = 0

        for episode in range(1, epochs + 1):

            for genome in self.genomes:
                genome.network = NeuralNetwork(genome)

            fitness_values = self._evaluate_genomes()

            for genome, fitness_value in zip(self.genomes, fitness_values):
                genome.fitness = fitness_value

            prom = np.mean(fitness_values)
            std_dev = np.std(fitness_values)
            best_fit = max(fitness_values)

            with open(output_file, 'a') as f:
                f.write(f"{episode};{prom};{std_dev:.3f};{best_fit}\n")

            print(f"Best fitness in epoch {episode}: {best_fit}")

            print(f"Epoch {episode}")
            if best_fit >= goal:
                self.save_genomes("results_" + str(epochs))
                break
            self.next_generation(distance_t)

    def test(self, _input: dict):
        if self.best_genome is not None:
            network = NeuralNetwork(self.best_genome)
            network.forward(_input)
        else:
            print("Error")

    def next_generation(self, distance_t: float):
        new_generation = []
        population_no_crossover = int(self.population_size * .25)
        
        for i in range(population_no_crossover):
            rand_genome = random.choice(self.genomes)
            rand_genome.mutate()
            new_generation.append(rand_genome)

        new_species = Species(distance_t, self.genomes, self.C1, self.C2, self.C3)
        new_generation.extend(new_species.speciation(self.population_size - population_no_crossover))
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
        else:
            print("Error loading model")

    def _get_connection(self, node1, node2):
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