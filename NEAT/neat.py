import os
from .connection_genes import ConnectionGenes
from .node_genes import NodeGenes
from .genome import Genome
from .species import Species
import numpy as np
from .utils import indice_max_probabilidad
import pickle
import random
from .neural_network import NeuralNetwork
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

class NEAT(nn.Module):
    def __init__(self, inputSize: int, outputSize: int, populationSize: int, C1: float, C2: float, C3: float):
        super(NEAT,self).__init__()
        self.input_size: int = inputSize
        self.output_size: int = outputSize
        self.population_size: int = populationSize
        self.device = torch.device("cuda")

        self.genomes: list[Genome] = []
        for i in range(populationSize):
            g = Genome(inputSize, outputSize)
            self.genomes.append(g)
        
        self.C1: float = C1
        self.C2: float = C2
        self.C3: float = C3

        self.best_genome: Genome = None

    def evaluate_genome(self,genome):
        env = gym.make("ALE/SpaceInvaders-v5",render_mode = "rgb_array")
        env = FlattenObservation(env)
        network = NeuralNetwork(genome, torch.device("cuda"))  # Move network to GPU
        network.to(torch.device("cuda"))  # Ensure weights and biases are on GPU

        state, info = env.reset()
        done = False
        score = 0
        while not done or truncated:
            # Convert state to tensor and move to GPU
            state_tensor = torch.tensor(state, dtype=torch.float).to(torch.device("cuda"))

            # Forward pass on GPU
            actions = network.forward(state_tensor)

            final_action = indice_max_probabilidad(actions.cpu().detach().numpy())
            final_action = final_action.item()
            #print(final_action)

            n_state, reward, done, truncated, info = env.step(final_action)
            state = n_state
            score += reward

        # Move any calculated gradients back to CPU before returning
        if network.training:
            network.zero_grad()
        genome.fitness = score
        return genome.fitness

    def train(self, epochs: int, goal: float, distance_t: float, output_file: str):
        with open(output_file, "w") as f:
            f.write("epoch;prom_fit;std_dev;max_fit\n")

        print(f"goal: {goal}, epochs: {epochs}, genomes: {len(self.genomes)}")
        best_fit: float = 0

        for episode in range(1, epochs + 1):
            fits_epoch = []
            print(f"episode: {episode}\n")

            if best_fit >= goal:
                self.save_genomes("results_" + str(epochs))
                print(f"Epoch {episode}: Best Fitness: {best_fit}, Goal: {goal}")
                break
            
            for i in range(len(self.genomes)):
                print(f"genome:{i}")
                curr_fit = self.evaluate_genome(self.genomes[i])
                
                if curr_fit >= best_fit:
                    best_fit = curr_fit
                    self.best_genome = self.genomes[i]
                
                fits_epoch.append(curr_fit)

            prom = np.mean(fits_epoch)
            std_dev = np.std(fits_epoch)
            max_fit = np.max(fits_epoch)
            
            ep = str(episode)
            prom = str(prom)
            std_dev = "{:.3f}".format(std_dev)
            max_fit = str(max_fit)

            with open(output_file, 'a') as f:
                f.write(ep + ";" + prom + ";" + std_dev + ";" + max_fit + "\n")
            self.next_generation(distance_t)

    def test(self, _input: dict):
        if self.best_genome is not None:
            network = NeuralNetwork(self.best_genome, self.device)
            network.forward(_input)
        else:
            print("Error")

    def next_generation(self, distance_t: float):
        new_generation: list[Genome] = []
        population_no_crossover = int(self.population_size * .05)
        
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
