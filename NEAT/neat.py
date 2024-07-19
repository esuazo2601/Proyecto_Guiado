import os
from .connection_genes import ConnectionGenes, Connection
from .node_genes import NodeGenes
from .genome import Genome
from .species import Species
from .neural_network import NeuralNetwork
import numpy as np
from multiprocessing import Pool, cpu_count
import pickle
import random
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import graphviz

def indice_max_probabilidad(probabilidades):
# Calcula la distribución de probabilidad acumulativa (CDF)
    cdf = np.cumsum(probabilidades)
    
    # Genera un número aleatorio en el rango [0, 1]
    aleatorio = np.random.random()
    
    # Encuentra el primer índice en la CDF donde el valor acumulado es mayor o igual al número aleatorio generado
    indice = np.searchsorted(cdf, aleatorio, side='left')
    
    return indice

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
        self.env = gym.make('SpaceInvaders-ramDeterministic-v4',render_mode=None)

    #Codigo para evaluar cada genoma
    def _evaluate_genome(self, genome):
        env = self.env
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            #Entrada de la red neuronal
            dict_input = {i: int(valor) for i, valor in enumerate(state)}
            net_out = NeuralNetwork(genome).forward(dict_input)
            #print(net_out)
            #Accion de salida
            action = np.argmax(net_out)
            n_state, reward, terminated, truncated, _ = env.step(action)
            state = n_state
            score += reward
            done = terminated or truncated

        return score

    #Función que evalúa los genomas en "paquetes"
    def _evaluate_genomes_batch(self, genomes_batch):
        with Pool(10) as pool:
                fitness_values = pool.map(self._evaluate_genome, genomes_batch)
        return fitness_values

    #Función que finalmente se encarga de evaluar a los genomas en los paquetes antes mencionados
    def _evaluate_genomes(self):
        fitness_values = np.zeros(self.population_size)
        for i in range(0, self.population_size, self.batch_size):
            end = min(i + self.batch_size, self.population_size)
            genomes_batch = self.genomes[i:end]
            print(f"Processing genomes {i + 1} to {end}")
            batch_fitness = self._evaluate_genomes_batch(genomes_batch)
            fitness_values[i:end] = batch_fitness
        return fitness_values

    def train(self, epochs, goal, distance_t, output_file):
        print(f"Obj: {goal}, episodes: {epochs}\n")
        with open(output_file, "w") as f:
            f.write("epoch;prom_fit;std_dev;best\n")

        best_overall = -np.inf  # Initialize best fitness to negative infinity

        #Se entrena en un rango de episodios determinado
        for episode in range(1, epochs + 1):
            print("Episode ", episode)
            #Por cada genoma se crea una red neuronal
            for genome in self.genomes:
                genome.network = NeuralNetwork(genome)

            fitness_values = self._evaluate_genomes()

            best_of_epoch = None
            best_fitness_of_epoch = -np.inf

            for genome, fitness_value in zip(self.genomes, fitness_values):
                genome.fitness = fitness_value
                if fitness_value > best_fitness_of_epoch:
                    best_of_epoch = genome
                    best_fitness_of_epoch = fitness_value

            #Cálculo del promedio, la desviación estándar y el mejor de la generación
            prom = np.mean(fitness_values)
            std_dev = np.std(fitness_values)
            current_best_fit = best_fitness_of_epoch

            if current_best_fit > best_overall:
                best_overall = current_best_fit
                self.best_genome = best_of_epoch  # Quedarse con el mejor genoma

            #Escritura en el archivo de stats
            with open(output_file, 'a') as f:
                f.write(f"{str(episode)};{str(prom)};{std_dev:.3f};{str(current_best_fit)}\n")

            print(f"Best fitness so far: {best_overall}") 

            #Toma de checkpoints cada 20 episodios o bien cuando se llega al objetivo
            if episode % 20 == 0 and episode != 0:
                self.save_genomes(f"checkpoint_{episode}")
            if best_overall >= goal:
                self.save_genomes(f"results_{best_overall}")
                break
            
            #Siguiente generación
            self.next_generation(distance_t=distance_t, mutation_rate=0.25)


    def test(self, _input: dict):
        if self.best_genome is not None:
            network = NeuralNetwork(self.best_genome)
            network.forward(_input)
        else:
            print("Error")

    # def next_generation(self, distance_t: float):
    #     new_generation = []
    #     population_no_crossover = int(self.population_size * .15)

    #     for i in range(population_no_crossover):
    #         rand_genome = random.choice(self.genomes)
    #         rand_genome.mutate()
    #         new_generation.append(rand_genome)

    #     new_species = Species(distance_t, self.genomes, self.C1, self.C2, self.C3)
    #     new_generation.extend(new_species.speciation(self.population_size - population_no_crossover))
    #     self.genomes = new_generation

    def next_generation(self, distance_t: float, mutation_rate: float):
        new_generation = []

        # Cantidad de elites
        num_elites = max(1, int(self.population_size * 0.20))  # Ajustar porcentaje

        # Ordenar en orden descendiente de fitness
        sorted_genomes = sorted(self.genomes, key=lambda x: x.fitness, reverse=True)

        # preservar los mejores para la siguiente generacion
        elites = sorted_genomes[:num_elites]
        new_generation.extend(elites)

        # Ejecutar mutaciones
        num_to_mutate = int(self.population_size * mutation_rate)
        genomes_to_mutate = random.sample(sorted_genomes[num_elites:], num_to_mutate)

        for genome in genomes_to_mutate:
            genome.mutate()
            new_generation.append(genome)

        # Generar el resto mediante cruza
        new_species = Species(distance_t, self.genomes, self.C1, self.C2, self.C3)
        new_generation.extend(new_species.speciation(self.population_size - len(new_generation)))
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
    

    def visualize_network(self, genome, filename):
        output_dir = './network_visualizations'
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        dot = graphviz.Digraph()
        node_positions = {}

        # Add input nodes
        for i in range(1, self.input_size + 1):
            node_id = i
            dot.node(str(node_id), f'Input {node_id}', shape='box', style='filled', color='lightblue')
            node_positions[node_id] = (0, -i)

        # Add output nodes
        for i in range(1, self.output_size + 1):
            node_id = self.input_size + i
            dot.node(str(node_id), f'Output {node_id}', shape='box', style='filled', color='lightgreen')
            node_positions[node_id] = (1, -i)

        # Add hidden nodes and connections
        for key, connection in genome.connections.genes.items():
            if connection.Enabled:
                in_node = connection.Input
                out_node = connection.Output

                if in_node not in node_positions:
                    node_positions[in_node] = (random.uniform(0.1, 0.9), random.uniform(-self.input_size, self.output_size))

                if out_node not in node_positions:
                    node_positions[out_node] = (random.uniform(0.1, 0.9), random.uniform(-self.input_size, self.output_size))

                dot.edge(str(in_node), str(out_node), label=f'{connection.Weight:.2f}')

        dot.attr(rankdir='LR')  # This changes the layout to left-to-right instead of top-to-bottom
        dot.render(filepath, format='png')
        print(f"Network visualization saved to {filepath}.png")