import os
from .connection_genes import ConnectionGenes, Connection
from .node_genes import NodeGenes
from .genome import Genome
from .species import Species
from .neural_network import NeuralNetwork
from .convolutional_layer import CNN
import matplotlib.pyplot as plt

import pickle
import random
import gymnasium as gym

# TODO: Elegir cual de los dos utilizar para procesamiento del renderizado del ambiente
#import tensorflow
import torch
import tensordict

class NEAT():
    def __init__(self, inputSize: int, outputSize: int, populationSize: int, C1: float, C2: float, C3: float):
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

        self.best_genome: Genome

    # Encargada de probar las redes creadas y actualizar el valor fitness de cada genoma
    def train(self, env, epochs: int, goal: float, distance_t: float):
        height, width, channels = env.observation_space.shape

        best_fit: float = 0
        for episode in range(1, epochs+1):

            if best_fit >= goal:
                self.save_genomes("results_" + str(epochs))
                print("Epoch {}: Best Fitness: {}, Goal: {}", episode, best_fit, goal)
                break

            for i in range(len(self.genomes)):
                network: NeuralNetwork = NeuralNetwork(self.genomes[i])
                state,dict = env.reset()
                done = False
                score = 0 

                while not done:
                    env.render()
                    
                    state_to_tensor = torch.tensor(state,dtype=torch.float)
                    state_to_tensor = state_to_tensor.unsqueeze(0)
                    permuted_tensor = state_to_tensor.permute(0, 3, 1, 2)
                    conv_layer = CNN()
                    output = conv_layer(permuted_tensor)
                    #print(output)
                    #output_array = output.squeeze(0).detach().numpy()

                    # Visualizar la salida de la red convolucional
                    """      plt.figure(figsize=(8, 8))
                    plt.imshow(output_array[0], cmap='gray')  # Muestra solo el primer canal
                    plt.axis('off')
                    plt.show() """



                    # Crea un diccionario donde las claves son enteros y los valores son los elementos de la lista
                    #flattened_output = flattened_output[flattened_output != 0]
                    #lista_tensor = flattened_output.tolist()
                    #resultado = {i: int(valor) for i, valor in enumerate(lista_tensor)} 
                    #print(len(resultado))
                    #action = network.forward(resultado)
                    
                    #print("Actions: ",action)
                    # Tomar la acción en el entorno y obtener la siguiente observación y recompensa
                    n_state, reward, done, truncated, info = env.step()
                    state = n_state
                    score += reward

                self.genomes[i].fitness = score

                if self.genomes[i].fitness > best_fit:
                    best_fit = self.genomes[i].fitness
                    self.best_genome = self.genomes[i]

            self.next_generation(distance_t)
            print("Epoch {}: Best Fitness: {}, Goal: {}", episode, best_fit, goal)


    # Encargada de probar el rendimiento del mejor genoma
    def test(self, _input: dict):
        if self.best_genome is not None:
            pass
            network: NeuralNetwork = NeuralNetwork(self.best_genome)
            network.forward(_input)

        else:
            print("Error")

    # Separa los procesos para generar la siguiente generación de Genomas
    def next_generation(self, distance_t: float):
        """
        Se encarga de evolucionar los Genomas luego de probar las redes creadas.
        Primero genera una poblacion que solo muta Genomas aleatorios de la poblacion original.
        Luego se preocupa de rellenar el resto de la poblacion con Genomas producto de de la cruza entre los dos con mejor compatibilidad.
        Finalmente les aplica una mutacion aleatoria.
        """
        new_generation: list[Genome] = []
        #! In each generation, 25% of offspring resulted from mutation without crossover
        population_no_crossover = int(self.population_size * .25)
        
        for i in range(population_no_crossover):
            rand_genome = random.choice(self.genomes)
            rand_genome.mutate()
            new_generation.append(rand_genome)

        new_species: Species = Species(distance_t, self.genomes, self.C1, self.C2, self.C3)
        
        new_generation.extend(
            new_species.speciation(self.population_size - population_no_crossover)) # Agrega los retoños que se generen de la especiacion

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
        
            self.best_genome = model.best_genome

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
