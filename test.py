from NEAT.neat import NEAT
import gymnasium as gym
import numpy as np
from NEAT.neural_network import NeuralNetwork


def indice_max_probabilidad(probabilidades):
# Calcula la distribución de probabilidad acumulativa (CDF)
    cdf = np.cumsum(probabilidades)
    
    # Genera un número aleatorio en el rango [0, 1]
    aleatorio = np.random.random()
    
    # Encuentra el primer índice en la CDF donde el valor acumulado es mayor o igual al número aleatorio generado
    indice = np.searchsorted(cdf, aleatorio, side='left')
    
    return indice

model = NEAT(
        inputSize=128,
        outputSize=6,
        populationSize=150,
        C1=1.2,
        C2=2.3,
        C3=3.5,
)

model.load_genomes("results_885.0")

print(model.best_genome.fitness)
print(model.best_genome)
net = NeuralNetwork(model.best_genome)

env = gym.make("SpaceInvaders-ramDeterministic-v4",render_mode="human")

state, info = env.reset()
done = False
truncated = False  # Inicializar la variable truncated
score = 0

model.visualize_network(model.best_genome,"best 885")

while not done and not truncated:
    state = {i: state[i] for i in range(len(state))}
    actions = net.forward(state)
    #print(actions)
    final_action = indice_max_probabilidad(actions)
    #print(final_action)
    n_state, reward, done, truncated, info = env.step(final_action)
    score += reward
    state = n_state

print(score)