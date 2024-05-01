import gymnasium as gym
import neat
import numpy as np
import os

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def indice_max_probabilidad(probabilidades):
# Calcula la distribución de probabilidad acumulativa (CDF)
    cdf = np.cumsum(probabilidades)
    
    # Genera un número aleatorio en el rango [0, 1]
    aleatorio = np.random.random()
    
    # Encuentra el primer índice en la CDF donde el valor acumulado es mayor o igual al número aleatorio generado
    indice = np.searchsorted(cdf, aleatorio, side='left')
    
    return indice

def eval_genome(genomes, config):
    env = gym.make('ALE/SpaceInvaders-v5', render_mode = "rgb_array")  # Crear el ambiente Space Invaders
    observation = env.reset()  # Reiniciar el ambiente

    for genome_id, genome in genomes:
      #print(len(genomes))
      genome.fitness = 0
      net = neat.nn.FeedForwardNetwork.create(genome, config)  # Crear la red neuronal

      total_reward = 0
      done = False
      obs_ram = env.unwrapped.ale.getRAM()
      #print(obs_ram)
      while not done:
          env.render()
          
          output =  net.activate(obs_ram)  # Activar la red neuronal para obtener la acción
          softmaxed = softmax(output)
          action = indice_max_probabilidad(softmaxed)

          #print(output)
          #print(softmaxed)
          #print(action)

          observation, reward,done,truncated ,info = env.step(action)  # Ejecutar la acción en el ambiente
          obs_ram = env.unwrapped.ale.getRAM()
          total_reward += reward  # Acumular el premio
          #print(total_reward)
        
      genome.fitness = total_reward
      #print(genome.fitness)


def train(config_file, epochs):
    # Cargar configuración NEAT
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    # Crear el población inicial
    p = neat.Population(config)

    # Añadir un reportero para monitorizar el progreso
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Entrenar NEAT
    winner = p.run(eval_genome, epochs)  # Entrenar durante 150 generaciones

    # Mostrar el mejor genoma
    print('\nBest genome:\n{!s}'.format(winner))


config_path = 'NEAT_PYTHON/config.txt'

# Entrenar la red neuronal utilizando NEAT
train(config_path, 1000)