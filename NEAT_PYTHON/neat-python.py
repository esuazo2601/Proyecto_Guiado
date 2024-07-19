import gymnasium as gym
import neat
import numpy as np
import os
import time

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

class MyReporter(neat.StatisticsReporter):
    def __init__(self, filename):
        super().__init__()

        # Nombre del archivo CSV
        self.filename = filename
        self.generation = 0
        with open(self.filename, 'a') as f:
          f.write("gen;prom_fitness;std_dev\n")

    def post_evaluate(self, config, population, species, best_genome):
        # Obtener el fitness de cada genoma en la población actual
        fitness_generacion_actual = [c.fitness for c in population.values()]

        # Calcular el promedio y la desviación estándar del fitness de la generación actual
        promedio = np.mean(fitness_generacion_actual)
        promedio = round(promedio,2)
        desviacion_estandar = np.std(fitness_generacion_actual)
        desviacion_estandar = round(desviacion_estandar,2)

        # Escribir la información en el archivo CSV
        with open(self.filename, 'a') as f:
            f.write(f'{self.generation};{promedio};{desviacion_estandar}\n')

        # Llamar al método de la clase base para mantener la funcionalidad original
        super().post_evaluate(config, population, species, best_genome)

def eval_genome(genomes, config):
    #start = time.time()
    env = gym.make('SpaceInvaders-ram-v4', render_mode = None)  # Crear el ambiente Space Invaders
    for genome_id, genome in genomes:
        state,_ = env.reset()  # Reiniciar el ambiente
        #print("Genome: ", genome_id)
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)  # Crear la red neuronal

        total_reward = 0
        done = False
        
        while not done:
            #env.render()
            output =  net.activate(state)  # Activar la red neuronal para obtener la acción
            softmaxed = softmax(output)
            action = np.argmax(softmaxed)
            n_state, reward, done, truncated, info = env.step(action)  # Ejecutar la acción en el ambiente
            state = n_state
            total_reward += reward  # Acumular el premio
        
        genome.fitness = total_reward
        #end = time.time()
        #print(f"TIME: {(end-start)}")

def train(config_file):
    # Cargar configuración NEAT
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    # Crear el población inicial
    p = neat.Population(config)
    my_reporter = MyReporter("fitness_history_1.txt")  # Iniciar el generador en la generación 0
    

    # Añadir un reportero para monitorizar el progreso
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(my_reporter)
    p.add_reporter(neat.Checkpointer(100))
    #stats = neat.StatisticsReporter()
    
    #p.add_reporter(stats)

    # Entrenar NEAT
    for generation in range(300):
        my_reporter.generation = generation  # Actualizar el número de generación en MyReporter
        winner = p.run(eval_genome, 1)  # Entrenar arg2 generaciones

    # Mostrar el mejor genoma
    print('\nBest genome:\n{!s}'.format(winner))

config_path = 'NEAT_PYTHON/config.txt'

# Entrenar la red neuronal utilizando NEAT
if __name__ == '__main__':
    train(config_path)