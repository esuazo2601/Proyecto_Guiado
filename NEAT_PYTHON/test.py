import neat
import pickle
import gymnasium
import numpy as np
import os
import gzip

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

# Ruta del archivo gzip del mejor genoma
best_genome_path = 'neat-checkpoint-284'

# Cambiar el directorio de trabajo al directorio donde se encuentra el script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Verificar el directorio de trabajo actual
print(f"Directorio de trabajo actual: {os.getcwd()}")

# Verificar si el archivo gzip existe
if not os.path.exists(best_genome_path):
    raise FileNotFoundError(f"No se encontró el archivo: {best_genome_path}")

# Cargar el mejor genoma desde el archivo gzip
with gzip.open(best_genome_path, 'rb') as f:
    best_genome = pickle.load(f)

# Ruta del archivo de configuración
config_path = 'config'  # Asegúrate de poner la ruta correcta a tu archivo de configuración

# Verificar si el archivo de configuración existe
if not os.path.exists(config_path):
    raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")

# Cargar el entorno Gym
env = gymnasium.make('SpaceInvaders-ramDeterministic-v4', render_mode='human')

# Configurar la red neural usando el genoma y la configuración
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_path)
net = neat.nn.FeedForwardNetwork.create(best_genome, config)

# Evaluar el genoma en el entorno
observation, info = env.reset()
done = False
total_reward = 0

while not done:
    env.render()
    # Preparar la observación para la red neural
    observation = observation.flatten()  # Aplanar la observación si es necesario

    # Obtener la acción de la red neural
    output = net.activate(observation)
    softmaxed = softmax(output)
    action = indice_max_probabilidad(softmaxed)

    # Ejecutar la acción en el entorno
    observation, reward, done, info = env.step(action)
    total_reward += reward

print(f'Total Reward: {total_reward}')
env.close()
