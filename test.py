from NEAT.neat import NEAT
import gymnasium as gym

env = gym.make("SpaceInvaders-v4",render_mode="human")

done = 0
stats,info = env.reset()
obs_ram = env.unwrapped.ale.getRAM()
model = NEAT(128,6,120,1,2,3)
model.load_genomes("results_400")

while not done:
  dict_input = {i: int(valor) for i, valor in enumerate(obs_ram)} 
  model.test(dict_input)
  observation, reward, done, truncated, info = env.step()  # Ejecutar la acción en el ambiente
  obs_ram = env.unwrapped.ale.getRAM()

