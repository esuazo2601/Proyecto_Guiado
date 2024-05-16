import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
import multiprocessing
from NEAT.neat import NEAT

if __name__ == '__main__':
    env = gym.make("ALE/SpaceInvaders-v5",render_mode="rgb_array")
    env = FlattenObservation(env)
    actions = env.action_space.n
    # 210 160 3 4 
    model = NEAT(
        inputSize=216,
        outputSize=actions,
        populationSize=100,
        C1=1.0,
        C2=2.0,
        C3=3.0
    )

    # Define la función de entrenamiento y entrena el modelo
    model.train(
        env=env,
        epochs=300,
        goal=1000,
        distance_t=0.3,
        output_file="fitness_history_1.txt"
    )