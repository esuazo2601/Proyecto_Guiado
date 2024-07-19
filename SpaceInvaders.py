from NEAT.neat import NEAT
#import gymnasium as gym
#from gymnasium.wrappers import FlattenObservation
#import numpy as np

if __name__ == '__main__':

    model = NEAT(
        inputSize=128,
        outputSize=6,
        populationSize=150,
        C1=1.2,
        C2=2.3,
        C3=3.5,
    )
    model.load_genomes("checkpoint_40")
    model.train(
        #env=env,
        epochs=200,
        goal=850,
        distance_t=2,
        output_file="stats.txt"
    )
